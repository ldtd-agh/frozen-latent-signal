"""UMAP visualisation of the fine-tuned embeddings (Task 5), colored by
ADC-derived tumor cell count -- the fine-tuned counterpart to
`scripts/plot_tcc_umap.py`'s frozen-embedding figure.

Every one of the 7 fine-tuning runs (BiomedCLIP, MedSigLIP, CLIP,
NeuroVFM LoRA; Merlin, Med3D, MedImageInsights projection
heads -- see RESULTS.md) only trained one 80/20 patient split (or, for
MedImageInsights, five independent such splits for the
robustness check), so only ~20% of patients ever got a *held-out*
fine-tuned embedding from any single run. Showing "the whole population"
with any one of those runs would mean showing ~80% of points through a
model that was trained ON them -- of course they'd look artificially
coherent.

So this trains `K_FOLDS` (default 5) folds per backbone via
`finetune.trainer.k_fold_patient_splits()`, which -- unlike 5 independent
random 80/20 draws -- partitions patients into disjoint groups that
together cover every patient exactly once. An earlier version of this
figure then pooled all 5 folds' held-out embeddings into one scatter per
backend to show the entire 332-scan population. That pooling turned out to
be its own confound: each fold is an independently-trained adapter with no
mechanism tying its embedding coordinate system to any other fold's, and a
k-NN-purity-by-fold check found this dominates several backbones' pooled
plots (e.g. Merlin: 100% of a point's nearest neighbours share its fold;
NeuroVFM: 98%) -- most of the "several small disconnected clusters"
visible in that version were fold identity, not TCC structure. This
version instead shows only one fold per backend (`_select_representative_fold`),
chosen as whichever fold's own Geary's $C$ is closest to that backend's
5-fold mean -- a single trained model, so no cross-model mixing is
possible, at the cost of showing only that fold's ~60-70 held-out scans
rather than all 332.

Reuses each backbone's exact fine-tuning recipe (temperature, lr, target
column, encode_fn) from its own script rather than re-deriving them, so
this is guaranteed consistent with the numbers already reported in
RESULTS.md.

**Coloring is a separate concern from the training target and must not be
confused with it.** BiomedCLIP/MedSigLIP/CLIP train against `slice_tcc`
(their mechanistically correct target -- unchanged, still what Table
"finetuning_summary" reports), but `slice_tcc` and `tcc` (Merlin/Med3D's
target) are structurally different quantities -- log10(slice_tcc) sits
roughly 1.4 lower than log10(tcc) for the same scan, since one slice's
cell count is necessarily far smaller than the whole volume's. Naively
coloring every panel by its own training target and sharing one
`vmin`/`vmax` across all 5 (as an earlier version of this script did)
silently caps the slice_tcc-colored panels at ~54% of the color scale --
no point in those three panels could ever appear yellow, regardless of
what fine-tuning did to the embeddings. This is a plotting artifact, not
a real difference in the underlying data: every panel is now recolored by
the *same* target (`COLOR_TARGET_COLUMN`, matching `scripts/plot_tcc_umap.py`'s
convention for the frozen figure) via `_uniform_color_values()`, which
re-derives that column's values in the exact per-scan row order the
backbone's own dataset construction produced -- verified by an assertion
that the reconstructed patient-id sequence matches exactly, not assumed.

Writes:
  results/figures/tcc_umap_finetuned.png / .pdf
  results/figures/tcc_umap_finetuned_<backend>.png
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from transformers import AutoProcessor, SiglipModel

from analysis.cellularity import build_coherence_dataset, build_knn_graph, gearys_c, permutation_test
from analysis.reduction import reduce
from analysis.visualise import plot_embedding_by_value, plot_embedding_grid
from data.datasets.cfb_gbm import CFBGBMDataset
from embeddings.store import EmbeddingStore
from finetune.data import CFBGBMSliceDataset, EmbeddingDataset
from finetune.lora import wrap_visual_tower_with_lora
from finetune.losses import soft_neighbor_tcc_loss
from finetune.neurovfm_data import NeuroVFMTokenDataset, build_token_cache, forward_pooled, pack_batch
from finetune.neurovfm_trainer import NeuroVFMConfig
from finetune.projection_head import ProjectionHead, projection_head_encode_fn
from finetune.trainer import TrainConfig, default_encode_fn, encode_indices, k_fold_patient_splits, train

from neurovfm.pipelines import load_encoder

import scripts.finetune_cfb_gbm as biomedclip_script
import scripts.finetune_clip_cfb_gbm as clip_script
import scripts.finetune_medsiglip_cfb_gbm as medsiglip_script
import scripts.finetune_neurovfm_cfb_gbm as neurovfm_script
import scripts.finetune_projection_head_cfb_gbm as proj_head_script

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent
STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
FIGURES_DIR = Path("results/figures")

K_FOLDS = 5
FOLD_SEED = 42
UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.1, random_state=42)
COLORBAR_LABEL = r"$\log_{10}$(TCC)"
GRID_N_COLS = 4  # 8 backends -> full 2x4 grid

# Coloring target for every panel -- independent of each backend's own
# training target (see module docstring). Matches scripts/plot_tcc_umap.py's
# convention for the frozen figure (every panel colored by the same
# quantity, for direct comparability).
COLOR_TARGET_COLUMN = "tcc"

BACKEND_LABELS = {
    "biomedclip": "BiomedCLIP (LoRA)",
    "medsiglip": "MedSigLIP (LoRA)",
    "clip": "CLIP (LoRA)",
    "merlin": "Merlin (proj. head)",
    "med3d": "Med3D (proj. head)",
    "medimageinsights": "MedImageInsights (proj. head)",
    "neurovfm": "NeuroVFM (LoRA)",
    "imagenet_vit": "ImageNet ViT (LoRA)",
}

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _uniform_color_values(table: pd.DataFrame, training_target_column: str, reference_patient_ids: list[str]) -> np.ndarray:
    """
    Re-derive ``COLOR_TARGET_COLUMN`` values, in the exact per-scan row
    order a ``CFBGBMSliceDataset(..., target_column=training_target_column)``
    construction over the same ``table`` produced -- so a backend trained
    against one target (e.g. ``slice_tcc``) can still be colored by a
    different, uniform one (``tcc``) without any risk of silently
    mismatching scans to the wrong color.

    Safe by construction, not by assumption: this replays the identical
    ``table.dropna/filter -> groupby("patient_id") -> iterrows()``
    traversal ``CFBGBMSliceDataset``/``build_coherence_dataset`` use
    internally, then asserts the resulting patient-id sequence matches
    ``reference_patient_ids`` element-for-element before returning
    anything. If CFB-GBM's ``tcc``/``slice_tcc`` validity sets ever
    diverge (checked to be identical, 332/332, at the time this was
    written) this raises immediately instead of producing a subtly
    mis-colored figure.
    """
    filtered = table.dropna(subset=[training_target_column])
    filtered = filtered[filtered[training_target_column] > 0]

    pids: list[str] = []
    colors: list[float] = []
    for pid, group in filtered.groupby("patient_id"):
        for _, row in group.iterrows():
            pids.append(pid)
            colors.append(row[COLOR_TARGET_COLUMN])

    if pids != list(reference_patient_ids):
        raise AssertionError(
            f"Recoloring row-order mismatch: reconstructed {len(pids)} patient ids don't match "
            f"the {len(reference_patient_ids)} the dataset actually used -- do not use this output, "
            f"investigate why 'tcc' and '{training_target_column}' no longer share the same valid-scan set."
        )
    return np.log10(np.asarray(colors, dtype=np.float64))


def _oof_open_clip_vit(
    backend_name: str, model_name: str, pretrained: "str | None", target_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    K-fold out-of-fold embeddings for an open_clip-based ViT (BiomedCLIP,
    CLIP). ``pretrained=None`` for BiomedCLIP's hf-hub identifier (carries
    its own weights, see `scripts/finetune_cfb_gbm.py`); CLIP needs
    ``pretrained="openai"`` explicitly (see `scripts/finetune_clip_cfb_gbm.py`).

    Returns ``(X, color_values, patient_ids)`` -- ``color_values`` is
    ``COLOR_TARGET_COLUMN``, not ``target_column`` (see module docstring);
    training still happens against ``target_column`` exactly as before.
    """
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb_dataset = CFBGBMDataset(
        IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", projection_2d=True, slice_selection="gtv_mask",
    )
    if backend_name == "imagenet_vit":
        from embeddings.backends.imagenet_vit import load_imagenet_vit
        model0, preprocess = load_imagenet_vit(model_name)
    else:
        kwargs = {"pretrained": pretrained} if pretrained is not None else {}
        model0, preprocess = open_clip.create_model_from_pretrained(model_name, **kwargs)
    dataset = CFBGBMSliceDataset(cfb_dataset, table, preprocess, target_column=target_column)
    color_values = _uniform_color_values(table, target_column, dataset.patient_ids)

    folds = k_fold_patient_splits(dataset.patient_ids, K_FOLDS, FOLD_SEED)
    x_parts, color_parts, pid_parts = [], [], []
    for i, (train_ids, val_ids) in enumerate(folds):
        model = copy.deepcopy(model0)
        model.visual = wrap_visual_tower_with_lora(model.visual, backend_name)
        config = TrainConfig(backend_name=backend_name, target_column=target_column)
        result = train(config, dataset, model, default_encode_fn, override_split=(train_ids, val_ids))
        val_idx = [j for j, pid in enumerate(dataset.patient_ids) if pid in val_ids]
        x_parts.append(encode_indices(model, dataset, val_idx, config.device, default_encode_fn))
        color_parts.append(color_values[val_idx])
        pid_parts.extend(dataset.patient_ids[j] for j in val_idx)
        log.info(
            "%-12s fold %d/%d  n_val=%3d  fold C=%.4f",
            backend_name, i + 1, K_FOLDS, len(val_idx), result["final"]["observed"],
        )

    return np.concatenate(x_parts), np.concatenate(color_parts), pid_parts


def _oof_medsiglip() -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb_dataset = CFBGBMDataset(
        IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", projection_2d=True, slice_selection="gtv_mask",
    )
    processor = AutoProcessor.from_pretrained(medsiglip_script.MODEL_NAME)

    def preprocess(pil_img):
        return processor(images=pil_img, return_tensors="pt")["pixel_values"][0]

    dataset = CFBGBMSliceDataset(cfb_dataset, table, preprocess, target_column=medsiglip_script.TARGET_COLUMN)
    color_values = _uniform_color_values(table, medsiglip_script.TARGET_COLUMN, dataset.patient_ids)

    folds = k_fold_patient_splits(dataset.patient_ids, K_FOLDS, FOLD_SEED)
    x_parts, color_parts, pid_parts = [], [], []
    for i, (train_ids, val_ids) in enumerate(folds):
        model = SiglipModel.from_pretrained(medsiglip_script.MODEL_NAME)
        model.vision_model = wrap_visual_tower_with_lora(model.vision_model, "medsiglip")
        config = TrainConfig(
            backend_name="medsiglip", target_column=medsiglip_script.TARGET_COLUMN,
            temperature=medsiglip_script.TEMPERATURE, batch_size=16,
        )
        result = train(
            config, dataset, model, medsiglip_script.encode_fn, override_split=(train_ids, val_ids),
        )
        val_idx = [j for j, pid in enumerate(dataset.patient_ids) if pid in val_ids]
        x_parts.append(encode_indices(model, dataset, val_idx, config.device, medsiglip_script.encode_fn))
        color_parts.append(color_values[val_idx])
        pid_parts.extend(dataset.patient_ids[j] for j in val_idx)
        log.info(
            "medsiglip    fold %d/%d  n_val=%3d  fold C=%.4f",
            i + 1, K_FOLDS, len(val_idx), result["final"]["observed"],
        )

    return np.concatenate(x_parts), np.concatenate(color_parts), pid_parts


def _oof_projection_head(backend_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    store = EmbeddingStore(STORE_DIR / f"cfb_gbm_{backend_name}.h5")
    x, y, patient_ids = build_coherence_dataset(
        store, table, target_column=proj_head_script.TARGET_COLUMN, log_transform=True,
    )
    dataset = EmbeddingDataset(x, y, patient_ids)
    # Merlin/Med3D already train against COLOR_TARGET_COLUMN ("tcc"), so this
    # is a no-op in value -- applied anyway for a single consistent code path
    # (see _uniform_color_values's docstring) rather than an unchecked
    # assumption that stays correct only by luck if either constant changes.
    color_values = _uniform_color_values(table, proj_head_script.TARGET_COLUMN, dataset.patient_ids)

    folds = k_fold_patient_splits(dataset.patient_ids, K_FOLDS, FOLD_SEED)
    x_parts, color_parts, pid_parts = [], [], []
    for i, (train_ids, val_ids) in enumerate(folds):
        model = ProjectionHead(dim=x.shape[1])
        config = TrainConfig(
            backend_name=backend_name, target_column=proj_head_script.TARGET_COLUMN,
            temperature=proj_head_script._TEMPERATURE[backend_name], lr=proj_head_script._LEARNING_RATE,
        )
        result = train(
            config, dataset, model, projection_head_encode_fn, checkpoint_key="",
            override_split=(train_ids, val_ids),
        )
        val_idx = [j for j, pid in enumerate(dataset.patient_ids) if pid in val_ids]
        x_parts.append(encode_indices(model, dataset, val_idx, config.device, projection_head_encode_fn))
        color_parts.append(color_values[val_idx])
        pid_parts.extend(dataset.patient_ids[j] for j in val_idx)
        log.info(
            "%-12s fold %d/%d  n_val=%3d  fold C=%.4f",
            backend_name, i + 1, K_FOLDS, len(val_idx), result["final"]["observed"],
        )

    return np.concatenate(x_parts), np.concatenate(color_parts), pid_parts


def _oof_neurovfm() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    K-fold out-of-fold embeddings for NeuroVFM. Structurally distinct from
    every other `_oof_*` function because NeuroVFM's token-level pipeline
    (ragged `[N_tokens, 1024]` per scan, packed batches, segment-mean
    pooling -- see `finetune/neurovfm_data.py`) doesn't fit
    `finetune.trainer.train()`'s fixed-size-`DataLoader` assumption, the
    same reason `scripts/finetune_neurovfm_cfb_gbm.py` writes its own
    training loop rather than reusing `train()`/`encode_indices()`. This
    mirrors that script's loop directly (same optimizer, loss, early
    stopping, eval cadence) but repeats it over `K_FOLDS` disjoint folds
    instead of one 80/20 split, and returns pooled embeddings for each
    fold's held-out scans instead of just a coherence statistic.
    """
    cfg = NeuroVFMConfig()  # same defaults scripts/finetune_neurovfm_cfb_gbm.py and its robustness check use

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb_dataset = CFBGBMDataset(
        IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", modalities=["t1gd"],
        require_adc=True, require_gtv=True,
    )
    encoder0, preproc = load_encoder("mlinslab/neurovfm-encoder")
    device = encoder0.device
    norm_module = encoder0.norm_module

    cache = build_token_cache(cfb_dataset, IMAGING_ROOT, preproc, neurovfm_script.CACHE_PATH)
    dataset = NeuroVFMTokenDataset(cfb_dataset, table, cache, target_column=neurovfm_script.TARGET_COLUMN)
    # NeuroVFM's training target ("tcc") already equals COLOR_TARGET_COLUMN,
    # so this is a no-op in value -- applied anyway for the same single
    # consistent code path _oof_projection_head uses (see
    # _uniform_color_values's docstring).
    color_values = _uniform_color_values(table, neurovfm_script.TARGET_COLUMN, dataset.patient_ids)

    def _evaluate_x(model, indices: list[int]) -> np.ndarray:
        model.eval()
        embs = []
        with torch.no_grad():
            for start in range(0, len(indices), cfg.eval_batch_size):
                chunk = indices[start : start + cfg.eval_batch_size]
                batch = pack_batch(dataset, chunk)
                pooled = forward_pooled(model, norm_module, batch, device)
                embs.append(pooled.cpu().numpy())
        return np.concatenate(embs, axis=0)

    def _evaluate_c(model, indices: list[int], n_permutations: int) -> float:
        x = _evaluate_x(model, indices)
        y = dataset.targets_t[indices].numpy()
        pids = [dataset.patient_ids[i] for i in indices]
        w = build_knn_graph(x, k=cfg.k, patient_ids=pids, exclude_same_patient=True)
        return permutation_test(y, w, statistic="gearys_c", n_permutations=n_permutations, seed=0)["observed"]

    folds = k_fold_patient_splits(dataset.patient_ids, K_FOLDS, FOLD_SEED)
    x_parts, color_parts, pid_parts = [], [], []
    rng = np.random.default_rng(cfg.seed)
    for i, (train_ids, val_ids) in enumerate(folds):
        train_idx = [j for j, pid in enumerate(dataset.patient_ids) if pid in train_ids]
        val_idx = [j for j, pid in enumerate(dataset.patient_ids) if pid in val_ids]

        model = wrap_visual_tower_with_lora(copy.deepcopy(encoder0.model), "neurovfm")
        model.to(device)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

        best_c = _evaluate_c(model, val_idx, cfg.n_permutations_during_training)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
        epochs_without_improvement = 0

        for epoch in range(1, cfg.max_epochs + 1):
            model.train()
            perm = rng.permutation(train_idx)
            n_batches = len(perm) // cfg.batch_size
            for b in range(n_batches):
                batch_idx = perm[b * cfg.batch_size : (b + 1) * cfg.batch_size].tolist()
                batch = pack_batch(dataset, batch_idx)
                pooled = forward_pooled(model, norm_module, batch, device)
                y = dataset.targets_t[batch_idx].to(device)
                pids = [dataset.patient_ids[j] for j in batch_idx]
                loss = soft_neighbor_tcc_loss(pooled, y, pids, temperature=cfg.temperature)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_c = _evaluate_c(model, val_idx, cfg.n_permutations_during_training)
            if val_c < best_c:
                best_c = val_c
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= cfg.patience:
                    break

        model.load_state_dict(best_state, strict=False)
        fold_x = _evaluate_x(model, val_idx)
        fold_c = _evaluate_c(model, val_idx, cfg.n_permutations_final)
        x_parts.append(fold_x)
        color_parts.append(color_values[val_idx])
        pid_parts.extend(dataset.patient_ids[j] for j in val_idx)
        log.info("neurovfm     fold %d/%d  n_val=%3d  fold C=%.4f", i + 1, K_FOLDS, len(val_idx), fold_c)

    return np.concatenate(x_parts), np.concatenate(color_parts), pid_parts


CACHE_DIR = FIGURES_DIR / "_oof_cache"


def _cached_oof(backend: str, compute_fn) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Per-backend disk cache for `_oof_*`'s (x, color_values, patient_ids)
    output. Each of the 8 backbones takes on the order of 10-20 minutes to
    train (5 folds); without this, re-running `main()` to add or fix one
    backend silently re-trains all of them from scratch every time (there
    is no other persisted artifact -- the final PNGs/PDFs aren't enough to
    rebuild the grid figure), which is both wasteful and risky on a
    time-limited node (a kill mid-run loses every backend computed so far,
    not just the one in progress). Not a correctness concern either way --
    this only skips recomputation when a complete prior result already
    exists on disk; it never changes what gets computed.
    """
    cache_path = CACHE_DIR / f"{backend}.npz"
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)  # pids are saved as a dtype=object string array
        log.info("%-12s loaded from cache (%s)", backend, cache_path)
        return data["x"], data["color"], list(data["pids"])

    x, color, pids = compute_fn()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, x=x, color=color, pids=np.asarray(pids, dtype=object))
    return x, color, pids


def _select_representative_fold(
    x: np.ndarray, color: np.ndarray, pids: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str], int, float]:
    """
    Pooling all 5 out-of-fold models' embeddings into one scatter plot
    confounds TCC structure with model identity: each fold is an
    independently-trained adapter with its own, unaligned embedding
    coordinate system, and a k-NN-purity-by-fold check found this
    dominates the pooled UMAP for most backbones (e.g. Merlin: 100% of a
    point's nearest neighbours share its fold; NeuroVFM: 98%) -- the
    "several small disconnected clusters" in earlier versions of this
    figure were mostly fold identity, not TCC.

    Instead, show only one fold per backbone -- a single trained model, so
    no cross-model mixing is possible -- chosen as whichever of the 5
    folds' own (same-patient-excluded, k=8) Geary's $C$ is closest to that
    backbone's 5-fold mean, so it's representative rather than cherry-picked
    for a flattering or unflattering look. Costs population coverage (one
    fold's ~60-70 held-out scans, not all 332) in exchange for a figure
    that isn't partly an artifact of how many models happened to be pooled.
    """
    unique_pids = sorted(set(pids))
    folds = k_fold_patient_splits(unique_pids, K_FOLDS, FOLD_SEED)
    pid_to_fold = {pid: i for i, (_, val_ids) in enumerate(folds) for pid in val_ids}
    fold_id = np.array([pid_to_fold[p] for p in pids])

    fold_cs = np.array([
        gearys_c(
            color[fold_id == i],
            build_knn_graph(x[fold_id == i], k=8, patient_ids=[p for p, m in zip(pids, fold_id == i) if m], exclude_same_patient=True),
        )
        for i in range(K_FOLDS)
    ])
    chosen = int(np.argmin(np.abs(fold_cs - fold_cs.mean())))

    mask = fold_id == chosen
    return x[mask], color[mask], [p for p, m in zip(pids, mask) if m], chosen, float(fold_cs[chosen])


def main() -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    oof_results: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}

    log.info("=== BiomedCLIP (LoRA), %d folds ===", K_FOLDS)
    oof_results["biomedclip"] = _cached_oof("biomedclip", lambda: _oof_open_clip_vit(
        "biomedclip", biomedclip_script.MODEL_NAME, None, biomedclip_script.TARGET_COLUMN,
    ))

    log.info("=== MedSigLIP (LoRA), %d folds ===", K_FOLDS)
    oof_results["medsiglip"] = _cached_oof("medsiglip", _oof_medsiglip)

    log.info("=== CLIP (LoRA), %d folds ===", K_FOLDS)
    oof_results["clip"] = _cached_oof("clip", lambda: _oof_open_clip_vit(
        "clip", clip_script.MODEL_NAME, clip_script.PRETRAINED, clip_script.TARGET_COLUMN,
    ))

    log.info("=== ImageNet ViT (LoRA; BiomedCLIP's init), %d folds ===", K_FOLDS)
    from embeddings.backends.imagenet_vit import DEFAULT_MODEL_NAME as _IMAGENET_VIT
    oof_results["imagenet_vit"] = _cached_oof("imagenet_vit", lambda: _oof_open_clip_vit(
        "imagenet_vit", _IMAGENET_VIT, None, biomedclip_script.TARGET_COLUMN,
    ))

    log.info("=== NeuroVFM (LoRA), %d folds ===", K_FOLDS)
    oof_results["neurovfm"] = _cached_oof("neurovfm", _oof_neurovfm)

    log.info("=== Merlin (projection head), %d folds ===", K_FOLDS)
    oof_results["merlin"] = _cached_oof("merlin", lambda: _oof_projection_head("merlin"))

    log.info("=== MedImageInsights (projection head), %d folds ===", K_FOLDS)
    oof_results["medimageinsights"] = _cached_oof(
        "medimageinsights", lambda: _oof_projection_head("medimageinsights"),
    )

    log.info("=== Med3D (projection head), %d folds ===", K_FOLDS)
    oof_results["med3d"] = _cached_oof("med3d", lambda: _oof_projection_head("med3d"))

    # Pick one representative fold per backend (see _select_representative_fold's
    # docstring) and UMAP-reduce only that fold's embeddings, instead of
    # pooling all 5 folds together.
    panels: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    for backend, (x, y, pids) in oof_results.items():
        x_fold, y_fold, pids_fold, chosen, chosen_c = _select_representative_fold(x, y, pids)
        log.info(
            "%-12s fold=%d (C=%.4f, closest to 5-fold mean)  n=%d  running UMAP …",
            backend, chosen, chosen_c, len(y_fold),
        )
        coords = reduce(x_fold, method="umap", n_components=2, **UMAP_KWARGS)
        panels.append((backend, coords, y_fold, len(y_fold)))

    vmin = min(y.min() for _, _, y, _ in panels)
    vmax = max(y.max() for _, _, y, _ in panels)

    for backend, coords, y, n in panels:
        fig = plot_embedding_by_value(
            coords, y,
            title=f"{BACKEND_LABELS.get(backend, backend)} (n={n})",
            value_label=COLORBAR_LABEL,
            vmin=vmin, vmax=vmax,
            output_path=FIGURES_DIR / f"tcc_umap_finetuned_{backend}.png",
        )
        plt.close(fig)

    grid_panels = [(f"{BACKEND_LABELS.get(b, b)} (n={n})", coords, y) for b, coords, y, n in panels]
    fig = plot_embedding_grid(
        grid_panels,
        n_cols=GRID_N_COLS,
        vmin=vmin, vmax=vmax,
        value_label=COLORBAR_LABEL,
        suptitle="UMAP projection of fine-tuned embeddings, all eight backbones (one representative out-of-fold split each), colored by ADC-derived tumor cell count",
        output_path=FIGURES_DIR / "tcc_umap_finetuned.png",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
