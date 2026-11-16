"""
Nested cross-validated fine-tuning (review M2): checkpoint selection and
reporting on disjoint patients.

The original protocol (5 random 80/20 splits, `*_robustness_check.py`) and
the out-of-fold pass (`plot_finetuned_umap.py`) both select the best epoch
on the same held-out patients they then report C on, so the reported
fine-tuned C is optimistically biased and Delta <= 0 holds by construction
(epoch 0 = frozen model is a candidate). Here:

  outer loop   5 patient-grouped folds (`k_fold_patient_splits`, seed 42 --
               the same folds as the existing out-of-fold analyses). The
               outer test fold is never seen during training or selection.
  inner split  the outer-train patients are split 80/20 (patient-level)
               into inner-train (gradient updates) and inner-validation
               (early stopping / best-checkpoint selection, same Geary's C
               criterion as before).
  reporting    frozen and selected-checkpoint embeddings of the outer test
               fold; C is computed from them in `analyze_nested_cv.py`.

Every backbone keeps exactly the hyperparameters of its original
robustness check (temperature, lr, batch size, epochs, patience). Epoch 0
is still a candidate on the inner-validation split, but that no longer
forces Delta <= 0 on the test fold.

Writes one file per backbone: results/nested_cv/<backend>.npz with, for
each outer fold i: x_frozen_i, x_ft_i (test-fold embeddings), y_i (training
target, log10), pids_i, and per-fold metadata (selected epoch, inner-val C).
Already-finished backbones are skipped (delete the npz to recompute).

Usage (GPU):
    module load CUDA Python
    source .venv/bin/activate
    export PYTHONPATH=<path/to/neurovfm-repo>:$PYTHONPATH
    python scripts/finetune_nested_cv.py [backend ...]     # default: all eight
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.cellularity import build_coherence_dataset, build_knn_graph, gearys_c, permutation_test  # noqa: E402
from data.datasets.cfb_gbm import CFBGBMDataset  # noqa: E402
from embeddings.store import EmbeddingStore  # noqa: E402
from finetune.data import CFBGBMSliceDataset, EmbeddingDataset  # noqa: E402
from finetune.lora import wrap_visual_tower_with_lora  # noqa: E402
from finetune.trainer import (  # noqa: E402
    TrainConfig, default_encode_fn, encode_indices, k_fold_patient_splits, patient_split, train,
)

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent
STORE_DIR = _PROJECT_ROOT / "stores" / "cfb_gbm_scan"
TCC_TABLE_PATH = _PROJECT_ROOT / "results" / "tcc_table.csv"
OUT_DIR = _PROJECT_ROOT / "results" / "nested_cv"

K_OUTER = 5
OUTER_SEED = 42           # same outer folds as the existing out-of-fold analyses
INNER_VAL_FRACTION = 0.2
INNER_SEED_OFFSET = 1000  # inner split seed = INNER_SEED_OFFSET + fold index

ALL_BACKENDS = ["medsiglip", "medimageinsights", "biomedclip", "neurovfm", "clip", "merlin", "med3d", "imagenet_vit"]

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("nested_cv")


def _folds(patient_ids: list[str]):
    """Yield (fold, inner_train_ids, inner_val_ids, test_ids)."""
    for i, (outer_train, test_ids) in enumerate(k_fold_patient_splits(patient_ids, K_OUTER, OUTER_SEED)):
        inner_train, inner_val = patient_split(sorted(outer_train), INNER_VAL_FRACTION, INNER_SEED_OFFSET + i)
        assert not (inner_train & inner_val) and not ((inner_train | inner_val) & test_ids)
        yield i, inner_train, inner_val, test_ids


def _quick_c(x: np.ndarray, y: np.ndarray, pids: list[str]) -> float:
    return float(gearys_c(y, build_knn_graph(x, k=8, patient_ids=pids, exclude_same_patient=True)))


# ---------------------------------------------------------------------------
# Backbones that use finetune.trainer.train()
# ---------------------------------------------------------------------------

def _run_with_train(dataset, make_model, config: TrainConfig, encode_fn, checkpoint_key: str) -> dict:
    out: dict = {}
    for i, inner_train, inner_val, test_ids in _folds(dataset.patient_ids):
        t0 = time.perf_counter()
        test_idx = [j for j, p in enumerate(dataset.patient_ids) if p in test_ids]
        model = make_model()
        model.to(config.device)
        x_frozen = encode_indices(model, dataset, test_idx, config.device, encode_fn)
        result = train(config, dataset, model, encode_fn, checkpoint_key=checkpoint_key,
                       override_split=(inner_train, inner_val))
        x_ft = encode_indices(model, dataset, test_idx, config.device, encode_fn)
        y = dataset.targets_t[test_idx].numpy()
        pids = [dataset.patient_ids[j] for j in test_idx]
        hist = result["history"]
        best_epoch = int(min(hist, key=lambda h: h["val_c"])["epoch"])
        out[i] = dict(x_frozen=x_frozen, x_ft=x_ft, y=y, pids=pids, meta=dict(
            fold=i, n_test=len(test_idx), n_inner_train_patients=len(inner_train),
            n_inner_val_patients=len(inner_val), selected_epoch=best_epoch, epochs_run=len(hist) - 1,
            inner_val_c_baseline=result["baseline"]["observed"], inner_val_c_selected=result["final"]["observed"],
            minutes=(time.perf_counter() - t0) / 60,
        ))
        log.info("fold %d  inner-val C %.3f -> %.3f (epoch %d)  |  TEST C %.3f -> %.3f  (n=%d, %.1f min)",
                 i, result["baseline"]["observed"], result["final"]["observed"], best_epoch,
                 _quick_c(x_frozen, y, pids), _quick_c(x_ft, y, pids), len(test_idx), out[i]["meta"]["minutes"])
        del model
        torch.cuda.empty_cache()
    return out


def run_open_clip(backend: str) -> dict:
    import open_clip

    if backend == "biomedclip":
        import scripts.finetune_cfb_gbm as s
        model0, preprocess = open_clip.create_model_from_pretrained(s.MODEL_NAME)
    elif backend == "imagenet_vit":
        # BiomedCLIP's own init checkpoint; same recipe as BiomedCLIP (TrainConfig defaults,
        # slice_tcc target), so the pair differs only in BiomedCLIP's PMC-15M training.
        import scripts.finetune_cfb_gbm as s
        from embeddings.backends.imagenet_vit import load_imagenet_vit
        model0, preprocess = load_imagenet_vit()
    else:
        import scripts.finetune_clip_cfb_gbm as s
        model0, preprocess = open_clip.create_model_from_pretrained(s.MODEL_NAME, pretrained=s.PRETRAINED)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb = CFBGBMDataset(IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", projection_2d=True, slice_selection="gtv_mask")
    dataset = CFBGBMSliceDataset(cfb, table, preprocess, target_column=s.TARGET_COLUMN)

    def make_model():
        m = copy.deepcopy(model0)
        m.visual = wrap_visual_tower_with_lora(m.visual, backend)
        return m

    config = TrainConfig(backend_name=backend, target_column=s.TARGET_COLUMN)
    return _run_with_train(dataset, make_model, config, default_encode_fn, "lora")


def run_medsiglip() -> dict:
    from transformers import AutoProcessor, SiglipModel

    import scripts.finetune_medsiglip_cfb_gbm as s

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb = CFBGBMDataset(IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", projection_2d=True, slice_selection="gtv_mask")
    processor = AutoProcessor.from_pretrained(s.MODEL_NAME)

    def preprocess(pil_img):
        return processor(images=pil_img, return_tensors="pt")["pixel_values"][0]

    dataset = CFBGBMSliceDataset(cfb, table, preprocess, target_column=s.TARGET_COLUMN)

    def make_model():
        m = SiglipModel.from_pretrained(s.MODEL_NAME)
        m.vision_model = wrap_visual_tower_with_lora(m.vision_model, "medsiglip")
        return m

    config = TrainConfig(backend_name="medsiglip", target_column=s.TARGET_COLUMN,
                         temperature=s.TEMPERATURE, batch_size=16)
    return _run_with_train(dataset, make_model, config, s.encode_fn, "lora")


def run_projection_head(backend: str) -> dict:
    import scripts.finetune_projection_head_cfb_gbm as s
    from finetune.projection_head import ProjectionHead, projection_head_encode_fn

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    x, y, pids = build_coherence_dataset(EmbeddingStore(STORE_DIR / f"cfb_gbm_{backend}.h5"), table,
                                         target_column=s.TARGET_COLUMN, log_transform=True)
    dataset = EmbeddingDataset(x, y, pids)
    config = TrainConfig(backend_name=backend, target_column=s.TARGET_COLUMN,
                         temperature=s._TEMPERATURE[backend], lr=s._LEARNING_RATE)
    return _run_with_train(dataset, lambda: ProjectionHead(dim=x.shape[1]), config,
                           projection_head_encode_fn, checkpoint_key="")


# ---------------------------------------------------------------------------
# NeuroVFM (own packed-token training loop, mirrors finetune/neurovfm_trainer.py)
# ---------------------------------------------------------------------------

def run_neurovfm() -> dict:
    from neurovfm.pipelines import load_encoder

    import scripts.finetune_neurovfm_cfb_gbm as s
    from finetune.losses import soft_neighbor_tcc_loss
    from finetune.neurovfm_data import NeuroVFMTokenDataset, build_token_cache, forward_pooled, pack_batch
    from finetune.neurovfm_trainer import NeuroVFMConfig

    cfg = NeuroVFMConfig()
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    cfb = CFBGBMDataset(IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", modalities=["t1gd"],
                        require_adc=True, require_gtv=True)
    encoder0, preproc = load_encoder("mlinslab/neurovfm-encoder")
    device, norm_module = encoder0.device, encoder0.norm_module
    cache = build_token_cache(cfb, IMAGING_ROOT, preproc, s.CACHE_PATH)
    dataset = NeuroVFMTokenDataset(cfb, table, cache, target_column=s.TARGET_COLUMN)

    def embed(model, indices):
        model.eval()
        embs = []
        with torch.no_grad():
            for start in range(0, len(indices), cfg.eval_batch_size):
                batch = pack_batch(dataset, indices[start:start + cfg.eval_batch_size])
                embs.append(forward_pooled(model, norm_module, batch, device).float().cpu().numpy())
        return np.concatenate(embs, axis=0)

    def c_of(model, indices, n_perm):
        x = embed(model, indices)
        y = dataset.targets_t[indices].numpy()
        pids = [dataset.patient_ids[j] for j in indices]
        w = build_knn_graph(x, k=cfg.k, patient_ids=pids, exclude_same_patient=True)
        return permutation_test(y, w, statistic="gearys_c", n_permutations=n_perm, seed=0)["observed"]

    out: dict = {}
    rng = np.random.default_rng(cfg.seed)
    for i, inner_train, inner_val, test_ids in _folds(dataset.patient_ids):
        t0 = time.perf_counter()
        tr = [j for j, p in enumerate(dataset.patient_ids) if p in inner_train]
        va = [j for j, p in enumerate(dataset.patient_ids) if p in inner_val]
        te = [j for j, p in enumerate(dataset.patient_ids) if p in test_ids]

        model = wrap_visual_tower_with_lora(copy.deepcopy(encoder0.model), "neurovfm").to(device)
        x_frozen = embed(model, te)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg.lr, weight_decay=cfg.weight_decay)
        base_c = best_c = c_of(model, va, cfg.n_permutations_during_training)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
        best_epoch, since, epochs_run = 0, 0, 0
        for epoch in range(1, cfg.max_epochs + 1):
            epochs_run = epoch
            model.train()
            perm = rng.permutation(tr)
            for b in range(len(perm) // cfg.batch_size):
                bidx = perm[b * cfg.batch_size:(b + 1) * cfg.batch_size].tolist()
                pooled = forward_pooled(model, norm_module, pack_batch(dataset, bidx), device)
                loss = soft_neighbor_tcc_loss(pooled, dataset.targets_t[bidx].to(device),
                                              [dataset.patient_ids[j] for j in bidx], temperature=cfg.temperature)
                opt.zero_grad()
                loss.backward()
                opt.step()
            val_c = c_of(model, va, cfg.n_permutations_during_training)
            if val_c < best_c:
                best_c, best_epoch, since = val_c, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
            else:
                since += 1
                if since >= cfg.patience:
                    break
        model.load_state_dict(best_state, strict=False)
        x_ft = embed(model, te)
        y = dataset.targets_t[te].numpy()
        pids = [dataset.patient_ids[j] for j in te]
        out[i] = dict(x_frozen=x_frozen, x_ft=x_ft, y=y, pids=pids, meta=dict(
            fold=i, n_test=len(te), n_inner_train_patients=len(inner_train), n_inner_val_patients=len(inner_val),
            selected_epoch=best_epoch, epochs_run=epochs_run, inner_val_c_baseline=base_c,
            inner_val_c_selected=best_c, minutes=(time.perf_counter() - t0) / 60,
        ))
        log.info("fold %d  inner-val C %.3f -> %.3f (epoch %d)  |  TEST C %.3f -> %.3f  (n=%d, %.1f min)",
                 i, base_c, best_c, best_epoch, _quick_c(x_frozen, y, pids), _quick_c(x_ft, y, pids),
                 len(te), out[i]["meta"]["minutes"])
        del model
        torch.cuda.empty_cache()
    return out


RUNNERS = {
    "biomedclip": lambda: run_open_clip("biomedclip"),
    "clip": lambda: run_open_clip("clip"),
    "imagenet_vit": lambda: run_open_clip("imagenet_vit"),
    "medsiglip": run_medsiglip,
    "neurovfm": run_neurovfm,
    "merlin": lambda: run_projection_head("merlin"),
    "medimageinsights": lambda: run_projection_head("medimageinsights"),
    "med3d": lambda: run_projection_head("med3d"),
}


def save(backend: str, out: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, f in out.items():
        arrays[f"x_frozen_{i}"] = f["x_frozen"].astype(np.float32)
        arrays[f"x_ft_{i}"] = f["x_ft"].astype(np.float32)
        arrays[f"y_{i}"] = np.asarray(f["y"], dtype=np.float64)
        arrays[f"pids_{i}"] = np.asarray(f["pids"], dtype=str)
    arrays["meta"] = np.asarray(json.dumps([out[i]["meta"] for i in sorted(out)]))
    path = OUT_DIR / f"{backend}.npz"
    np.savez(path, **arrays)
    return path


def main() -> None:
    backends = sys.argv[1:] or ALL_BACKENDS
    log.info("device: %s", "cuda" if torch.cuda.is_available() else "cpu")
    for b in backends:
        path = OUT_DIR / f"{b}.npz"
        if path.exists():
            log.info("=== %s: already done (%s), skipping ===", b, path)
            continue
        log.info("=== %s: nested CV, %d outer folds ===", b, K_OUTER)
        t0 = time.perf_counter()
        out = RUNNERS[b]()
        log.info("=== %s done in %.1f min -> %s ===", b, (time.perf_counter() - t0) / 60, save(b, out))


if __name__ == "__main__":
    main()
