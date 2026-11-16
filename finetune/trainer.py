"""LoRA fine-tuning loop against the soft-neighbor TCC loss.

First, deliberately minimal pass at PLANNING.md's Task 5: checks whether
optimizing Task 3's coherence objective directly improves held-out Geary's
C at all, before adding the temporal/anchor loss terms or extending to
other backbones. See RESULTS.md for the result.

Patient-level train/val split (never split a patient's own scans across
both sides -- would leak, since same-patient scans are close in TCC by
construction). Early-stopping metric is held-out, same-patient-excluded
Geary's C (lower = more coherent) -- not training loss -- matching Task 3's
own confound control and PLANNING.md's explicit design choice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from analysis.cellularity import build_knn_graph, permutation_test

from .data import CFBGBMSliceDataset
from .losses import soft_neighbor_tcc_loss

log = logging.getLogger(__name__)

# model, image_batch -> [B, D] embeddings, postprocessed exactly like that
# backend's own `embed()` (e.g. L2-normalised for BiomedCLIP, raw pooler
# output for MedSigLIP -- see each backend's embeddings/backends/*.py) so
# the "frozen baseline" measured here matches what Task 3 actually
# evaluated. Default matches BiomedCLIP (open_clip's `encode_image`).
EncodeFn = Callable[[object, torch.Tensor], torch.Tensor]


def default_encode_fn(model, images: torch.Tensor) -> torch.Tensor:
    z = model.encode_image(images)
    return z / z.norm(dim=-1, keepdim=True)


@dataclass
class TrainConfig:
    backend_name: str = "biomedclip"
    target_column: str = "slice_tcc"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 48
    max_epochs: int = 40
    temperature: float = 1.0
    val_fraction: float = 0.2
    k: int = 8
    seed: int = 42
    patience: int = 8
    n_permutations_during_training: int = 199
    n_permutations_final: int = 999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def patient_split(patient_ids: list[str], val_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    unique = sorted(set(patient_ids))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_fraction)))
    return set(unique[n_val:]), set(unique[:n_val])  # train, val


def k_fold_patient_splits(patient_ids: list[str], k: int, seed: int) -> list[tuple[set[str], set[str]]]:
    """
    Partition the unique patient set into `k` disjoint folds and return `k`
    ``(train_ids, val_ids)`` pairs whose ``val_ids`` are pairwise disjoint
    and collectively exhaustive (every patient appears in exactly one
    fold's val set) -- unlike `patient_split()`'s independent random
    draws, where 5 draws at val_fraction=0.2 leave ~33% of patients absent
    from every val set by chance ((1-0.2)^5 ≈ 0.33). Needed for a
    full-population out-of-fold UMAP figure: every scan must get a
    fine-tuned embedding from a model that never saw it during training.
    """
    rng = np.random.default_rng(seed)
    unique = sorted(set(patient_ids))
    rng.shuffle(unique)
    folds = np.array_split(np.array(unique), k)
    splits = []
    for i in range(k):
        val_ids = set(folds[i].tolist())
        train_ids = set(unique) - val_ids
        splits.append((train_ids, val_ids))
    return splits


def _encode_all(
    model, images: torch.Tensor, device: str, encode_fn: EncodeFn, batch_size: int = 64
) -> np.ndarray:
    """No-grad forward pass over a (possibly large) image batch, chunked to fit memory."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            chunk = images[i : i + batch_size].to(device)
            z = encode_fn(model, chunk)
            outs.append(z.cpu().numpy())
    return np.concatenate(outs, axis=0)


def encode_indices(
    model, dataset: CFBGBMSliceDataset, indices: list[int], device: str, encode_fn: EncodeFn = default_encode_fn,
) -> np.ndarray:
    """No-grad forward pass over `dataset[indices]`, returning [len(indices), D] embeddings."""
    images = torch.stack([dataset.images[i] for i in indices])
    return _encode_all(model, images, device, encode_fn)


def evaluate_coherence(
    model,
    dataset: CFBGBMSliceDataset,
    indices: list[int],
    device: str,
    k: int,
    n_permutations: int,
    encode_fn: EncodeFn = default_encode_fn,
    seed: int = 0,
) -> dict:
    y = dataset.targets_t[indices].numpy()
    patient_ids = [dataset.patient_ids[i] for i in indices]

    x = encode_indices(model, dataset, indices, device, encode_fn)
    w = build_knn_graph(x, k=k, patient_ids=patient_ids, exclude_same_patient=True)
    return permutation_test(y, w, statistic="gearys_c", n_permutations=n_permutations, seed=seed)


def train(
    config: TrainConfig,
    dataset: CFBGBMSliceDataset,
    model,
    encode_fn: EncodeFn = default_encode_fn,
    save_dir: "str | None" = None,
    peft_attr: str = "visual",
    checkpoint_key: str = "lora",
    override_split: "tuple[set[str], set[str]] | None" = None,
) -> dict:
    """
    ``save_dir``, if given, saves the best checkpoint's LoRA adapter (via
    peft's own ``save_pretrained`` -- portable, reloadable with
    ``PeftModel.from_pretrained``, includes the LoraConfig) to that
    directory. ``peft_attr`` names the attribute on ``model`` holding the
    peft-wrapped submodule (``"visual"`` for open_clip backbones,
    ``"vision_model"`` for MedSigLIP/other HF backbones).

    ``checkpoint_key`` selects which of ``model.state_dict()``'s keys are
    tracked as "the best checkpoint" (any key containing this substring) --
    default ``"lora"`` for the LoRA-wrapped backbones above. Pass ``""``
    (matches every key) for a plain trainable module with no LoRA wrapper,
    e.g. a projection head -- **do not leave this at the default** for a
    non-LoRA model, since an empty match silently no-ops the early-stopping
    restore (the model is left at its last epoch's state, not its best).

    ``override_split``, if given, is a ``(train_ids, val_ids)`` pair used
    verbatim instead of drawing a fresh random split from
    ``config.val_fraction``/``config.seed`` -- e.g. one fold from
    `k_fold_patient_splits()`, for a full-population out-of-fold pass.
    """
    if override_split is not None:
        train_ids, val_ids = override_split
    else:
        train_ids, val_ids = patient_split(dataset.patient_ids, config.val_fraction, config.seed)
    train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in train_ids]
    val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_ids]
    log.info(
        "Train/val split: %d train scans (%d patients), %d val scans (%d patients)",
        len(train_idx), len(train_ids), len(val_idx), len(val_ids),
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    model.to(config.device)

    baseline = evaluate_coherence(
        model, dataset, val_idx, config.device, config.k, config.n_permutations_final, encode_fn
    )
    log.info(
        "Baseline (pretrained, no fine-tuning) held-out C=%+.4f p=%.3f (n=%d val scans -- "
        "not directly comparable to Task 3's full-population number)",
        baseline["observed"], baseline["p_value"], len(val_idx),
    )

    best_c = baseline["observed"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if checkpoint_key in k}
    epochs_without_improvement = 0
    history = [{"epoch": 0, "train_loss": None, "val_c": baseline["observed"], "val_p": baseline["p_value"]}]

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        epoch_losses = []
        for images, y, patient_ids in train_loader:
            images = images.to(config.device)
            y = y.to(config.device)

            z = encode_fn(model, images)

            loss = soft_neighbor_tcc_loss(z, y, list(patient_ids), temperature=config.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_result = evaluate_coherence(
            model, dataset, val_idx, config.device, config.k, config.n_permutations_during_training, encode_fn
        )
        mean_loss = float(np.mean(epoch_losses))
        log.info(
            "epoch %2d/%d  train_loss=%.4f  held-out C=%+.4f p=%.3f",
            epoch, config.max_epochs, mean_loss, val_result["observed"], val_result["p_value"],
        )
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "val_c": val_result["observed"], "val_p": val_result["p_value"]}
        )

        if val_result["observed"] < best_c:
            best_c = val_result["observed"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if checkpoint_key in k}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, config.patience)
                break

    model.load_state_dict(best_state, strict=False)
    final = evaluate_coherence(
        model, dataset, val_idx, config.device, config.k, config.n_permutations_final, encode_fn
    )
    log.info(
        "Best checkpoint, re-evaluated with %d permutations: C=%+.4f p=%.3f",
        config.n_permutations_final, final["observed"], final["p_value"],
    )

    if save_dir is not None:
        getattr(model, peft_attr).save_pretrained(save_dir)
        log.info("Saved LoRA adapter (best checkpoint) to %s", save_dir)

    return {"baseline": baseline, "final": final, "history": history, "val_idx": val_idx}


def run_multi_seed_robustness(
    model_factory: Callable[[], object],
    wrap_fn: Callable[[object], None],
    dataset: CFBGBMSliceDataset,
    config_template: TrainConfig,
    seeds: list[int],
    encode_fn: EncodeFn = default_encode_fn,
    save_dir_template: "str | None" = None,
    peft_attr: str = "visual",
    checkpoint_key: str = "lora",
) -> "pd.DataFrame":
    """
    Rerun `train()` across several patient-level splits (one per seed) to
    check the fine-tuning result isn't a lucky-split artifact. `model_factory`
    returns a fresh, unwrapped copy of the base model each call; `wrap_fn`
    LoRA-wraps it in place (e.g. ``lambda m: setattr(m, "visual",
    wrap_visual_tower_with_lora(m.visual, "biomedclip"))``).

    ``save_dir_template``, if given, is formatted with ``seed=`` (e.g.
    ``"results/adapters/clip_seed{seed}"``) and each seed's best checkpoint
    is saved there -- see `train()`'s ``save_dir``.
    """
    rows: list[dict] = []
    for seed in seeds:
        model = model_factory()
        wrap_fn(model)
        config = replace(config_template, seed=seed)
        save_dir = save_dir_template.format(seed=seed) if save_dir_template else None
        result = train(config, dataset, model, encode_fn, save_dir=save_dir, peft_attr=peft_attr, checkpoint_key=checkpoint_key)
        baseline_c = result["baseline"]["observed"]
        final_c = result["final"]["observed"]
        log.warning(
            "seed=%4d  baseline C=%+.4f p=%.3f  -> fine-tuned C=%+.4f p=%.3f  (n_val=%d)",
            seed, baseline_c, result["baseline"]["p_value"], final_c, result["final"]["p_value"], len(result["val_idx"]),
        )
        rows.append({"seed": seed, "baseline_c": baseline_c, "final_c": final_c, "delta": final_c - baseline_c})
    return pd.DataFrame(rows)
