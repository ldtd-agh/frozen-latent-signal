"""Reusable single-split LoRA fine-tuning loop for NeuroVFM.

Extracted from the original `scripts/finetune_neurovfm_cfb_gbm.py` (which
used to inline this loop in its own `main()`) so the exact same recipe can
be reused by both the single-split production run and the 5-seed
robustness check (`scripts/finetune_neurovfm_robustness_check.py`) --
mirroring the split between `finetune/trainer.py`'s shared `train()` and
each ViT backend's thin wrapper script. NeuroVFM can't reuse `trainer.py`
directly: its ragged per-token tensors and packed-batch pooling
(`finetune/neurovfm_data.py`) don't fit `train()`'s fixed-size-`DataLoader`
assumption, so this module exists instead.

`train()` here takes a freshly loaded, unwrapped `encoder` (from
`neurovfm.pipelines.load_encoder(...)`) and wraps its vision tower with a
new LoRA adapter in place -- callers running several seeds must call
`load_encoder(...)` again per seed rather than reusing one instance.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from dataclasses import dataclass

from analysis.cellularity import build_knn_graph, permutation_test

from .lora import trainable_parameter_count, wrap_visual_tower_with_lora
from .losses import soft_neighbor_tcc_loss
from .neurovfm_data import forward_pooled, pack_batch
from .trainer import patient_split

log = logging.getLogger(__name__)


@dataclass
class NeuroVFMConfig:
    backend_name: str = "neurovfm"
    temperature: float = 0.213
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    max_epochs: int = 40
    patience: int = 8
    k: int = 8
    n_permutations_during_training: int = 199
    n_permutations_final: int = 999
    val_fraction: float = 0.2
    seed: int = 42
    eval_batch_size: int = 16  # chunk size for packing held-out scans through the model, no_grad


def evaluate(model, norm_module, dataset, indices, device, k, n_permutations, eval_batch_size, seed=0) -> dict:
    model.eval()
    embs = []
    with torch.no_grad():
        for start in range(0, len(indices), eval_batch_size):
            chunk = indices[start : start + eval_batch_size]
            batch = pack_batch(dataset, chunk)
            pooled = forward_pooled(model, norm_module, batch, device)
            embs.append(pooled.cpu().numpy())
    x = np.concatenate(embs, axis=0)
    y = dataset.targets_t[indices].numpy()
    pids = [dataset.patient_ids[i] for i in indices]
    w = build_knn_graph(x, k=k, patient_ids=pids, exclude_same_patient=True)
    return permutation_test(y, w, statistic="gearys_c", n_permutations=n_permutations, seed=seed)


def train(
    config: NeuroVFMConfig,
    dataset,
    encoder,
    device,
    save_dir: "str | None" = None,
) -> dict:
    norm_module = encoder.norm_module

    train_ids, val_ids = patient_split(dataset.patient_ids, config.val_fraction, config.seed)
    train_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in train_ids]
    val_idx = [i for i, pid in enumerate(dataset.patient_ids) if pid in val_ids]
    log.info(
        "Train/val split: %d train scans (%d patients), %d val scans (%d patients)",
        len(train_idx), len(train_ids), len(val_idx), len(val_ids),
    )

    model = wrap_visual_tower_with_lora(encoder.model, config.backend_name)
    model.to(device)
    trainable, total = trainable_parameter_count(model)
    log.info(
        "LoRA-wrapped %s: %d / %d params trainable (%.2f%%)",
        config.backend_name, trainable, total, 100 * trainable / total,
    )

    baseline = evaluate(
        model, norm_module, dataset, val_idx, device, config.k, config.n_permutations_final, config.eval_batch_size,
    )
    log.info(
        "Baseline (pretrained, no fine-tuning) held-out C=%+.4f p=%.3f (n=%d val scans -- "
        "not directly comparable to Task 3's full-population number)",
        baseline["observed"], baseline["p_value"], len(val_idx),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=config.lr, weight_decay=config.weight_decay,
    )

    best_c = baseline["observed"]
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
    epochs_without_improvement = 0
    history = [{"epoch": 0, "train_loss": None, "val_c": baseline["observed"], "val_p": baseline["p_value"]}]

    rng = np.random.default_rng(config.seed)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        epoch_losses = []
        n_batches = len(perm) // config.batch_size
        for b in range(n_batches):
            batch_idx = perm[b * config.batch_size : (b + 1) * config.batch_size].tolist()
            batch = pack_batch(dataset, batch_idx)
            pooled = forward_pooled(model, norm_module, batch, device)
            y = dataset.targets_t[batch_idx].to(device)
            pids = [dataset.patient_ids[i] for i in batch_idx]

            loss = soft_neighbor_tcc_loss(pooled, y, pids, temperature=config.temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_result = evaluate(
            model, norm_module, dataset, val_idx, device, config.k,
            config.n_permutations_during_training, config.eval_batch_size,
        )
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        log.info(
            "epoch %2d/%d  train_loss=%.4f  held-out C=%+.4f p=%.3f",
            epoch, config.max_epochs, mean_loss, val_result["observed"], val_result["p_value"],
        )
        history.append(
            {"epoch": epoch, "train_loss": mean_loss, "val_c": val_result["observed"], "val_p": val_result["p_value"]}
        )

        if val_result["observed"] < best_c:
            best_c = val_result["observed"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora" in k}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, config.patience)
                break

    model.load_state_dict(best_state, strict=False)
    final = evaluate(
        model, norm_module, dataset, val_idx, device, config.k, config.n_permutations_final, config.eval_batch_size,
    )
    log.info(
        "Best checkpoint, re-evaluated with %d permutations: C=%+.4f p=%.3f",
        config.n_permutations_final, final["observed"], final["p_value"],
    )

    if save_dir is not None:
        model.save_pretrained(str(save_dir))
        log.info("Saved LoRA adapter to %s", save_dir)

    return {"baseline": baseline, "final": final, "history": history, "val_idx": val_idx}
