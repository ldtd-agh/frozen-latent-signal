"""Fine-tuning loss functions.

Only the primary loss from PLANNING.md's Task 5 is implemented so far: a
differentiable relaxation of Geary's C, the exact statistic Task 3 uses to
evaluate coherence (see `analysis/cellularity.py`). This is deliberately
scoped to answering "does optimizing the coherence objective directly move
the needle at all" before adding the temporal (within-patient displacement)
and anchor/distillation loss terms PLANNING.md also describes.
"""

from __future__ import annotations

import torch


def soft_neighbor_tcc_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    patient_ids: list[str],
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    For every item i in the batch, predict y_i as a softmax-weighted average
    of every OTHER item j's true y_j, weighted by
    ``softmax(-||z_i - z_j||^2 / temperature)``. Same-patient pairs are
    excluded from the weighting -- mirrors Task 3's confound control,
    without it the model could "solve" the task by clustering patient
    identity alone (a patient's own scans are close in TCC for free).

    loss = MSE(y_pred, y). No hard k to tune -- temperature plays that role,
    so this is a soft, differentiable stand-in for Geary's C's k-NN graph.

    Parameters
    ----------
    z: [B, D] batch of (L2-normalised) embeddings
    y: [B] batch of (log10) TCC targets
    patient_ids: length-B list of patient id strings
    """
    b = z.shape[0]
    dist_sq = torch.cdist(z, z, p=2) ** 2  # [B, B]

    same_patient = torch.tensor(
        [[pi == pj for pj in patient_ids] for pi in patient_ids],
        device=z.device,
        dtype=torch.bool,
    )
    self_mask = torch.eye(b, device=z.device, dtype=torch.bool)
    mask = same_patient | self_mask

    # Guard against a batch composition where every OTHER item happens to
    # share a patient with i (only possible with a tiny batch or one
    # patient dominating it) -- falls back to excluding only self for that
    # row so softmax never sees an all -inf row.
    row_fully_masked = mask.all(dim=1)
    logits = (-dist_sq / temperature).masked_fill(mask, float("-inf"))
    if row_fully_masked.any():
        fallback_logits = (-dist_sq / temperature).masked_fill(self_mask, float("-inf"))
        logits = torch.where(row_fully_masked.unsqueeze(1), fallback_logits, logits)

    weights = torch.softmax(logits, dim=1)  # [B, B], rows sum to 1
    y_pred = weights @ y

    return torch.nn.functional.mse_loss(y_pred, y)
