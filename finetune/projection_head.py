"""Projection-head fine-tuning for frozen 3-D CNN backbones (Merlin, Med3D).

LoRA doesn't apply to these directly (Merlin is a 3-D I3D-ResNet-152,
Med3D a 3-D ResNet-10 -- convolutional, not attention-based; see
`finetune/lora.py`). PLANNING.md's original proposal was a conv-bottleneck
adapter inserted after every block -- more expressive, but a lot more
engineering (hooking into each backbone's block structure) for what this
first pass wants to answer: does *any* cheap, targeted adaptation help
these backbones the way LoRA helped the ViTs, and specifically -- for
Med3D, which showed *zero* frozen coherence (see RESULTS.md) -- is that a
property of the backbone's frozen features, or just of raw Euclidean
proximity in the untouched embedding space?

So this keeps the backbone entirely frozen and trains only a small
residual MLP on top of its pooled output:

    output = x + net(x)      net: Linear(D, H) -> ReLU -> Linear(H, D)

`net`'s final layer is zero-initialised, so training starts from the
frozen embedding exactly (output == x at step 0) and only learns an
additive correction -- the same "start at zero, learn a small delta"
philosophy as LoRA, applied to a plain MLP instead of low-rank attention
updates.

Because the backbone is frozen, there's no reason to touch raw volumes at
all: this fine-tunes directly on the same precomputed HDF5 embeddings
Task 3's coherence test already used (`analysis.cellularity.
build_coherence_dataset`), via `finetune.data.EmbeddingDataset` -- no
GPU-heavy backbone forward pass per step, unlike the ViT LoRA runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Residual MLP head: ``output = x + net(x)``, ``net`` zero-initialised."""

    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def projection_head_encode_fn(model: ProjectionHead, x: torch.Tensor) -> torch.Tensor:
    return model(x)
