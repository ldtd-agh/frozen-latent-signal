"""LoRA wrapping for ViT-based frozen-embedding backbones.

Only attention-based backbones are supported here -- Merlin's 3-D
I3D-ResNet-152 is convolutional and needs a different adapter mechanism (a
small conv bottleneck, not built yet). `target_modules` below were found by
dumping `named_modules()` for each backbone's vision tower and picking every
`nn.Linear` inside the attention/MLP blocks (excludes patch-embed and any
final projection/head layers not shared by both naming conventions):

- BiomedCLIP (open_clip `TimmModel` wrapping a timm ViT): `trunk.blocks.N.attn.qkv`,
  `trunk.blocks.N.attn.proj`, `trunk.blocks.N.mlp.fc1/fc2`.
- ImageNet ViT (`vit_base_patch16_224.augreg_in21k_ft_in1k`, BiomedCLIP's own
  initialization, `embeddings/backends/imagenet_vit.py`): the bare timm ViT,
  so the same `blocks.N.attn.qkv`/`attn.proj`/`mlp.fc1/fc2` (plus
  `patch_embed.proj`, which the `"proj"` suffix also matches, as it does in
  BiomedCLIP's trunk). BiomedCLIP additionally gets LoRA on its 768->512
  `head.proj`, which this checkpoint has no counterpart of.
- MedSigLIP (HuggingFace `SiglipVisionModel`): `encoder.layers.N.self_attn.{q,k,v,out}_proj`,
  `encoder.layers.N.mlp.fc1/fc2`.
- CLIP (`ViT-B-16-quickgelu`, open_clip's *native* `VisionTransformer`, not
  the timm wrapper BiomedCLIP uses -- different implementation despite the
  same nominal ViT-B/16 architecture): attention is a fused
  `torch.nn.MultiheadAttention` whose QKV projection is a raw
  `in_proj_weight` parameter, not a `nn.Linear` submodule peft can target
  the normal way -- only `attn.out_proj` (itself a real `nn.Linear`
  subclass) is addressable. MLP layers are named `mlp.c_fc`/`mlp.c_proj`
  (CLIP's original naming), not `fc1`/`fc2`. So CLIP's LoRA adapts the
  attention *output* projection and the full MLP, but not the QKV
  projection -- a real, deliberate difference from BiomedCLIP/MedSigLIP's
  coverage, not an oversight (see RESULTS.md for whether it mattered).
- NeuroVFM (`mlinslab/neurovfm-encoder`, a 3-D ViT-Base, `scripts/finetune_neurovfm_cfb_gbm.py`):
  despite a "fused" QKV projection (one `nn.Linear` producing all of
  Q/K/V, via flash-attn's `FusedDense` -- which subclasses `nn.Linear`
  directly, confirmed by inspecting its MRO rather than assumed from the
  name), the module itself is still a plainly addressable `self.qkv`
  submodule, unlike CLIP's native `nn.MultiheadAttention`
  (whose fused QKV is a raw `Parameter`, not a submodule at all). Its MLP
  (`flash_attn.modules.mlp.FusedMLP`) is "fused" only in its CUDA
  forward/backward kernel -- it still holds ordinary `fc1`/`fc2`
  `nn.Linear` submodules for weight storage (confirmed the same way).
  Full coverage, same as BiomedCLIP's case. Runs under a separate conda
  env (not this project's main `.venv`) -- see that script's module
  docstring for why.

peft matches `target_modules` by suffix against the full dotted module name,
so the short strings below (`"qkv"`, `"q_proj"`, ...) are enough -- no need
to enumerate every block index.
"""

from __future__ import annotations

from peft import LoraConfig, get_peft_model

_TARGET_MODULES: dict[str, list[str]] = {
    "biomedclip": ["qkv", "proj", "fc1", "fc2"],
    "imagenet_vit": ["qkv", "proj", "fc1", "fc2"],  # BiomedCLIP's timm init checkpoint: identical module names and coverage
    "clip": ["out_proj", "c_fc", "c_proj"],  # native VisionTransformer, not the timm wrapper -- see module docstring
    "medsiglip": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    "neurovfm": ["qkv", "proj", "fc1", "fc2"],  # fused QKV, but still an addressable nn.Linear -- see module docstring
}

_DEFAULT_RANK = 8
_DEFAULT_ALPHA = 16
_DEFAULT_DROPOUT = 0.05


def wrap_visual_tower_with_lora(
    visual_module,
    backend_name: str,
    r: int = _DEFAULT_RANK,
    lora_alpha: int = _DEFAULT_ALPHA,
    lora_dropout: float = _DEFAULT_DROPOUT,
):
    """
    Wrap a backbone's vision tower with LoRA adapters, freezing every other
    parameter. Returns the peft-wrapped module -- assign it back onto the
    parent model (e.g. ``model.visual = wrap_visual_tower_with_lora(model.visual,
    "biomedclip")``) and forward through the parent exactly as before
    (``model.encode_image(...)``); the parent's forward still delegates to
    this submodule by attribute, so nothing else needs to change.
    """
    if backend_name not in _TARGET_MODULES:
        raise ValueError(
            f"No LoRA target_modules configured for {backend_name!r}. "
            f"Choose from {list(_TARGET_MODULES)} or add a new entry."
        )
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=_TARGET_MODULES[backend_name],
        bias="none",
    )
    return get_peft_model(visual_module, config)


def trainable_parameter_count(model) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
