"""ImageNet-21k ViT-B/16 backend: BiomedCLIP's own initialization checkpoint.

BiomedCLIP's vision tower is timm's ``vit_base_patch16_224`` and was
initialized from Google's supervised AugReg ImageNet-21k weights (Steiner et
al. 2021), not from OpenAI CLIP. The BiomedCLIP paper's supplement says it
"chose to initialize ViT-B/16 with ImageNet-pretrained weights" without
naming the checkpoint. A layer-wise weight correlation identifies the AugReg
IN-21k family: r = +0.68 for the patch embedding, +0.77 for the positional
embedding, and +0.42 -> +0.13 for QKV from block 0 to block 11. Every other
public ViT-B/16 gives r ~ 0, including OpenAI/LAION CLIP, MAE, DINO,
AugReg IN-1k-only, and the original ViT IN-21k.

The three AugReg IN-21k variants (``augreg_in21k`` and its two IN-1k
fine-tunes) correlate identically, because the fine-tunes barely move the
trunk. The default is ``augreg_in21k_ft_in1k``, which was timm's default
``vit_base_patch16_224`` weight when BiomedCLIP was trained.

Kept as close to BiomedCLIP as the two checkpoints allow, so the pair
isolates BiomedCLIP's PMC-15M image-text training:
  - same timm architecture and weights at initialization;
  - same input: the GTV-mask slice, min-max scaled to uint8 RGB;
  - same geometric preprocessing as BiomedCLIP's open_clip transform
    (bicubic resize to 224, center-crop 224), not timm's crop_pct=0.9, which
    would crop away the slice borders; the normalization constants are the
    checkpoint's own (mean = std = 0.5), since those are the statistics its
    weights were trained on;
  - embedding = the trunk's CLS token after the final LayerNorm (768-d), i.e.
    exactly what BiomedCLIP's trunk outputs before its PMC-15M-trained
    768->512 projection (which has no counterpart here), L2-normalized like
    BiomedCLIP's.

`load_imagenet_vit()` returns an open_clip-shaped wrapper (``.visual``,
``.encode_image``) so the LoRA fine-tuning path (`finetune.lora`,
`finetune.trainer.default_encode_fn`) can treat it exactly like BiomedCLIP.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend

DEFAULT_MODEL_NAME = "vit_base_patch16_224.augreg_in21k_ft_in1k"


def load_imagenet_vit(model_name: str = DEFAULT_MODEL_NAME):
    """Return ``(model, preprocess)`` with an open_clip-like ``model``."""
    import timm  # type: ignore[import]
    import torch
    from torchvision import transforms

    class TimmImageEncoder(torch.nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.visual = trunk

        def encode_image(self, images):
            return self.visual(images)

    trunk = timm.create_model(model_name, pretrained=True, num_classes=0)
    cfg = trunk.pretrained_cfg
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])
    return TimmImageEncoder(trunk), preprocess


class ImageNetViTBackend(EmbeddingBackend):
    """Frozen embeddings from the ImageNet-21k ViT-B/16 that BiomedCLIP was initialized from."""

    name = "imagenet_vit"
    embedding_dim = 768

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: str | None = None) -> None:
        self.model_name = model_name
        self._device = device
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import torch

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = load_imagenet_vit(self.model_name)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, volume: np.ndarray) -> np.ndarray:
        """Same input handling as `BioMedCLIPBackend.embed` (2-D slice; 3-D/4-D fall back to mid-slice)."""
        import torch
        from PIL import Image  # type: ignore[import]

        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        vol = volume.astype(np.float32)
        if vol.ndim == 4:
            vol = vol[0]
        if vol.ndim == 3:
            vol = vol[vol.shape[0] // 2]

        slice_norm = ((vol - vol.min()) / (vol.max() - vol.min() + 1e-8) * 255).astype(np.uint8)
        pil_img = Image.fromarray(slice_norm).convert("RGB")
        image_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model.encode_image(image_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        import torch

        self._model = None
        self._preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
