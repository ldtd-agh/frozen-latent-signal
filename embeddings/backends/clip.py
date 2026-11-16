"""Vanilla (non-medical) CLIP embedding backend (open_clip-based).

Non-medical baseline for the CFB-GBM coherence test: same architecture as
BioMedCLIPBackend (ViT-B/16), but pretrained on general web image-text
pairs (OpenAI's original CLIP training data) rather than biomedical
image-text pairs. Holding architecture fixed and varying only the
pretraining domain isolates whether medical-domain pretraining is
contributing anything beyond generic vision pretraining for a signal like
ADC-derived tumour cellularity coherence.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend


class CLIPBackend(EmbeddingBackend):
    """
    Wraps vanilla OpenAI CLIP's vision encoder via open_clip.

    Expects a 2-D image as input — use ``projection_2d=True`` on the dataset
    to get the right slice before calling this backend. As a convenience
    fallback, 3-D ``[D, H, W]`` and 4-D ``[C, D, H, W]`` volumes are also
    accepted and reduced to their mid-axial slice, but projection at the
    dataset level (with seg-mask guidance) is always preferred.

    Requires: pip install open_clip_torch Pillow torch

    Parameters
    ----------
    model_name: open_clip architecture name (default:
                ``'ViT-B-16-quickgelu'`` — same ViT-B/16 architecture
                BiomedCLIP fine-tunes from, with the QuickGELU activation
                OpenAI's original checkpoint was trained with; the plain
                ``'ViT-B-16'`` entry uses standard GELU and open_clip warns
                on a QuickGELU mismatch if paired with ``pretrained='openai'``).
    pretrained: open_clip pretrained-weights tag (default: ``'openai'`` —
                OpenAI's original CLIP checkpoint).
    device:     torch device string (default: auto-detect CUDA)
    """

    name = "clip"
    embedding_dim = 512  # ViT-B/16 projection dim (same as BiomedCLIP)

    def __init__(
        self,
        model_name: str = "ViT-B-16-quickgelu",
        pretrained: str = "openai",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._device = device
        self._model = None
        self._preprocess = None

    def load(self) -> None:
        import torch
        from open_clip import create_model_from_pretrained  # type: ignore[import]

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model, self._preprocess = create_model_from_pretrained(
            self.model_name, pretrained=self.pretrained
        )
        self._model.eval()
        self._model.to(self._device)

    def embed(self, volume: np.ndarray) -> np.ndarray:
        """
        Encode a 2-D slice with CLIP's vision encoder.

        Accepted shapes
        ---------------
        ``[H, W]``        — single-channel 2-D (ideal: use ``projection_2d=True`` on dataset)
        ``[C, H, W]``     — multi-channel 2-D (first channel used)
        ``[D, H, W]``     — 3-D volume fallback (mid-axial slice used)
        ``[C, D, H, W]``  — 4-D volume fallback (first channel, mid-axial slice)
        """
        import torch
        from PIL import Image  # type: ignore[import]

        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        vol = volume.astype(np.float32)

        # Collapse to [H, W]
        if vol.ndim == 4:
            vol = vol[0]          # [C, D, H, W] → [D, H, W]
        if vol.ndim == 3:
            vol = vol[vol.shape[0] // 2]   # [D, H, W] or [C, H, W] → [H, W]

        slice_2d = vol  # [H, W]

        # Normalise to [0, 255] uint8 and convert to RGB PIL image
        slice_norm = (
            (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255
        ).astype(np.uint8)
        pil_img = Image.fromarray(slice_norm).convert("RGB")

        image_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            image_features = self._model.encode_image(image_tensor)
            # L2-normalise to match CLIP's zero-shot usage (and BiomedCLIP's convention)
            image_features /= image_features.norm(dim=-1, keepdim=True)

        return image_features.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        import torch

        del self._model
        del self._preprocess
        self._model = None
        self._preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
