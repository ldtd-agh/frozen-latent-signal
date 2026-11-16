"""MedSigLIP embedding backend (HuggingFace transformers)."""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend

# SigLIP-so400m projection head output dimension
_EMBEDDING_DIM = 1152


class MedSigLIPBackend(EmbeddingBackend):
    """
    Wraps the MedSigLIP image encoder via HuggingFace AutoModel.

    MedSigLIP is a 2-D model (SigLIP-so400m-patch14-448 fine-tuned on medical
    images).  3-D ``[D, H, W]`` and 4-D ``[C, D, H, W]`` volumes are reduced to
    their mid-axial slice before encoding; pass ``projection_2d=True`` on the
    dataset for seg-mask-guided slice selection instead.

    The processor resizes images to 448 × 448 internally — no external TF
    dependency required.  ``outputs.image_embeds`` are already L2-normalised
    by SigLIP's architecture.

    Requires: pip install torch transformers Pillow

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (default: ``"google/medsiglip-448"``).
    device:
        PyTorch device string (default: auto-detect CUDA).
    """

    name = "medsiglip"
    embedding_dim = _EMBEDDING_DIM

    def __init__(
        self,
        model_name: str = "google/medsiglip-448",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self._device_str = device
        self._device = None
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # EmbeddingBackend interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import SiglipModel, AutoProcessor  # type: ignore[import]

        self._device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SiglipModel.from_pretrained(self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, volume: np.ndarray) -> np.ndarray:
        """
        Encode a 2-D slice with MedSigLIP's image encoder.

        Accepted shapes
        ---------------
        ``[H, W]``        — single-channel 2-D
        ``[C, H, W]``     — multi-channel 2-D (first channel used)
        ``[D, H, W]``     — 3-D volume (mid-axial slice used)
        ``[C, D, H, W]``  — 4-D volume (first channel, mid-axial slice)

        Returns
        -------
        float32 array ``[1152]``, L2-normalised (SigLIP architecture).
        """
        import torch

        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        pil_img = self._to_pil(volume)
        inputs = self._processor(images=pil_img, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        vec = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return vec.squeeze(0).cpu().numpy().astype(np.float32)

    def embed_batch(self, volumes: list[np.ndarray]) -> np.ndarray:
        """Embed a list of volumes in a single forward pass."""
        import torch

        if self._model is None:
            raise RuntimeError("Call load() before embed_batch().")

        pil_imgs = [self._to_pil(v) for v in volumes]
        inputs = self._processor(images=pil_imgs, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)

        vec = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        return vec.cpu().numpy().astype(np.float32)  # [B, 1152]

    def unload(self) -> None:
        import torch

        del self._model
        del self._processor
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(volume: np.ndarray):
        """Convert an ndarray of any accepted shape to an RGB PIL Image."""
        from PIL import Image  # type: ignore[import]

        vol = volume.astype(np.float32)

        # Collapse to [H, W]
        if vol.ndim == 4:
            vol = vol[0]                    # [C, D, H, W] → [D, H, W]
        if vol.ndim == 3:
            vol = vol[vol.shape[0] // 2]   # [D, H, W] or [C, H, W] → [H, W]
        # vol: [H, W]

        lo, hi = vol.min(), vol.max()
        slice_u8 = ((vol - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
        return Image.fromarray(slice_u8).convert("RGB")
