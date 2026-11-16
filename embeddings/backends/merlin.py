"""Merlin (NVIDIA) volumetric medical imaging foundation model backend.

Architecture notes (from source inspection)
--------------------------------------------
* Merlin is an Inflated-3D ResNet-152 (I3D) trained on abdominal CT scans via
  contrastive image–report learning.
* ImageEmbedding=True head: AdaptiveAvgPool3d(1,1,1) → [B, 2048] projection.
* Input convention (MONAI): [B, C, H, W, D]  — D is the *last* spatial axis.
  Internally the forward permutes to [B, C, D, H, W] before the 3-D convs, so D
  is treated as the "temporal / depth" axis (the coarsely-sampled sup-inf axis).
* The final avgpool is AdaptiveAvgPool3d(1,1,1), meaning **any spatial size is
  accepted** — no mandatory target resolution.

Preprocessing for brain MRI (Lumiere dataset)
----------------------------------------------
Merlin's DataLoader resample CT to (1.5, 1.5, 3 mm) and clip HU [-1000, 1000].
For atlas-registered brain MRI volumes (already z-score normalised to [0, 1] by
LumiereDataset), we apply a lightweight equivalent:

  1. Already normalised to [0, 1] — no HU windowing needed.
  2. Transpose: SimpleITK array [Z, Y, X] → Merlin [X, Y, Z] = [H, W, D], so
     the inferior-superior axis (Z, which Merlin down-samples to 3 mm) is last.
  3. Optional trilinear resize to `target_size` (default 224 × 224 × 60) to
     match Merlin's training distribution and keep GPU memory manageable.
     Keeping D ≈ 60 matches the ≈ 182/3 slices Merlin sees from abdominal CT.
"""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend

# Merlin I3D-ResNet-152 + AdaptiveAvgPool3d(1,1,1) + projection head
_EMBEDDING_DIM = 2048

# Target (H, W, D) after trilinear resize.
# H=224, W=224 match Merlin's training spatial footprint.
# D=60 ≈ 182 mm / 3 mm/slice — the coarse inf-sup sampling Merlin was trained on.
_DEFAULT_TARGET_SIZE: tuple[int, int, int] = (224, 224, 60)


class MerlinBackend(EmbeddingBackend):
    """
    Wraps the Merlin ImageEmbedding head for volumetric scan embedding.

    Output: float32 [2048] per scan.

    Parameters
    ----------
    device:
        PyTorch device string (default: auto-detect CUDA).
    target_size:
        ``(H, W, D)`` to which each volume is trilinearly resampled before
        inference.  Set to ``None`` to skip resampling (uses native volume
        size; may OOM on large volumes).  Default: ``(224, 224, 60)``.
    """

    name = "merlin"
    embedding_dim = _EMBEDDING_DIM

    def __init__(
        self,
        device: str | None = None,
        target_size: tuple[int, int, int] | None = _DEFAULT_TARGET_SIZE,
    ) -> None:
        self._device_str = device
        self._device = None
        self.target_size = target_size
        self._model = None

    # ------------------------------------------------------------------
    # EmbeddingBackend interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from merlin import Merlin  # type: ignore[import]

        self._device = self._device_str or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = Merlin(ImageEmbedding=True)
        self._model.eval()
        self._model.to(self._device)

    def embed(self, volume: np.ndarray) -> np.ndarray:
        """
        Embed a single volumetric scan.

        Parameters
        ----------
        volume:
            float32 array [D, H, W] or [C, D, H, W] (from LumiereDataset).
            Values should be in [0, 1] (LumiereDataset normalize=True ensures this).

        Returns
        -------
        float32 array [2048].
        """
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        vol = volume.astype(np.float32)

        # Drop channel dim if multi-channel (we embed one modality at a time)
        if vol.ndim == 4:
            vol = vol[0]           # [C, D, H, W] → [D, H, W], keep first channel
        # vol: [D, H, W]  (SimpleITK convention: Z=inf-sup, Y, X)

        # ---- Axis rearrangement ----------------------------------------
        # SimpleITK gives [Z, Y, X].
        # Merlin forward expects [B, C, H, W, D] where D (last) = inf-sup axis.
        # → transpose [Z, Y, X] to [X, Y, Z] = [H, W, D].
        if vol.ndim == 2:
            vol = vol[:, :, np.newaxis]        # [H, W] → [H, W, D=1] (2-D slice)
        else:
            vol = np.transpose(vol, (2, 1, 0)) # [Z, Y, X] → [H, W, D]

        # ---- Tensor: [1, 1, H, W, D] -----------------------------------
        tensor = (
            torch.from_numpy(vol)
            .unsqueeze(0)          # batch dim  → [1, H, W, D]
            .unsqueeze(0)          # channel dim → [1, 1, H, W, D]
            .to(self._device)
        )

        # ---- Optional trilinear resize ---------------------------------
        if self.target_size is not None:
            tensor = F.interpolate(
                tensor,
                size=self.target_size,
                mode="trilinear",
                align_corners=False,
            )

        # ---- Forward pass ----------------------------------------------
        with torch.no_grad():
            out = self._model(tensor)   # → [1, B, 2048]

        # out[0] = [B, 2048], for B=1: squeeze to [2048]
        embedding = out[0].squeeze(0).cpu().numpy()
        return embedding.astype(np.float32)

    def embed_batch(self, volumes: list[np.ndarray]) -> np.ndarray:
        """Embed a list of volumes in a single GPU batch."""
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Call load() before embed_batch().")

        tensors = []
        for volume in volumes:
            vol = volume.astype(np.float32)
            if vol.ndim == 4:
                vol = vol[0]
            if vol.ndim == 2:
                vol = vol[:, :, np.newaxis]
            else:
                vol = np.transpose(vol, (2, 1, 0))      # [H, W, D]
            t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)   # [1, 1, H, W, D]
            if self.target_size is not None:
                t = F.interpolate(t, size=self.target_size, mode="trilinear", align_corners=False)
            tensors.append(t)

        batch = torch.cat(tensors, dim=0).to(self._device)    # [B, 1, H, W, D]

        with torch.no_grad():
            out = self._model(batch)   # → [1, B, 2048]

        return out[0].cpu().numpy().astype(np.float32)   # [B, 2048]

    def unload(self) -> None:
        import torch

        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
