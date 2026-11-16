"""Med3D (ResNet-based) 3-D segmentation pre-trained backbone backend."""

from __future__ import annotations

import numpy as np

from .base import EmbeddingBackend


class Med3DBackend(EmbeddingBackend):
    """
    Wraps a Med3D ResNet backbone for volumetric embedding.

    The encoder is the convolutional backbone of Med3D (resnet_10/18/34/50).
    Global average pooling over the final feature map gives the embedding.
    Requires: pip install -e ".[med3d]"

    Parameters
    ----------
    model_variant: 'resnet10' | 'resnet18' | 'resnet34' | 'resnet50'
    pretrained_path: Path to Med3D pretrained .pth checkpoint.
    device: torch device string (default: auto-detect CUDA).
    input_shape: (D, H, W) to resample input volumes before embedding.
    """

    name = "med3d"
    embedding_dim = 512  # ResNet-10/18 final feature maps → 512-d via GAP

    def __init__(
        self,
        model_variant: str = "resnet10",
        pretrained_path: str | None = None,
        device: str | None = None,
        input_shape: tuple[int, int, int] = (64, 128, 128),
    ) -> None:
        self.model_variant = model_variant
        self.pretrained_path = pretrained_path
        self._device = device
        self.input_shape = input_shape
        self._model = None
        self._gap = None

    def load(self) -> None:
        import torch
        import torch.nn as nn
        from monai.networks.nets import resnet10, resnet18, resnet34, resnet50  # type: ignore[import]

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")

        factory = {"resnet10": resnet10, "resnet18": resnet18,
                   "resnet34": resnet34, "resnet50": resnet50}
        backbone = factory[self.model_variant](
            pretrained=False,
            spatial_dims=3,
            n_input_channels=1,
            num_classes=2,   # dummy; we remove the head
        )

        if self.pretrained_path:
            state = torch.load(self.pretrained_path, map_location="cpu")
            # Med3D checkpoints are often wrapped in 'state_dict'
            if "state_dict" in state:
                state = state["state_dict"]
            # Strip 'module.' prefix from DataParallel
            state = {k.replace("module.", ""): v for k, v in state.items()}
            backbone.load_state_dict(state, strict=False)

        # Remove classification head; keep up to last avgpool / layer4
        self._model = nn.Sequential(*list(backbone.children())[:-1])
        self._model.eval()
        self._model.to(self._device)
        self._gap = nn.AdaptiveAvgPool3d(1).to(self._device)

        # Determine actual embedding dim by a forward pass
        dummy = torch.zeros(1, 1, *self.input_shape).to(self._device)
        with torch.no_grad():
            out = self._gap(self._model(dummy))
        self.embedding_dim = int(out.view(1, -1).shape[1])

    def embed(self, volume: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        vol = volume.astype(np.float32)
        if vol.ndim == 4:
            vol = vol[0]  # [D, H, W]

        # Resample to expected input_shape via trilinear interpolation
        tensor = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        tensor = F.interpolate(tensor, size=self.input_shape, mode="trilinear", align_corners=False)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            feat = self._model(tensor)         # [1, C, d, h, w]
            feat = self._gap(feat)             # [1, C, 1, 1, 1]
            feat = feat.view(1, -1)            # [1, embedding_dim]

        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        import torch

        del self._model
        del self._gap
        self._model = None
        self._gap = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
