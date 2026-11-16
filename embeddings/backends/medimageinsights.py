"""MedImageInsight (UniCL / DaViT) embedding backend.

Volumes are sliced along the depth axis, each slice embedded independently,
and the resulting vectors are mean-pooled to produce one [1024]-dim embedding.
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

from embeddings.backends.base import EmbeddingBackend

_REPO_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "MedImageInsights")
_REPO_DIR = os.path.normpath(_REPO_DIR)
_DEFAULT_MODEL_DIR = os.path.join(_REPO_DIR, "2024.09.27")
_DEFAULT_VISION_MODEL = "medimageinsigt-v1.0.0.pt"


class MedImageInsightsBackend(EmbeddingBackend):
    """
    MedImageInsight backend — 2D UniCL/DaViT model, mean-pooled over slices.

    Parameters
    ----------
    model_dir:
        Path to the versioned model directory that contains config.yaml,
        vision_model/, and language_model/.
    vision_model_name:
        Filename of the vision checkpoint inside model_dir/vision_model/.
    n_slices:
        How many depth slices to sample per volume.
        None → use every slice (may be slow on thick CT stacks).
    """

    name = "medimageinsights"
    embedding_dim = 1024

    def __init__(
        self,
        model_dir: str = _DEFAULT_MODEL_DIR,
        vision_model_name: str = _DEFAULT_VISION_MODEL,
        n_slices: int | None = None,
        slice_batch_size: int = 16,
    ) -> None:
        self.model_dir = model_dir
        self.vision_model_name = vision_model_name
        self.n_slices = n_slices
        self.slice_batch_size = slice_batch_size
        self._model = None
        self._preprocess = None
        self._device = None

    def load(self) -> None:
        if _REPO_DIR not in sys.path:
            sys.path.insert(0, _REPO_DIR)

        from medimageinsightmodel import MedImageInsight  # noqa: PLC0415

        classifier = MedImageInsight(
            model_dir=self.model_dir,
            vision_model_name=self.vision_model_name,
            language_model_name="",
        )
        classifier.load_model()

        self._device = classifier.device
        self._preprocess = classifier.preprocess
        self._model = classifier.model
        self._model.eval()

    def _to_pil_slices(self, volume: np.ndarray) -> list[Image.Image]:
        """Return a list of RGB PIL images, one per selected depth slice."""
        if volume.ndim == 4:
            volume = volume.mean(axis=0)  # [C, D, H, W] -> [D, H, W]

        v_min, v_max = float(volume.min()), float(volume.max())
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        volume = (volume * 255).astype(np.uint8)  # [D, H, W]

        n_depth = volume.shape[0]
        if self.n_slices is not None and self.n_slices < n_depth:
            indices = np.linspace(0, n_depth - 1, self.n_slices, dtype=int)
        else:
            indices = np.arange(n_depth)

        return [Image.fromarray(volume[i], mode="L").convert("RGB") for i in indices]

    def embed(self, volume: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call load() before embed().")

        slices = self._to_pil_slices(volume)
        all_embeddings = []
        for i in range(0, len(slices), self.slice_batch_size):
            batch = slices[i : i + self.slice_batch_size]
            tensors = torch.stack([self._preprocess(s) for s in batch]).to(self._device)
            with torch.no_grad():
                all_embeddings.append(self._model.encode_image(tensors))
        embeddings = torch.cat(all_embeddings, dim=0)  # [N_slices, 1024]

        return embeddings.median(dim=0).values.cpu().numpy().astype(np.float32)

    def unload(self) -> None:
        self._model = None
        self._preprocess = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
