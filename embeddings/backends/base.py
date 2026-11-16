"""Abstract base class for all embedding backends."""

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    """
    Interface for volumetric medical image embedding models.

    Subclass this and implement `load()` and `embed()`.
    Register the subclass in `config/experiment.yaml` under `backends`.
    """

    name: str
    """Short identifier, e.g. 'merlin', 'medclip'."""

    embedding_dim: int
    """Dimensionality of the output embedding vector."""

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights into memory (and onto GPU if available).
        Called exactly once before any `embed()` / `embed_batch()` calls.
        """

    @abstractmethod
    def embed(self, volume: np.ndarray) -> np.ndarray:
        """
        Embed a single volumetric scan.

        Parameters
        ----------
        volume: float32 array shaped [D, H, W] or [C, D, H, W].

        Returns
        -------
        float32 array shaped [embedding_dim].
        """

    def embed_batch(self, volumes: list[np.ndarray]) -> np.ndarray:
        """
        Embed a list of volumes.

        Default implementation loops over `embed()`.
        Override for GPU batching to improve throughput.

        Returns
        -------
        float32 array shaped [N, embedding_dim].
        """
        return np.stack([self.embed(v) for v in volumes])

    def unload(self) -> None:
        """
        Optional: free GPU memory / release model resources.
        Called after the embedding run completes.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dim={self.embedding_dim})"
