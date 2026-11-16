"""Dimensionality reduction wrappers: UMAP, PCA, t-SNE."""

from __future__ import annotations

import numpy as np
from joblib import Memory

memory = Memory(location='.cache')

@memory.cache
def reduce(
    vectors: np.ndarray,
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Reduce high-dimensional embeddings to a lower-dimensional space.

    Parameters
    ----------
    vectors:      float32 [N, D]
    method:       'umap' | 'pca' | 'tsne'
    n_components: Target dimensionality (2 or 3).
    **kwargs:     Passed directly to the underlying reducer constructor.

    Returns
    -------
    float32 array [N, n_components]
    """
    method = method.lower()
    if method == "umap":
        return _umap(vectors, n_components, **kwargs)
    elif method == "pca":
        return _pca(vectors, n_components, **kwargs)
    elif method in ("tsne", "t-sne"):
        return _tsne(vectors, n_components, **kwargs)
    else:
        raise ValueError(f"Unknown reduction method: {method!r}. Choose 'umap', 'pca', or 'tsne'.")


# ======================================================================
# Private implementations
# ======================================================================

def _umap(vectors: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    try:
        import umap  # type: ignore[import]
    except ImportError:
        raise ImportError("umap-learn is required: pip install umap-learn")

    defaults = dict(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=42)
    defaults.update(kwargs)
    reducer = umap.UMAP(**defaults)
    return reducer.fit_transform(vectors).astype(np.float32)


def _pca(vectors: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, **kwargs)
    return pca.fit_transform(vectors).astype(np.float32)


def _tsne(vectors: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    from sklearn.manifold import TSNE

    defaults = dict(n_components=n_components, perplexity=30, random_state=42)
    defaults.update(kwargs)
    tsne = TSNE(**defaults)
    return tsne.fit_transform(vectors).astype(np.float32)
