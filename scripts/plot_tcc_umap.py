"""
UMAP embedding visualisation coloured by ADC-derived tumour cell count (TCC).

For each of the 7 frozen-embedding backends, projects the 334-scan
scan-paired embedding space (stores/cfb_gbm_scan/) to 2D via UMAP and
colours each point by log10(TCC) (results/tcc_table.csv) — the same
alignment and log-transform used in the Task 3 coherence test
(`analysis.cellularity.build_coherence_dataset`), so the figure visualises
exactly the signal that test measures.

Publication figure: one panel per backend, a single shared log10(TCC)
colour scale across all panels (directly comparable across backends since
it's the same underlying scans), viridis colormap (perceptually uniform,
colorblind-safe, high contrast — dark purple = low cellularity, yellow =
high). Wrapped into a grid (`GRID_N_COLS`, 4 -> 2x4, one cell unused for 7
backends) rather than one wide strip. Saved as PNG (300 dpi) and PDF
(vector) for print, plus one individual PNG per backend.

Writes:
  results/figures/tcc_umap.png / .pdf         — combined grid figure
  results/figures/tcc_umap_<backend>.png      — individual panels
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.cellularity import build_coherence_dataset
from analysis.reduction import reduce
from analysis.visualise import plot_embedding_by_value, plot_embedding_grid
from embeddings.store import EmbeddingStore

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
FIGURES_DIR = Path("results/figures")

# Includes 'clip' -- a non-medical baseline (vanilla OpenAI CLIP ViT-B/16,
# same architecture BiomedCLIP fine-tunes from) -- and 'med3d', a second
# 3D-volumetric medical backbone (different pretraining recipe than Merlin).
# 'neurovfm' is the one backbone actually pretrained on brain MRI/CT.
BACKENDS = ["biomedclip", "medsiglip", "clip", "imagenet_vit", "neurovfm", "merlin", "medimageinsights", "med3d"]
BACKEND_LABELS = {
    "merlin": "Merlin",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
    "medimageinsights": "MedImageInsights",
    "clip": "CLIP (non-medical)",
    "med3d": "Med3D",
    "neurovfm": "NeuroVFM",
    "imagenet_vit": "ImageNet ViT (non-medical)",
}

UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.1, random_state=42)
COLORBAR_LABEL = r"$\log_{10}$(TCC)"
GRID_N_COLS = 4  # 8 backends -> full 2x4 grid

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    log.info("Loaded TCC table: %d rows", len(table))

    # Pass 1: UMAP-reduce every backend and collect (coords, y) so the final
    # figure can share one log10(TCC) colour scale across all panels.
    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue

        store = EmbeddingStore(store_path)
        X, y, patient_ids = build_coherence_dataset(store, table, log_transform=True)
        log.info("%-16s  n=%d  running UMAP …", backend, len(y))
        coords = reduce(X, method="umap", n_components=2, **UMAP_KWARGS)
        panels.append((backend, coords, y))

    if not panels:
        raise SystemExit("No backend stores found — run scripts/embed_cfb_gbm.py first.")

    vmin = min(y.min() for _, _, y in panels)
    vmax = max(y.max() for _, _, y in panels)

    # Individual per-backend panels (useful when a paper only has room for one).
    for backend, coords, y in panels:
        fig = plot_embedding_by_value(
            coords, y,
            title=BACKEND_LABELS.get(backend, backend),
            value_label=COLORBAR_LABEL,
            vmin=vmin, vmax=vmax,
            output_path=FIGURES_DIR / f"tcc_umap_{backend}.png",
        )
        plt.close(fig)

    # Combined grid comparison figure with one shared colorbar.
    grid_panels = [(BACKEND_LABELS.get(b, b), coords, y) for b, coords, y in panels]
    fig = plot_embedding_grid(
        grid_panels,
        n_cols=GRID_N_COLS,
        vmin=vmin, vmax=vmax,
        value_label=COLORBAR_LABEL,
        suptitle="UMAP projection of frozen embeddings, colored by ADC-derived tumor cell count",
        output_path=FIGURES_DIR / "tcc_umap.png",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
