"""Scatter plots of dimensionality-reduced embeddings, colored by a continuous value
(used for the UMAP figures, colored by log10 TCC)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

# Single-hue sequential ramp (light -> dark blue), an alternative to viridis
# for continuous magnitude encoding (e.g. tumor cell count).
_SEQUENTIAL_BLUE_STEPS: list[str] = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]


def sequential_blue_cmap():
    """Perceptually-graded single-hue blue colormap for continuous magnitude."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", _SEQUENTIAL_BLUE_STEPS)


def plot_embedding_by_value(
    reduced: np.ndarray,
    values: np.ndarray,
    output_path: str | None = None,
    title: str = "Embedding (UMAP)",
    value_label: str = "value",
    figsize: tuple[float, float] = (7.0, 6.0),
    point_size: float = 32.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: "str | object" = "viridis",
    ax: "plt.Axes | None" = None,
) -> "plt.Figure":
    """
    2-D scatter of a dimensionality-reduced embedding, each point coloured
    by a continuous value (e.g. log10 tumour cell count), with a colorbar.

    `cmap` defaults to viridis (perceptually uniform, colorblind-safe, high
    contrast) — pass `sequential_blue_cmap()` instead for a more muted
    single-hue ramp.

    Pass `ax` to draw into an existing subplot (e.g. one panel of a
    multi-backend comparison figure) instead of creating a new figure;
    `output_path` and the colorbar are then the caller's responsibility.
    """
    import matplotlib.pyplot as plt

    owns_figure = ax is None
    if owns_figure:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc = ax.scatter(
        reduced[:, 0], reduced[:, 1], c=values, cmap=cmap,
        s=point_size, vmin=vmin, vmax=vmax,
        edgecolors="white", linewidths=0.3, alpha=0.9,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    if owns_figure:
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(value_label, fontsize=10)
        plt.tight_layout()
        if output_path:
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Saved → {output_path}")

    return fig


def plot_embedding_grid(
    panels: list[tuple[str, np.ndarray, np.ndarray]],
    n_cols: int = 4,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: "str | object" = "viridis",
    value_label: str = "value",
    suptitle: str | None = None,
    point_size: float = 26.0,
    panel_size: tuple[float, float] = (4.2, 4.4),
    output_path: str | None = None,
) -> "plt.Figure":
    """
    Multi-panel grid of `plot_embedding_by_value`-style scatter plots (one
    dimensionality-reduced embedding per panel, all sharing one colour scale
    and colorbar) -- wraps to `n_cols` per row instead of a single row, so a
    7-panel comparison reads as a compact 2x4 grid rather than a single very
    wide strip. A short last row (when `len(panels)` doesn't divide evenly
    by `n_cols`) is centered under the rows above it, via a double-resolution
    `GridSpec` (each panel spans 2 sub-columns; a row with `m < n_cols`
    panels is left-padded by `(n_cols - m)` sub-columns), rather than left-
    aligned with the remainder left blank.

    `panels`: list of ``(title, coords, values)`` -- `coords` is `[N, 2]`,
    `values` is `[N]` (e.g. log10(TCC)).
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    n_rows = -(-n // n_cols)  # ceil division
    fig = plt.figure(figsize=(panel_size[0] * n_cols, panel_size[1] * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols * 2)

    axes = []
    sc = None
    for idx, (title, coords, values) in enumerate(panels):
        row, col_in_row = divmod(idx, n_cols)
        row_count = min(n_cols, n - row * n_cols)  # panels actually in this row
        offset = n_cols - row_count  # sub-column left-pad centering a short row
        start = offset + col_in_row * 2
        ax = fig.add_subplot(gs[row, start:start + 2])
        sc = ax.scatter(
            coords[:, 0], coords[:, 1], c=values, cmap=cmap, vmin=vmin, vmax=vmax,
            s=point_size, edgecolors="white", linewidths=0.3, alpha=0.9,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        axes.append(ax)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(value_label, fontsize=11)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, y=1.02)

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = str(output_path).rsplit(".", 1)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved → {output_path}")
        print(f"Saved → {pdf_path}")

    return fig
