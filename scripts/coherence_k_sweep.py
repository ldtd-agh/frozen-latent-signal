"""
Geary's C sensitivity to k (k-NN graph size), across all backends including
the non-medical CLIP baseline — a robustness check on Task 3's coherence
test, and a direct comparison of medical-domain vs. generic pretraining at
every k.

Run against both TCC targets (see analysis.cellularity.compute_scan_tcc):
  - 'tcc'       whole-volume, the fair target for Merlin/MedImageInsights
                (which see the full 3D volume).
  - 'slice_tcc' restricted to the single GTV-argmax axial slice, the fair
                target for BiomedCLIP/MedSigLIP/CLIP (which
                only ever see that one slice).
Both are swept for every backend so the difference is visible, not assumed.

TCC and embeddings are already computed (see scripts/embed_cfb_gbm.py and
scripts/cellularity_coherence.py); this only reruns the cheap part
(k-NN graph + Geary's C + permutation test) for each (backend, target, k)
triple, so it's fast even though it multiplies the number of permutation
tests run.

Writes:
  results/coherence_k_sweep.csv
  results/figures/coherence_k_sweep_tcc.png / .pdf
  results/figures/coherence_k_sweep_slice_tcc.png / .pdf
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from analysis.cellularity import build_coherence_dataset, build_knn_graph, permutation_test
from embeddings.store import EmbeddingStore

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
RESULTS_DIR = Path("results")
FIGURES_DIR = Path("results/figures")

# Order matters: matched 1:1 against the fixed categorical hue order below.
# 'med3d' is a second 3D-volumetric medical backbone (different pretraining
# recipe than Merlin), added to help separate "3D architecture matters" from
# "Merlin's specific pretraining matters" -- it's a regular peer here, not a
# baseline, so it gets a normal solid line like the other medical backbones
# (only the two non-medical baselines, 'clip' and 'imagenet_vit' -- BiomedCLIP's
# ImageNet-21k initialization -- get the dashed treatment).
BACKENDS = ["merlin", "biomedclip", "medsiglip", "medimageinsights", "clip", "med3d", "neurovfm", "imagenet_vit"]
BACKEND_LABELS = {
    "merlin": "Merlin",
    "biomedclip": "BiomedCLIP",
    "medsiglip": "MedSigLIP",
    "medimageinsights": "MedImageInsights",
    "clip": "CLIP (non-medical baseline)",
    "med3d": "Med3D",
    "neurovfm": "NeuroVFM",
    "imagenet_vit": "ImageNet ViT (non-medical baseline)",
}
TARGETS = ["tcc", "slice_tcc"]
TARGET_TITLES = {
    "tcc": "Target: whole-volume TCC",
    "slice_tcc": "Target: single-slice TCC (matches the 2D backends' actual input)",
}
# Fixed categorical order (validated for adjacent-pair colorblind safety) —
# never cycled/reassigned by filtering. CLIP gets a dashed line on top of its
# color to flag it as the non-medical reference condition, not a peer.
_CATEGORICAL_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#008300", "#4a3aa7", "#8c564b", "#c2185b"]

K_VALUES = [3, 5, 8, 12, 16, 20, 25, 30]
N_PERMUTATIONS = 999
STATISTIC = "gearys_c"
SEED = 42

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def _sweep_target(table: "pd.DataFrame", target: str) -> list[dict]:
    rows: list[dict] = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue

        store = EmbeddingStore(store_path)
        X, y, patient_ids = build_coherence_dataset(store, table, target_column=target, log_transform=True)
        log.info("%-28s target=%-10s n=%d", BACKEND_LABELS.get(backend, backend), target, len(y))
        if len(y) == 0:
            continue

        for k in K_VALUES:
            W_full = build_knn_graph(X, k=k)
            full = permutation_test(
                y, W_full, statistic=STATISTIC, n_permutations=N_PERMUTATIONS, seed=SEED
            )
            W_excl = build_knn_graph(X, k=k, patient_ids=patient_ids, exclude_same_patient=True)
            excl = permutation_test(
                y, W_excl, statistic=STATISTIC, n_permutations=N_PERMUTATIONS, seed=SEED
            )
            log.info(
                "  k=%-3d  full C=%+.3f p=%.3f  |  same-patient-excluded C=%+.3f p=%.3f",
                k, full["observed"], full["p_value"], excl["observed"], excl["p_value"],
            )
            rows.append(
                {
                    "backend": backend,
                    "target": target,
                    "k": k,
                    "full_observed": full["observed"],
                    "full_p_value": full["p_value"],
                    "same_patient_excluded_observed": excl["observed"],
                    "same_patient_excluded_p_value": excl["p_value"],
                }
            )
    return rows


def _plot_target(df: "pd.DataFrame", target: str) -> None:
    import matplotlib.pyplot as plt

    color_by_backend = {b: _CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)] for i, b in enumerate(BACKENDS)}

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for backend in BACKENDS:
        sub = df[df["backend"] == backend]
        if sub.empty:
            continue
        label = BACKEND_LABELS.get(backend, backend)
        color = color_by_backend[backend]
        is_baseline = backend in ("clip", "imagenet_vit")  # the two non-medical baselines
        style = dict(
            marker="o", markersize=5, linewidth=2.4 if is_baseline else 1.8,
            linestyle="--" if is_baseline else "-", color=color, label=label,
        )
        axes[0].plot(sub["k"], sub["full_observed"], **style)
        axes[1].plot(sub["k"], sub["same_patient_excluded_observed"], **style)

    for ax, subtitle in zip(axes, ["Full k-NN graph", "Same-patient neighbors excluded"]):
        ax.axhline(1.0, color="#898781", linewidth=1, linestyle=":", zorder=0)
        ax.set_title(subtitle, fontsize=12)
        ax.set_xlabel("k (neighbors)")
        ax.set_xticks(K_VALUES)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_ylabel("Geary's C")
    axes[1].legend(fontsize=8, loc="best", frameon=False)
    fig.suptitle(
        f"Geary's C vs. k-NN graph size — {TARGET_TITLES[target]}",
        fontsize=13, y=1.03,
    )
    plt.tight_layout()

    png_path = FIGURES_DIR / f"coherence_k_sweep_{target}.png"
    pdf_path = FIGURES_DIR / f"coherence_k_sweep_{target}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", png_path.resolve())
    log.info("Saved %s", pdf_path.resolve())


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    log.info("Loaded TCC table: %d rows", len(table))

    all_rows: list[dict] = []
    for target in TARGETS:
        if target not in table.columns:
            log.warning("Target column %r not in TCC table, skipping.", target)
            continue
        all_rows.extend(_sweep_target(table, target))

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "coherence_k_sweep.csv", index=False)
    log.info("Saved %s", (RESULTS_DIR / "coherence_k_sweep.csv").resolve())

    for target in TARGETS:
        sub = df[df["target"] == target]
        if not sub.empty:
            _plot_target(sub, target)


if __name__ == "__main__":
    main()
