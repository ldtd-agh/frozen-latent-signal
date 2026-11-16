"""
Example-patient figure (Statement of Significance Q9).

One CFB-GBM scan shown as the pipeline sees it: T1Gd (the only model input),
ADC (used only to compute the TCC target) and the GTV mask, in two
orthogonal planes. The axial row is the exact slice the 2-D backbones
receive (the GTV-argmax slice, `select_best_axial_slice`); the coronal row
passes through the GTV centroid and marks the selected slice's level with a
dashed line.

The default example is a typical scan: the t0 scan whose log10 TCC is
closest to the median among patients with all three timepoints.

Writes: tex_source/images/example_patient.pdf (+ .png preview)
Usage:  python scripts/plot_example_patient.py [patient_id] [timepoint]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk

from data.datasets.cfb_gbm import _CANONICAL_ORIENTATION, _scan_path

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TCC_TABLE_PATH = Path("results/tcc_table.csv")
OUT_PATH = Path("tex_source/images/example_patient.pdf")
CROP_MARGIN_MM = 8.0  # margin around the brain mask's bounding box

GTV_COLOR = "#e8590c"
SLICE_LINE_COLOR = "#ffd43b"


def default_example() -> tuple[str, str]:
    t = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    t = t[t["tcc"] > 0]
    three = t.groupby("patient_id").size().loc[lambda s: s == 3].index
    t0 = t[t["patient_id"].isin(three) & (t["timepoint_label"] == "t0")].copy()
    target = np.median(np.log10(t0["tcc"]))
    row = t0.iloc[np.argmin(np.abs(np.log10(t0["tcc"]) - target))]
    return row["patient_id"], row["timepoint_label"]


def load(patient_num: int, timepoint: str, modality: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Array [D, H, W] in the dataset's canonical orientation, and spacing (sx, sy, sz) in mm."""
    img = sitk.DICOMOrient(sitk.ReadImage(str(_scan_path(IMAGING_ROOT, patient_num, timepoint, modality))),
                           _CANONICAL_ORIENTATION)
    return sitk.GetArrayFromImage(img).astype(np.float32), img.GetSpacing()


def window(vol: np.ndarray, mask: np.ndarray, lo: float = 1, hi: float = 99.5) -> tuple[float, float]:
    vals = vol[mask & (vol != 0)]
    return tuple(np.percentile(vals, [lo, hi]))


def main() -> None:
    if len(sys.argv) >= 3:
        pid, tp = sys.argv[1], sys.argv[2]
    else:
        pid, tp = default_example()
    num = int(pid)

    t1, sp = load(num, tp, "t1gd")
    adc, _ = load(num, tp, "adc")
    gtv, _ = load(num, tp, "gtv")
    gtv = gtv > 0
    brain = load(num, tp, "brain_mask")[0] > 0 if _scan_path(IMAGING_ROOT, num, tp, "brain_mask").exists() else t1 > 0
    sx, sy, sz = sp  # array axes are [z, y, x]

    # Same rule as select_best_axial_slice: axial slice with the most GTV voxels.
    z_sel = int(np.argmax(gtv.sum(axis=(1, 2))))
    y_cor = int(round(np.argwhere(gtv)[:, 1].mean()))

    # Crop to the brain bounding box (+ margin) in each in-plane axis.
    nz = np.argwhere(brain)
    lo, hi = nz.min(0), nz.max(0)
    m = np.array([CROP_MARGIN_MM / sz, CROP_MARGIN_MM / sy, CROP_MARGIN_MM / sx]).astype(int)
    lo, hi = np.maximum(lo - m, 0), np.minimum(hi + m, np.array(t1.shape) - 1)
    zs, ys, xs = slice(lo[0], hi[0] + 1), slice(lo[1], hi[1] + 1), slice(lo[2], hi[2] + 1)

    t1_win, adc_win = window(t1, brain), window(adc, brain)

    def axial(v):
        return v[z_sel, ys, xs]

    def coronal(v):
        return v[zs, y_cor, xs]

    tcc_row = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str}).query(
        "patient_id == @pid and timepoint_label == @tp").iloc[0]
    vol_cm3 = gtv.sum() * sx * sy * sz / 1000.0

    planes = [
        ("Axial (selected 2-D slice)", axial, (sy, sx)),
        ("Coronal (through GTV centroid)", coronal, (sz, sx)),
    ]
    cols = [("T1Gd (model input)", "t1"), ("ADC ($\\times10^{-6}$ mm$^2$/s)", "adc"), ("T1Gd + GTV mask", "gtv")]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [
                                 (hi[1] - lo[1]) * sy, (hi[0] - lo[0]) * sz]})
    for r, (plane_name, cut, (row_mm, col_mm)) in enumerate(planes):
        aspect = row_mm / col_mm
        for c, (title, kind) in enumerate(cols):
            ax = axes[r, c]
            if kind == "adc":
                im = ax.imshow(cut(adc), cmap="gray", vmin=adc_win[0], vmax=adc_win[1],
                               aspect=aspect, origin="lower", interpolation="nearest")
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
                cb.ax.tick_params(labelsize=6)
            else:
                ax.imshow(cut(t1), cmap="gray", vmin=t1_win[0], vmax=t1_win[1],
                          aspect=aspect, origin="lower", interpolation="nearest")
            if kind == "gtv":
                g = cut(gtv.astype(float))
                ax.imshow(np.ma.masked_where(g == 0, g), cmap=matplotlib.colors.ListedColormap([GTV_COLOR]),
                          alpha=0.35, aspect=aspect, origin="lower", interpolation="nearest")
                ax.contour(g, levels=[0.5], colors=[GTV_COLOR], linewidths=0.9, origin="lower")
            if r == 1:  # mark the selected axial slice's level on the coronal view
                ax.axhline(z_sel - lo[0], color=SLICE_LINE_COLOR, lw=0.8, ls="--")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 0:
                ax.set_ylabel(plane_name, fontsize=8)

    fig.suptitle(
        f"Patient {pid}, {tp}: GTV {vol_cm3:.1f} cm$^3$, TCC = {tcc_row['tcc']:.2e} cells "
        f"(log$_{{10}}$ = {np.log10(tcc_row['tcc']):.2f})",
        fontsize=8.5,
    )
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    fig.savefig(OUT_PATH.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"patient={pid} timepoint={tp} z_sel={z_sel} y_cor={y_cor} spacing={sp} "
          f"GTV={vol_cm3:.2f} cm3 TCC={tcc_row['tcc']:.3e} slice_index(table)={tcc_row['slice_index']}")
    print(f"Wrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
