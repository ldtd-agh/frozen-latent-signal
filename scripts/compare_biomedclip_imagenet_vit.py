"""
Paired comparison of BiomedCLIP with its own ImageNet-21k initialization
(`imagenet_vit`; see embeddings/backends/imagenet_vit.py). CPU only.

Both backbones embed the same scans, so their Geary's C values can be
compared on identical patients with a paired, patient-level bootstrap (same
resampling scheme as `analyze_nested_cv.bootstrap_mean_delta`: patients are
resampled with replacement, all of a patient's scans travel together, and
copies keep their patient id so the same-patient exclusion still applies):

  frozen      full 332-scan population, native single-slice TCC target, k=8,
              same-patient-excluded: C(BiomedCLIP) - C(ImageNet ViT).
  fine-tuned  nested-CV outer test folds (results/nested_cv/*.npz): per fold
              C(BiomedCLIP) - C(ImageNet ViT), for frozen and for fine-tuned
              test-fold embeddings, averaged over the five folds.

Negative differences favor BiomedCLIP. Writes results/imagenet_vit/pair_bootstrap.csv.
Usage: python scripts/compare_biomedclip_imagenet_vit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.cellularity import build_coherence_dataset  # noqa: E402
from embeddings.store import EmbeddingStore  # noqa: E402
from scripts.analyze_nested_cv import c_only, load  # noqa: E402

A, B = "biomedclip", "imagenet_vit"
STORE_DIR = _PROJECT_ROOT / "stores" / "cfb_gbm_scan"
TCC_TABLE_PATH = _PROJECT_ROOT / "results" / "tcc_table.csv"
OUT = _PROJECT_ROOT / "results" / "imagenet_vit" / "pair_bootstrap.csv"
N_BOOT = 2000
SEED = 0


def _patient_index(pids) -> tuple[np.ndarray, dict]:
    pids = np.asarray(pids)
    uniq = np.unique(pids)
    return uniq, {p: np.where(pids == p)[0] for p in uniq}


def _boot(sets: list[tuple[np.ndarray, np.ndarray, np.ndarray, list]], rng) -> np.ndarray:
    """sets: list of (xa, xb, y, pids) on identical scans; returns bootstrap of the mean C(a)-C(b)."""
    idx = [_patient_index(s[3]) for s in sets]
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        diffs = []
        for (xa, xb, y, pids), (uniq, idx_of) in zip(sets, idx):
            ii = np.concatenate([idx_of[p] for p in rng.choice(uniq, size=len(uniq), replace=True)])
            pp = [pids[i] for i in ii]
            diffs.append(c_only(xa[ii], y[ii], pp) - c_only(xb[ii], y[ii], pp))
        out[b] = np.mean(diffs)
    return out


def _row(name: str, sets, rng) -> dict:
    obs = np.mean([c_only(xa, y, p) - c_only(xb, y, p) for xa, xb, y, p in sets])
    boot = _boot(sets, rng)
    r = dict(comparison=name, observed_diff=obs, ci95_lo=np.percentile(boot, 2.5), ci95_hi=np.percentile(boot, 97.5),
             boot_frac_diff_ge0=float((boot >= 0).mean()))
    print(f"{name:28s} C({A}) - C({B}) = {obs:+.3f}  95% CI [{r['ci95_lo']:+.3f}, {r['ci95_hi']:+.3f}]  "
          f"P(boot >= 0) = {r['boot_frac_diff_ge0']:.4f}")
    return r


def main() -> None:
    rng = np.random.default_rng(SEED)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    xa, ya, pa = build_coherence_dataset(EmbeddingStore(STORE_DIR / f"cfb_gbm_{A}.h5"), table, target_column="slice_tcc")
    xb, yb, pb = build_coherence_dataset(EmbeddingStore(STORE_DIR / f"cfb_gbm_{B}.h5"), table, target_column="slice_tcc")
    assert list(pa) == list(pb) and np.allclose(ya, yb), "stores are not aligned scan-for-scan"
    rows = [_row("frozen, full population", [(xa, xb, ya, list(pa))], rng)]

    fa, _ = load(A)
    fb, _ = load(B)
    for f, g in zip(fa, fb):
        assert f["pids"] == g["pids"] and np.allclose(f["y"], g["y"]), "nested-CV test folds are not aligned"
    rows.append(_row("frozen, nested-CV test folds", [(f["x0"], g["x0"], f["y"], f["pids"]) for f, g in zip(fa, fb)], rng))
    rows.append(_row("fine-tuned, nested-CV test folds", [(f["x1"], g["x1"], f["y"], f["pids"]) for f, g in zip(fa, fb)], rng))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
