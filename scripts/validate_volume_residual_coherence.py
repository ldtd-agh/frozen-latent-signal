"""
Volume-only and volume-residualized coherence (review M1).

TCC is dominated by GTV volume (R^2 = 0.92 for whole-volume TCC, see
`validate_tcc_decomposition.py`), so coherence with log TCC could be nothing
more than coherence with tumour size. This script runs the exact same
coherence test as the headline table (`cellularity_coherence.py`: k=8
k-NN graph in each frozen embedding, Geary's C, 999-permutation null, full
graph and same-patient-excluded graph) on four alternative targets per
backend, each computed on the backend's native field of view
(whole-volume for 3-D / volume-pooled backends, single GTV-argmax slice for
the 2-D backends):

  log_tcc        log10 TCC -- the headline target (reproduces Table 3)
  log_volume     log10 GTV volume (2-D: log10 slice tumour area) alone
  resid_volume   residual of log_tcc on log_volume (OLS)
  resid_vol_adc  residual of log_tcc on log_volume + 6 within-mask ADC
                 statistics (mean, median, std, p10, p90, skew)
  adc_mean       within-mask mean ADC alone (no volume information)
  resid_adc      residual of adc_mean on log_volume (ADC with any size
                 association removed)

Only coherence with the residual targets (or with ADC alone) can support a
claim that an embedding tracks something about cellularity beyond tumour
burden.

Residualization is in-sample OLS over the analysis cohort; it uses only the
target and the per-scan features, never the embeddings, so it cannot leak
embedding information into the target.

Reads:  results/tcc_table.csv, results/validation/tcc_decomposition_features.csv,
        stores/cfb_gbm_scan/cfb_gbm_<backend>.h5
Writes: results/validation/volume_residual_coherence.csv

Usage: python scripts/validate_volume_residual_coherence.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.cellularity import build_knn_graph, permutation_test
from embeddings.store import EmbeddingStore

STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
FEATURES_PATH = Path("results/validation/tcc_decomposition_features.csv")
OUT_PATH = Path("results/validation/volume_residual_coherence.csv")

BACKENDS = ["medsiglip", "medimageinsights", "biomedclip", "neurovfm", "imagenet_vit", "clip", "merlin", "med3d"]
NATIVE_TARGET = {
    "merlin": "tcc", "medimageinsights": "tcc", "med3d": "tcc", "neurovfm": "tcc",
    "biomedclip": "slice_tcc", "medsiglip": "slice_tcc", "clip": "slice_tcc",
    "imagenet_vit": "slice_tcc",
}
# Per field of view: (TCC column, size column, ADC-stat prefix)
FOV = {
    "tcc": ("tcc", "volume_cm3", "adc_"),
    "slice_tcc": ("slice_tcc", "slice_area_cm2", "slice_adc_"),
}
ADC_STATS = ["mean", "median", "std", "p10", "p90", "skew"]
TARGETS = ["log_tcc", "log_volume", "resid_volume", "resid_vol_adc", "adc_mean", "resid_adc"]

K = 8
N_PERMUTATIONS = 999
SEED = 42

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)


def _ols_residual(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Residual of y on [1, X] (OLS) and the in-sample R^2."""
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    r2 = 1.0 - resid.var() / y.var()
    return resid, float(r2)


def build_targets(fov: str) -> tuple[pd.DataFrame, dict]:
    """Per-scan table (patient_id, timepoint_days, <TARGETS>) for one field of view."""
    tcc_col, size_col, prefix = FOV[fov]
    tcc = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    feats = pd.read_csv(FEATURES_PATH, dtype={"patient_id": str})
    df = tcc.merge(feats, on=["patient_id", "timepoint_days"], how="inner", validate="one_to_one")
    adc_cols = [prefix + s for s in ADC_STATS]
    df = df.dropna(subset=[tcc_col, size_col, *adc_cols])
    df = df[(df[tcc_col] > 0) & (df[size_col] > 0)].reset_index(drop=True)

    out = df[["patient_id", "timepoint_days"]].copy()
    out["log_tcc"] = np.log10(df[tcc_col].to_numpy(float))
    out["log_volume"] = np.log10(df[size_col].to_numpy(float))
    out["adc_mean"] = df[prefix + "mean"].to_numpy(float)

    out["resid_volume"], r2_v = _ols_residual(out["log_tcc"].to_numpy(), out[["log_volume"]].to_numpy())
    Xva = np.column_stack([out["log_volume"].to_numpy(), df[adc_cols].to_numpy(float)])
    out["resid_vol_adc"], r2_va = _ols_residual(out["log_tcc"].to_numpy(), Xva)
    out["resid_adc"], r2_adc = _ols_residual(out["adc_mean"].to_numpy(), out[["log_volume"]].to_numpy())
    info = {"n": len(out), "r2_volume": r2_v, "r2_vol_adc": r2_va, "r2_adc_on_volume": r2_adc}
    log.info("%-9s n=%d  in-sample R^2: volume=%.3f  volume+ADC=%.3f  adc_mean~volume=%.3f",
             fov, len(out), r2_v, r2_va, r2_adc)
    return out, info


def align(store: EmbeddingStore, table: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Embeddings for the rows of `table` present in `store` (same matching rule as
    `analysis.cellularity.build_coherence_dataset`), and the matching rows."""
    X_rows, keep = [], []
    stored = set(store.patients())
    for i, row in table.iterrows():
        if row["patient_id"] not in stored:
            continue
        vectors, timepoints, _ = store.load_patient(row["patient_id"])
        idx = np.where(np.isclose(timepoints, row["timepoint_days"], atol=1.0))[0]
        if idx.size == 0:
            continue
        X_rows.append(vectors[idx[0]])
        keep.append(i)
    return np.stack(X_rows).astype(np.float32), table.loc[keep].reset_index(drop=True)


def main() -> None:
    tables = {fov: build_targets(fov) for fov in FOV}
    rows = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue
        fov = NATIVE_TARGET[backend]
        table, info = tables[fov]
        X, sub = align(EmbeddingStore(store_path), table)
        pids = sub["patient_id"].to_numpy(dtype=object)
        graphs = {
            "full": build_knn_graph(X, k=K),
            "excl": build_knn_graph(X, k=K, patient_ids=pids, exclude_same_patient=True),
        }
        for target in TARGETS:
            y = sub[target].to_numpy(float)
            row = {"backend": backend, "fov": fov, "target": target, "n": len(y), "k": K, **info}
            for gname, W in graphs.items():
                res = permutation_test(y, W, statistic="gearys_c", n_permutations=N_PERMUTATIONS, seed=SEED)
                row[f"{gname}_C"] = res["observed"]
                row[f"{gname}_p"] = res["p_value"]
                row[f"{gname}_null_mean"] = res["null_mean"]
            rows.append(row)
            log.info("%-16s %-13s n=%d  full C=%.3f p=%.3f | excl C=%.3f p=%.3f",
                     backend, target, len(y), row["full_C"], row["full_p"], row["excl_C"], row["excl_p"])

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    pivot = out.pivot(index="backend", columns="target", values="excl_C").reindex(
        [b for b in BACKENDS if b in set(out["backend"])])[TARGETS]
    pivot_p = out.pivot(index="backend", columns="target", values="excl_p").reindex(pivot.index)[TARGETS]
    print("\nSame-patient-excluded Geary's C (k=8)\n", pivot.round(3).to_string())
    print("\np-values\n", pivot_p.round(3).to_string())
    print(f"\nWrote {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
