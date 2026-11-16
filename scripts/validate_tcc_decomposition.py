"""
Construct-validity check for TCC (review response, Tier 1 item 1): how much
of TCC's variation is explained by GTV volume and within-mask ADC summary
statistics alone -- the "ingredients" TCC is built from -- versus requiring
something beyond them.

Not a validation that TCC measures true histological cellularity (that
needs an independent biological reference, see VALIDATION_REVIEW.md); this
only tests whether TCC collapses to a simple function of volume/ADC, which
would undercut treating it as a richer signal.

For each of `tcc` (whole-volume) and `slice_tcc` (single GTV-argmax axial
slice), fits patient-grouped-CV ridge regressions of log10(target) on:
  A. volume only      (log10(volume_cm3) / log10(slice_area_cm2))
  B. volume + ADC      (A's feature + adc_{mean,median,std,p10,p90,skew})
and reports pooled out-of-fold R^2, MAE, Spearman rho.

Writes:
  results/validation/tcc_decomposition_features.csv  -- per-scan features
  results/validation/tcc_decomposition_regression.csv -- per (target, feature_set) metrics

Usage: python scripts/validate_tcc_decomposition.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from analysis.cellularity import build_decomposition_table
from data.datasets.cfb_gbm import CFBGBMDataset
from finetune.trainer import k_fold_patient_splits

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"
FEATURES_PATH = VALIDATION_DIR / "tcc_decomposition_features.csv"

FORCE_RECOMPUTE = False
K_FOLDS = 5
FOLD_SEED = 42

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)

FEATURE_SETS = {
    "tcc": {
        "A_volume_only": ["log_volume"],
        "B_volume_adc": ["log_volume", "adc_mean", "adc_median", "adc_std", "adc_p10", "adc_p90", "adc_skew"],
    },
    "slice_tcc": {
        "A_volume_only": ["log_slice_area"],
        "B_volume_adc": [
            "log_slice_area", "slice_adc_mean", "slice_adc_median", "slice_adc_std",
            "slice_adc_p10", "slice_adc_p90", "slice_adc_skew",
        ],
    },
}


def grouped_cv_ridge(X, y, groups, k=K_FOLDS, seed=FOLD_SEED) -> dict:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    from sklearn.metrics import r2_score, mean_absolute_error

    folds = k_fold_patient_splits(list(groups), k, seed)
    y_true_all, y_pred_all = [], []
    for train_ids, val_ids in folds:
        train_idx = [i for i, g in enumerate(groups) if g in train_ids]
        val_idx = [i for i, g in enumerate(groups) if g in val_ids]
        if not train_idx or not val_idx:
            continue
        scaler = StandardScaler().fit(X[train_idx])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[train_idx]), y[train_idx])
        y_pred_all.append(model.predict(scaler.transform(X[val_idx])))
        y_true_all.append(y[val_idx])

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "spearman": float(rho),
    }


def main() -> None:
    import numpy as np
    import pandas as pd

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)

    if FEATURES_PATH.exists() and not FORCE_RECOMPUTE:
        log.info("Reusing cached decomposition features: %s", FEATURES_PATH.resolve())
        features = pd.read_csv(FEATURES_PATH, dtype={"patient_id": str})
    else:
        log.info("Building CFB-GBM scan-level dataset (pairing='scan') …")
        dataset = CFBGBMDataset(IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan")
        log.info("Computing volume/ADC decomposition features for every qualifying scan …")
        features = build_decomposition_table(dataset)
        features.to_csv(FEATURES_PATH, index=False)
        log.info("Wrote %s", FEATURES_PATH.resolve())

    tcc_table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    merged = tcc_table.merge(features, on=["patient_id", "timepoint_days"], how="inner")
    merged["log_volume"] = np.log10(merged["volume_cm3"].clip(lower=1e-6))
    merged["log_slice_area"] = np.log10(merged["slice_area_cm2"].clip(lower=1e-6))

    rows = []
    for target, feature_sets in FEATURE_SETS.items():
        sub = merged.dropna(subset=[target]).copy()
        sub = sub[sub[target] > 0]
        sub["log_target"] = np.log10(sub[target])

        for name, cols in feature_sets.items():
            avail = sub.dropna(subset=cols)
            if len(avail) < K_FOLDS * 2:
                log.warning("%s / %s: too few complete rows (n=%d), skipping", target, name, len(avail))
                continue
            X = avail[cols].to_numpy(dtype=np.float64)
            y = avail["log_target"].to_numpy(dtype=np.float64)
            groups = avail["patient_id"].tolist()
            metrics = grouped_cv_ridge(X, y, groups)
            log.info(
                "%-10s  %-16s  n=%3d  R2=%+.3f  MAE=%.3f  Spearman=%+.3f",
                target, name, metrics["n"], metrics["r2"], metrics["mae"], metrics["spearman"],
            )
            rows.append({"target": target, "feature_set": name, "features": ",".join(cols), **metrics})

    out = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "tcc_decomposition_regression.csv"
    out.to_csv(out_path, index=False)

    print()
    print("=" * 100)
    print("  TCC decomposition (construct validity) -- held-out ridge regression of log10(target)")
    print("=" * 100)
    print(out.to_string(index=False))
    print()
    print(f"  Features : {FEATURES_PATH.resolve()}")
    print(f"  Metrics  : {out_path.resolve()}")


if __name__ == "__main__":
    main()
