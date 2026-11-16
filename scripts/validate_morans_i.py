"""
Moran's I as a second spatial-autocorrelation statistic (review response,
Tier 1 item 3), reported alongside the existing Geary's C headline results
in `results/coherence_results.csv` for the same frozen backends/targets.
`morans_i()` already existed in `analysis/cellularity.py` but was never
actually run/reported -- this just calls `coherence_test_backend` a second
time with `statistic="morans_i"`, same k, same permutation null, same
same-patient-exclusion confound control as the Geary's C results already
in the paper.

Not independent evidence of downstream usefulness on its own (same graph,
same neighbors, just a different formula) -- see VALIDATION_REVIEW.md.

Writes: results/validation/morans_i_results.csv

Usage: python scripts/validate_morans_i.py
"""

from __future__ import annotations

import logging
from pathlib import Path

from analysis.cellularity import coherence_test_backend

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"
STORE_DIR = Path("stores/cfb_gbm_scan")

NATIVE_TARGET = {
    "merlin": "tcc",
    "medimageinsights": "tcc",
    "biomedclip": "slice_tcc",
    "medsiglip": "slice_tcc",
    "clip": "slice_tcc",
    "imagenet_vit": "slice_tcc",
    "med3d": "tcc",
    "neurovfm": "tcc",
}
BACKENDS = list(NATIVE_TARGET)
K = 8
N_PERMUTATIONS = 999

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    import pandas as pd

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})

    rows = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue
        target = NATIVE_TARGET[backend]
        result = coherence_test_backend(
            store_path, table, target_column=target, k=K,
            statistic="morans_i", n_permutations=N_PERMUTATIONS,
        )
        full, excl = result["full_graph"], result["same_patient_excluded"]
        if full is None:
            log.warning("%s: too few samples, skipping", backend)
            continue
        log.info(
            "%-16s  target=%-10s  n=%3d  full: I=%+.3f p=%.3f  |  same-patient-excluded: I=%+.3f p=%.3f",
            backend, target, result["n_samples"],
            full["observed"], full["p_value"], excl["observed"], excl["p_value"],
        )
        rows.append({
            "backend": backend, "target": target, "n_samples": result["n_samples"], "k": K,
            "full_observed": full["observed"], "full_p_value": full["p_value"],
            "same_patient_excluded_observed": excl["observed"],
            "same_patient_excluded_p_value": excl["p_value"],
        })

    out = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "morans_i_results.csv"
    out.to_csv(out_path, index=False)
    print()
    print("=" * 100)
    print("  Moran's I (frozen embeddings, native target per backend)")
    print("=" * 100)
    print(out.to_string(index=False))
    print(f"\n  Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
