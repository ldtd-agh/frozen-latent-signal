"""
REVISIONS_1.md reviewer point 1: "make sure the permutation null actually
tests the absence of an embedding-distance/TCC-distance association and
accounts for patient clustering."

The existing permutation null (`analysis.cellularity.permutation_test`,
used for every headline Geary's C / Moran's I number in
`tab:headline_coherence` and `tab:second_metrics`) shuffles y at the
individual-scan level (`rng.permutation(y)`). CFB-GBM has 204 patients
contributing 332 scans (100 patients x1 scan, 80 x2, 24 x3) -- a flat
scan-level shuffle treats a patient's 2-3 temporally-correlated scans as
independent exchangeable units, which is exactly the clustering concern
the reviewer raises. This is true even though the "same-patient-excluded"
graph variant already drops same-patient *edges* -- that's a fix to the
graph, not to the null's exchangeability unit.

This script reruns both Geary's C and Moran's I, full graph and
same-patient-excluded graph, for all eight backbends on their native
target, with `analysis.cellularity.cluster_permutation_test` (patient-
block permutation: entire patients' observed y-blocks are swapped with
other same-scan-count patients', never torn apart scan-by-scan -- see
that function's docstring) instead of the original `permutation_test`,
and reports both p-values side by side so a reader can see directly
whether the conclusion changes.

Writes: results/validation/cluster_permutation_robustness.csv

Usage: python scripts/validate_cluster_permutation.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from analysis.cellularity import (
    build_coherence_dataset, build_knn_graph, cluster_permutation_test, permutation_test,
)

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"
STORE_DIR = Path("stores/cfb_gbm_scan")

NATIVE_TARGET = {
    "merlin": "tcc", "medimageinsights": "tcc", "med3d": "tcc", "neurovfm": "tcc",
    "biomedclip": "slice_tcc", "medsiglip": "slice_tcc", "clip": "slice_tcc",
    "imagenet_vit": "slice_tcc",
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
    from embeddings.store import EmbeddingStore

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})

    rows = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue
        target = NATIVE_TARGET[backend]
        store = EmbeddingStore(store_path)
        X, y, patient_ids = build_coherence_dataset(store, table, target_column=target)
        if len(y) < K + 1:
            log.warning("%s: too few samples, skipping", backend)
            continue

        W_full = build_knn_graph(X, k=K)
        W_excl = build_knn_graph(X, k=K, patient_ids=patient_ids, exclude_same_patient=True)

        for statistic in ("gearys_c", "morans_i"):
            for graph_name, W in (("full", W_full), ("same_patient_excluded", W_excl)):
                old = permutation_test(y, W, statistic=statistic, n_permutations=N_PERMUTATIONS)
                new = cluster_permutation_test(
                    y, W, patient_ids, statistic=statistic, n_permutations=N_PERMUTATIONS,
                )
                log.info(
                    "%-16s  %-9s  %-22s  observed=%+.3f  scan-level p=%.4f  patient-block p=%.4f  %s",
                    backend, statistic, graph_name, old["observed"], old["p_value"], new["p_value"],
                    "SAME" if (old["p_value"] < 0.05) == (new["p_value"] < 0.05) else "**CONCLUSION FLIPS**",
                )
                rows.append({
                    "backend": backend, "target": target, "statistic": statistic, "graph": graph_name,
                    "n_samples": int(len(y)), "observed": old["observed"],
                    "scan_level_p": old["p_value"], "patient_block_p": new["p_value"],
                    "scan_level_null_std": old["null_std"], "patient_block_null_std": new["null_std"],
                })

    df = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "cluster_permutation_robustness.csv"
    df.to_csv(out_path, index=False)
    print()
    print("=" * 120)
    print("  Scan-level vs. patient-block permutation null -- headline Geary's C / Moran's I")
    print("=" * 120)
    print(df.to_string(index=False))
    print(f"\n  Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
