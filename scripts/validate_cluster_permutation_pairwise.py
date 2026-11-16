"""
REVISIONS_1.md reviewer point 1, applied specifically to the pairwise
embedding-distance vs.\ |TCC difference| test (`tab:pairwise_distance`,
Section sec:second_metrics) -- the "new headline finding" the reviewer's
wording points at most directly, since it is newer than the original
Geary's C table and this is the first time its permutation null is
checked against patient clustering (see
`scripts/validate_cluster_permutation.py` for the same check applied to
the original Geary's C / Moran's I headline table -- same conclusion
there: no flip).

`validate_pairwise_distance_target.py`'s `permutation_pvalue` shuffles y
at the scan level (`rng.permutation(y)`), same issue as the original
Geary's C null. This reruns that same frozen-embedding test with
`analysis.cellularity.patient_block_permutation` instead (entire
patients' observed y-blocks swapped with other same-scan-count patients',
same-patient pairs still excluded from `keep` on both sides) and reports
both p-values side by side.

Writes: results/validation/cluster_permutation_pairwise.csv

Usage: python scripts/validate_cluster_permutation_pairwise.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from analysis.cellularity import build_coherence_dataset, patient_block_permutation
from validate_pairwise_distance_target import NATIVE_TARGET, BACKENDS, TCC_TABLE_PATH, STORE_DIR, N_PERMUTATIONS

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


def cluster_permutation_pvalue(y: np.ndarray, patient_ids: np.ndarray, dist_keep: np.ndarray, keep_mask: np.ndarray,
                                observed: float, n_perm: int = N_PERMUTATIONS, seed: int = 42) -> float:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = patient_block_permutation(y, patient_ids, rng)
        ydiff_p = pdist(yp[:, None], metric="cityblock")[keep_mask]
        rho_p, _ = spearmanr(dist_keep, ydiff_p)
        if rho_p >= observed:
            count += 1
    return (1 + count) / (n_perm + 1)


def main() -> None:
    import pandas as pd
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from embeddings.store import EmbeddingStore

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})

    rows = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            continue
        target = NATIVE_TARGET[backend]
        store = EmbeddingStore(store_path)
        X, y, patient_ids = build_coherence_dataset(store, table, target_column=target)
        patient_ids = np.asarray(patient_ids)
        n = len(y)

        dist = pdist(X, metric="euclidean")
        ydiff = pdist(y[:, None], metric="cityblock")
        iu = np.triu_indices(n, k=1)
        keep = patient_ids[iu[0]] != patient_ids[iu[1]]

        rho, _ = spearmanr(dist[keep], ydiff[keep])

        # old: scan-level permutation (replicate validate_pairwise_distance_target.py exactly)
        rng_old = np.random.default_rng(42)
        count_old = 0
        for _ in range(N_PERMUTATIONS):
            yp = rng_old.permutation(y)
            ydiff_p = pdist(yp[:, None], metric="cityblock")[keep]
            rho_p, _ = spearmanr(dist[keep], ydiff_p)
            if rho_p >= rho:
                count_old += 1
        p_old = (1 + count_old) / (N_PERMUTATIONS + 1)

        p_new = cluster_permutation_pvalue(y, patient_ids, dist[keep], keep, rho)

        log.info(
            "%-16s  target=%-10s  n_pairs=%6d  rho=%+.3f  scan-level p=%.3f  patient-block p=%.3f  %s",
            backend, target, keep.sum(), rho, p_old, p_new,
            "SAME" if (p_old < 0.05) == (p_new < 0.05) else "**CONCLUSION FLIPS**",
        )
        rows.append({
            "backend": backend, "target": target, "n_pairs": int(keep.sum()), "rho": rho,
            "scan_level_p": p_old, "patient_block_p": p_new,
        })

    df = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "cluster_permutation_pairwise.csv"
    df.to_csv(out_path, index=False)
    print()
    print("=" * 110)
    print("  Pairwise distance-to-target test: scan-level vs. patient-block permutation null")
    print("=" * 110)
    print(df.to_string(index=False))
    print(f"\n  Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
