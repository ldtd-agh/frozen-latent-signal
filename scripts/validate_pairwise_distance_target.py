"""
A third flavor of evidence, deliberately different from both Geary's
C/Moran's I (local k-NN graph) and the ridge-regression probe (global
linear recoverability): pairwise embedding-distance vs.\ pairwise
|log10(TCC) difference| correlation, across *all* eligible pairs, not just
each point's k nearest neighbors. This was one of the original external
review's suggestions (NEW_AMENDS.md, "pairwise distance-target
association") that was deprioritized earlier as redundant with the k-NN
based checks -- worth actually running now that the regression probe
diverged from Geary's C/Moran's I, specifically to see whether this
metric tracks those two (another vote for "it's all the same signal") or
moves independently (a genuinely different angle on representation
quality).

Metric: Spearman correlation between ||z_i - z_j|| and
|y_i - y_j| (y = log10(TCC)) over all same-target, different-patient pairs
-- positive correlation means "farther apart in embedding space tends to
mean farther apart in TCC", i.e.\ the embedding's metric structure (not
just its k-NN topology) respects the target.

Frozen: single consistent embedding space, full O(n^2) pairwise
correlation (n=332, same-patient pairs excluded).

Fine-tuned: computed *within* each of the 5 k-fold OOF folds separately
(never across two independently-trained fold-models' embeddings -- the
same pooling confound already fixed for the UMAP figure). This script's
own `run_finetuned` scores every fold against the OOF cache's cached
uniform whole-volume color, which is NOT each backbone's native target
for the four single-slice backbones -- superseded for that reason by
`scripts/validate_pairwise_distance_significance.py`, which recovers each
backbone's true native-target value instead and is the source of this
paper's reported fine-tuned numbers (Table tab:pairwise_distance_finetuned
in main.tex). Keep using *this* script's `run_frozen` for the frozen-only
numbers (Table tab:pairwise_distance); its `run_finetuned` is left here
only as a quick diagnostic, not a source of paper numbers.

Significance (frozen only, cheap enough at this N): a patient-level
permutation test -- shuffle y across patients 999 times, keep embedding
distances fixed, recompute the correlation each time.

Writes: results/validation/pairwise_distance_target.csv

Usage: python scripts/validate_pairwise_distance_target.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from analysis.cellularity import build_coherence_dataset
from finetune.trainer import k_fold_patient_splits

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"
STORE_DIR = Path("stores/cfb_gbm_scan")
OOF_CACHE_DIR = RESULTS_DIR / "figures" / "_oof_cache"

NATIVE_TARGET = {
    "merlin": "tcc", "medimageinsights": "tcc", "med3d": "tcc", "neurovfm": "tcc",
    "biomedclip": "slice_tcc", "medsiglip": "slice_tcc", "clip": "slice_tcc",
    "imagenet_vit": "slice_tcc",
}
BACKENDS = list(NATIVE_TARGET)
OOF_K, OOF_SEED = 5, 42
N_PERMUTATIONS = 999

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


def pairwise_corr(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    patient_ids = np.asarray(patient_ids)
    n = len(y)
    dist = pdist(X, metric="euclidean")
    ydiff = pdist(y[:, None], metric="cityblock")  # |y_i - y_j|

    iu = np.triu_indices(n, k=1)
    same_patient = patient_ids[iu[0]] == patient_ids[iu[1]]
    keep = ~same_patient

    rho, _ = spearmanr(dist[keep], ydiff[keep])
    return float(rho), dist[keep], ydiff[keep]


def permutation_pvalue(y: np.ndarray, patient_ids: np.ndarray, dist_keep: np.ndarray, keep_mask: np.ndarray,
                        observed: float, n_perm: int = N_PERMUTATIONS, seed: int = 42) -> float:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    n = len(y)
    iu = np.triu_indices(n, k=1)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        ydiff_p = pdist(yp[:, None], metric="cityblock")[keep_mask]
        rho_p, _ = spearmanr(dist_keep, ydiff_p)
        if rho_p >= observed:
            count += 1
    return (1 + count) / (n_perm + 1)


def run_frozen(rows: list[dict]) -> None:
    import pandas as pd
    from embeddings.store import EmbeddingStore

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            continue
        target = NATIVE_TARGET[backend]
        store = EmbeddingStore(store_path)
        X, y, patient_ids = build_coherence_dataset(store, table, target_column=target)
        patient_ids = np.asarray(patient_ids)
        n = len(y)

        from scipy.spatial.distance import pdist
        dist = pdist(X, metric="euclidean")
        ydiff = pdist(y[:, None], metric="cityblock")
        iu = np.triu_indices(n, k=1)
        keep = patient_ids[iu[0]] != patient_ids[iu[1]]

        from scipy.stats import spearmanr
        rho, _ = spearmanr(dist[keep], ydiff[keep])
        p = permutation_pvalue(y, patient_ids, dist[keep], keep, rho)

        log.info("frozen     %-16s  target=%-10s  n_pairs=%6d  rho=%+.3f  p=%.3f", backend, target, keep.sum(), rho, p)
        rows.append({"embedding": "frozen", "backend": backend, "target": target, "n_pairs": int(keep.sum()), "rho": rho, "p_value": p})


def run_finetuned(rows: list[dict]) -> None:
    for backend in BACKENDS:
        cache_path = OOF_CACHE_DIR / f"{backend}.npz"
        if not cache_path.exists():
            continue
        data = np.load(cache_path, allow_pickle=True)
        X, y, pids = data["x"], data["color"], np.asarray(data["pids"])

        unique_pids = sorted(set(pids.tolist()))
        outer_folds = k_fold_patient_splits(unique_pids, OOF_K, OOF_SEED)
        pid_to_fold = {pid: i for i, (_, val_ids) in enumerate(outer_folds) for pid in val_ids}
        fold_id = np.array([pid_to_fold[p] for p in pids])

        fold_rhos = []
        for fid in range(OOF_K):
            mask = fold_id == fid
            Xf, yf, pf = X[mask], y[mask], pids[mask]
            rho, _, _ = pairwise_corr(Xf, yf, pf)
            fold_rhos.append(rho)

        fold_rhos = np.array(fold_rhos)
        log.info("fine-tuned %-16s  fold rhos=%s  mean=%+.3f  std=%.3f", backend,
                  np.round(fold_rhos, 3).tolist(), fold_rhos.mean(), fold_rhos.std())
        rows.append({"embedding": "fine-tuned", "backend": backend, "target": "tcc (uniform)",
                      "n_pairs": None, "rho": float(fold_rhos.mean()), "rho_std": float(fold_rhos.std()),
                      "p_value": None})


def main() -> None:
    import pandas as pd

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    run_frozen(rows)
    run_finetuned(rows)

    df = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "pairwise_distance_target.csv"
    df.to_csv(out_path, index=False)
    print()
    print("=" * 100)
    print("  Pairwise embedding-distance vs. pairwise |TCC difference| (Spearman rho)")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\n  Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
