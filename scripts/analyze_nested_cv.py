"""
Analysis of `finetune_nested_cv.py` output (review M2, minor 4). CPU only.

For each backbone with results/nested_cv/<backend>.npz:

  per fold   frozen and fine-tuned same-patient-excluded Geary's C (k=8) on
             the outer test fold (never used for training or checkpoint
             selection), each with a 999-permutation p-value.
  summary    mean frozen / fine-tuned C and mean Delta across the 5 folds;
             exact Wilcoxon signed-rank test over the 5 fold pairs (two-sided,
             and one-sided for "fine-tuned C < frozen C"); a patient-level
             bootstrap 95% CI for the mean Delta: within each test fold,
             patients are resampled with replacement (all of a patient's
             scans travel together; copies keep the same patient id, so the
             same-patient exclusion stops copies from neighbouring each
             other), C is recomputed for both embeddings on the resampled
             set, and the fold Deltas are averaged.
  collapse   frozen vs. fine-tuned test-fold embeddings: mean per-dimension
             std, mean pairwise Euclidean distance, number of dead
             (std < 1e-4) dimensions, and mean per-scan cosine similarity
             between each scan's frozen and fine-tuned embedding. Averaged
             over folds. (Same check as Section "Ruling out embedding
             collapse"; this is where NeuroVFM's is computed.)
  rho        global pairwise-distance Spearman rho (different-patient pairs,
             `validate_pairwise_distance_target.pairwise_corr`) for frozen
             and fine-tuned test-fold embeddings; mean over folds and exact
             Wilcoxon over the 5 fold pairs. Checkpoints were selected on
             Geary's C on the inner split, so neither statistic was selected
             on the test patients.

Writes results/nested_cv/{per_fold,summary,collapse}.csv
Usage: python scripts/analyze_nested_cv.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import wilcoxon

from analysis.cellularity import build_knn_graph, gearys_c, permutation_test
from scripts.validate_pairwise_distance_target import pairwise_corr

DIR = Path("results/nested_cv")
BACKENDS = ["medsiglip", "medimageinsights", "biomedclip", "neurovfm", "clip", "merlin", "med3d", "imagenet_vit"]
K = 8
N_PERM = 999
N_BOOT = 2000
SEED = 0
DEAD_STD = 1e-4


def load(backend: str) -> tuple[list[dict], list[dict]]:
    d = np.load(DIR / f"{backend}.npz")
    meta = json.loads(str(d["meta"]))
    folds = [dict(x0=d[f"x_frozen_{i}"], x1=d[f"x_ft_{i}"], y=d[f"y_{i}"], pids=d[f"pids_{i}"].tolist())
             for i in range(len(meta))]
    return folds, meta


def c_test(x, y, pids) -> dict:
    return permutation_test(y, build_knn_graph(x, k=K, patient_ids=pids, exclude_same_patient=True),
                            statistic="gearys_c", n_permutations=N_PERM, seed=SEED)


def c_only(x, y, pids) -> float:
    return float(gearys_c(y, build_knn_graph(x, k=K, patient_ids=pids, exclude_same_patient=True)))


def bootstrap_mean_delta(folds: list[dict], rng: np.random.Generator) -> np.ndarray:
    # Pre-index each fold's scans by patient.
    by_patient = []
    for f in folds:
        pids = np.asarray(f["pids"])
        uniq = np.unique(pids)
        by_patient.append((uniq, {p: np.where(pids == p)[0] for p in uniq}))
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        deltas = []
        for f, (uniq, idx_of) in zip(folds, by_patient):
            draw = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([idx_of[p] for p in draw])
            pids = [f["pids"][i] for i in idx]
            deltas.append(c_only(f["x1"][idx], f["y"][idx], pids) - c_only(f["x0"][idx], f["y"][idx], pids))
        out[b] = np.mean(deltas)
    return out


def collapse_stats(x0: np.ndarray, x1: np.ndarray) -> dict:
    def cos(a, b):
        return np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)

    s0, s1 = x0.std(0), x1.std(0)
    return dict(
        std_frozen=s0.mean(), std_ft=s1.mean(), std_ratio=s1.mean() / s0.mean(),
        pdist_frozen=pdist(x0).mean(), pdist_ft=pdist(x1).mean(), pdist_ratio=pdist(x1).mean() / pdist(x0).mean(),
        dead_frozen=int((s0 < DEAD_STD).sum()), dead_ft=int((s1 < DEAD_STD).sum()), dim=x0.shape[1],
        cos_frozen_vs_ft=cos(x0, x1).mean(),
    )


def wilcoxon_p(a, b, alternative):
    if np.allclose(a, b):
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(wilcoxon(a, b, alternative=alternative, method="exact").pvalue)


def main() -> None:
    per_fold, summary, collapse = [], [], []
    for backend in BACKENDS:
        if not (DIR / f"{backend}.npz").exists():
            print(f"[skip] {backend}: no results yet")
            continue
        folds, meta = load(backend)
        c0s, c1s, rho0s, rho1s = [], [], [], []
        for f, m in zip(folds, meta):
            r0, r1 = c_test(f["x0"], f["y"], f["pids"]), c_test(f["x1"], f["y"], f["pids"])
            rho0 = pairwise_corr(f["x0"], f["y"], np.asarray(f["pids"]))[0]
            rho1 = pairwise_corr(f["x1"], f["y"], np.asarray(f["pids"]))[0]
            rho0s.append(rho0)
            rho1s.append(rho1)
            c0s.append(r0["observed"])
            c1s.append(r1["observed"])
            per_fold.append(dict(backend=backend, fold=m["fold"], n_test=m["n_test"],
                                 test_c_frozen=r0["observed"], test_p_frozen=r0["p_value"],
                                 test_c_ft=r1["observed"], test_p_ft=r1["p_value"],
                                 delta=r1["observed"] - r0["observed"], rho_frozen=rho0, rho_ft=rho1,
                                 selected_epoch=m["selected_epoch"],
                                 epochs_run=m["epochs_run"], inner_val_c_baseline=m["inner_val_c_baseline"],
                                 inner_val_c_selected=m["inner_val_c_selected"]))
            collapse.append(dict(backend=backend, fold=m["fold"], **collapse_stats(f["x0"], f["x1"])))
        c0s, c1s, rho0s, rho1s = map(np.array, (c0s, c1s, rho0s, rho1s))
        boot = bootstrap_mean_delta(folds, np.random.default_rng(SEED))
        summary.append(dict(
            backend=backend, n_folds=len(folds),
            mean_c_frozen=c0s.mean(), mean_c_ft=c1s.mean(), mean_delta=(c1s - c0s).mean(),
            ci95_lo=np.percentile(boot, 2.5), ci95_hi=np.percentile(boot, 97.5),
            boot_frac_delta_ge0=float((boot >= 0).mean()),
            n_folds_improved=int((c1s < c0s).sum()), n_folds_epoch0=int(sum(m["selected_epoch"] == 0 for m in meta)),
            p_two_sided=wilcoxon_p(c1s, c0s, "two-sided"), p_one_sided=wilcoxon_p(c1s, c0s, "less"),
            mean_rho_frozen=rho0s.mean(), mean_rho_ft=rho1s.mean(), mean_rho_delta=(rho1s - rho0s).mean(),
            n_folds_rho_improved=int((rho1s > rho0s).sum()),
            rho_p_two_sided=wilcoxon_p(rho1s, rho0s, "two-sided"), rho_p_one_sided=wilcoxon_p(rho1s, rho0s, "greater"),
        ))
        s = summary[-1]
        print(f"{backend:17s} C {s['mean_c_frozen']:.3f} -> {s['mean_c_ft']:.3f}  Δ={s['mean_delta']:+.3f} "
              f"[{s['ci95_lo']:+.3f}, {s['ci95_hi']:+.3f}]  improved {s['n_folds_improved']}/5  "
              f"epoch0 {s['n_folds_epoch0']}/5  p2={s['p_two_sided']:.4f}  |  rho {s['mean_rho_frozen']:.3f} -> "
              f"{s['mean_rho_ft']:.3f} ({s['n_folds_rho_improved']}/5 up)")

    pd.DataFrame(per_fold).to_csv(DIR / "per_fold.csv", index=False)
    pd.DataFrame(summary).to_csv(DIR / "summary.csv", index=False)
    col = pd.DataFrame(collapse)
    col.to_csv(DIR / "collapse.csv", index=False)
    if len(col):
        print("\nCollapse check (mean over folds):")
        print(col.groupby("backend")[["std_ratio", "pdist_ratio", "dead_frozen", "dead_ft", "cos_frozen_vs_ft"]]
              .mean().round(3).to_string())


if __name__ == "__main__":
    main()
