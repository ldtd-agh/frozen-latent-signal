"""
Frozen-embedding benchmark for `imagenet_vit`, BiomedCLIP's own ImageNet-21k
initialization (see embeddings/backends/imagenet_vit.py), side by side with
BiomedCLIP and CLIP.

Runs the paper's frozen tests with the same settings as the scripts behind
each table, on the native single-slice TCC target:
  headline     Geary's C and Moran's I, k=8, full and same-patient-excluded
               graphs, 999 permutations (cellularity_coherence.py, validate_morans_i.py)
  pairwise     global pairwise-distance Spearman rho over different-patient
               pairs, 999 permutations (validate_pairwise_distance_target.py)
  k sweep      Geary's C at k in {3,...,30}, both graphs (coherence_k_sweep.py)
  mid-slice    Geary's C / Moran's I on the geometric mid-volume slice
               (validate_slice_selection_sensitivity.py)
  vol / ADC    Geary's C for log TCC, log area, mean ADC and residuals
               (validate_volume_residual_coherence.py)

BiomedCLIP and CLIP are recomputed rather than read from the existing CSVs, as
a check that this script reproduces them. Outputs go to results/imagenet_vit/
only; no CSV behind the paper is touched.

Usage: python scripts/evaluate_imagenet_vit_frozen.py   (CPU, a few minutes)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.cellularity import build_coherence_dataset, build_knn_graph, coherence_test_backend, permutation_test  # noqa: E402
from embeddings.store import EmbeddingStore  # noqa: E402
from scripts import validate_volume_residual_coherence as vr  # noqa: E402
from scripts.validate_pairwise_distance_target import permutation_pvalue  # noqa: E402

BACKENDS = ["imagenet_vit", "biomedclip", "clip"]
TARGET = "slice_tcc"
STORE_DIR = _PROJECT_ROOT / "stores" / "cfb_gbm_scan"
MID_STORE_DIR = _PROJECT_ROOT / "stores" / "cfb_gbm_scan_midslice"
TCC_TABLE_PATH = _PROJECT_ROOT / "results" / "tcc_table.csv"
OUT_DIR = _PROJECT_ROOT / "results" / "imagenet_vit"
K = 8
K_VALUES = [3, 5, 8, 12, 16, 20, 25, 30]
N_PERM = 999
SEED = 42

logging.basicConfig(format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("imagenet_vit_frozen")


def store(backend: str, mid: bool = False) -> Path:
    return (MID_STORE_DIR if mid else STORE_DIR) / f"cfb_gbm_{backend}.h5"


def headline(table: pd.DataFrame, mid: bool = False) -> pd.DataFrame:
    rows = []
    for b in BACKENDS:
        if not store(b, mid).exists():
            log.warning("missing %s", store(b, mid))
            continue
        for stat in ["gearys_c", "morans_i"]:
            r = coherence_test_backend(store(b, mid), table, target_column=TARGET, k=K, statistic=stat,
                                       n_permutations=N_PERM)
            rows.append(dict(backend=b, slice="mid" if mid else "gtv_mask", statistic=stat, n=r["n_samples"],
                             full=r["full_graph"]["observed"], full_p=r["full_graph"]["p_value"],
                             excl=r["same_patient_excluded"]["observed"],
                             excl_p=r["same_patient_excluded"]["p_value"]))
            log.info("%-6s %-13s %-9s full %.3f (p=%.3f) | excl %.3f (p=%.3f)", rows[-1]["slice"], b, stat,
                     rows[-1]["full"], rows[-1]["full_p"], rows[-1]["excl"], rows[-1]["excl_p"])
    return pd.DataFrame(rows)


def pairwise(table: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr

    rows = []
    for b in BACKENDS:
        X, y, pids = build_coherence_dataset(EmbeddingStore(store(b)), table, target_column=TARGET)
        pids = np.asarray(pids)
        iu = np.triu_indices(len(y), k=1)
        keep = pids[iu[0]] != pids[iu[1]]
        dist = pdist(X)[keep]
        rho = float(spearmanr(dist, pdist(y[:, None], metric="cityblock")[keep])[0])
        p = permutation_pvalue(y, pids, dist, keep, rho)
        rows.append(dict(backend=b, n_pairs=int(keep.sum()), rho=rho, p_value=p))
        log.info("pairwise %-13s rho=%+.3f p=%.3f (n_pairs=%d)", b, rho, p, keep.sum())
    return pd.DataFrame(rows)


def k_sweep(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in BACKENDS:
        X, y, pids = build_coherence_dataset(EmbeddingStore(store(b)), table, target_column=TARGET)
        for k in K_VALUES:
            row = dict(backend=b, k=k)
            for g, W in [("full", build_knn_graph(X, k=k)),
                         ("excl", build_knn_graph(X, k=k, patient_ids=pids, exclude_same_patient=True))]:
                r = permutation_test(y, W, statistic="gearys_c", n_permutations=N_PERM, seed=SEED)
                row[g], row[f"{g}_p"] = r["observed"], r["p_value"]
            rows.append(row)
    return pd.DataFrame(rows)


def volume_adc() -> pd.DataFrame:
    tab, info = vr.build_targets(TARGET)
    rows = []
    for b in BACKENDS:
        X, sub = vr.align(EmbeddingStore(store(b)), tab)
        pids = sub["patient_id"].to_numpy(dtype=object)
        graphs = {"full": build_knn_graph(X, k=K),
                  "excl": build_knn_graph(X, k=K, patient_ids=pids, exclude_same_patient=True)}
        for t in vr.TARGETS:
            row = dict(backend=b, target=t, n=len(sub))
            for g, W in graphs.items():
                r = permutation_test(sub[t].to_numpy(float), W, statistic="gearys_c", n_permutations=N_PERM,
                                     seed=SEED)
                row[f"{g}_C"], row[f"{g}_p"] = r["observed"], r["p_value"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    results = {
        "headline": headline(table),
        "midslice": headline(table, mid=True),
        "pairwise": pairwise(table),
        "k_sweep": k_sweep(table),
        "volume_adc": volume_adc(),
    }
    for name, df in results.items():
        df.to_csv(OUT_DIR / f"frozen_{name}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Headline (slice TCC, k=8) ===")
    print(results["headline"].round(3).to_string(index=False))
    print("\n=== Mid-volume slice ===")
    print(results["midslice"].round(3).to_string(index=False))
    print("\n=== Pairwise-distance rho ===")
    print(results["pairwise"].round(3).to_string(index=False))
    print("\n=== k sweep: Geary's C, same-patient-excluded (full graph in parentheses) ===")
    ks = results["k_sweep"]
    piv = ks.pivot(index="k", columns="backend", values="excl")[BACKENDS]
    pivf = ks.pivot(index="k", columns="backend", values="full")[BACKENDS]
    print((piv.round(3).astype(str) + " (" + pivf.round(3).astype(str) + ")").to_string())
    print("\n=== Volume / ADC targets: same-patient-excluded Geary's C (p) ===")
    va = results["volume_adc"]
    print((va.pivot(index="target", columns="backend", values="excl_C")[BACKENDS].round(3).astype(str) + " (" +
           va.pivot(index="target", columns="backend", values="excl_p")[BACKENDS].round(3).astype(str) + ")")
          .reindex(vr.TARGETS).to_string())
    print(f"\nWrote {OUT_DIR}/frozen_*.csv")


if __name__ == "__main__":
    main()
