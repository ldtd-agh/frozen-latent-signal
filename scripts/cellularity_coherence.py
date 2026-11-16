"""
CFB-GBM population-level embedding coherence test (Task 3: TumorTwin TCC +
Geary's C / Moran's I).

For each of the frozen-embedding stores produced by running
`scripts/embed_cfb_gbm.py` with ``PAIRING = "scan"`` (stores/cfb_gbm_scan/,
205 patients / 334 scans as of the TCIA v4 release), build a k-NN graph in
that backend's embedding space and test whether tumour cell count (TCC) is
spatially autocorrelated over it more than a permutation null would
predict -- once over the full graph, once with same-patient neighbors
excluded (the confound control: same-patient scans are anatomically
similar and temporally close in TCC, which could otherwise look like
coherence for free).

Two TCC targets, run for every backend:
  - ``tcc``       whole-volume total cell count -- the mechanistically
                   correct target for backends that see the full 3D volume
                   (Merlin, MedImageInsights).
  - ``slice_tcc`` TCC restricted to the single GTV-argmax axial slice --
                   the mechanistically correct target for the 2D-slice
                   backends (BiomedCLIP, MedSigLIP, CLIP, ImageNet ViT),
                   which never see the rest of the volume ``tcc`` integrates
                   over. Comparing a 2D backend against whole-volume TCC
                   conflates "does the embedding track tumour biology" with
                   "this backend never saw most of the target's support".
Both are reported for every backend (not just each one's "native" target)
so the difference itself is visible.

No backbone training happens here: this is the frozen-embedding step from
PLANNING.md, meant to be evaluated before committing to the LoRA
fine-tuning subsystem (Task 5).

Writes:
  results/tcc_table.csv          -- one row per scan (patient, timepoint,
                                     tcc, slice_tcc, slice_index, tumour
                                     size, RANO)
  results/coherence_results.csv  -- one row per (backend, target) with
                                     Geary's C / Moran's I, p-value and
                                     null stats for both graph variants
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from analysis.cellularity import build_tcc_table, coherence_test_backend
from data.datasets.cfb_gbm import CFBGBMDataset

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent

STORE_DIR = Path("stores/cfb_gbm_scan")
RESULTS_DIR = Path("results")
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"

# TCC computation takes ~10-15s/scan (TumorTwin's nibabel/pydantic overhead), so
# ~an hour for the full scan set. If TCC_TABLE_PATH already exists (e.g. from a
# prior run of this script, or a standalone build_tcc_table() call), reuse it
# instead of recomputing from scratch. Set to True to force a fresh computation
# (e.g. after the TSVs or the ADC/GTV files change).
FORCE_RECOMPUTE_TCC = False

# Includes 'clip' -- a non-medical baseline (vanilla OpenAI CLIP ViT-B/16,
# same architecture BiomedCLIP fine-tunes from) -- to check whether medical
# pretraining is contributing anything beyond generic vision pretraining --
# and 'med3d' -- a second 3D-volumetric backbone (MedicalNet ResNet-10,
# different pretraining recipe than Merlin) to help separate "3D
# architecture matters" from "Merlin's specific pretraining matters".
BACKENDS = ["merlin", "biomedclip", "medsiglip", "medimageinsights", "clip", "med3d", "neurovfm", "imagenet_vit"]
# Each backend's mechanistically-correct target -- see module docstring.
# (Both targets are still computed for every backend; this only flags which
# one is the fair comparison for that backend's actual field of view.)
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
TARGETS = ["tcc", "slice_tcc"]
K = 8
N_PERMUTATIONS = 999
STATISTIC = "gearys_c"

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    import pandas as pd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cached = TCC_TABLE_PATH.exists() and not FORCE_RECOMPUTE_TCC
    if cached:
        table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
        if "slice_tcc" not in table.columns:
            log.info("Cached TCC table predates slice_tcc -- forcing a recompute.")
            cached = False

    if cached:
        log.info("Reusing existing TCC table: %s", TCC_TABLE_PATH.resolve())
        n_valid = int(table["tcc"].notna().sum())
        log.info("Loaded TCC for %d/%d scans", n_valid, len(table))
    else:
        log.info("Building CFB-GBM scan-level dataset (pairing='scan') …")
        dataset = CFBGBMDataset(IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan")
        n_scans = sum(len(v) for v in dataset._scan_filter.values())
        log.info("Dataset ready: %d patients, %d qualifying scans", len(dataset.patients()), n_scans)

        log.info("Computing TCC via TumorTwin for every qualifying scan (this is slow: "
                  "~10-15s/scan observed, so ~%d scans could take an hour or more) …", n_scans)
        table = build_tcc_table(dataset)
        n_valid = int(table["tcc"].notna().sum())
        log.info("TCC computed for %d/%d scans", n_valid, len(table))
        table.to_csv(TCC_TABLE_PATH, index=False)

    rows: list[dict] = []
    for backend in BACKENDS:
        store_path = STORE_DIR / f"cfb_gbm_{backend}.h5"
        if not store_path.exists():
            log.warning("Store not found, skipping: %s", store_path)
            continue

        for target in TARGETS:
            result = coherence_test_backend(
                store_path, table, target_column=target,
                k=K, statistic=STATISTIC, n_permutations=N_PERMUTATIONS,
            )
            full = result["full_graph"]
            excl = result["same_patient_excluded"]
            if full is None:
                log.warning(
                    "%-16s  %-10s  too few samples (n=%d) for k=%d, skipping",
                    backend, target, result["n_samples"], K,
                )
                continue

            native = " *" if NATIVE_TARGET.get(backend) == target else ""
            log.info(
                "%-16s  %-10s%s  n=%3d  full: %s=%+.3f p=%.3f  |  same-patient-excluded: %s=%+.3f p=%.3f",
                backend, target, native, result["n_samples"],
                STATISTIC, full["observed"], full["p_value"],
                STATISTIC, excl["observed"], excl["p_value"],
            )
            rows.append(
                {
                    "backend": backend,
                    "target": target,
                    "native": NATIVE_TARGET.get(backend) == target,
                    "n_samples": result["n_samples"],
                    "k": K,
                    "statistic": STATISTIC,
                    "full_observed": full["observed"],
                    "full_p_value": full["p_value"],
                    "full_null_mean": full["null_mean"],
                    "same_patient_excluded_observed": excl["observed"],
                    "same_patient_excluded_p_value": excl["p_value"],
                    "same_patient_excluded_null_mean": excl["null_mean"],
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "coherence_results.csv", index=False)

    print()
    print("=" * 100)
    print("  CFB-GBM embedding coherence test — summary")
    print("=" * 100)
    print(summary.to_string(index=False))
    print()
    print(f"  Full per-scan TCC table : {(RESULTS_DIR / 'tcc_table.csv').resolve()}")
    print(f"  Coherence summary       : {(RESULTS_DIR / 'coherence_results.csv').resolve()}")


if __name__ == "__main__":
    main()
