"""
REVISIONS_1.md reviewer point 2, second half: once
`scripts/embed_cfb_gbm_midslice.py` has produced
`stores/cfb_gbm_scan_midslice/cfb_gbm_<backend>.h5` for the four 2D
backbones (BiomedCLIP, MedSigLIP, CLIP, ImageNet ViT), this runs the same
Geary's C / Moran's I coherence test (`analysis.cellularity.
coherence_test_backend`, same k, same permutation null, same same-patient
exclusion control) on the mid-slice embeddings and reports it next to the
already-published gtv_mask numbers (`results/coherence_results.csv`),
so the question "does the frozen-coherence ranking depend on picking the
most-tumor slice rather than a geometric mid-volume slice?" has a direct
side-by-side answer.

Does NOT touch the published gtv_mask stores (`stores/cfb_gbm_scan/`) or
`main.tex` -- this is a sensitivity check, not a replacement for the main
analysis (reviewer's own framing).

Writes: results/validation/slice_selection_sensitivity.csv

Usage: python scripts/validate_slice_selection_sensitivity.py
(after the mid-slice stores exist -- see embed_cfb_gbm_midslice.py)
"""

from __future__ import annotations

import logging
from pathlib import Path

from analysis.cellularity import coherence_test_backend

RESULTS_DIR = Path("results")
VALIDATION_DIR = RESULTS_DIR / "validation"
TCC_TABLE_PATH = RESULTS_DIR / "tcc_table.csv"
GTV_STORE_DIR = Path("stores/cfb_gbm_scan")
MID_STORE_DIR = Path("stores/cfb_gbm_scan_midslice")

# Only the four 2D backbones whose slice-selection procedure is in question.
NATIVE_TARGET = {
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

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})

    rows = []
    for backend in BACKENDS:
        target = NATIVE_TARGET[backend]
        gtv_path = GTV_STORE_DIR / f"cfb_gbm_{backend}.h5"
        mid_path = MID_STORE_DIR / f"cfb_gbm_{backend}.h5"

        if not mid_path.exists():
            log.warning(
                "%s: mid-slice store not found at %s -- run embed_cfb_gbm_midslice.py on a GPU node first",
                backend, mid_path,
            )
            continue

        gtv_result = coherence_test_backend(gtv_path, table, target_column=target, k=K, n_permutations=N_PERMUTATIONS)
        mid_result = coherence_test_backend(mid_path, table, target_column=target, k=K, n_permutations=N_PERMUTATIONS)

        gtv_excl, mid_excl = gtv_result["same_patient_excluded"], mid_result["same_patient_excluded"]
        log.info(
            "%-16s  gtv_mask: C=%.3f p=%.3f  |  mid-slice: C=%.3f p=%.3f  (n_gtv=%d, n_mid=%d)",
            backend, gtv_excl["observed"], gtv_excl["p_value"], mid_excl["observed"], mid_excl["p_value"],
            gtv_result["n_samples"], mid_result["n_samples"],
        )
        rows.append({
            "backend": backend, "target": target,
            "gtv_mask_C": gtv_excl["observed"], "gtv_mask_p": gtv_excl["p_value"], "gtv_mask_n": gtv_result["n_samples"],
            "mid_slice_C": mid_excl["observed"], "mid_slice_p": mid_excl["p_value"], "mid_slice_n": mid_result["n_samples"],
        })

    if not rows:
        print("No mid-slice stores found yet -- nothing to compare. See embed_cfb_gbm_midslice.py.")
        return

    df = pd.DataFrame(rows)
    out_path = VALIDATION_DIR / "slice_selection_sensitivity.csv"
    df.to_csv(out_path, index=False)
    print()
    print("=" * 100)
    print("  Slice-selection sensitivity: gtv_mask (published) vs. mid-volume slice, same-patient-excluded")
    print("=" * 100)
    print(df.to_string(index=False))
    print(f"\n  Saved {out_path.resolve()}")


if __name__ == "__main__":
    main()
