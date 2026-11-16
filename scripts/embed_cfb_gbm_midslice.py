"""
REVISIONS_1.md reviewer point 2: "test sensitivity to the 2D slice-
selection procedure." The four 2D backbones (BiomedCLIP, MedSigLIP,
CLIP, ImageNet ViT) currently receive the axial slice with the greatest
number of GTV-positive voxels (`slice_selection="gtv_mask"`,
`scripts/embed_cfb_gbm.py`'s `BACKEND_CONFIGS`) -- a defensible choice to
guarantee the slice contains tumor, but one that uses the segmentation
mask to pick a slice based on tumor extent, which the reviewer notes
could make tumor-burden information easier for these models to recover
given TCC is itself overwhelmingly volume/ADC-dominated
(Section~\\ref{sec:tcc}, `results/validation/tcc_decomposition_*.csv`).

This is a copy of `scripts/embed_cfb_gbm.py`, trimmed to only the four 2D
backbones, with `slice_selection` switched to `"mid"` (the geometric
mid-volume slice each backend's own dataset code already supports as an
alternative -- no new slice-selection logic needed) and PAIRING forced to
`"scan"` (the 332-scan/204-patient set every validate_*.py script and
Table~\\ref{tab:headline_coherence} use, not the 24-patient trajectory
set). Writes to a SEPARATE store directory
(`stores/cfb_gbm_scan_midslice/`) so the existing gtv_mask-selected
stores backing the published results are never touched.

Requires a GPU (BioMedCLIP/MedSigLIP/CLIP/ImageNet ViT forward passes
over 332 scans) -- this interactive session has none (`nvidia-smi`: "No
devices were found"), so this script is prepared but NOT run yet. Submit
as a GPU batch/interactive job, then run
`scripts/validate_slice_selection_sensitivity.py` against the resulting
store to compare the mid-slice Geary's C ranking to
Table~\\ref{tab:headline_coherence}'s gtv_mask ranking.

Usage (on a GPU node):
    python scripts/embed_cfb_gbm_midslice.py biomedclip
    python scripts/embed_cfb_gbm_midslice.py medsiglip
    python scripts/embed_cfb_gbm_midslice.py clip
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from data.datasets.cfb_gbm import CFBGBMDataset
from embeddings.backends import BioMedCLIPBackend, CLIPBackend, ImageNetViTBackend, MedSigLIPBackend
from embeddings.pipeline import EmbeddingPipeline
from embeddings.store import EmbeddingStore

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent

PAIRING = "scan"
STORE_DIR = Path("stores/cfb_gbm_scan_midslice")

BACKEND_NAME = "biomedclip"
PATIENT_IDS: list[str] | None = None
SKIP_CACHED = True
BATCH_SIZE = 1

_MODALITIES = ["t1gd"]
# Same four 2D backbones Table~\ref{tab:headline_coherence} evaluates with
# slice_selection="gtv_mask" -- only the slice-selection procedure changes.
BACKEND_CONFIGS: dict[str, tuple] = {
    "biomedclip":  (BioMedCLIPBackend, True, "mid"),
    "medsiglip":   (MedSigLIPBackend, True, "mid"),
    "clip":        (CLIPBackend, True, "mid"),
    "imagenet_vit": (ImageNetViTBackend, True, "mid"),
}

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S", level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    backend_name = sys.argv[1] if len(sys.argv) > 1 else BACKEND_NAME
    if backend_name not in BACKEND_CONFIGS:
        raise SystemExit(f"Unknown backend {backend_name!r}. Choose from {list(BACKEND_CONFIGS)}.")
    backend_cls, projection_2d, slice_selection = BACKEND_CONFIGS[backend_name]

    t_start = time.perf_counter()

    log.info("Initialising CFB-GBM dataset (pairing=%r, slice_selection=%r) …", PAIRING, slice_selection)
    dataset = CFBGBMDataset(
        imaging_root=IMAGING_ROOT,
        tsv_dir=TSV_DIR,
        modalities=_MODALITIES,
        require_adc=True,
        require_gtv=True,
        require_rano=False,
        normalize=True,
        patient_ids=PATIENT_IDS,
        projection_2d=projection_2d,
        slice_selection=slice_selection,
        pairing=PAIRING,
    )
    log.info("Dataset ready: %d patients", len(dataset.patients()))

    backend = backend_cls()

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = STORE_DIR / f"cfb_gbm_{backend_name}.h5"
    store = EmbeddingStore(store_path, backend_name=backend_name)
    log.info("Embedding store: %s", store_path)

    pipeline = EmbeddingPipeline(
        dataset=dataset, backend=backend, store=store, batch_size=BATCH_SIZE,
        skip_cached=SKIP_CACHED, verbose=True, min_timepoints=1,
    )
    stats = pipeline.run()

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"  CFB-GBM mid-slice embedding complete — backend={backend_name}")
    print("=" * 60)
    print(f"  Patients embedded : {stats['embedded']}")
    print(f"  Patients skipped  : {stats['skipped']}")
    print(f"  Patients failed   : {stats['failed']}")
    print(f"  Store             : {store_path.resolve()}")
    print(f"  Total time        : {elapsed:.1f} s")


if __name__ == "__main__":
    main()
