"""
CFB-GBM longitudinal GBM — frozen-embedding pipeline.

Modality choice: T1Gd (gadolinium-enhanced T1), single channel
=======================================================================
Same rationale as `scripts/embed.py`'s CT1 choice for Lumiere: T1Gd is the
primary RANO criterion (enhancing tumour core) and the most information-dense
single sequence for tracking treatment response/progression. Kept identical
across all 5 backends below so cross-backend comparison in the downstream
ADC-cellularity probe (Task 3) isn't confounded by different inputs.

Dataset: two pairing modes, edit `PAIRING` below.
  - 'trajectory' (Task 1/2's original scope): exactly 3 fixed timepoints per
    patient (t0/t1/t2), filtered to the "usable subset" — ADC + GTV mask +
    both RANO transitions present at all 3 timepoints (24/264 patients as of
    the TCIA v4 release). Store: stores/cfb_gbm/.
  - 'scan' (Task 3's coherence test — no trajectory pairing needed): every
    individual (patient, timepoint) scan with ADC + GTV present is embedded
    on its own (205/264 patients, 334 scans as of the TCIA v4 release).
    Store: stores/cfb_gbm_scan/ — kept separate from the trajectory store
    since it's a different, larger scan set and Task 4 still needs the
    original trajectory-paired embeddings.
See `data/datasets/cfb_gbm.py` and `tests/test_cfb_gbm.py` for how these
numbers are derived and verified.

Projection: Merlin (3-D I3D-ResNet), MedImageInsights (mean/median-pools
its own depth slices internally) and Med3D (3-D ResNet) get the full
volume. BiomedCLIP, MedSigLIP, CLIP and ImageNet ViT are 2-D ViT encoders,
so they get a single axial slice selected via the GTV mask (most-tumour-
voxels slice) instead of relying on their internal mid-slice fallback.

`clip` is a non-medical baseline (vanilla OpenAI CLIP ViT-B/16, same
architecture BiomedCLIP fine-tunes from) — not one of the paper's medical
backbones, added to check whether medical-domain pretraining is
contributing anything beyond generic vision pretraining for the Task 3
coherence signal.

`med3d` is a second 3D-volumetric medical backbone (MedicalNet ResNet-10,
Chen et al. 2019, pretrained on 23 medical-imaging datasets spanning
multiple modalities/organs — not brain-MRI-specific, but 3D like Merlin
and unlike it in pretraining recipe) — added to help separate "3D
architecture matters" from "Merlin's specific pretraining matters" for
Task 3's open Merlin puzzle (see RESULTS.md). Weights pulled from the
official-mirror HF repo `TencentMedicalNet/MedicalNet-Resnet10` (MIT
licensed, matches arXiv:1904.00625) via `huggingface_hub`'s cache — no
manual download step needed, unlike MedImageInsights' git-lfs submodule.

Usage
-----
Edit `BACKEND_NAME` (and `PAIRING`, if needed) below and run, or override
the backend from the command line to batch through all backends without
hand-editing each time:

    python scripts/embed_cfb_gbm.py merlin
    python scripts/embed_cfb_gbm.py biomedclip
    python scripts/embed_cfb_gbm.py medsiglip
    python scripts/embed_cfb_gbm.py medimageinsights
    python scripts/embed_cfb_gbm.py clip
    python scripts/embed_cfb_gbm.py med3d
    CFB_GBM_PAIRING=scan python scripts/embed_cfb_gbm.py imagenet_vit
"""

from __future__ import annotations

import functools
import logging
import os
import sys
import time
from pathlib import Path
from typing import Literal

from huggingface_hub import hf_hub_download

from data.datasets.cfb_gbm import CFBGBMDataset
from embeddings.backends import (
    BioMedCLIPBackend,
    CLIPBackend,
    ImageNetViTBackend,
    Med3DBackend,
    MedImageInsightsBackend,
    MedSigLIPBackend,
    MerlinBackend,
)
from embeddings.pipeline import EmbeddingPipeline
from embeddings.store import EmbeddingStore

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent

# 'trajectory' (Task 1/2, 24 patients/72 scans) or 'scan' (Task 3, 205
# patients/334 scans) — see module docstring.
# Override without editing: CFB_GBM_PAIRING=scan python scripts/embed_cfb_gbm.py <backend>
PAIRING: Literal["trajectory", "scan"] = os.environ.get("CFB_GBM_PAIRING", "trajectory")  # type: ignore[assignment]
STORE_DIR = Path("stores/cfb_gbm") if PAIRING == "trajectory" else Path("stores/cfb_gbm_scan")

# Which backend to run this invocation. Edit here, or override via
# `python scripts/embed_cfb_gbm.py <name>`.
BACKEND_NAME = "merlin"

# Optional: restrict to a subset of (already-usable) patients for a quick
# test run. Set to None to process all usable patients.
PATIENT_IDS: list[str] | None = None

# Set to True to process only patients not yet in the store (resume-safe).
SKIP_CACHED = True

# Batch size for GPU inference. 1 is safe; increase if GPU memory allows.
BATCH_SIZE = 1

# Med3D (MedicalNet ResNet-10) pretrained checkpoint -- pulled from the
# official-mirror HF repo, cached locally by huggingface_hub (no manual
# download step, unlike MedImageInsights' git-lfs submodule).
_MED3D_CHECKPOINT = hf_hub_download(
    repo_id="TencentMedicalNet/MedicalNet-Resnet10", filename="resnet_10.pth"
)

# name -> (backend factory, projection_2d, slice_selection)
# All backends use the same single modality (T1Gd) for cross-backend
# comparability — see module docstring.
_MODALITIES = ["t1gd"]
BACKEND_CONFIGS: dict[str, tuple] = {
    "merlin":           (MerlinBackend, False, "mid"),
    "biomedclip":       (BioMedCLIPBackend, True, "gtv_mask"),
    "medsiglip":        (MedSigLIPBackend, True, "gtv_mask"),
    "medimageinsights": (MedImageInsightsBackend, False, "mid"),
    "clip":             (CLIPBackend, True, "gtv_mask"),
    # BiomedCLIP's own ImageNet-21k initialization -- see embeddings/backends/imagenet_vit.py
    "imagenet_vit":     (ImageNetViTBackend, True, "gtv_mask"),
    "med3d":            (functools.partial(Med3DBackend, pretrained_path=_MED3D_CHECKPOINT), False, "mid"),
}

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    backend_name = sys.argv[1] if len(sys.argv) > 1 else BACKEND_NAME
    if backend_name not in BACKEND_CONFIGS:
        raise SystemExit(
            f"Unknown backend {backend_name!r}. Choose from {list(BACKEND_CONFIGS)}."
        )
    backend_cls, projection_2d, slice_selection = BACKEND_CONFIGS[backend_name]

    t_start = time.perf_counter()

    log.info("Initialising CFB-GBM dataset (pairing=%r) …", PAIRING)
    dataset = CFBGBMDataset(
        imaging_root=IMAGING_ROOT,
        tsv_dir=TSV_DIR,
        modalities=_MODALITIES,
        require_adc=True,
        require_gtv=True,
        require_rano=True,  # ignored when pairing='scan'
        normalize=True,
        patient_ids=PATIENT_IDS,
        projection_2d=projection_2d,
        slice_selection=slice_selection,
        pairing=PAIRING,
    )
    n_scans = sum(len(v) for v in dataset._scan_filter.values()) if dataset._scan_filter else None
    log.info(
        "Dataset ready: %d patients%s, T1Gd, projection_2d=%s",
        len(dataset.patients()),
        f", {n_scans} scans" if n_scans is not None else "",
        projection_2d,
    )

    # In 'scan' pairing a patient can have as few as 1 qualifying timepoint
    # by design -- purging below 3 would silently delete most of the run.
    min_timepoints = 3 if PAIRING == "trajectory" else 1

    backend = backend_cls()

    STORE_DIR.mkdir(parents=True, exist_ok=True)
    store_path = STORE_DIR / f"cfb_gbm_{backend_name}.h5"
    store = EmbeddingStore(store_path, backend_name=backend_name)
    log.info("Embedding store: %s", store_path)

    log.info("Starting embedding pipeline …")
    pipeline = EmbeddingPipeline(
        dataset=dataset,
        backend=backend,
        store=store,
        batch_size=BATCH_SIZE,
        skip_cached=SKIP_CACHED,
        verbose=True,
        min_timepoints=min_timepoints,
    )
    stats = pipeline.run()

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"  CFB-GBM embedding pipeline complete — backend={backend_name}")
    print("=" * 60)
    print(f"  Patients embedded : {stats['embedded']}")
    print(f"  Patients skipped  : {stats['skipped']}  (already cached)")
    print(f"  Patients failed   : {stats['failed']}")
    print(f"  Store             : {store_path.resolve()}")
    print(f"  Total time        : {elapsed:.1f} s")
    print()

    _print_sample(store, dataset)


def _print_sample(store: EmbeddingStore, dataset: CFBGBMDataset) -> None:
    """Print shape + timepoints for the first stored patient as a sanity check."""
    patients = store.patients()
    if not patients:
        return

    pid = patients[0]
    vecs, tps, ratings = store.load_patient(pid)
    final_rano = dataset.final_outcome(pid)

    print(f"  Sample patient    : {pid}")
    print(f"  Timepoints (days) : {tps.tolist()}")
    print(f"  Embedding shape   : {vecs.shape}  (T x D)")
    print(f"  RANO ratings      : {ratings}")
    print(f"  Final RANO        : {final_rano}")
    print()


if __name__ == "__main__":
    main()
