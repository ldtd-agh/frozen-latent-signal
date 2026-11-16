"""CFB-GBM frozen-embedding extraction for NeuroVFM (mlinslab/neurovfm-encoder).

NeuroVFM is a brain MRI/CT 3-D ViT-Base foundation model (Vol-JEPA
pretraining on 5.24M clinical neuroimaging volumes, Dao-AILab flash-attn
backed) -- the first genuinely *in-domain* (brain-pretrained) backbone in
this benchmark, unlike Merlin (abdominal CT) or the generic 2-D medical
VLMs. See https://huggingface.co/mlinslab/neurovfm-encoder and
https://github.com/MLNeurosurg/neurovfm.

This script does NOT run in the project's main `.venv`. NeuroVFM pins
torch==2.5.0 / python==3.10.14 exactly and hard-imports `flash_attn` at
module load time (even though inference itself passes
`use_flash_attn=False`) -- flash-attn 2.6.3 has no prebuilt wheel past
torch 2.4, so it cannot coexist with this project's torch 2.12 install.
Run this script with the dedicated conda env instead:

    conda activate <path/to/neurovfm-env>
    python scripts/embed_neurovfm_cfb_gbm.py

(Env setup, one-time: conda env with python==3.10.14; pip install
torch==2.5.0 torchvision==0.20.0 --index-url .../cu121; torch-scatter==2.1.2
from the PyG wheel index; flash-attn==2.6.3 built from source against
CUDA/12.1.1 (`MAX_JOBS=8 pip install --no-build-isolation flash-attn==2.6.3`);
its separate `fused_dense_lib` CUDA extension -- not included in the
flash-attn wheel/sdist build, needed because NeuroVFM's `SelfAttention`
hard-codes `FusedDense` for the QKV/output projections, not config-gated --
built from a `git clone --branch v2.6.3 github.com/Dao-AILab/flash-attention`
checkout's `csrc/fused_dense_lib` subdirectory; `pip install -e` the cloned
`github.com/MLNeurosurg/neurovfm` repo; `pip install h5py` for this
project's `EmbeddingStore` format.)

Population and I/O format match every other backbone exactly, so Task 3's
coherence test (`analysis/cellularity.py`) and everything downstream
consumes this store identically: `CFBGBMDataset(pairing="scan")`'s same
205-patient/334-scan population (T1Gd + ADC + GTV present), written to
`stores/cfb_gbm_scan/cfb_gbm_neurovfm.h5` via the same `EmbeddingStore`.

Volume handling is the one real deviation from the `EmbeddingBackend`
interface every other backbone uses. NeuroVFM's own `StudyPreprocessor`
needs the *original* NIfTI file (reorients to RPI, resamples to its
pretraining spacing of 1x1x4mm, crops to patch multiples, tokenizes into
4x16x16 patches) -- it cannot work from the plain `[D,H,W]` numpy array
`CFBGBMDataset.scans_for_patient()` hands other backends, since that array
has already been reoriented by this project's own loader and carries no
spacing metadata by the time it reaches `embed()`. Rather than bolt a
second, parallel preprocessing path onto the shared pipeline for one
backbone, this script bypasses pixel loading entirely on our side:
`CFBGBMDataset.scan_records()` (metadata-only, same filtering, resolves
`adc_path`/`gtv_path`/timepoint/outcome without ever touching pixel data)
gives us the per-scan identity we need, and the T1Gd path itself is
rebuilt with the dataset module's own `_scan_path()` helper -- the exact
same path-construction logic the dataset uses internally, not a
reimplementation -- and handed directly to NeuroVFM's preprocessor.

Pooling: `EncoderPipeline.embed()` returns per-token embeddings
`[N_total, 768]` with no built-in global pool (this checkpoint's
`cls_token` is `None`; the repo's only pooling module, `models/mil.py`'s
attention-based MIL pooler, is a *trained* component tied to the
diagnostic-head classification task, not appropriate for a generic frozen
embedding). We mean-pool across tokens to a single 768-dim vector per
scan -- the same "simple, unweighted, no extra learned weights" principle
already used for Merlin's and MedImageInsights' pooling elsewhere in this
project.

Usage: python scripts/embed_neurovfm_cfb_gbm.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# This project's root -- so `data.*` / `embeddings.*` import under the
# separate neurovfm conda env too (it already has numpy/pandas/SimpleITK
# as its own pinned deps; only h5py needed adding).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.datasets.cfb_gbm import CFBGBMDataset, _scan_path  # noqa: E402
from data.datasets.base import PatientScan  # noqa: E402
from embeddings.store import EmbeddingStore  # noqa: E402

from neurovfm.pipelines import load_encoder  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent
STORE_PATH = _PROJECT_ROOT / "stores" / "cfb_gbm_scan" / "cfb_gbm_neurovfm.h5"
BACKEND_NAME = "neurovfm"
MODALITY = "mri"  # NeuroVFM's preprocessor doesn't distinguish T1/T1Gd/T2/etc -- just "mri" vs "ct"

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    t_start = time.perf_counter()

    log.info("Initialising CFB-GBM dataset (pairing='scan') …")
    dataset = CFBGBMDataset(
        IMAGING_ROOT, tsv_dir=TSV_DIR, modalities=["t1gd"],
        require_adc=True, require_gtv=True, pairing="scan",
    )
    patients = dataset.patients()
    log.info("Dataset ready: %d patients (scan pairing)", len(patients))

    log.info("Loading NeuroVFM encoder …")
    encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")

    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store = EmbeddingStore(STORE_PATH, backend_name=BACKEND_NAME)
    log.info("Embedding store: %s", STORE_PATH)

    embedded = skipped = failed = 0
    for i, patient_id in enumerate(patients, 1):
        if store.has(patient_id):
            skipped += 1
            log.info("[%d/%d] Skip  %s (cached)", i, len(patients), patient_id)
            continue

        try:
            records = dataset.scan_records(patient_id)
            for record in records:
                tp_label = record["metadata"]["timepoint_label"]
                t1gd_path = _scan_path(IMAGING_ROOT, record["patient_num"], tp_label, "t1gd")
                if not t1gd_path.exists():
                    log.warning("Missing T1Gd file, skipping: %s", t1gd_path)
                    continue

                batch = preproc.load_study([str(t1gd_path)], modality=MODALITY)
                with torch.inference_mode():
                    embs = encoder.embed(batch)  # [N_tokens, 768]
                pooled = embs.float().mean(dim=0).cpu().numpy().astype(np.float32)

                scan = PatientScan(
                    patient_id=patient_id,
                    timepoint=record["timepoint"],
                    image=np.zeros((1,), dtype=np.float32),  # unused by store.save()
                    modality="MRI",
                    outcome=record["outcome"],
                    metadata=record["metadata"],
                )
                store.save(scan, pooled)

            embedded += 1
            log.info("[%d/%d] Done  %s (%d scans)", i, len(patients), patient_id, len(records))
        except Exception as exc:
            failed += 1
            log.warning("[%d/%d] FAIL  %s: %s", i, len(patients), patient_id, exc)

    elapsed = time.perf_counter() - t_start
    print()
    print("=" * 60)
    print(f"  CFB-GBM embedding pipeline complete — backend={BACKEND_NAME}")
    print("=" * 60)
    print(f"  Patients embedded : {embedded}")
    print(f"  Patients skipped  : {skipped}  (already cached)")
    print(f"  Patients failed   : {failed}")
    print(f"  Store             : {STORE_PATH.resolve()}")
    print(f"  Total time        : {elapsed:.1f} s")


if __name__ == "__main__":
    main()
