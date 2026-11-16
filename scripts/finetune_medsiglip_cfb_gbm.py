"""LoRA fine-tuning entry point for MedSigLIP -- second confirmation of the
Task 5 result found on BiomedCLIP (see RESULTS.md).

Mirrors `scripts/finetune_cfb_gbm.py` but for MedSigLIP (HuggingFace
`SiglipModel`, not open_clip), which differs in three ways that matter here:

1. Preprocessing goes through `AutoProcessor` (returns a dict with
   `pixel_values`, not a bare tensor) -- adapted to `CFBGBMSliceDataset`'s
   generic `preprocess(pil_image) -> tensor` interface with a small wrapper
   below.
2. The vision tower is `model.vision_model`, and the forward path is
   `model.get_image_features(pixel_values=...).pooler_output`, not
   `model.encode_image(...)`.
3. **Needs bf16 autocast + a smaller batch.** MedSigLIP (so400m, 433M
   params) at 448x448 resolution (~1024 patch tokens vs. BiomedCLIP's ~196)
   uses ~2.9 GB/sample in fp32 during training -- batch 16 alone OOMs a
   40GB A100. bf16 autocast (wrapped inside `encode_fn` below, cast back to
   fp32 before the loss) gets batch 16 down to ~31GB, which fits with
   headroom; batch 24+ still OOMs even under bf16. BATCH_SIZE is set
   directly in `TrainConfig` below rather than left at the BiomedCLIP
   script's default of 48.
4. **Not L2-normalised.** `embeddings/backends/medsiglip.py`'s `embed()`
   uses the raw `pooler_output` (confirmed by direct inspection: norms are
   ~6-7, not 1) -- despite that module's docstring claiming SigLIP output is
   pre-normalised. Task 3's stored MedSigLIP embeddings are this raw,
   unnormalised vector, so `encode_fn` here matches that exactly rather than
   "fixing" it -- changing it would make this run's baseline incomparable to
   Task 3's published number. One consequence: raw pairwise squared
   distances are ~12x larger in scale than BiomedCLIP's normalised ones
   (empirically ~10 vs ~0.6 mean, checked on a handful of real scans) --
   `TEMPERATURE` below is set accordingly, not copied from the BiomedCLIP
   script's default of 1.0.

Usage: python scripts/finetune_medsiglip_cfb_gbm.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoProcessor, SiglipModel

from data.datasets.cfb_gbm import CFBGBMDataset
from finetune.data import CFBGBMSliceDataset
from finetune.lora import trainable_parameter_count, wrap_visual_tower_with_lora
from finetune.trainer import TrainConfig, train

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent
TCC_TABLE_PATH = Path("results/tcc_table.csv")
RESULTS_DIR = Path("results")

BACKEND_NAME = "medsiglip"
MODEL_NAME = "google/medsiglip-448"
TARGET_COLUMN = "slice_tcc"  # MedSigLIP's native target -- see analysis/cellularity.py
TEMPERATURE = 10.0  # calibrated to MedSigLIP's raw (unnormalised) embedding scale -- see module docstring

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def encode_fn(model, images: torch.Tensor) -> torch.Tensor:
    """Matches `MedSigLIPBackend.embed()` exactly: raw (unnormalised) pooler output.

    bf16 autocast keeps this backend's forward/backward pass inside the
    A100's memory budget (see module docstring); cast back to fp32 before
    returning so the softmax/MSE in `soft_neighbor_tcc_loss` isn't done in
    reduced precision.
    """
    with torch.autocast(device_type="cuda" if images.is_cuda else "cpu", dtype=torch.bfloat16):
        out = model.get_image_features(pixel_values=images).pooler_output
    return out.float()


def main() -> None:
    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    log.info("Loaded TCC table: %d rows", len(table))

    log.info("Building CFB-GBM scan-level dataset (pairing='scan') …")
    cfb_dataset = CFBGBMDataset(
        IMAGING_ROOT,
        tsv_dir=TSV_DIR,
        pairing="scan",
        projection_2d=True,
        slice_selection="gtv_mask",
    )

    log.info("Loading %s …", MODEL_NAME)
    model = SiglipModel.from_pretrained(MODEL_NAME)
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def preprocess(pil_img):
        return processor(images=pil_img, return_tensors="pt")["pixel_values"][0]

    log.info("Building slice dataset (loads + preprocesses every scan once) …")
    dataset = CFBGBMSliceDataset(cfb_dataset, table, preprocess, target_column=TARGET_COLUMN)
    log.info("Dataset ready: %d scans, %d patients", len(dataset), len(set(dataset.patient_ids)))

    model.vision_model = wrap_visual_tower_with_lora(model.vision_model, BACKEND_NAME)
    trainable, total = trainable_parameter_count(model.vision_model)
    log.info(
        "LoRA-wrapped %s vision tower: %d / %d params trainable (%.2f%%)",
        BACKEND_NAME, trainable, total, 100 * trainable / total,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        backend_name=BACKEND_NAME, target_column=TARGET_COLUMN, temperature=TEMPERATURE,
        batch_size=16,  # see module docstring: MedSigLIP's memory footprint caps this well below BiomedCLIP's 48
    )
    adapter_dir = RESULTS_DIR / "adapters" / BACKEND_NAME
    result = train(config, dataset, model, encode_fn, save_dir=str(adapter_dir), peft_attr="vision_model")

    print()
    print("=" * 70)
    print(f"  LoRA fine-tuning ({BACKEND_NAME}) — held-out Geary's C")
    print("=" * 70)
    print(f"  Baseline (pretrained, no fine-tuning): C={result['baseline']['observed']:+.4f}  p={result['baseline']['p_value']:.3f}")
    print(f"  After fine-tuning (best checkpoint):   C={result['final']['observed']:+.4f}  p={result['final']['p_value']:.3f}")
    print(f"  Held-out set: {len(result['val_idx'])} scans")
    print()

    history_path = RESULTS_DIR / f"finetune_{BACKEND_NAME}_history.csv"
    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    log.info("Saved %s", history_path.resolve())


if __name__ == "__main__":
    main()
