"""LoRA fine-tuning entry point for vanilla (non-medical) CLIP.

Interrogates the BiomedCLIP/MedSigLIP fine-tuning results (RESULTS.md):
those two showed a large, robust improvement from LoRA + the coherence
loss, but CLIP already *beat* both of them frozen (see the CLIP baseline
section of RESULTS.md). If fine-tuning brings all three to roughly the
same place, the "medical pretraining has a fixable weakness" framing needs
qualifying -- it would suggest any decent ViT converges here with this
recipe, regardless of pretraining domain.

CLIP (`embeddings/backends/clip.py`) is architecturally identical to
BiomedCLIP: same open_clip `ViT-B-16-quickgelu` + timm ViT, same
`encode_image` + L2-normalise convention -- so this reuses
`finetune.trainer.default_encode_fn` unchanged and `finetune/lora.py`'s
"clip" target-module entry (identical list to "biomedclip"'s).

Usage: python scripts/finetune_clip_cfb_gbm.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import open_clip
import pandas as pd

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

BACKEND_NAME = "clip"
MODEL_NAME = "ViT-B-16-quickgelu"
PRETRAINED = "openai"
TARGET_COLUMN = "slice_tcc"  # CLIP's native target -- see analysis/cellularity.py

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


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

    log.info("Loading %s (pretrained=%s) …", MODEL_NAME, PRETRAINED)
    model, preprocess = open_clip.create_model_from_pretrained(MODEL_NAME, pretrained=PRETRAINED)

    log.info("Building slice dataset (loads + preprocesses every scan once) …")
    dataset = CFBGBMSliceDataset(cfb_dataset, table, preprocess, target_column=TARGET_COLUMN)
    log.info("Dataset ready: %d scans, %d patients", len(dataset), len(set(dataset.patient_ids)))

    model.visual = wrap_visual_tower_with_lora(model.visual, BACKEND_NAME)
    trainable, total = trainable_parameter_count(model.visual)
    log.info(
        "LoRA-wrapped %s visual tower: %d / %d params trainable (%.2f%%)",
        BACKEND_NAME, trainable, total, 100 * trainable / total,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(backend_name=BACKEND_NAME, target_column=TARGET_COLUMN)
    adapter_dir = RESULTS_DIR / "adapters" / BACKEND_NAME
    result = train(config, dataset, model, save_dir=str(adapter_dir), peft_attr="visual")

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
