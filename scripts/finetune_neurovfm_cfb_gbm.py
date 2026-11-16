"""LoRA fine-tuning entry point for NeuroVFM (single 80/20 split).

Runs in this project's main `.venv` (Python 3.11.5, loaded via
`module load Python/3.11.5` for its bundled shared libs -- see
`scripts/embed_neurovfm_cfb_gbm.py`'s module docstring and `RESULTS.md`'s
"Eighth run" section for the environment-migration story) plus the
vendored `neurovfm` package on `PYTHONPATH` (not pip-installed into the
venv):

    module load Python/3.11.5
    source .venv/bin/activate
    export PYTHONPATH=<path/to/neurovfm-repo>:$PYTHONPATH
    python scripts/finetune_neurovfm_cfb_gbm.py

Despite NeuroVFM's "fused" QKV (`flash_attn.ops.fused_dense.FusedDense`)
and "fused" MLP (`flash_attn.modules.mlp.FusedMLP`), both hold ordinary
addressable `nn.Linear` submodules (`qkv`/`proj`/`fc1`/`fc2`) -- confirmed
by inspecting their MRO and `named_modules()` directly, not assumed from
the name -- unlike CLIP's native `nn.MultiheadAttention`,
whose fused QKV is a raw `Parameter`, not a submodule peft can target.
Full LoRA coverage, same as BiomedCLIP's case (`finetune/lora.py`).

The actual training loop lives in `finetune/neurovfm_trainer.py` (not
`finetune.trainer.train()`: that function's `DataLoader` expects
fixed-size stackable tensors, but NeuroVFM's own preprocessing produces a
*ragged* `[N_tokens, 1024]` tensor per scan) -- this script is now a thin
single-seed wrapper around it, shared with
`scripts/finetune_neurovfm_robustness_check.py`'s 5-seed sweep.

Usage: python scripts/finetune_neurovfm_cfb_gbm.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from data.datasets.cfb_gbm import CFBGBMDataset  # noqa: E402
from finetune.neurovfm_data import NeuroVFMTokenDataset, build_token_cache  # noqa: E402
from finetune.neurovfm_trainer import NeuroVFMConfig, train  # noqa: E402

from neurovfm.pipelines import load_encoder  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGING_ROOT = Path(os.environ["CFB_GBM_ROOT"]) / "CFB-GBM"
TSV_DIR = IMAGING_ROOT.parent
TCC_TABLE_PATH = _PROJECT_ROOT / "results" / "tcc_table.csv"
CACHE_PATH = _PROJECT_ROOT / "stores" / "neurovfm_tokens_cache.pt"
RESULTS_DIR = _PROJECT_ROOT / "results"

BACKEND_NAME = "neurovfm"
TARGET_COLUMN = "tcc"  # whole-volume -- NeuroVFM's native target, same convention as Merlin/MedImageInsights/Med3D
SEED = 42  # matches the original single-split run; NeuroVFMConfig's other defaults (temperature=0.213, etc.) are unchanged

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    log.info("Building CFB-GBM scan-level dataset (pairing='scan') …")
    cfb_dataset = CFBGBMDataset(
        IMAGING_ROOT, tsv_dir=TSV_DIR, pairing="scan", modalities=["t1gd"],
        require_adc=True, require_gtv=True,
    )

    log.info("Loading NeuroVFM encoder …")
    encoder, preproc = load_encoder("mlinslab/neurovfm-encoder")

    log.info("Building/loading token cache (one-time per scan; resume-safe) …")
    cache = build_token_cache(cfb_dataset, IMAGING_ROOT, preproc, CACHE_PATH)

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    dataset = NeuroVFMTokenDataset(cfb_dataset, table, cache, target_column=TARGET_COLUMN)
    log.info("Dataset ready: %d scans, %d patients", len(dataset), len(set(dataset.patient_ids)))

    config = NeuroVFMConfig(backend_name=BACKEND_NAME, seed=SEED)
    adapter_dir = RESULTS_DIR / "adapters" / BACKEND_NAME
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result = train(config, dataset, encoder, encoder.device, save_dir=str(adapter_dir))
    baseline, final, history, val_idx = result["baseline"], result["final"], result["history"], result["val_idx"]

    print()
    print("=" * 70)
    print(f"  LoRA fine-tuning ({BACKEND_NAME}) — held-out Geary's C")
    print("=" * 70)
    print(f"  Baseline (pretrained, no fine-tuning): C={baseline['observed']:+.4f}  p={baseline['p_value']:.3f}")
    print(f"  After fine-tuning (best checkpoint):   C={final['observed']:+.4f}  p={final['p_value']:.3f}")
    print(f"  Held-out set: {len(val_idx)} scans")
    print()

    history_path = RESULTS_DIR / f"finetune_{BACKEND_NAME}_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    log.info("Saved %s", history_path.resolve())


if __name__ == "__main__":
    main()
