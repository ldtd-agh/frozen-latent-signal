"""Multi-seed robustness check for `scripts/finetune_projection_head_cfb_gbm.py`.

See `scripts/finetune_robustness_check.py` (the BiomedCLIP version) for why
this exists. `model_factory`/`wrap_fn` are trivial here (a fresh
`ProjectionHead` is cheap to construct, no backbone to load or wrap) --
`run_multi_seed_robustness` is still the right tool since the split logic,
training loop and history bookkeeping are identical.

Usage: python scripts/finetune_projection_head_robustness_check.py <merlin|med3d>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from analysis.cellularity import build_coherence_dataset
from embeddings.store import EmbeddingStore
from finetune.data import EmbeddingDataset
from finetune.projection_head import ProjectionHead, projection_head_encode_fn
from finetune.trainer import TrainConfig, run_multi_seed_robustness

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
RESULTS_DIR = Path("results")

BACKEND_NAME = "merlin"  # or "med3d" / "medimageinsights" -- override via argv[1]
TARGET_COLUMN = "tcc"
_TEMPERATURE = {"merlin": 85.0, "med3d": 36.0, "medimageinsights": 0.053}
_LEARNING_RATE = 1e-3  # see finetune_projection_head_cfb_gbm.py's module-level comment
SEEDS = [0, 1, 2, 42, 123]

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,  # per-epoch logs from finetune.trainer are noisy across 5 runs
)
log = logging.getLogger(__name__)


def main() -> None:
    backend_name = sys.argv[1] if len(sys.argv) > 1 else BACKEND_NAME
    store_path = STORE_DIR / f"cfb_gbm_{backend_name}.h5"
    if not store_path.exists():
        raise SystemExit(f"Store not found: {store_path} -- run scripts/embed_cfb_gbm.py {backend_name} first.")

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    store = EmbeddingStore(store_path)
    X, y, patient_ids = build_coherence_dataset(store, table, target_column=TARGET_COLUMN, log_transform=True)
    dim = X.shape[1]
    dataset = EmbeddingDataset(X, y, patient_ids)

    def model_factory():
        return ProjectionHead(dim=dim)

    def wrap_fn(model):
        pass  # nothing to wrap -- the whole model is already trainable

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config_template = TrainConfig(
        backend_name=backend_name, target_column=TARGET_COLUMN,
        temperature=_TEMPERATURE[backend_name], lr=_LEARNING_RATE,
    )
    df = run_multi_seed_robustness(
        model_factory, wrap_fn, dataset, config_template, SEEDS, projection_head_encode_fn,
        checkpoint_key="",  # plain module, not LoRA -- see finetune/trainer.py's train() docstring
    )

    print()
    print(df.to_string(index=False))

    out_path = RESULTS_DIR / f"finetune_{backend_name}_projection_head_robustness.csv"
    df.to_csv(out_path, index=False)
    log.warning("Saved %s", out_path.resolve())


if __name__ == "__main__":
    main()
