"""Projection-head fine-tuning entry point for a frozen backbone whose
embedding store is already computed (Merlin, Med3D, or MedImageInsights)
-- see `finetune/projection_head.py` for why this exists and how it
differs from the ViT LoRA scripts.

MedImageInsights was added here rather than given true LoRA fine-tuning:
it's a custom third-party UniCL/DaViT architecture (not open_clip/HF), and
its embedding pipeline runs *every slice* of a volume through the encoder
and median-pools the results -- unfamiliar internals plus a non-standard
pooling op to backprop through would make true LoRA both slow and risky.
Since its frozen embedding store already exists, the projection head
applies with zero new code, exactly like Merlin/Med3D.

Trains directly on the precomputed embedding store from
`scripts/embed_cfb_gbm.py` (`stores/cfb_gbm_scan/cfb_gbm_<backend>.h5`) --
no volumes, no backbone forward pass, so this is much faster than the ViT
LoRA runs (seconds, not minutes, per epoch).

Usage: python scripts/finetune_projection_head_cfb_gbm.py <merlin|med3d|medimageinsights>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from analysis.cellularity import build_coherence_dataset
from embeddings.store import EmbeddingStore
from finetune.data import EmbeddingDataset
from finetune.projection_head import ProjectionHead, projection_head_encode_fn
from finetune.trainer import TrainConfig, train

# ---------------------------------------------------------------------------
# Configuration — edit before running
# ---------------------------------------------------------------------------

STORE_DIR = Path("stores/cfb_gbm_scan")
TCC_TABLE_PATH = Path("results/tcc_table.csv")
RESULTS_DIR = Path("results")

BACKEND_NAME = "merlin"  # or "med3d" / "medimageinsights" -- override via argv[1]
# All three see the whole volume (MedImageInsights via its own
# median-slice-pooling protocol) -- 'tcc' (whole-volume) is their native
# target, matching Task 3's NATIVE_TARGET (analysis/cellularity.py
# and scripts/cellularity_coherence.py).
TARGET_COLUMN = "tcc"

# None of the three L2-normalise their output (see embeddings/backends/
# merlin.py, med3d.py, medimageinsights.py), and their raw embedding
# scales differ a lot -- calibrated empirically to each backend's real
# mean pairwise squared distance (same practice as MedSigLIP's TEMPERATURE
# in finetune_medsiglip_cfb_gbm.py), not copied from the ViT scripts'
# temperature=1.0. MedImageInsights' embeddings happen to already sit at a
# near-unit norm (mean ~0.81), giving it a much smaller scale than
# Merlin/Med3D -- checked directly, not assumed to be similar just because
# all three are "3-D-ish" backbones.
_TEMPERATURE = {"merlin": 85.0, "med3d": 36.0, "medimageinsights": 0.053}
# The ViT scripts' default lr=1e-4 was too conservative for this small MLP
# head (checked directly: lr=1e-3 gives Merlin a real improvement over
# lr=1e-4's, 1e-2 is too aggressive and reverts to baseline; Med3D is flat
# at every lr tried -- 1e-4, 1e-3, 1e-2 -- confirming its null result isn't
# an LR artifact). See RESULTS.md.
_LEARNING_RATE = 1e-3

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def main() -> None:
    backend_name = sys.argv[1] if len(sys.argv) > 1 else BACKEND_NAME
    store_path = STORE_DIR / f"cfb_gbm_{backend_name}.h5"
    if not store_path.exists():
        raise SystemExit(f"Store not found: {store_path} -- run scripts/embed_cfb_gbm.py {backend_name} first.")

    table = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    log.info("Loaded TCC table: %d rows", len(table))

    store = EmbeddingStore(store_path)
    X, y, patient_ids = build_coherence_dataset(store, table, target_column=TARGET_COLUMN, log_transform=True)
    log.info("Loaded %d frozen %s embeddings (dim=%d), target=%s", len(y), backend_name, X.shape[1], TARGET_COLUMN)

    dataset = EmbeddingDataset(X, y, patient_ids)
    model = ProjectionHead(dim=X.shape[1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        backend_name=backend_name, target_column=TARGET_COLUMN,
        temperature=_TEMPERATURE[backend_name], lr=_LEARNING_RATE,
    )
    result = train(
        config, dataset, model, encode_fn=projection_head_encode_fn, checkpoint_key="",
    )

    print()
    print("=" * 70)
    print(f"  Projection-head fine-tuning ({backend_name}) — held-out Geary's C")
    print("=" * 70)
    print(f"  Baseline (frozen backbone, no head):   C={result['baseline']['observed']:+.4f}  p={result['baseline']['p_value']:.3f}")
    print(f"  After fine-tuning (best checkpoint):   C={result['final']['observed']:+.4f}  p={result['final']['p_value']:.3f}")
    print(f"  Held-out set: {len(result['val_idx'])} scans")
    print()

    adapter_dir = RESULTS_DIR / "adapters" / f"{backend_name}_projection_head"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), adapter_dir / "head.pt")
    log.info("Saved projection head weights to %s", adapter_dir / "head.pt")

    history_path = RESULTS_DIR / f"finetune_{backend_name}_projection_head_history.csv"
    pd.DataFrame(result["history"]).to_csv(history_path, index=False)
    log.info("Saved %s", history_path.resolve())


if __name__ == "__main__":
    main()
