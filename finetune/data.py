"""On-the-fly slice + TCC dataset for LoRA fine-tuning.

Wraps `CFBGBMDataset(pairing="scan")` to produce (preprocessed_image,
log10(TCC), patient_id) triples for every scan with a valid target in the
TCC table. Images are loaded, normalised, and run through the backend's own
`preprocess` transform once at construction time -- ~330 scans of a single
2-D T1Gd slice each is small enough to sit fully in memory, so training
epochs don't repeat NIfTI I/O or renormalisation.

Slice normalisation (min-max to uint8, RGB) mirrors
`embeddings/backends/biomedclip.py`'s `embed()` exactly, so the fine-tuned
model sees the same input distribution its frozen-embedding counterpart
(and Task 3's coherence numbers) were computed from.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from data.datasets.cfb_gbm import CFBGBMDataset


def _slice_to_pil(slice_2d: np.ndarray) -> Image.Image:
    slice_norm = (
        (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8) * 255
    ).astype(np.uint8)
    return Image.fromarray(slice_norm).convert("RGB")


class CFBGBMSliceDataset(Dataset):
    """
    One entry per (patient, timepoint) scan with a valid `target_column`
    value in `tcc_table`. `preprocess` is the backend's own image transform
    (open_clip's `create_model_from_pretrained(...)` second return value).
    """

    def __init__(
        self,
        dataset: CFBGBMDataset,
        tcc_table: pd.DataFrame,
        preprocess,
        target_column: str = "slice_tcc",
        log_transform: bool = True,
    ) -> None:
        table = tcc_table.dropna(subset=[target_column])
        table = table[table[target_column] > 0]

        self.images: list[torch.Tensor] = []
        self.targets: list[float] = []
        self.patient_ids: list[str] = []

        for pid, group in table.groupby("patient_id"):
            if pid not in dataset.patients():
                continue
            scans_by_tp = {
                s.metadata["timepoint_label"]: s for s in dataset.scans_for_patient(pid)
            }
            for _, row in group.iterrows():
                scan = scans_by_tp.get(row["timepoint_label"])
                if scan is None:
                    continue
                pil_img = _slice_to_pil(scan.image.astype(np.float32))
                self.images.append(preprocess(pil_img))
                y = row[target_column]
                self.targets.append(float(np.log10(y)) if log_transform else float(y))
                self.patient_ids.append(pid)

        self.targets_t = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.images[idx], self.targets_t[idx], self.patient_ids[idx]


class EmbeddingDataset(Dataset):
    """
    Precomputed-embedding counterpart to `CFBGBMSliceDataset`, for
    fine-tuning a small trainable head on top of an entirely frozen
    backbone (e.g. `finetune/projection_head.py`) -- since the backbone
    never needs a gradient, there's no reason to reload/reprocess raw
    volumes each run: this wraps the exact same frozen vectors Task 3's
    coherence test used (`analysis.cellularity.build_coherence_dataset`'s
    output), so `finetune.trainer.train()`/`evaluate_coherence()` can be
    reused unchanged (they only ever touch `.images`/`.targets_t`/
    `.patient_ids`, whether "images" holds pixels or embeddings).

    `X`, `y`, `patient_ids`: outputs of `build_coherence_dataset()` --
    `y` is already the (log10, by that function's own default) target.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, patient_ids: "np.ndarray | list[str]") -> None:
        self.images = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.targets_t = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.patient_ids = list(patient_ids)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.images[idx], self.targets_t[idx], self.patient_ids[idx]
