"""Token-level dataset + packed-batch helpers for LoRA fine-tuning NeuroVFM.

NeuroVFM's own `StudyPreprocessor` turns one NIfTI volume into a *ragged*
`[N_tokens, 1024]` tensor (patch count varies by volume size after its
1x1x4mm resampling) plus matching `[N_tokens, 3]` coordinates -- nothing
like the fixed-size `[C,H,W]` tensors `CFBGBMSliceDataset`
(`finetune/data.py`) stacks trivially via the default `DataLoader`
collate. `finetune.trainer.train()` is built around that fixed-size
assumption (plain `DataLoader(Subset(dataset, idx), batch_size=...)`, no
custom collate hook), so it isn't reused here -- `scripts/finetune_neurovfm_cfb_gbm.py`
writes its own loop rather than forcing a shared helper to fit a shape
it wasn't designed for.

Preprocessing (NIfTI load, reorient, resample, tokenize) is the expensive,
CPU-bound part (~10s/scan, confirmed by direct timing) and is identical
every time for the same scan -- there's no augmentation -- so it is run
once per scan and cached to disk (`build_token_cache`), not repeated
every epoch or every one of the 5 robustness-check seeds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.datasets.cfb_gbm import CFBGBMDataset, _scan_path

log = logging.getLogger(__name__)


def build_token_cache(
    dataset: CFBGBMDataset,
    imaging_root: Path,
    preproc,
    cache_path: Path,
) -> dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]]:
    """
    Run NeuroVFM's `StudyPreprocessor` once per (patient_id, timepoint_label)
    scan in `dataset`'s population and cache the resulting (tokens, coords)
    pair to `cache_path`. Resume-safe: already-cached keys are skipped.
    """
    cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    if cache_path.exists():
        cache = torch.load(cache_path, weights_only=False)
        log.info("Loaded existing cache: %d scans", len(cache))

    patients = dataset.patients()
    n_new = 0
    for i, patient_id in enumerate(patients, 1):
        for record in dataset.scan_records(patient_id):
            tp_label = record["metadata"]["timepoint_label"]
            key = (patient_id, tp_label)
            if key in cache:
                continue

            t1gd_path = _scan_path(imaging_root, record["patient_num"], tp_label, "t1gd")
            if not t1gd_path.exists():
                log.warning("Missing T1Gd file, skipping: %s", t1gd_path)
                continue

            batch = preproc.load_study([str(t1gd_path)], modality="mri")
            cache[key] = (batch["img"].clone(), batch["coords"].clone())
            n_new += 1

            if n_new % 20 == 0:
                torch.save(cache, cache_path)
                log.info("[%d/%d patients] cached %d new scans so far …", i, len(patients), n_new)

    torch.save(cache, cache_path)
    log.info("Token cache complete: %d scans total (%d newly added) -> %s", len(cache), n_new, cache_path)
    return cache


class NeuroVFMTokenDataset(Dataset):
    """
    One entry per (patient, timepoint) scan with a valid `target_column`
    value in `tcc_table`, backed by a pre-built token cache
    (`build_token_cache`). `.tokens[i]` / `.coords[i]` are ragged
    `[N_i, 1024]` / `[N_i, 3]` tensors -- callers must pack them (see
    `pack_batch` below) rather than stacking directly.
    """

    def __init__(
        self,
        dataset: CFBGBMDataset,
        tcc_table: pd.DataFrame,
        cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]],
        target_column: str = "tcc",
        log_transform: bool = True,
    ) -> None:
        table = tcc_table.dropna(subset=[target_column])
        table = table[table[target_column] > 0]

        self.tokens: list[torch.Tensor] = []
        self.coords: list[torch.Tensor] = []
        self.targets: list[float] = []
        self.patient_ids: list[str] = []

        for pid, group in table.groupby("patient_id"):
            if pid not in dataset.patients():
                continue
            for _, row in group.iterrows():
                key = (pid, row["timepoint_label"])
                if key not in cache:
                    continue
                tokens, coords = cache[key]
                self.tokens.append(tokens)
                self.coords.append(coords)
                y = row[target_column]
                self.targets.append(float(np.log10(y)) if log_transform else float(y))
                self.patient_ids.append(pid)

        self.targets_t = torch.tensor(self.targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int):
        return self.tokens[idx], self.coords[idx], self.targets_t[idx], self.patient_ids[idx]


def pack_batch(dataset: NeuroVFMTokenDataset, indices: list[int]) -> dict:
    """
    Concatenate several scans' ragged (tokens, coords) into one packed
    batch dict matching the format `StudyPreprocessor.load_study()` /
    `EncoderPipeline.embed()` already expect (`img`, `coords`,
    `series_cu_seqlens`, `series_max_len`, `mode`, `path`) -- same
    concatenation logic `StudyPreprocessor` uses internally for multiple
    volumes in one study, applied here across a training minibatch instead.
    """
    toks = [dataset.tokens[i] for i in indices]
    crds = [dataset.coords[i] for i in indices]
    lengths = [t.shape[0] for t in toks]

    img = torch.cat(toks, dim=0)
    coords = torch.cat(crds, dim=0)
    cu_seqlens = torch.zeros(len(lengths) + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.tensor(lengths, dtype=torch.int32).cumsum(0)

    return {
        "img": img,
        "coords": coords,
        "series_cu_seqlens": cu_seqlens,
        "series_max_len": max(lengths),
        "mode": ["mri"] * len(indices),
        "path": [f"scan_{i}" for i in indices],
        "size": [None] * len(indices),
    }


def forward_pooled(
    model,
    norm_module,
    batch: dict,
    device: str,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Normalize -> model forward -> per-series mean pool, mirroring
    `EncoderPipeline.embed()`'s preprocessing exactly but pooling each
    series (scan) to one vector instead of returning raw per-token output
    -- `encoder.embed()` has no built-in pool (see module docstring in
    `scripts/finetune_neurovfm_cfb_gbm.py`), and a flat mean over the
    whole *batch's* tokens would wrongly mix different scans together, so
    pooling must respect `cu_seqlens` segment boundaries.

    Returns [len(cu_seqlens)-1, D] -- one embedding per scan in the batch.
    """
    tokens = batch["img"].to(device)
    coords = batch["coords"].to(device)
    cu_seqlens = batch["series_cu_seqlens"].to(device)
    max_len = batch["series_max_len"]

    tokens = norm_module.normalize(tokens, batch["mode"], batch["path"], cu_seqlens=cu_seqlens, sizes=batch.get("size"))

    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    with torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=amp_dtype):
        out = model(tokens, coords, masks=None, cu_seqlens=cu_seqlens, max_seqlen=max_len, use_flash_attn=False)
    out = out.float()

    pooled = []
    cu_seqlens_cpu = cu_seqlens.cpu()
    for i in range(len(cu_seqlens_cpu) - 1):
        start, end = int(cu_seqlens_cpu[i]), int(cu_seqlens_cpu[i + 1])
        pooled.append(out[start:end].mean(dim=0))
    return torch.stack(pooled)
