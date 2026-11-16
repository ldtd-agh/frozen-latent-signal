"""Abstract base classes for longitudinal medical imaging datasets."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, overload

import numpy as np


def select_best_axial_slice(
    volume: np.ndarray,
    seg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Extract the most informative axial (2-D) slice from a volumetric scan.

    Selection priority
    ------------------
    1. If *seg_mask* ``[D, H, W]`` is provided and non-empty, pick the depth
       slice with the most tumour voxels (any non-zero label).
    2. Fall back to the mid-axial slice otherwise.

    Parameters
    ----------
    volume:
        ``[D, H, W]`` (single channel) or ``[C, D, H, W]`` (multi-channel).
    seg_mask:
        Integer array ``[D, H, W]`` where non-zero values mark tumour tissue.
        Pass ``None`` to always use the mid-slice fallback.

    Returns
    -------
    np.ndarray
        ``[H, W]`` for single-channel input, ``[C, H, W]`` for multi-channel.
    """
    multichannel = volume.ndim == 4           # [C, D, H, W]
    depth = volume.shape[1] if multichannel else volume.shape[0]

    if seg_mask is not None and seg_mask.ndim == 3:
        scores = np.sum(seg_mask != 0, axis=(1, 2))   # tumour voxels per slice → [D]
        idx = int(np.argmax(scores)) if scores.max() > 0 else depth // 2
    else:
        idx = depth // 2

    return volume[:, idx] if multichannel else volume[idx]


@dataclass
class PatientScan:
    """A single volumetric scan for one patient at one timepoint."""

    patient_id: str
    """Stable identifier across timepoints."""

    timepoint: float
    """Elapsed time (days from t0, or fraction number)."""

    image: np.ndarray
    """Volumetric scan [D, H, W] or [C, D, H, W], float32."""

    modality: str
    """'CT' | 'CBCT' | ..."""

    delivered_dose: float | None = None
    """Gy delivered up to this timepoint."""

    outcome: str | None = None
    """'CR' | 'PR' | 'PD' | None"""

    metadata: dict = field(default_factory=dict)
    """Any extra fields."""


class LongitudinalDataset(ABC):
    """Yields scans grouped by patient in temporal order."""

    @abstractmethod
    def patients(self) -> list[str]:
        """Return all patient IDs in the dataset."""

    @abstractmethod
    def scans_for_patient(self, patient_id: str) -> list[PatientScan]:
        """Return scans for one patient, sorted by timepoint ascending."""

    def __iter__(self) -> Iterator[tuple[str, list[PatientScan]]]:
        for pid in self.patients():
            yield pid, self.scans_for_patient(pid)

    def all_scans(self) -> Iterator[PatientScan]:
        """Iterate over every scan across all patients."""
        for _, scans in self:
            yield from scans

    @overload
    def __getitem__(self, index: int) -> tuple[str, list["PatientScan"]]: ...
    @overload
    def __getitem__(self, index: slice) -> list[tuple[str, list["PatientScan"]]]: ...

    def __getitem__(self, index):
        """
        Return the nth patient's (patient_id, scans) pair, or a list of them for a slice.

        Examples
        --------
        >>> pid, scans = dataset[0]          # first patient
        >>> pid, scans = dataset[-1]         # last patient
        >>> subset = dataset[0:3]            # first three patients as a list
        """
        pids = self.patients()
        if isinstance(index, slice):
            return [(pid, self.scans_for_patient(pid)) for pid in pids[index]]
        if isinstance(index, int):
            if index < -len(pids) or index >= len(pids):
                raise IndexError(f"dataset index {index} out of range for {len(pids)} patients")
            return pids[index], self.scans_for_patient(pids[index])
        raise TypeError(f"indices must be int or slice, not {type(index).__name__}")

    def __len__(self) -> int:
        return len(self.patients())

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_patients={len(self.patients())})"
        )
