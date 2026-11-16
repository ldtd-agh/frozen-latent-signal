"""HDF5-backed embedding store — one file per backend, flat patient layout."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from data.datasets.base import PatientScan


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_strings(grp, name: str, strings: list[str]) -> None:
    """Write a list of strings as a variable-length UTF-8 dataset."""
    import h5py
    dt = h5py.string_dtype(encoding="utf-8")
    grp.create_dataset(name, data=np.array(strings, dtype=object), dtype=dt)


def _read_strings(dataset) -> list[str]:
    """Read a string dataset; decodes bytes for h5py < 3 compatibility."""
    return [r.decode("utf-8") if isinstance(r, bytes) else str(r) for r in dataset[:]]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class EmbeddingStore:
    """
    Persistent store for patient embeddings, backed by a single HDF5 file.

    One file = one backend.  The backend name is stored as a file-level
    attribute (f.attrs["backend"]) for informational purposes.

    HDF5 layout
    -----------
    embeddings.h5
    └── <patient_id>/             e.g. "Patient-001"
        ├── vectors      float32[T, D]
        ├── timepoints   float32[T]
        └── rano_rating  str[T]    "" when no rating available

    Notes
    -----
    - Writes are atomic at the scan level: interruption preserves partial work.
    - `has()` lets the pipeline skip patients already embedded.
    """

    def __init__(self, path: str | Path, backend_name: str | None = None) -> None:
        self.path = Path(path)
        if backend_name is not None:
            import h5py
            with h5py.File(self.path, "a") as f:
                f.attrs["backend"] = backend_name

    # ------------------------------------------------------------------
    # Backend metadata
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Backend name stored in the file, or empty string if not set."""
        import h5py
        if not self.path.exists():
            return ""
        with h5py.File(self.path, "r") as f:
            return str(f.attrs.get("backend", ""))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has(self, patient_id: str) -> bool:
        """Return True if this patient already has embeddings stored."""
        import h5py
        if not self.path.exists():
            return False
        with h5py.File(self.path, "r") as f:
            return patient_id in f

    def save(self, scan: "PatientScan", vector: np.ndarray) -> None:
        """
        Append or update one embedding for (patient_id, timepoint).

        If the patient group already exists the new timepoint is appended;
        if the timepoint already exists it is overwritten.
        """
        import h5py

        with h5py.File(self.path, "a") as f:
            grp = f.require_group(scan.patient_id)

            if "vectors" in grp and "timepoints" in grp:
                existing_vecs = grp["vectors"][:]
                existing_tps = grp["timepoints"][:]
                existing_ratings = (
                    _read_strings(grp["rano_rating"])
                    if "rano_rating" in grp
                    else [""] * len(existing_tps)
                )

                tp_arr = float(scan.timepoint)
                match_idx = np.where(np.isclose(existing_tps, tp_arr))[0]

                if match_idx.size > 0:
                    # Overwrite existing timepoint
                    idx = match_idx[0]
                    existing_vecs[idx] = vector.astype(np.float32)
                    existing_ratings[idx] = scan.outcome or ""
                    del grp["vectors"]
                    grp.create_dataset("vectors", data=existing_vecs)
                    del grp["rano_rating"]
                    _write_strings(grp, "rano_rating", existing_ratings)
                    return

                # Append new timepoint
                sort_order = np.argsort(np.append(existing_tps, tp_arr))
                new_vecs = np.vstack(
                    [existing_vecs, vector.astype(np.float32)[np.newaxis]]
                )[sort_order]
                new_tps = np.append(existing_tps, tp_arr)[sort_order]
                new_ratings = np.array(
                    existing_ratings + [scan.outcome or ""]
                )[sort_order].tolist()

                del grp["vectors"]
                del grp["timepoints"]
                del grp["rano_rating"]
                grp.create_dataset("vectors", data=new_vecs)
                grp.create_dataset("timepoints", data=new_tps.astype(np.float32))
                _write_strings(grp, "rano_rating", new_ratings)

            else:
                # First timepoint for this patient
                grp.create_dataset(
                    "vectors", data=vector.astype(np.float32)[np.newaxis]
                )
                grp.create_dataset(
                    "timepoints", data=np.array([scan.timepoint], dtype=np.float32)
                )
                _write_strings(grp, "rano_rating", [scan.outcome or ""])

    def save_patient(
        self,
        patient_id: str,
        vectors: np.ndarray,
        timepoints: np.ndarray,
        rano_ratings: list[str] | None = None,
    ) -> None:
        """
        Write all timepoints for a patient at once.
        Overwrites any existing data for this patient.
        """
        import h5py

        if rano_ratings is None:
            rano_ratings = [""] * len(timepoints)

        with h5py.File(self.path, "a") as f:
            if patient_id in f:
                del f[patient_id]
            grp = f.create_group(patient_id)
            grp.create_dataset("vectors", data=vectors.astype(np.float32))
            grp.create_dataset("timepoints", data=timepoints.astype(np.float32))
            _write_strings(grp, "rano_rating", rano_ratings)

    def load_patient(
        self, patient_id: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Load all embeddings for one patient.

        Returns
        -------
        (vectors [T, D], timepoints [T], rano_ratings [T])
        """
        import h5py

        with h5py.File(self.path, "r") as f:
            if patient_id not in f:
                raise KeyError(f"No embeddings found for {patient_id!r}")
            grp = f[patient_id]
            rano_ratings = (
                _read_strings(grp["rano_rating"])
                if "rano_rating" in grp
                else [""] * len(grp["timepoints"])
            )
            return grp["vectors"][:], grp["timepoints"][:], rano_ratings

    def load_all(self) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
        """
        Load all patients.

        Returns
        -------
        {patient_id: (vectors [T, D], timepoints [T], rano_ratings [T])}
        """
        import h5py

        result: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        if not self.path.exists():
            return result
        with h5py.File(self.path, "r") as f:
            for pid in f:
                grp = f[pid]
                if not hasattr(grp, "keys") or "vectors" not in grp:
                    continue
                rano_ratings = (
                    _read_strings(grp["rano_rating"])
                    if "rano_rating" in grp
                    else [""] * len(grp["timepoints"])
                )
                result[pid] = (grp["vectors"][:], grp["timepoints"][:], rano_ratings)
        return result

    def patients(self) -> list[str]:
        """List all patient IDs stored in this file."""
        import h5py

        if not self.path.exists():
            return []
        with h5py.File(self.path, "r") as f:
            return [pid for pid in f if isinstance(f[pid], h5py.Group) and "vectors" in f[pid]]

    def delete_patient(self, patient_id: str) -> None:
        """Remove all data for a single patient."""
        import h5py

        with h5py.File(self.path, "a") as f:
            if patient_id in f:
                del f[patient_id]

    def purge_below_min_timepoints(self, min_timepoints: int) -> list[str]:
        """
        Delete every patient whose stored timepoint count is below `min_timepoints`.
        Returns the list of removed patient IDs.
        """
        import h5py

        removed: list[str] = []
        if not self.path.exists():
            return removed
        with h5py.File(self.path, "a") as f:
            to_delete = [
                pid for pid in f
                if isinstance(f[pid], h5py.Group)
                and "timepoints" in f[pid]
                and len(f[pid]["timepoints"]) < min_timepoints
            ]
            for pid in to_delete:
                del f[pid]
                removed.append(pid)
        return removed

    def __repr__(self) -> str:
        return f"EmbeddingStore(path={self.path!r}, backend={self.backend!r})"
