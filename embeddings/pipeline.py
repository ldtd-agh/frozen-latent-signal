"""Orchestrates dataset → backend → store with caching and progress reporting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.datasets.base import LongitudinalDataset
    from embeddings.backends.base import EmbeddingBackend
    from embeddings.store import EmbeddingStore

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Drives the embedding run: iterates over every patient/scan in a dataset,
    calls the backend, and writes to the store — skipping already-cached scans.

    Parameters
    ----------
    dataset:  A `LongitudinalDataset` instance.
    backend:  A loaded (or not-yet-loaded) `EmbeddingBackend` instance.
    store:    An `EmbeddingStore` instance pointing to the HDF5 file.
    batch_size: Number of scans to embed at once (uses `embed_batch`).
    skip_cached: If True (default), skip patients already in the store.
    verbose:  Print progress to stdout if True.
    """

    def __init__(
        self,
        dataset: "LongitudinalDataset",
        backend: "EmbeddingBackend",
        store: "EmbeddingStore",
        batch_size: int = 1,
        skip_cached: bool = True,
        verbose: bool = True,
        min_timepoints: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.backend = backend
        self.store = store
        self.batch_size = batch_size
        self.skip_cached = skip_cached
        self.verbose = verbose
        self.min_timepoints = min_timepoints

    def run(self) -> dict[str, int]:
        """
        Execute the full pipeline.

        Returns
        -------
        {'embedded': int, 'skipped': int, 'failed': int}
        """
        self._log("Loading backend…")
        self.backend.load()

        stats = {"embedded": 0, "skipped": 0, "failed": 0}
        patients = self.dataset.patients()
        total = len(patients)

        for i, patient_id in enumerate(patients, 1):
            if self.skip_cached and self.store.has(patient_id):
                self._log(f"[{i}/{total}] Skip  {patient_id} (cached)")
                stats["skipped"] += 1
                continue

            try:
                scans = self.dataset.scans_for_patient(patient_id)
                self._embed_patient(patient_id, scans)
                stats["embedded"] += 1
                self._log(f"[{i}/{total}] Done  {patient_id} ({len(scans)} scans)")
            except Exception as exc:
                stats["failed"] += 1
                logger.warning("Failed %s: %s", patient_id, exc)
                self._log(f"[{i}/{total}] FAIL  {patient_id}: {exc}")

        self.backend.unload()

        if self.min_timepoints is not None:
            removed = self.store.purge_below_min_timepoints(self.min_timepoints)
            if removed:
                self._log(
                    f"Purged {len(removed)} patient(s) with fewer than "
                    f"{self.min_timepoints} timepoints: {removed}"
                )
            stats["purged"] = len(removed)

        self._log(
            f"Pipeline complete — embedded={stats['embedded']}, "
            f"skipped={stats['skipped']}, failed={stats['failed']}"
        )
        return stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _embed_patient(self, patient_id: str, scans) -> None:
        """Embed all scans for one patient and write to the store."""
        import numpy as np

        if self.batch_size > 1:
            volumes = [s.image for s in scans]
            vectors = self.backend.embed_batch(volumes)  # [T, D]
        else:
            vectors = np.stack([self.backend.embed(s.image) for s in scans])

        for scan, vec in zip(scans, vectors):
            self.store.save(scan, vec)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        logger.info(msg)
