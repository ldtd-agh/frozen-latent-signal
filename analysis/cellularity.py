"""
Total cell count (TCC) via TumorTwin, and a population-level embedding
coherence test against it (PLANNING.md Task 3 -- redesigned from an earlier
linear-probe framing, see git history for that version).

Cellularity / TCC
------------------
Per-scan tumour cell count is computed with TumorTwin
(github.com/OncologyModelingGroup/TumorTwin): `ADC_to_cellularity` converts
an ADC map to a per-voxel cellularity index within a GTV mask (normalized
inverse ADC, standard in the diffusion-MRI glioma literature -- see
`tumortwin.preprocessing.cellularity` for the exact formulation), then
`compute_total_cell_count` sums it over the whole image, scaled by a
per-voxel carrying capacity (`compute_carrying_capacity`, derived from the
scan's own voxel spacing -- *not* TumorTwin's demo-dataset default, which
corresponds to a different, coarser voxel size).

CFB-GBM provides a single GTV mask (no separate non-enhancing ROI), so it's
passed as `roi_enhance`; `roi_nonenhance=None` means no voxels get
TumorTwin's fixed CELLULARITY_NON_ENHANCE=0.16 literature constant -- see
PLANNING.md's open question on this.

Core method: population-level spatial autocorrelation, not a regression probe
--------------------------------------------------------------------------
This is deliberately **not** a train/test regression probe. Every valid
scan (any patient, any timepoint with ADC+GTV present -- see
`data.datasets.cfb_gbm`'s `pairing='scan'` mode) is one node; a k-NN graph
is built in a backend's frozen-embedding space; Geary's C / Moran's I then
ask whether TCC is spatially autocorrelated over that graph (neighbors in
embedding space have similar TCC) more than a permutation null would
predict by chance. No within-patient pairing is needed, so N is every
usable scan (~334 across ~205 patients for CFB-GBM), not just fully-
longitudinal patients (~24) -- the main advantage over the earlier probe.

Same-patient confound control (do not skip): same-patient scans are
anatomically similar and have temporally-close TCC, so an embedding could
score well by clustering patient identity rather than tumour biology.
`coherence_test_backend` reports the statistic twice per backbone: once
over the full k-NN graph, once with same-patient neighbors excluded from
the graph entirely. A large gap between the two is itself a finding
(anatomy-dominated vs. biology-dominated embedding).

Geary's C and Moran's I are implemented directly here (a few lines of numpy
over an arbitrary precomputed graph) rather than via squidpy: squidpy's
`spatial_autocorr` targets AnnData-based spatial-transcriptomics pipelines
and pulls in a large scanpy/anndata/numba stack for what is otherwise a
short, standard, auditable computation.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import csr_matrix

    from data.datasets.cfb_gbm import CFBGBMDataset
    from embeddings.store import EmbeddingStore

DEFAULT_CELL_SIZE_MM = 1e-6
DEFAULT_PACKING_FRACTION = 0.75


# ---------------------------------------------------------------------------
# Per-scan TCC (I/O -- reads NIfTI files via TumorTwin)
# ---------------------------------------------------------------------------

def compute_scan_tcc(
    adc_path: str | Path,
    gtv_path: str | Path,
    cell_size_mm: float = DEFAULT_CELL_SIZE_MM,
    packing_fraction: float = DEFAULT_PACKING_FRACTION,
) -> dict | None:
    """
    Compute one scan's total cell count (TCC) via TumorTwin's
    ADC_to_cellularity + compute_total_cell_count -- both over the whole
    volume (``tcc``) and restricted to the single axial slice with the most
    GTV voxels (``slice_tcc``, ``slice_index``).

    The slice-level figure matters because 4 of the 8 embedding backends
    (BiomedCLIP, MedSigLIP, CLIP, ImageNet ViT) only ever see one 2D slice
    of the volume (selected the same way, by GTV voxel count), so comparing
    their embeddings against whole-volume TCC conflates "does this backend's
    embedding correlate with tumour biology" with "this backend never saw
    most of the voxels the target was computed from". `slice_tcc` is the
    matching, mechanistically-fair target for those four; Merlin and
    MedImageInsights do see the full volume, so `tcc` is still the right
    target for them (see RESULTS.md for the fuller discussion).

    The slice axis is taken as the one with the largest voxel spacing (the
    through-plane/axial direction for a standard MRI acquisition) rather
    than a hardcoded array axis. Verified against
    `data.datasets.base.select_best_axial_slice`'s SimpleITK/RAS-based
    selection (what the real embedding pipeline actually uses) to pick the
    identical slice index on CFB-GBM's files.

    Returns None (with a warning) if the ADC/GTV pair fails TumorTwin's
    shape/spacing consistency check, or if the resulting whole-volume TCC
    is not a finite positive number. `slice_tcc` is set to None (without
    failing the whole record) if it alone is non-positive/non-finite.
    """
    from tumortwin.postprocessing import compute_total_cell_count
    from tumortwin.preprocessing import ADC_to_cellularity, compute_carrying_capacity
    from tumortwin.types.imaging import NibabelNifti

    try:
        adc = NibabelNifti.from_file(str(adc_path))
        gtv = NibabelNifti.from_file(str(gtv_path))
        cellularity = ADC_to_cellularity(adc, roi_enhance=gtv, roi_nonenhance=None)
        carrying_capacity = compute_carrying_capacity(
            adc, cell_size_mm=cell_size_mm, packing_fraction=packing_fraction
        )
        tcc = float(compute_total_cell_count(cellularity, carrying_capacity))

        slice_axis = int(np.argmax([adc.spacing.x, adc.spacing.y, adc.spacing.z]))
        gtv_arr = gtv.array
        other_axes = tuple(a for a in range(3) if a != slice_axis)
        gtv_counts = gtv_arr.sum(axis=other_axes)
        slice_index = (
            int(np.argmax(gtv_counts))
            if gtv_counts.max() > 0
            else gtv_arr.shape[slice_axis] // 2
        )
        cellularity_slice = np.take(cellularity.array, slice_index, axis=slice_axis)
        slice_tcc = float(carrying_capacity * cellularity_slice.sum())
    except AssertionError as exc:
        warnings.warn(f"Could not compute TCC for {adc_path}: {exc}")
        return None

    if not np.isfinite(tcc) or tcc <= 0:
        warnings.warn(f"Non-positive/non-finite TCC for {adc_path}: {tcc}")
        return None

    if not np.isfinite(slice_tcc) or slice_tcc <= 0:
        warnings.warn(f"Non-positive/non-finite slice_tcc for {adc_path}: {slice_tcc}")
        slice_tcc = None

    return {
        "tcc": tcc,
        "slice_tcc": slice_tcc,
        "slice_index": slice_index,
        "slice_axis": slice_axis,
    }


# ---------------------------------------------------------------------------
# Per-patient / per-dataset table building
# ---------------------------------------------------------------------------

def scan_tcc_rows(
    dataset: "CFBGBMDataset",
    patient_id: str,
    **tcc_kwargs,
) -> list[dict]:
    """
    One row per scan for a single patient: TCC plus a few metadata fields.

    Uses `dataset.scan_records()` rather than `scans_for_patient()` -- TCC
    only needs `adc_path`/`gtv_path`/RANO/timepoint, not the requested
    imaging modality's pixel data, and loading+normalising that volume for
    every one of ~300+ scans just to discard it would roughly triple the
    I/O of building this table for no benefit.
    """
    records = dataset.scan_records(patient_id)

    rows: list[dict] = []
    for rec in records:
        adc_path = rec["metadata"].get("adc_path")
        gtv_path = rec["metadata"].get("gtv_path")
        result = (
            compute_scan_tcc(adc_path, gtv_path, **tcc_kwargs)
            if adc_path and gtv_path
            else None
        )
        rows.append(
            {
                "patient_id": patient_id,
                "timepoint_label": rec["metadata"].get("timepoint_label"),
                "timepoint_days": rec["timepoint"],
                "tcc": result["tcc"] if result else None,
                "slice_tcc": result["slice_tcc"] if result else None,
                "slice_index": result["slice_index"] if result else None,
                "tumor_size_cm3": rec["metadata"].get("tumor_size_cm3"),
                "rano": rec["outcome"],
            }
        )
    return rows


def compute_scan_decomposition(
    adc_path: str | Path,
    gtv_path: str | Path,
) -> dict | None:
    """
    Construct-validity helper (reviewer-requested decomposition, not used by
    the main TCC/coherence pipeline): per-scan GTV volume and within-mask
    ADC summary statistics, computed directly from the ADC/GTV NIfTI pair
    with no TumorTwin cellularity/carrying-capacity step. These are the
    "ingredients" TCC is built from (see `compute_scan_tcc`'s docstring) --
    used to test how much of TCC's variation they explain on their own.

    Mirrors `compute_scan_tcc`'s whole-volume/slice split: `volume_cm3` +
    whole-volume ADC stats are the fair comparison for backends that see
    the full 3D volume, `slice_area_cm2` + slice-only ADC stats for
    backends that only see the single GTV-argmax axial slice. The slice is
    selected identically to `compute_scan_tcc` (largest-spacing axis,
    GTV-argmax index) so the two feature sets line up with `tcc`/
    `slice_tcc` scan-for-scan.

    Returns None (with a warning) on a shape/spacing mismatch between the
    ADC and GTV images, or if the GTV mask is empty.
    """
    from tumortwin.types.imaging import NibabelNifti, Unit

    try:
        adc = NibabelNifti.from_file(str(adc_path))
        gtv = NibabelNifti.from_file(str(gtv_path))
        if adc.shape != gtv.shape:
            raise AssertionError(f"ADC/GTV shape mismatch: {adc.shape} vs {gtv.shape}")

        match adc.spacing.unit:
            case Unit.MICRON:
                vol_scale = 1.0e-3**3
            case Unit.MILLIMETER:
                vol_scale = 1.0
            case Unit.METER:
                vol_scale = 1.0e3**3
            case Unit.UNKNOWN:
                vol_scale = 1.0
            case _:
                raise AssertionError(f"Unsupported spacing unit: {adc.spacing.unit}")
        voxel_mm3 = vol_scale * adc.spacing.x * adc.spacing.y * adc.spacing.z

        gtv_arr = gtv.array
        adc_arr = adc.array
        mask = gtv_arr > 0
        if mask.sum() == 0:
            raise AssertionError("Empty GTV mask")

        volume_cm3 = float(mask.sum()) * voxel_mm3 / 1000.0
        whole_adc = adc_arr[mask]
        whole_stats = _adc_summary_stats(whole_adc)

        slice_axis = int(np.argmax([adc.spacing.x, adc.spacing.y, adc.spacing.z]))
        other_axes = tuple(a for a in range(3) if a != slice_axis)
        gtv_counts = gtv_arr.sum(axis=other_axes)
        slice_index = (
            int(np.argmax(gtv_counts)) if gtv_counts.max() > 0 else gtv_arr.shape[slice_axis] // 2
        )
        in_plane_spacing = [adc.spacing.x, adc.spacing.y, adc.spacing.z]
        pixel_mm2 = vol_scale ** (2 / 3) * np.prod(
            [in_plane_spacing[a] for a in other_axes]
        )

        gtv_slice = np.take(gtv_arr, slice_index, axis=slice_axis) > 0
        adc_slice = np.take(adc_arr, slice_index, axis=slice_axis)
        slice_area_cm2 = float(gtv_slice.sum()) * pixel_mm2 / 100.0
        slice_stats = _adc_summary_stats(adc_slice[gtv_slice]) if gtv_slice.sum() > 0 else None
    except AssertionError as exc:
        warnings.warn(f"Could not compute decomposition features for {adc_path}: {exc}")
        return None

    result = {"volume_cm3": volume_cm3, "slice_area_cm2": slice_area_cm2}
    result.update({f"adc_{k}": v for k, v in whole_stats.items()})
    if slice_stats is not None:
        result.update({f"slice_adc_{k}": v for k, v in slice_stats.items()})
    else:
        result.update({f"slice_adc_{k}": None for k in ("mean", "median", "std", "p10", "p90", "skew")})
    return result


def _adc_summary_stats(values: np.ndarray) -> dict:
    from scipy.stats import skew

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "skew": float(skew(values)),
    }


def build_decomposition_table(dataset: "CFBGBMDataset") -> "pd.DataFrame":
    """
    One row per scan with `compute_scan_decomposition`'s features plus
    identifying columns (`patient_id`, `timepoint_days`), so this can be
    merged with `build_tcc_table`'s output on those two columns.
    """
    import pandas as pd

    rows: list[dict] = []
    for pid in dataset.patients():
        for rec in dataset.scan_records(pid):
            adc_path = rec["metadata"].get("adc_path")
            gtv_path = rec["metadata"].get("gtv_path")
            feats = (
                compute_scan_decomposition(adc_path, gtv_path)
                if adc_path and gtv_path
                else None
            )
            row = {"patient_id": pid, "timepoint_days": rec["timepoint"]}
            if feats:
                row.update(feats)
            rows.append(row)
    return pd.DataFrame(rows)


def build_tcc_table(dataset: "CFBGBMDataset", **tcc_kwargs) -> "pd.DataFrame":
    """
    Build the full per-scan TCC table for every patient in `dataset`.

    Use a `CFBGBMDataset` constructed with ``pairing='scan'`` to get the
    full, non-trajectory-paired N (see `data.datasets.cfb_gbm`).
    """
    import pandas as pd

    rows: list[dict] = []
    for pid in dataset.patients():
        rows.extend(scan_tcc_rows(dataset, pid, **tcc_kwargs))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# k-NN graph in embedding space
# ---------------------------------------------------------------------------

def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 8,
    patient_ids: Sequence[str] | None = None,
    exclude_same_patient: bool = False,
) -> "csr_matrix":
    """
    Symmetric, unweighted k-NN adjacency in embedding space:
    ``w_ij = 1`` if j is among i's k nearest neighbors (Euclidean) or vice
    versa, else 0.

    If `exclude_same_patient` is True, `patient_ids` (same length and node
    order as `embeddings`) is required, and same-patient pairs are excluded
    from the neighbor search itself -- every node still gets exactly k
    neighbors, all from other patients. This is Task 3's confound control
    (see module docstring).
    """
    import scipy.sparse as sp

    n = embeddings.shape[0]
    if exclude_same_patient:
        if patient_ids is None or len(patient_ids) != n:
            raise ValueError(
                "patient_ids (len == n_nodes) is required when exclude_same_patient=True"
            )
        patient_ids = np.asarray(patient_ids)

    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        if exclude_same_patient:
            candidates = np.where(patient_ids != patient_ids[i])[0]
        else:
            candidates = np.concatenate([np.arange(i), np.arange(i + 1, n)])
        if len(candidates) < k:
            raise ValueError(
                f"Node {i} has only {len(candidates)} eligible neighbor candidates, "
                f"fewer than k={k}. Reduce k or check patient_ids."
            )
        dists = np.linalg.norm(embeddings[candidates] - embeddings[i], axis=1)
        nearest = candidates[np.argsort(dists)[:k]]
        rows.extend([i] * k)
        cols.extend(nearest.tolist())

    data = np.ones(len(rows), dtype=np.float64)
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)  # symmetrize: an edge exists if either direction found it
    W.setdiag(0)
    W.eliminate_zeros()
    return W


# ---------------------------------------------------------------------------
# Spatial autocorrelation statistics
# ---------------------------------------------------------------------------

def gearys_c(y: np.ndarray, W: "csr_matrix") -> float:
    """
    Geary's C spatial-autocorrelation statistic over an arbitrary weighted
    graph (need not be geographic -- here, a k-NN graph in embedding space).

        C = (n - 1) / (2 * sum(W)) * sum_ij w_ij (y_i - y_j)^2 / sum_i (y_i - ybar)^2

    C is centered at 1 under spatial randomness; C < 1 indicates positive
    autocorrelation (neighbors have similar y), C > 1 negative.

    Reference: Geary (1954), "The contiguity ratio and statistical mapping."
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    W_sum = W.sum()
    if W_sum == 0:
        return float("nan")

    Wc = W.tocoo()
    num = np.sum(Wc.data * (y[Wc.row] - y[Wc.col]) ** 2)
    denom = np.sum((y - y.mean()) ** 2)
    if denom == 0:
        return float("nan")

    return float((n - 1) / (2 * W_sum) * num / denom)


def morans_i(y: np.ndarray, W: "csr_matrix") -> float:
    """
    Moran's I spatial-autocorrelation statistic over an arbitrary weighted
    graph.

        I = n / sum(W) * sum_ij w_ij (y_i - ybar)(y_j - ybar) / sum_i (y_i - ybar)^2

    I's expected value under randomness is -1/(n-1) (~0 for large n);
    I above that indicates positive (clustered) autocorrelation.

    Reference: Moran (1950), "Notes on continuous stochastic phenomena."
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    W_sum = W.sum()
    if W_sum == 0:
        return float("nan")

    yc = y - y.mean()
    Wc = W.tocoo()
    num = np.sum(Wc.data * yc[Wc.row] * yc[Wc.col])
    denom = np.sum(yc ** 2)
    if denom == 0:
        return float("nan")

    return float(n / W_sum * num / denom)


_STATISTICS = {"gearys_c": gearys_c, "morans_i": morans_i}


def patient_block_permutation(y: np.ndarray, patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    One draw from a patient-cluster-respecting permutation null: instead of
    shuffling individual scan-level values (`rng.permutation(y)`, which
    treats every scan as an independent exchangeable unit even though many
    patients contribute 2-3 temporally-correlated scans), reassign each
    patient's *entire* block of observed y-values to a different,
    randomly-chosen patient's row-positions -- restricted to swaps between
    patients with the same scan count, so every block always has an exact
    donor (CFB-GBM's scan-count distribution is {1: 100, 2: 80, 3: 24}
    patients -- no singleton strata, so every patient is swappable).

    This is what REVISIONS_1.md's reviewer point 1 is asking for: a null
    that doesn't under/overestimate the variance of the test statistic by
    pretending within-patient scans are independent draws.
    """
    patient_ids = np.asarray(patient_ids)
    y_perm = np.empty_like(y)

    idx_by_pid: dict = {}
    for pid in np.unique(patient_ids):
        idx_by_pid[pid] = np.where(patient_ids == pid)[0]

    by_size: dict[int, list] = {}
    for pid, idx in idx_by_pid.items():
        by_size.setdefault(len(idx), []).append(pid)

    for _, pids in by_size.items():
        pids = np.asarray(pids, dtype=object)
        donors = rng.permutation(pids)
        for recipient, donor in zip(pids, donors):
            y_perm[idx_by_pid[recipient]] = y[idx_by_pid[donor]]

    return y_perm


def cluster_permutation_test(
    y: np.ndarray,
    W: "csr_matrix",
    patient_ids: np.ndarray,
    statistic: Literal["gearys_c", "morans_i"] = "gearys_c",
    n_permutations: int = 999,
    seed: int = 42,
) -> dict:
    """
    Same statistic and same one-sided empirical p-value convention as
    `permutation_test`, but drawing the null from `patient_block_permutation`
    instead of a flat scan-level shuffle -- see that function's docstring
    and REVISIONS_1.md reviewer point 1. Intended as a direct, apples-to-
    apples robustness check against `permutation_test`'s existing numbers,
    not a silent replacement for them.
    """
    stat_fn = _STATISTICS[statistic]
    y = np.asarray(y, dtype=np.float64)
    patient_ids = np.asarray(patient_ids)
    observed = stat_fn(y, W)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        null_stats[i] = stat_fn(patient_block_permutation(y, patient_ids, rng), W)

    if statistic == "gearys_c":
        p_value = (1 + np.sum(null_stats <= observed)) / (n_permutations + 1)
    else:
        p_value = (1 + np.sum(null_stats >= observed)) / (n_permutations + 1)

    return {
        "statistic": statistic,
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.nanmean(null_stats)),
        "null_std": float(np.nanstd(null_stats)),
        "n_permutations": n_permutations,
    }


def permutation_test(
    y: np.ndarray,
    W: "csr_matrix",
    statistic: Literal["gearys_c", "morans_i"] = "gearys_c",
    n_permutations: int = 999,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo permutation test for spatial autocorrelation: shuffle y
    across nodes (keeping the graph W fixed), recompute the statistic each
    time, and derive an empirical, one-sided p-value against the null of
    no autocorrelation -- in the direction consistent with clustering
    (C < observed for Geary's C, I > observed for Moran's I).

    Returns
    -------
    dict with 'observed', 'p_value', 'null_mean', 'null_std', 'n_permutations'.
    """
    stat_fn = _STATISTICS[statistic]
    y = np.asarray(y, dtype=np.float64)
    observed = stat_fn(y, W)

    rng = np.random.default_rng(seed)
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        null_stats[i] = stat_fn(rng.permutation(y), W)

    if statistic == "gearys_c":
        p_value = (1 + np.sum(null_stats <= observed)) / (n_permutations + 1)
    else:
        p_value = (1 + np.sum(null_stats >= observed)) / (n_permutations + 1)

    return {
        "statistic": statistic,
        "observed": float(observed),
        "p_value": float(p_value),
        "null_mean": float(np.nanmean(null_stats)),
        "null_std": float(np.nanstd(null_stats)),
        "n_permutations": n_permutations,
    }


# ---------------------------------------------------------------------------
# Backend-level orchestration
# ---------------------------------------------------------------------------

def build_coherence_dataset(
    store: "EmbeddingStore",
    tcc_table: "pd.DataFrame",
    target_column: str = "tcc",
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align stored frozen embeddings with the TCC table.

    `target_column` selects which cellularity target to use: ``'tcc'``
    (whole-volume, the right target for backends that see the full 3D
    volume -- Merlin, MedImageInsights) or ``'slice_tcc'`` (restricted to
    the single GTV-argmax axial slice, the right target for the 2D-slice
    backends -- BiomedCLIP, MedSigLIP, CLIP, ImageNet ViT -- since they
    never see the rest of the volume `tcc` is computed from).

    TCC spans several orders of magnitude across tumour sizes; by default
    (`log_transform=True`) it's log10-transformed before use as the graph
    node feature, so Geary's C / Moran's I aren't dominated by a handful of
    very large tumours (standard practice for such heavy-tailed count
    features, e.g. in spatial-transcriptomics coherence tests).

    Returns
    -------
    X: float32 [N, D] embeddings
    y: float64 [N] (log10) target values
    patient_ids: str [N], for the same-patient-exclusion confound control
    """
    table = tcc_table.dropna(subset=[target_column])
    table = table[table[target_column] > 0]

    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    patient_rows: list[str] = []

    for pid, group in table.groupby("patient_id"):
        if pid not in store.patients():
            continue
        vectors, timepoints, _ = store.load_patient(pid)
        for _, row in group.iterrows():
            idx = np.where(np.isclose(timepoints, row["timepoint_days"], atol=1.0))[0]
            if idx.size == 0:
                continue
            X_rows.append(vectors[idx[0]])
            y_rows.append(row[target_column])
            patient_rows.append(pid)

    if not X_rows:
        return (
            np.empty((0, 0), dtype=np.float32),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=object),
        )

    X = np.stack(X_rows).astype(np.float32)
    y = np.asarray(y_rows, dtype=np.float64)
    if log_transform:
        y = np.log10(y)
    patient_ids = np.asarray(patient_rows, dtype=object)
    return X, y, patient_ids


def coherence_test_backend(
    store_path: str | Path,
    tcc_table: "pd.DataFrame",
    target_column: str = "tcc",
    k: int = 8,
    statistic: Literal["gearys_c", "morans_i"] = "gearys_c",
    n_permutations: int = 999,
    log_transform: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run Task 3's coherence test for one backend: build a k-NN graph in its
    embedding space, then compute the spatial-autocorrelation statistic
    for the chosen target (see `build_coherence_dataset` for
    `target_column`) over that graph with a permutation null -- twice: once
    over the full graph, once with same-patient neighbors excluded (see
    module docstring). A large gap between the two indicates the embedding
    is mostly clustering patient identity rather than tumour biology.
    """
    from embeddings.store import EmbeddingStore

    store = EmbeddingStore(store_path)
    X, y, patient_ids = build_coherence_dataset(
        store, tcc_table, target_column=target_column, log_transform=log_transform
    )

    result = {
        "backend": store.backend,
        "target": target_column,
        "statistic": statistic,
        "n_samples": int(len(y)),
        "k": k,
        "full_graph": None,
        "same_patient_excluded": None,
    }
    if len(y) < k + 1:
        return result

    W_full = build_knn_graph(X, k=k)
    result["full_graph"] = permutation_test(
        y, W_full, statistic=statistic, n_permutations=n_permutations, seed=seed
    )

    W_excl = build_knn_graph(X, k=k, patient_ids=patient_ids, exclude_same_patient=True)
    result["same_patient_excluded"] = permutation_test(
        y, W_excl, statistic=statistic, n_permutations=n_permutations, seed=seed
    )

    return result
