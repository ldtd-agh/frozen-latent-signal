"""
Unit + integration tests for analysis/cellularity.py (TumorTwin TCC +
Geary's C / Moran's I population coherence test).

Pure-function and synthetic-fixture tests never touch real imaging data,
TumorTwin, or embedding stores. The integration tests at the bottom require
the real CFB-GBM dataset (and, for the coherence test, an already-computed
embedding store) and are skipped automatically when unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from analysis.cellularity import (
    build_coherence_dataset,
    build_knn_graph,
    build_tcc_table,
    coherence_test_backend,
    compute_scan_tcc,
    gearys_c,
    morans_i,
    permutation_test,
    scan_tcc_rows,
)
_TSV_DIR = Path(os.environ.get("CFB_GBM_ROOT", ""))
_IMAGING_ROOT = _TSV_DIR / "CFB-GBM"
DATA_AVAILABLE = _IMAGING_ROOT.exists() and _TSV_DIR.exists()

_STORE_PATH = Path("stores/cfb_gbm/cfb_gbm_medsiglip.h5")
STORE_AVAILABLE = _STORE_PATH.exists()


# ===========================================================================
# Geary's C / Moran's I (pure, no I/O)
# ===========================================================================

def _two_pairs_graph() -> sp.csr_matrix:
    """4 nodes, two disconnected edges: 0-1 and 2-3."""
    return sp.csr_matrix(
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=float,
        )
    )


class TestGearysC:
    def test_perfect_clustering_gives_zero(self):
        W = _two_pairs_graph()
        y = np.array([10.0, 10.0, 0.0, 0.0])  # identical within each connected pair
        assert gearys_c(y, W) == pytest.approx(0.0, abs=1e-9)

    def test_no_variance_returns_nan(self):
        W = _two_pairs_graph()
        y = np.ones(4)
        assert np.isnan(gearys_c(y, W))

    def test_empty_graph_returns_nan(self):
        W = sp.csr_matrix((4, 4))
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(gearys_c(y, W))


class TestMoransI:
    def test_perfect_clustering_gives_one(self):
        W = _two_pairs_graph()
        y = np.array([10.0, 10.0, 0.0, 0.0])
        assert morans_i(y, W) == pytest.approx(1.0, abs=1e-9)

    def test_no_variance_returns_nan(self):
        W = _two_pairs_graph()
        y = np.ones(4)
        assert np.isnan(morans_i(y, W))

    def test_empty_graph_returns_nan(self):
        W = sp.csr_matrix((4, 4))
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(morans_i(y, W))


# ===========================================================================
# build_knn_graph (pure, no I/O)
# ===========================================================================

class TestBuildKnnGraph:
    def test_min_degree_at_least_k(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        W = build_knn_graph(X, k=3)
        degrees = np.asarray(W.sum(axis=1)).flatten()
        assert (degrees >= 3).all()

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 2))
        W = build_knn_graph(X, k=4)
        assert (W != W.T).nnz == 0

    def test_no_self_loops(self):
        rng = np.random.default_rng(2)
        X = rng.standard_normal((10, 2))
        W = build_knn_graph(X, k=2)
        assert W.diagonal().sum() == 0

    def test_exclude_same_patient(self):
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.standard_normal((5, 3)) + np.array([10.0, 0.0, 0.0]),
                rng.standard_normal((5, 3)) + np.array([-10.0, 0.0, 0.0]),
            ]
        )
        patient_ids = np.array(["A"] * 5 + ["B"] * 5)
        W = build_knn_graph(X, k=3, patient_ids=patient_ids, exclude_same_patient=True)
        Wc = W.tocoo()
        for i, j in zip(Wc.row, Wc.col):
            assert patient_ids[i] != patient_ids[j]

    def test_exclude_same_patient_requires_patient_ids(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(ValueError):
            build_knn_graph(X, k=2, exclude_same_patient=True)

    def test_k_too_large_for_exclusion_raises(self):
        X = np.random.default_rng(0).standard_normal((5, 2))
        patient_ids = np.array(["A", "A", "A", "B", "B"])
        # patient-A nodes only have 2 other-patient (B) candidates < k=3
        with pytest.raises(ValueError):
            build_knn_graph(X, k=3, patient_ids=patient_ids, exclude_same_patient=True)


# ===========================================================================
# permutation_test (pure, no I/O)
# ===========================================================================

class TestPermutationTest:
    def test_detects_clustering(self):
        rng = np.random.default_rng(0)
        # two well-separated blobs -> a k-NN graph should mostly connect within-blob
        X = np.vstack(
            [rng.standard_normal((15, 3)) + 10, rng.standard_normal((15, 3)) - 10]
        )
        y = np.array([1.0] * 15 + [0.0] * 15)
        W = build_knn_graph(X, k=4)
        result = permutation_test(y, W, statistic="gearys_c", n_permutations=200, seed=0)
        assert result["p_value"] < 0.05
        assert result["observed"] < result["null_mean"]

    def test_reproducible_with_seed(self):
        W = _two_pairs_graph()
        y = np.array([1.0, 2.0, 3.0, 4.0])
        r1 = permutation_test(y, W, n_permutations=50, seed=7)
        r2 = permutation_test(y, W, n_permutations=50, seed=7)
        assert r1 == r2

    def test_morans_i_direction(self):
        rng = np.random.default_rng(0)
        X = np.vstack(
            [rng.standard_normal((15, 3)) + 10, rng.standard_normal((15, 3)) - 10]
        )
        y = np.array([1.0] * 15 + [0.0] * 15)
        W = build_knn_graph(X, k=4)
        result = permutation_test(y, W, statistic="morans_i", n_permutations=200, seed=0)
        assert result["p_value"] < 0.05
        assert result["observed"] > result["null_mean"]


# ===========================================================================
# scan_tcc_rows / build_tcc_table (synthetic dataset, compute_scan_tcc mocked)
# ===========================================================================

class _FakeDataset:
    """Duck-typed stand-in for CFBGBMDataset's scan_records()-based interface."""

    def __init__(self, records_by_patient: dict[str, list[dict]]):
        self._records = records_by_patient

    def patients(self) -> list[str]:
        return list(self._records.keys())

    def scan_records(self, patient_id: str) -> list[dict]:
        return self._records[patient_id]


def _make_record(patient_id: str, timepoint_label: str, timepoint_days: float, outcome) -> dict:
    return {
        "patient_num": int(patient_id),
        "timepoint": timepoint_days,
        "outcome": outcome,
        "metadata": {
            "timepoint_label": timepoint_label,
            "adc_path": f"/fake/{patient_id}_{timepoint_label}_adc.nii.gz",
            "gtv_path": f"/fake/{patient_id}_{timepoint_label}_gtv.nii.gz",
            "tumor_size_cm3": 40.0,
        },
    }


@pytest.fixture
def fake_scan_tcc(monkeypatch):
    values = {
        "t0": {"tcc": 1.0e9, "slice_tcc": 4.0e8, "slice_index": 60, "slice_axis": 2},
        "t1": {"tcc": 2.0e9, "slice_tcc": 8.0e8, "slice_index": 61, "slice_axis": 2},
        "t2": {"tcc": 0.5e9, "slice_tcc": 2.0e8, "slice_index": 59, "slice_axis": 2},
    }

    def _fake(adc_path, gtv_path, **kwargs):
        for tp, v in values.items():
            if tp in str(adc_path):
                return v
        return None

    monkeypatch.setattr("analysis.cellularity.compute_scan_tcc", _fake)
    return values


class TestScanTccRows:
    def test_values_and_metadata(self, fake_scan_tcc):
        records = [
            _make_record("001", "t0", 0.0, None),
            _make_record("001", "t1", 112.0, "SD"),
        ]
        dataset = _FakeDataset({"001": records})
        rows = scan_tcc_rows(dataset, "001")
        by_tp = {r["timepoint_label"]: r for r in rows}
        assert by_tp["t0"]["tcc"] == pytest.approx(1.0e9)
        assert by_tp["t1"]["tcc"] == pytest.approx(2.0e9)
        assert by_tp["t0"]["slice_tcc"] == pytest.approx(4.0e8)
        assert by_tp["t1"]["slice_tcc"] == pytest.approx(8.0e8)
        assert by_tp["t1"]["slice_index"] == 61
        assert by_tp["t1"]["rano"] == "SD"
        assert by_tp["t0"]["tumor_size_cm3"] == 40.0

    def test_missing_paths_yield_none(self, fake_scan_tcc):
        record = _make_record("001", "t0", 0.0, None)
        record["metadata"]["adc_path"] = None
        dataset = _FakeDataset({"001": [record]})
        rows = scan_tcc_rows(dataset, "001")
        assert rows[0]["tcc"] is None

    def test_build_table_multi_patient(self, fake_scan_tcc):
        dataset = _FakeDataset(
            {
                "001": [_make_record("001", "t0", 0.0, None)],
                "002": [_make_record("002", "t0", 0.0, None), _make_record("002", "t1", 100.0, "PD")],
            }
        )
        table = build_tcc_table(dataset)
        assert len(table) == 3
        assert set(table["patient_id"]) == {"001", "002"}


# ===========================================================================
# build_coherence_dataset (synthetic store, no I/O)
# ===========================================================================

class _FakeStore:
    def __init__(self, data: dict[str, tuple[np.ndarray, np.ndarray]]):
        self._data = data

    def patients(self) -> list[str]:
        return list(self._data.keys())

    def load_patient(self, pid: str):
        vectors, timepoints = self._data[pid]
        return vectors, timepoints, [""] * len(timepoints)


def _make_synthetic_coherence_data(n_patients=10, dim=4, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    store_data = {}
    rows = []
    for i in range(n_patients):
        pid = f"{i:03d}"
        n_tp = rng.integers(1, 4)  # 1-3 scans, uneven -- like real CFB-GBM coverage
        tps = np.sort(rng.choice([0.0, 100.0, 200.0], size=n_tp, replace=False)).astype(np.float32)
        tcc = rng.uniform(1e8, 1e10, size=n_tp)
        vecs = rng.standard_normal((n_tp, dim)).astype(np.float32)
        store_data[pid] = (vecs, tps)
        for t, v in zip(tps, tcc):
            rows.append({"patient_id": pid, "timepoint_days": float(t), "tcc": float(v)})
    return _FakeStore(store_data), pd.DataFrame(rows)


class TestBuildCoherenceDataset:
    def test_shapes_and_log_transform(self):
        store, table = _make_synthetic_coherence_data()
        X, y, patient_ids = build_coherence_dataset(store, table, log_transform=True)
        assert X.shape[0] == len(table)
        assert y.shape[0] == len(table)
        assert patient_ids.shape[0] == len(table)
        # log10 of values in [1e8, 1e10] -> roughly [8, 10]
        assert y.min() >= 7.5
        assert y.max() <= 10.5

    def test_no_log_transform(self):
        store, table = _make_synthetic_coherence_data()
        X, y, _ = build_coherence_dataset(store, table, log_transform=False)
        assert y.min() >= 1e8

    def test_drops_nonpositive_and_missing_tcc(self):
        store, table = _make_synthetic_coherence_data()
        table.loc[0, "tcc"] = np.nan
        table.loc[1, "tcc"] = -5.0
        X, y, _ = build_coherence_dataset(store, table)
        assert X.shape[0] == len(table) - 2

    def test_missing_patient_in_store_skipped(self):
        store, table = _make_synthetic_coherence_data()
        extra = table.iloc[[0]].copy()
        extra["patient_id"] = "999"
        import pandas as pd

        table = pd.concat([table, extra], ignore_index=True)
        _, _, patient_ids = build_coherence_dataset(store, table)
        assert "999" not in patient_ids


class TestCoherenceTestBackend:
    def test_too_few_samples_returns_none_results(self, tmp_path, monkeypatch):
        import pandas as pd

        from embeddings.store import EmbeddingStore

        store_path = tmp_path / "tiny.h5"
        store = EmbeddingStore(store_path, backend_name="fake")
        store.save_patient("001", np.random.default_rng(0).standard_normal((1, 4)).astype(np.float32), np.array([0.0]))
        table = pd.DataFrame([{"patient_id": "001", "timepoint_days": 0.0, "tcc": 1e9}])

        result = coherence_test_backend(store_path, table, k=8)
        assert result["n_samples"] == 1
        assert result["full_graph"] is None
        assert result["same_patient_excluded"] is None


# ===========================================================================
# Integration tests (require real data / real TumorTwin / real embedding stores)
# ===========================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="CFB-GBM data not available")
class TestTccRealData:
    def test_compute_scan_tcc_is_positive_and_plausible(self):
        adc_path = _IMAGING_ROOT / "015" / "t0" / "15_t0_adc.nii.gz"
        gtv_path = _IMAGING_ROOT / "015" / "t0" / "15_t0_gtv.nii.gz"
        result = compute_scan_tcc(adc_path, gtv_path)
        assert result is not None
        assert 1e7 < result["tcc"] < 1e12
        # slice_tcc is a single-slab subset of the whole-volume TCC -> strictly smaller
        assert result["slice_tcc"] is not None
        assert 0 < result["slice_tcc"] < result["tcc"]
        assert isinstance(result["slice_index"], int)

    def test_slice_index_matches_real_pipelines_slice_selection(self):
        """
        The slice compute_scan_tcc() picks (via TumorTwin/nibabel + argmax
        voxel spacing) must be the exact same physical slice
        select_best_axial_slice() picks (via SimpleITK + RAS reorientation)
        for the real 2D embedding backends -- otherwise slice_tcc isn't
        actually the target those backends were shown.
        """
        import SimpleITK as sitk

        adc_path = _IMAGING_ROOT / "015" / "t0" / "15_t0_adc.nii.gz"
        gtv_path = _IMAGING_ROOT / "015" / "t0" / "15_t0_gtv.nii.gz"

        result = compute_scan_tcc(adc_path, gtv_path)

        img = sitk.ReadImage(str(gtv_path))
        img = sitk.DICOMOrient(img, "RAS")
        arr = sitk.GetArrayFromImage(img)  # [D, H, W], D = axis 0
        counts = arr.sum(axis=(1, 2))
        expected_index = int(counts.argmax())

        assert result["slice_index"] == expected_index

    def test_build_tcc_table_scan_pairing(self):
        from data.datasets.cfb_gbm import CFBGBMDataset, build_manifest, usable_scans

        manifest = build_manifest(_TSV_DIR)
        scan_patients = sorted({pid for pid, _ in usable_scans(manifest)})[:2]
        dataset = CFBGBMDataset(
            _IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="scan", patient_ids=scan_patients
        )
        table = build_tcc_table(dataset)
        assert len(table) >= 2
        valid = table["tcc"].dropna()
        assert len(valid) > 0
        assert (valid > 0).all()


@pytest.mark.skipif(
    not (DATA_AVAILABLE and STORE_AVAILABLE),
    reason="CFB-GBM data or a computed embedding store not available",
)
class TestCoherenceRealStore:
    def test_coherence_test_backend_runs(self):
        from data.datasets.cfb_gbm import CFBGBMDataset

        # Use the (old, trajectory-paired) 24-patient dataset here just to
        # exercise the wiring cheaply against whatever store already
        # exists; the real Task 3 run uses pairing='scan' with a freshly
        # re-embedded, much larger store (see scripts/cellularity_coherence.py).
        dataset = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="trajectory")
        table = build_tcc_table(dataset)
        result = coherence_test_backend(_STORE_PATH, table, k=5, n_permutations=99)
        assert result["backend"] == "medsiglip"
        assert result["n_samples"] > 0
        if result["full_graph"] is not None:
            assert "p_value" in result["full_graph"]
            assert "p_value" in result["same_patient_excluded"]
