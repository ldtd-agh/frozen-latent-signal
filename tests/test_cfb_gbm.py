"""
Unit + integration tests for CFBGBMDataset.

Manifest-building tests use self-contained fixture TSVs (via `tmp_path`) so
they never depend on the real, cluster-mounted TCIA download. The
integration tests below require the real dataset and are skipped
automatically when it is not present.
"""

from __future__ import annotations

import os
import csv
from pathlib import Path

import numpy as np
import pytest

from data.datasets.cfb_gbm import (
    CFBGBMDataset,
    MRI_MODALITIES,
    TIMEPOINTS,
    build_manifest,
    usable_patients,
    usable_scans,
    _clean_rano,
    _scan_path,
    _parse_float,
    _zscore_normalize,
)

_TSV_DIR = Path(os.environ.get("CFB_GBM_ROOT", ""))
_IMAGING_ROOT = _TSV_DIR / "CFB-GBM"
DATA_AVAILABLE = _IMAGING_ROOT.exists() and _TSV_DIR.exists()


# ===========================================================================
# Pure-function unit tests (no I/O)
# ===========================================================================

class TestCleanRano:
    def test_extracts_code(self):
        assert _clean_rano("Stable Disease (SD)") == "SD"
        assert _clean_rano("Progressive Disease (PD)") == "PD"
        assert _clean_rano("Partial Response (PR)") == "PR"
        assert _clean_rano("Complete Response (CR)") == "CR"

    def test_none_and_blank(self):
        assert _clean_rano(None) is None
        assert _clean_rano("") is None
        assert _clean_rano("   ") is None

    def test_no_parens_passthrough(self):
        assert _clean_rano("SD") == "SD"


class TestParseFloat:
    def test_valid(self):
        assert _parse_float("41.96") == 41.96

    def test_blank_and_none(self):
        assert _parse_float("") is None
        assert _parse_float("   ") is None
        assert _parse_float(None) is None


class TestScanPath:
    def test_zero_padding_and_unpadded_prefix(self):
        p = _scan_path(Path("/root"), 15, "t1", "adc")
        assert p == Path("/root/015/t1/15_t1_adc.nii.gz")

    def test_three_digit_patient(self):
        p = _scan_path(Path("/root"), 264, "t0", "t1gd")
        assert p == Path("/root/264/t0/264_t0_t1gd.nii.gz")


class TestZscoreNormalize:
    def test_output_range(self):
        rng = np.random.default_rng(0)
        vol = rng.standard_normal((8, 16, 16)).astype(np.float32)
        out = _zscore_normalize(vol)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_zero_vol_returns_zeros(self):
        vol = np.zeros((8, 8, 8), dtype=np.float32)
        assert np.all(_zscore_normalize(vol) == 0.0)


# ===========================================================================
# Manifest-building tests (self-contained fixture TSVs, no real data)
# ===========================================================================

def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        writer.writerows(rows)


@pytest.fixture
def fixture_tsv_dir(tmp_path: Path) -> Path:
    """
    Build a tiny synthetic TSV set mirroring the real CFB-GBM schema, with
    a deliberate mix of complete/incomplete patients so filtering logic is
    actually exercised:

      001 — fully usable: adc + t1gd at t0/t1/t2, gtv at t0/t1/t2, both RANO
      002 — missing ADC at t2 -> not usable
      003 — missing GTV at t1 -> not usable
      004 — only t0->t1 RANO rated (no t2 imaging follow-up) -> not usable
    """
    mri_header = [
        "id_patient", "temporality", "time_diff_t0 (weeks)",
        "t1gd", "t1eg", "t1tse", "t2star", "t2tse", "flair", "adc", "mri_machine",
    ]
    mri_rows = [
        ["1", "t0", "0", "1", "0", "0", "1", "0", "1", "1", "SCANNER"],
        ["1", "t1", "16", "1", "0", "0", "1", "0", "0", "1", "SCANNER"],
        ["1", "t2", "24", "1", "0", "0", "0", "0", "0", "1", "SCANNER"],
        ["2", "t0", "0", "1", "0", "0", "1", "0", "1", "1", "SCANNER"],
        ["2", "t1", "17", "1", "0", "0", "1", "0", "0", "1", "SCANNER"],
        ["2", "t2", "26", "1", "0", "0", "0", "0", "0", "0", "SCANNER"],  # no ADC
        ["3", "t0", "0", "1", "0", "0", "1", "0", "1", "1", "SCANNER"],
        ["3", "t1", "15", "1", "0", "0", "1", "0", "0", "1", "SCANNER"],
        ["3", "t2", "25", "1", "0", "0", "0", "0", "0", "1", "SCANNER"],
        ["4", "t0", "0", "1", "0", "0", "1", "0", "1", "1", "SCANNER"],
        ["4", "t1", "18", "1", "0", "0", "1", "0", "0", "1", "SCANNER"],
        # patient 4 has no t2 MRI row at all
    ]
    _write_tsv(tmp_path / "CFB-GBM_mri_availability_v01.tsv", mri_header, mri_rows)

    tia_header = ["id_patient", "temporality", "gtv", "gtv_type", "rtdose", "treatment_machine", "tps"]
    tia_rows = [
        ["1", "t0", "1", "humanSegmentation", "1", "T1", "PRECISION"],
        ["1", "t1", "1", "humanSegmentation", "0", "", ""],
        ["1", "t2", "1", "humanSegmentation", "0", "", ""],
        ["2", "t0", "1", "humanSegmentation", "1", "T1", "PRECISION"],
        ["2", "t1", "1", "humanSegmentation", "0", "", ""],
        ["2", "t2", "1", "humanSegmentation", "0", "", ""],
        ["3", "t0", "1", "humanSegmentation", "1", "T1", "PRECISION"],
        ["3", "t1", "0", "", "0", "", ""],  # no GTV at t1
        ["3", "t2", "1", "humanSegmentation", "0", "", ""],
        ["4", "t0", "1", "humanSegmentation", "1", "T1", "PRECISION"],
        ["4", "t1", "1", "humanSegmentation", "0", "", ""],
    ]
    _write_tsv(tmp_path / "CFB-GBM_treatment_imaging_availability_v01.tsv", tia_header, tia_rows)

    rano_header = [
        "id_patient", "size_t0 (cm3)", "size_t1 (cm3)", "size_t2 (cm3)",
        "reduction_rate_t0_to_t1", "rano_t0_to_t1",
        "reduction_rate_t0_to_t2", "rano_t0_to_t2",
        "reduction_rate_t1_to_t2", "rano_t1_to_t2",
    ]
    rano_rows = [
        ["1", "40.0", "38.0", "35.0", "0.05", "Stable Disease (SD)",
         "0.12", "Stable Disease (SD)", "0.07", "Stable Disease (SD)"],
        ["2", "50.0", "48.0", "", "0.04", "Stable Disease (SD)", "", "", "", ""],
        ["3", "60.0", "90.0", "95.0", "-0.5", "Progressive Disease (PD)",
         "-0.58", "Progressive Disease (PD)", "-0.05", "Progressive Disease (PD)"],
        ["4", "30.0", "20.0", "", "0.33", "Partial Response (PR)", "", "", "", ""],
    ]
    _write_tsv(tmp_path / "CFB-GBM_rano_criteria_v01.tsv", rano_header, rano_rows)

    return tmp_path


class TestBuildManifest:
    def test_parses_all_patients(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert set(manifest.keys()) == {"001", "002", "003", "004"}

    def test_zero_pads_keys(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert manifest["001"]["patient_num"] == 1

    def test_time_diff_weeks_parsed(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert manifest["001"]["mri"]["t0"]["time_diff_weeks"] == 0.0
        assert manifest["001"]["mri"]["t1"]["time_diff_weeks"] == 16.0
        assert manifest["001"]["mri"]["t2"]["time_diff_weeks"] == 24.0

    def test_missing_timepoint_defaults_false_and_none(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert manifest["004"]["mri"]["t2"]["time_diff_weeks"] is None
        assert manifest["004"]["mri"]["t2"]["adc"] is False

    def test_rano_codes_cleaned(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert manifest["001"]["rano"]["rano_t0_to_t1"] == "SD"
        assert manifest["003"]["rano"]["rano_t1_to_t2"] == "PD"
        assert manifest["004"]["rano"]["rano_t0_to_t1"] == "PR"
        assert manifest["004"]["rano"]["rano_t1_to_t2"] is None

    def test_tumor_size_parsed(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert manifest["001"]["tumor_size_cm3"]["t2"] == 35.0
        assert manifest["002"]["tumor_size_cm3"]["t2"] is None


class TestUsablePatients:
    def test_default_filters_to_fully_complete(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        usable = usable_patients(manifest)
        assert usable == ["001"]

    def test_missing_adc_excluded(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        usable = usable_patients(manifest)
        assert "002" not in usable

    def test_missing_gtv_excluded(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        usable = usable_patients(manifest)
        assert "003" not in usable

    def test_incomplete_rano_excluded(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        usable = usable_patients(manifest)
        assert "004" not in usable

    def test_relaxed_filters_include_more(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        usable = usable_patients(manifest, require_gtv=False, require_rano=False)
        assert set(usable) >= {"001", "003"}

    def test_extra_modality_requirement(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        # patient 001 never has t2tse at t2 -> excluded when required
        usable = usable_patients(manifest, modalities=["t1gd", "t2tse"], require_gtv=False, require_rano=False)
        assert "001" not in usable


class TestUsableScans:
    """
    Scan-level counterpart to TestUsablePatients, on the same fixture:
    001 fully complete; 002 missing ADC only at t2; 003 missing GTV only
    at t1; 004 has no t2 imaging at all and only one RANO transition.
    """

    def test_fully_complete_patient_all_three(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        scans = usable_scans(manifest)
        assert ("001", "t0") in scans
        assert ("001", "t1") in scans
        assert ("001", "t2") in scans

    def test_missing_adc_at_one_timepoint_still_includes_others(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        scans = usable_scans(manifest)
        assert ("002", "t0") in scans
        assert ("002", "t1") in scans
        assert ("002", "t2") not in scans  # ADC missing only at t2

    def test_missing_gtv_at_one_timepoint_still_includes_others(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        scans = usable_scans(manifest)
        assert ("003", "t0") in scans
        assert ("003", "t1") not in scans  # GTV missing only at t1
        assert ("003", "t2") in scans

    def test_no_rano_and_no_third_timepoint_still_included(self, fixture_tsv_dir):
        # Patient 004 fails usable_patients() entirely (no t2 imaging at
        # all, and only one of the two RANO transitions) -- but
        # usable_scans() doesn't need trajectory pairing or RANO.
        manifest = build_manifest(fixture_tsv_dir)
        scans = usable_scans(manifest)
        assert ("004", "t0") in scans
        assert ("004", "t1") in scans
        assert ("004", "t2") not in scans

    def test_total_count(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        assert len(usable_scans(manifest)) == 9  # 3 + 2 + 2 + 2

    def test_more_patients_contribute_than_usable_patients(self, fixture_tsv_dir):
        manifest = build_manifest(fixture_tsv_dir)
        scan_patients = {pid for pid, tp in usable_scans(manifest)}
        traj_patients = set(usable_patients(manifest))
        assert scan_patients == {"001", "002", "003", "004"}
        assert traj_patients == {"001"}


# ===========================================================================
# Integration tests (require real data)
# ===========================================================================

@pytest.mark.skipif(not DATA_AVAILABLE, reason="CFB-GBM data not available")
class TestCFBGBMDataset:

    def test_usable_patient_count(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        # Exact usable-N determined from the real TSVs (ADC + GTV + both RANO
        # transitions present at all 3 timepoints, T1Gd requested).
        assert len(ds.patients()) == 24

    def test_patient_ids_format(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        for pid in ds.patients():
            assert len(pid) == 3 and pid.isdigit()

    def test_scans_sorted_and_three_timepoints(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        pid = ds.patients()[0]
        scans = ds.scans_for_patient(pid)
        assert len(scans) == 3
        tps = [s.timepoint for s in scans]
        assert tps == sorted(tps)
        assert tps[0] == 0.0

    def test_scan_records_matches_scans_for_patient_metadata(self):
        """scan_records() (metadata-only) must agree with scans_for_patient()
        (loads pixels too) on everything except the image itself."""
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        pid = ds.patients()[0]
        scans = ds.scans_for_patient(pid)
        records = {r["metadata"]["timepoint_label"]: r for r in ds.scan_records(pid)}
        assert len(records) == len(scans)
        for s in scans:
            r = records[s.metadata["timepoint_label"]]
            assert r["timepoint"] == s.timepoint
            assert r["outcome"] == s.outcome
            assert r["metadata"]["adc_path"] == s.metadata["adc_path"]
            assert r["metadata"]["gtv_path"] == s.metadata["gtv_path"]
            assert r["metadata"]["tumor_size_cm3"] == s.metadata["tumor_size_cm3"]

    def test_image_shape_single_modality(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, modalities=["t1gd"])
        scans = ds.scans_for_patient(ds.patients()[0])
        for s in scans:
            assert s.image.ndim == 3, "Expected [D, H, W]"

    def test_image_shape_multi_modality(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, modalities=["t1gd", "t2star"])
        scans = ds.scans_for_patient(ds.patients()[0])
        for s in scans:
            assert s.image.ndim == 4
            assert s.image.shape[0] == 2

    def test_projection_2d(self):
        ds = CFBGBMDataset(
            _IMAGING_ROOT, tsv_dir=_TSV_DIR, modalities=["t1gd"],
            projection_2d=True, slice_selection="gtv_mask",
        )
        scans = ds.scans_for_patient(ds.patients()[0])
        for s in scans:
            assert s.image.ndim == 2

    def test_adc_and_gtv_paths_present(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        scans = ds.scans_for_patient(ds.patients()[0])
        for s in scans:
            assert s.metadata["adc_path"] is not None
            assert Path(s.metadata["adc_path"]).exists()
            assert s.metadata["gtv_path"] is not None
            assert Path(s.metadata["gtv_path"]).exists()

    def test_outcome_labels(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        scans = ds.scans_for_patient(ds.patients()[0])
        by_tp = {s.metadata["timepoint_label"]: s for s in scans}
        assert by_tp["t0"].outcome is None
        assert by_tp["t1"].outcome in ("SD", "PD", "PR", "CR")
        assert by_tp["t2"].outcome in ("SD", "PD", "PR", "CR")

    def test_final_outcome(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        outcomes = ds.all_final_outcomes()
        assert len(outcomes) == len(ds.patients())
        assert all(v in ("SD", "PD", "PR", "CR") for v in outcomes.values())

    def test_registration_matches_across_modalities(self):
        """ADC / GTV / requested modality share geometry within a timepoint."""
        import SimpleITK as sitk

        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR)
        pid = ds.patients()[0]
        patient_num = ds._manifest[pid]["patient_num"]
        for tp in TIMEPOINTS:
            t1gd = sitk.ReadImage(str(_scan_path(_IMAGING_ROOT, patient_num, tp, "t1gd")))
            adc = sitk.ReadImage(str(_scan_path(_IMAGING_ROOT, patient_num, tp, "adc")))
            assert t1gd.GetSize() == adc.GetSize()
            assert t1gd.GetOrigin() == adc.GetOrigin()


@pytest.mark.skipif(not DATA_AVAILABLE, reason="CFB-GBM data not available")
class TestCFBGBMDatasetScanPairing:
    """
    pairing='scan' real-data checks. Deliberately avoid calling
    scans_for_patient() across the whole ~205-patient set here (that loads
    ~334 full-resolution volumes and takes several minutes) -- construction
    itself is manifest-only and cheap, so most assertions go through
    ds.patients()/ds._scan_filter; only a single patient's scans are
    actually loaded, as a lightweight smoke test.
    """

    def test_scan_pairing_gives_more_patients_than_trajectory(self):
        traj_ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="trajectory")
        scan_ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="scan")
        assert len(traj_ds.patients()) == 24
        assert len(scan_ds.patients()) == 205
        assert set(traj_ds.patients()) <= set(scan_ds.patients())

    def test_scan_filter_totals_match_usable_scans(self):
        ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="scan")
        assert sum(len(v) for v in ds._scan_filter.values()) == 334

    def test_partial_trajectory_patient_only_yields_its_valid_scans(self):
        """A patient present under 'scan' but not 'trajectory' pairing has
        fewer than 3 qualifying timepoints -- confirm scans_for_patient()
        only loads those, not all 3."""
        traj_ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="trajectory")
        scan_ds = CFBGBMDataset(_IMAGING_ROOT, tsv_dir=_TSV_DIR, pairing="scan")
        partial_only = set(scan_ds.patients()) - set(traj_ds.patients())
        assert partial_only  # sanity: such patients exist

        pid = sorted(partial_only)[0]
        expected_tps = scan_ds._scan_filter[pid]
        assert 1 <= len(expected_tps) < 3

        scans = scan_ds.scans_for_patient(pid)
        assert {s.metadata["timepoint_label"] for s in scans} == expected_tps
        for s in scans:
            assert s.metadata["adc_path"] is not None
            assert s.metadata["gtv_path"] is not None
