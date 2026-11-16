"""
CFB-GBM longitudinal GBM dataset adapter.

Dataset structure
------------------
<imaging_root>/
    <PPP>/                              # zero-padded 3-digit patient number, e.g. "015"
        t0/
            <N>_t0_<modality>.nii.gz    # <N> = patient number, NOT zero-padded
        t1/
            <N>_t1_<modality>.nii.gz
        t2/
            <N>_t2_<modality>.nii.gz

Modality file suffixes: t1gd, t1eg, t1tse, t2star, t2tse, flair, adc (all
potentially at every timepoint), plus ct / rtdose (t0 only), gtv (tumour
segmentation mask) and brain_mask.

Supporting TSVs (in <tsv_dir>, defaults to the parent of <imaging_root>):
    *mri_availability*.tsv                 — per (patient, timepoint) MRI
                                              modality flags + actual elapsed
                                              time ('time_diff_t0 (weeks)').
    *treatment_imaging_availability*.tsv   — per (patient, timepoint) GTV /
                                              RTDOSE flags.
    *rano_criteria*.tsv                    — per-patient tumour size and RANO
                                              rating for each transition.

Timepoints
----------
Exactly 3 fixed timepoints per patient: t0 (pre-treatment), t1 (~4 months
post), t2 (~6 months post) — no week-folder parsing needed, unlike Lumiere.
Elapsed days are derived per-patient from the `time_diff_t0 (weeks)` column
(actual values vary considerably: t1 ranges 5-35 weeks, t2 ranges 14-55
weeks — do not assume the nominal 4/6-month figures hold for every patient).

Registration
------------
All modalities for a given (patient, timepoint) share the same
size/spacing/origin/direction — already registered to that patient's own T0
T1Gd space by TCIA. This is *within-patient* registration only; there is no
cross-patient atlas space equivalent to Lumiere's DeepBraTumIA atlas.

Cellularity
-----------
This adapter stays dumb about ADC-derived cellularity: it exposes the raw
ADC volume path and GTV mask path via `PatientScan.metadata`
(`adc_path`, `gtv_path`); cellularity computation lives in
`analysis/cellularity.py`.

RANO
----
The RANO TSV rates *transitions*, not absolute timepoints
(`rano_t0_to_t1`, `rano_t1_to_t2`, `rano_t0_to_t2` — Stable/Progressive/
Partial Response/Complete Response). `PatientScan.outcome` at t1 holds the
t0->t1 transition rating; at t2, the t1->t2 rating. t0 has no prior
transition so `outcome=None`. The full set of transition codes is also
copied into `metadata` for convenience.

Usable subset
-------------
`CFBGBMDataset.patients()` returns only patients with the requested
modalities, ADC, and GTV mask present at all three timepoints, and both
RANO transitions rated (mirrors Lumiere's `min_timepoints` filtering, but
on a richer availability condition). As of the TCIA CFB-GBM v4 release this
comes to 24/264 patients — call `usable_patients()` directly against a
freshly parsed manifest to recompute if the TSVs are updated.

Reference
---------
CFB-GBM (TCIA collection), DOI 10.7937/v9pn-2f72.
https://www.cancerimagingarchive.net/collection/cfb-gbm/
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path
from typing import Sequence, Literal

import numpy as np

from .base import LongitudinalDataset, PatientScan, select_best_axial_slice

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEPOINTS: tuple[str, ...] = ("t0", "t1", "t2")

# Modalities available (potentially) at every timepoint. `ct` / `rtdose` are
# t0-only and therefore excluded — they can never satisfy an "all 3
# timepoints" availability requirement.
MRI_MODALITIES: tuple[str, ...] = (
    "t1gd", "t1eg", "t1tse", "t2star", "t2tse", "flair", "adc",
)

DEFAULT_MODALITIES: list[str] = ["t1gd"]

_RANO_CODE_RE = re.compile(r"\(([A-Za-z]{2})\)\s*$")

_CANONICAL_ORIENTATION = "RAS"


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class CFBGBMDataset(LongitudinalDataset):
    """
    Longitudinal GBM MRI dataset (CFB-GBM, TCIA).

    Each scan yields a float32 volume of shape:
        [C, H, W]   if ``projection_2d=True`` and ``len(modalities) > 1``
        [H, W]      if ``projection_2d=True`` and ``len(modalities) == 1``
        [C, D, H, W] / [D, H, W]  otherwise (full volume)

    Parameters
    ----------
    imaging_root:
        Path to the directory containing the zero-padded patient folders
        (e.g. ``.../CFB-GBM/``, holding ``001/``, ``002/``, ...).
    tsv_dir:
        Directory containing the CFB-GBM TSVs. Defaults to
        ``imaging_root.parent`` (the standard TCIA download layout).
    modalities:
        Subset of `MRI_MODALITIES` to include as image channels. Order
        defines channel order in the stacked array. Default: ``["t1gd"]``
        (contrast-enhanced T1 — the primary RANO criterion, analogous to
        Lumiere's CT1 choice).
    require_adc / require_gtv / require_rano:
        Whether ADC, the GTV mask, and both RANO transitions must be
        present at all 3 timepoints for a patient to be included. All
        default to True (the "usable subset" from PLANNING.md). Loosen
        these for exploratory runs that don't need the full analysis
        pipeline. Only consulted when ``pairing='trajectory'``.
    pairing:
        ``'trajectory'`` (default): a patient is included only if
        `modalities` (+ ADC/GTV if required) are present at **all 3**
        timepoints, and (if `require_rano`) both RANO transitions are
        rated -- the "usable subset" needed for within-patient analyses
        (Task 1/2's original 24-patient set). ``require_rano`` is ignored
        in this mode.
        ``'scan'``: no trajectory pairing at all -- every individual
        (patient, timepoint) scan with `modalities` (+ ADC/GTV if
        required) present is included on its own, regardless of what's
        available at that patient's other timepoints or whether RANO is
        rated. `require_rano` is ignored in this mode. Gives a much
        larger N for population-level analyses that don't need
        within-patient pairing (e.g. Task 3's coherence test) --
        ~334 scans across ~205 patients vs. 72 scans across 24 patients
        for T1Gd+ADC+GTV, as of the TCIA v4 release.
    normalize:
        If True (default), z-score each channel independently (excluding
        zero/background voxels), clip to ±5σ, then rescale to [0, 1].
    patient_ids:
        Optional list of patient IDs (zero-padded 3-digit strings, e.g.
        ``["015", "025"]``) to restrict to. Must already satisfy the
        `require_*` filters. Default: all usable patients.
    projection_2d:
        Collapse each volume to a single axial slice (needed for 2-D
        embedding backends). Default False (full 3-D volume).
    slice_selection:
        ``'gtv_mask'`` picks the slice with the most GTV voxels;
        ``'mid'`` (default) uses the mid-axial slice.
    """

    def __init__(
        self,
        imaging_root: str | Path,
        tsv_dir: str | Path | None = None,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        require_adc: bool = True,
        require_gtv: bool = True,
        require_rano: bool = True,
        normalize: bool = True,
        patient_ids: list[str] | None = None,
        projection_2d: bool = False,
        slice_selection: Literal["gtv_mask", "mid"] = "mid",
        pairing: Literal["trajectory", "scan"] = "trajectory",
    ) -> None:
        self.imaging_root = Path(imaging_root)
        self.tsv_dir = Path(tsv_dir) if tsv_dir is not None else self.imaging_root.parent
        self.modalities = list(modalities)
        self.require_adc = require_adc
        self.require_gtv = require_gtv
        self.require_rano = require_rano
        self.normalize = normalize
        self.projection_2d = projection_2d

        if pairing not in ("trajectory", "scan"):
            raise ValueError(f"pairing must be 'trajectory' or 'scan', got {pairing!r}")
        self.pairing = pairing

        if slice_selection not in ("gtv_mask", "mid"):
            raise ValueError(
                f"slice_selection must be 'gtv_mask' or 'mid', got {slice_selection!r}"
            )
        self.slice_selection = slice_selection

        for m in self.modalities:
            if m not in MRI_MODALITIES:
                raise ValueError(
                    f"Unknown modality: {m!r}.  Choose from {list(MRI_MODALITIES)}."
                )

        if not self.imaging_root.exists():
            raise FileNotFoundError(f"Imaging root not found: {self.imaging_root}")

        self._manifest = build_manifest(self.tsv_dir)

        if self.pairing == "trajectory":
            usable = usable_patients(
                self._manifest,
                modalities=self.modalities,
                require_adc=self.require_adc,
                require_gtv=self.require_gtv,
                require_rano=self.require_rano,
            )
            self._scan_filter: dict[str, set[str]] | None = None
        else:  # "scan"
            scan_pairs = usable_scans(
                self._manifest,
                modalities=self.modalities,
                require_adc=self.require_adc,
                require_gtv=self.require_gtv,
            )
            self._scan_filter = {}
            for pid, tp in scan_pairs:
                self._scan_filter.setdefault(pid, set()).add(tp)
            usable = sorted(self._scan_filter.keys())

        if patient_ids is not None:
            missing = set(patient_ids) - set(usable)
            if missing:
                warnings.warn(f"Requested patients not usable/found: {sorted(missing)}")
            self._patients = [p for p in patient_ids if p in set(usable)]
        else:
            self._patients = usable

    # ------------------------------------------------------------------
    # LongitudinalDataset interface
    # ------------------------------------------------------------------

    def patients(self) -> list[str]:
        return self._patients

    def scans_for_patient(self, patient_id: str) -> list[PatientScan]:
        """Return the (up to 3) scans for one patient, sorted by timepoint."""
        scans: list[PatientScan] = []
        for stub in self.scan_records(patient_id):
            gtv_mask = _load_mask(Path(stub["metadata"]["gtv_path"])) if stub["metadata"]["gtv_path"] else None
            try:
                image = self._load_image(stub["patient_num"], stub["metadata"]["timepoint_label"], gtv_mask)
            except FileNotFoundError as exc:
                warnings.warn(f"Skipping {patient_id}/{stub['metadata']['timepoint_label']}: {exc}")
                continue

            scans.append(
                PatientScan(
                    patient_id=patient_id,
                    timepoint=stub["timepoint"],
                    image=image,
                    modality="MRI",
                    delivered_dose=None,
                    outcome=stub["outcome"],
                    metadata=stub["metadata"],
                )
            )

        scans.sort(key=lambda s: s.timepoint)
        return scans

    def scan_records(self, patient_id: str) -> list[dict]:
        """
        Metadata-only counterpart to `scans_for_patient()`: same timepoint
        selection, day/outcome computation and path resolution, but never
        touches pixel data. Cheap to call across the whole population --
        used by analyses (e.g. Task 3's TCC computation) that only need
        `adc_path`/`gtv_path`/RANO/timepoint, not the requested imaging
        modality volume.

        Returns
        -------
        List of ``{"patient_num": int, "timepoint": float, "outcome": str
        | None, "metadata": dict}`` dicts, one per qualifying timepoint,
        NOT sorted by timepoint (unlike `scans_for_patient()`).
        """
        entry = self._manifest.get(patient_id)
        if entry is None:
            raise KeyError(f"Unknown patient: {patient_id}")
        patient_num = entry["patient_num"]

        records: list[dict] = []
        for tp in TIMEPOINTS:
            if self._scan_filter is not None and tp not in self._scan_filter.get(patient_id, ()):
                continue

            if tp == "t0":
                days = 0.0
            else:
                weeks = entry["mri"][tp]["time_diff_weeks"]
                if weeks is None:
                    warnings.warn(f"Skipping {patient_id}/{tp}: no time_diff_t0 recorded")
                    continue
                days = weeks * 7.0

            outcome = None
            if tp == "t1":
                outcome = entry["rano"]["rano_t0_to_t1"]
            elif tp == "t2":
                outcome = entry["rano"]["rano_t1_to_t2"]

            adc_path = _scan_path(self.imaging_root, patient_num, tp, "adc")
            gtv_path = _scan_path(self.imaging_root, patient_num, tp, "gtv")

            records.append(
                {
                    "patient_num": patient_num,
                    "timepoint": days,
                    "outcome": outcome,
                    "metadata": {
                        "dataset": "CFB-GBM",
                        "timepoint_label": tp,
                        "modalities": self.modalities,
                        "adc_path": str(adc_path) if adc_path.exists() else None,
                        "gtv_path": str(gtv_path) if gtv_path.exists() else None,
                        "tumor_size_cm3": entry["tumor_size_cm3"][tp],
                        "rano_t0_to_t1": entry["rano"]["rano_t0_to_t1"],
                        "rano_t0_to_t2": entry["rano"]["rano_t0_to_t2"],
                        "rano_t1_to_t2": entry["rano"]["rano_t1_to_t2"],
                        "projection_2d": self.projection_2d,
                    },
                }
            )

        return records

    # ------------------------------------------------------------------
    # Convenience: patient-level RANO outcome
    # ------------------------------------------------------------------

    def final_outcome(self, patient_id: str) -> str | None:
        """Return the latest available RANO transition rating (t1->t2, falling
        back to t0->t1) for a patient, or None if neither is available."""
        entry = self._manifest.get(patient_id)
        if entry is None:
            return None
        return entry["rano"]["rano_t1_to_t2"] or entry["rano"]["rano_t0_to_t1"]

    def all_final_outcomes(self) -> dict[str, str | None]:
        """Return {patient_id: final_outcome} for all included patients."""
        return {pid: self.final_outcome(pid) for pid in self._patients}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_image(
        self,
        patient_num: int,
        timepoint: str,
        gtv_mask: np.ndarray | None,
    ) -> np.ndarray:
        """Load, optionally normalise/project, the requested modality channels."""
        import SimpleITK as sitk  # noqa: N813

        channels: list[np.ndarray] = []
        for mod in self.modalities:
            path = _scan_path(self.imaging_root, patient_num, timepoint, mod)
            if not path.exists():
                raise FileNotFoundError(f"Missing {mod} file: {path}")
            vol = _read_nifti(path, sitk)
            if self.normalize:
                vol = _zscore_normalize(vol)
            channels.append(vol)

        volume = channels[0] if len(channels) == 1 else np.stack(channels, axis=0)
        # [D, H, W] or [C, D, H, W]

        if self.projection_2d:
            mask_for_proj = gtv_mask if self.slice_selection == "gtv_mask" else None
            volume = select_best_axial_slice(volume, mask_for_proj)

        return volume


# ---------------------------------------------------------------------------
# Manifest building (pure — touches only the TSVs, never imaging files)
# ---------------------------------------------------------------------------

def build_manifest(tsv_dir: str | Path) -> dict[str, dict]:
    """
    Parse the CFB-GBM TSVs into a per-patient availability + RANO manifest.

    Never touches imaging files — safe to call before downloading/mounting
    any NIfTI data, to determine the usable-N up front.

    Returns
    -------
    {patient_dir (zero-padded 3-digit str): {
        "patient_num": int,
        "mri": {tp: {"time_diff_weeks": float | None, <modality>: bool, ...}},
        "gtv": {tp: bool},
        "rano": {"rano_t0_to_t1": str | None, "rano_t0_to_t2": str | None,
                  "rano_t1_to_t2": str | None},
        "tumor_size_cm3": {tp: float | None},
    }}
    """
    tsv_dir = Path(tsv_dir)
    mri_rows = _read_tsv(_find_tsv(tsv_dir, "mri_availability"))
    tia_rows = _read_tsv(_find_tsv(tsv_dir, "treatment_imaging_availability"))
    rano_rows = _read_tsv(_find_tsv(tsv_dir, "rano_criteria"))

    manifest: dict[str, dict] = {}

    def _entry(patient_num: int) -> dict:
        pid = f"{patient_num:03d}"
        return manifest.setdefault(
            pid,
            {
                "patient_num": patient_num,
                "mri": {
                    tp: {"time_diff_weeks": None, **{m: False for m in MRI_MODALITIES}}
                    for tp in TIMEPOINTS
                },
                "gtv": {tp: False for tp in TIMEPOINTS},
                "rano": {
                    "rano_t0_to_t1": None,
                    "rano_t0_to_t2": None,
                    "rano_t1_to_t2": None,
                },
                "tumor_size_cm3": {tp: None for tp in TIMEPOINTS},
            },
        )

    for row in mri_rows:
        patient_num = int(row["id_patient"].strip())
        tp = row["temporality"].strip()
        if tp not in TIMEPOINTS:
            continue
        entry = _entry(patient_num)
        td = row.get("time_diff_t0 (weeks)", "").strip()
        entry["mri"][tp]["time_diff_weeks"] = float(td) if td else None
        for m in MRI_MODALITIES:
            entry["mri"][tp][m] = row.get(m, "").strip() == "1"

    for row in tia_rows:
        patient_num = int(row["id_patient"].strip())
        tp = row["temporality"].strip()
        if tp not in TIMEPOINTS:
            continue
        entry = _entry(patient_num)
        entry["gtv"][tp] = row.get("gtv", "").strip() == "1"

    for row in rano_rows:
        patient_num = int(row["id_patient"].strip())
        entry = _entry(patient_num)
        for key in ("rano_t0_to_t1", "rano_t0_to_t2", "rano_t1_to_t2"):
            entry["rano"][key] = _clean_rano(row.get(key))
        entry["tumor_size_cm3"]["t0"] = _parse_float(row.get("size_t0 (cm3)"))
        entry["tumor_size_cm3"]["t1"] = _parse_float(row.get("size_t1 (cm3)"))
        entry["tumor_size_cm3"]["t2"] = _parse_float(row.get("size_t2 (cm3)"))

    return manifest


def usable_patients(
    manifest: dict[str, dict],
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    require_adc: bool = True,
    require_gtv: bool = True,
    require_rano: bool = True,
) -> list[str]:
    """
    Filter a manifest (from `build_manifest`) to patients with `modalities`
    (plus ADC if `require_adc`) and the GTV mask present at all 3
    timepoints, and both RANO transitions rated (if `require_rano`).

    Pure function, no I/O — operates purely on the in-memory manifest dict.
    """
    needed = set(modalities)
    if require_adc:
        needed.add("adc")

    usable: list[str] = []
    for pid, entry in manifest.items():
        if not all(
            entry["mri"][tp].get(m, False) for tp in TIMEPOINTS for m in needed
        ):
            continue
        if require_gtv and not all(entry["gtv"][tp] for tp in TIMEPOINTS):
            continue
        if require_rano and not (
            entry["rano"]["rano_t0_to_t1"] and entry["rano"]["rano_t1_to_t2"]
        ):
            continue
        usable.append(pid)
    return sorted(usable)


def usable_scans(
    manifest: dict[str, dict],
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    require_adc: bool = True,
    require_gtv: bool = True,
) -> list[tuple[str, str]]:
    """
    Scan-level counterpart to `usable_patients()`: every individual
    (patient_dir, timepoint) pair with `modalities` (plus ADC if
    `require_adc`) and the GTV mask present *at that timepoint* -- no
    all-3-timepoints or RANO constraint, so a patient can contribute 1, 2,
    or 3 qualifying scans independently.

    This is the right query for population-level analyses that don't need
    within-patient trajectory pairing (e.g. Task 3's Geary's C / Moran's I
    coherence test): N is every valid scan rather than every
    fully-longitudinal patient, which is a much larger usable N given
    CFB-GBM's partial ADC/GTV coverage (~334 scans across ~205 patients
    for T1Gd+ADC+GTV vs. 72 scans across 24 patients the trajectory-paired
    query gives, as of the TCIA v4 release).

    Pure function, no I/O — operates purely on the in-memory manifest dict.
    """
    needed = set(modalities)
    if require_adc:
        needed.add("adc")

    scans: list[tuple[str, str]] = []
    for pid, entry in manifest.items():
        for tp in TIMEPOINTS:
            if not all(entry["mri"][tp].get(m, False) for m in needed):
                continue
            if require_gtv and not entry["gtv"][tp]:
                continue
            scans.append((pid, tp))
    return sorted(scans)


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------

def _scan_path(imaging_root: Path, patient_num: int, timepoint: str, modality: str) -> Path:
    """Build the on-disk path for one (patient, timepoint, modality) NIfTI file."""
    return (
        imaging_root
        / f"{patient_num:03d}"
        / timepoint
        / f"{patient_num}_{timepoint}_{modality}.nii.gz"
    )


def _find_tsv(tsv_dir: Path, keyword: str) -> Path:
    """Find the (versioned) TSV file whose name contains *keyword*."""
    matches = sorted(tsv_dir.glob(f"*{keyword}*.tsv"))
    if not matches:
        raise FileNotFoundError(f"No '*{keyword}*.tsv' file found in {tsv_dir}")
    return matches[-1]  # lexicographically latest version suffix


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _parse_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    raw = raw.strip()
    return float(raw) if raw else None


def _clean_rano(raw: str | None) -> str | None:
    """
    Extract the RANO code from a verbose label, e.g.
    'Stable Disease (SD)' -> 'SD'. Returns None for missing/blank input,
    or the trimmed raw string unchanged if no '(XX)' suffix is found.
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    m = _RANO_CODE_RE.search(raw)
    return m.group(1).upper() if m else raw


def _read_nifti(path: Path, sitk) -> np.ndarray:
    """Read a NIfTI file, reorient to RAS, and return a float32 [D, H, W] array."""
    img = sitk.ReadImage(str(path))
    img = sitk.DICOMOrient(img, _CANONICAL_ORIENTATION)
    arr = sitk.GetArrayFromImage(img)   # [D, H, W] in canonical RAS layout
    return arr.astype(np.float32)


def _zscore_normalize(vol: np.ndarray, clip_sigma: float = 5.0) -> np.ndarray:
    """
    Z-score normalise a single-channel volume, excluding background (zero)
    voxels, then clip to [-clip_sigma, clip_sigma] and rescale to [0, 1].
    """
    mask = vol != 0.0
    if mask.sum() < 10:
        lo, hi = vol.min(), vol.max()
        return ((vol - lo) / (hi - lo + 1e-8)).astype(np.float32)

    mu = vol[mask].mean()
    std = vol[mask].std()
    if std < 1e-8:
        return np.zeros_like(vol, dtype=np.float32)

    vol = (vol - mu) / std
    vol = np.clip(vol, -clip_sigma, clip_sigma)
    vol = (vol + clip_sigma) / (2.0 * clip_sigma)   # -> [0, 1]
    return vol.astype(np.float32)


def _load_mask(path: Path) -> np.ndarray | None:
    """Load a binary/int segmentation mask (e.g. GTV) as int16 [D, H, W], or None."""
    import SimpleITK as sitk  # noqa: N813

    if not path.exists():
        return None
    try:
        img = sitk.ReadImage(str(path))
        img = sitk.DICOMOrient(img, _CANONICAL_ORIENTATION)
        arr = sitk.GetArrayFromImage(img)
        return arr.astype(np.int16)
    except Exception as exc:
        warnings.warn(f"Could not load mask {path}: {exc}")
        return None
