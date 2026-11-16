"""
Cohort sex and age distribution (SAGER reporting requirement).

Reports gender (as recorded in CFB-GBM's clinical TSV: F / M) and age at t0
for (a) the full CFB-GBM collection and (b) the 204-patient analysis cohort
(patients with at least one scan with a valid TCC in results/tcc_table.csv),
plus the per-scan sex split of the 332-scan analysis set.

Writes: results/validation/cohort_demographics.csv
Usage:  python scripts/cohort_demographics.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TSV_DIR = Path(os.environ["CFB_GBM_ROOT"])
CLINICAL_TSV = TSV_DIR / "CFB-GBM_clinical_data_v03_20260619.tsv"
TCC_TABLE_PATH = Path("results/tcc_table.csv")
OUT_PATH = Path("results/validation/cohort_demographics.csv")


def summarize(df: pd.DataFrame, name: str) -> dict:
    age = df["age_at_t0 (years)"]
    sex = df["gender"]
    return {
        "cohort": name,
        "n_patients": len(df),
        "n_female": int((sex == "F").sum()),
        "n_male": int((sex == "M").sum()),
        "n_sex_missing": int(sex.isna().sum()),
        "pct_female": 100 * (sex == "F").mean(),
        "age_mean": age.mean(),
        "age_sd": age.std(),
        "age_median": age.median(),
        "age_q1": age.quantile(0.25),
        "age_q3": age.quantile(0.75),
        "age_min": age.min(),
        "age_max": age.max(),
        "n_age_missing": int(age.isna().sum()),
        "age_mean_female": age[sex == "F"].mean(),
        "age_mean_male": age[sex == "M"].mean(),
    }


def main() -> None:
    clin = pd.read_csv(CLINICAL_TSV, sep="\t")
    clin["patient_id"] = clin["id_patient"].astype(int).map(lambda i: f"{i:03d}")

    tcc = pd.read_csv(TCC_TABLE_PATH, dtype={"patient_id": str})
    tcc = tcc[tcc["tcc"].notna() & (tcc["tcc"] > 0)]
    cohort_ids = set(tcc["patient_id"])

    analysis = clin[clin["patient_id"].isin(cohort_ids)]
    missing = cohort_ids - set(analysis["patient_id"])
    if missing:
        print(f"WARNING: {len(missing)} cohort patients missing from clinical TSV: {sorted(missing)}")

    rows = [summarize(clin, "full_collection"), summarize(analysis, "analysis_cohort")]
    scans = tcc.merge(clin[["patient_id", "gender"]], on="patient_id", how="left")
    rows[1]["n_scans"] = len(scans)
    rows[1]["n_scans_female"] = int((scans["gender"] == "F").sum())
    rows[1]["n_scans_male"] = int((scans["gender"] == "M").sum())

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
