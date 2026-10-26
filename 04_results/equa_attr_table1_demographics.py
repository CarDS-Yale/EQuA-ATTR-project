


#!/usr/bin/env python3
"""
Table 1: demographics + comorbidities for your final cohort (or any patient list).

Takes the output of equa_attr_final_cohort_classification.py, filters to whichever
classification tier you specify (default: CONFIRMED_POSITIVE), and summarizes:
  - age (from Patients.txt birth date, relative to a reference date), sex, race,
    ethnicity
  - comorbidities via ICD-10 prefix matching across the DX files already used by
    the cohort script: hypertension, diabetes, CHF, stroke/TIA, vascular disease,
    chronic kidney disease

Column names are auto-detected and PRINTED so you can verify them against your
real files rather than trust blind assumptions — this has not been run against
your actual Patients.txt yet, only built from the FILES dict structure already
used elsewhere in this pipeline.

Usage:
  python3 equa_attr_table1_demographics.py \
      --classification-file attr_cm_final_classification.csv \
      --tier CONFIRMED_POSITIVE
"""

import os
import argparse
import re
import pandas as pd

RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
PATIENTS_FILE = "CarDS_2837199_Patients.txt"
DX_FILES = ["CarDS_2837199_Hosp_Enc_DX.txt", "CarDS_2837199_Outpatient_Enc_DX.txt"]

COMORBIDITY_ICD_PATTERNS = {
    "Hypertension": re.compile(r"^I1[0-5]"),
    "Diabetes mellitus": re.compile(r"^E0[89]|^E1[0-3]"),
    "Congestive heart failure": re.compile(r"^I50"),
    "Stroke/TIA/thromboembolism": re.compile(r"^I6[0-9]|^G45|^I74|^I82"),
    "Vascular disease": re.compile(r"^I7[0-9]"),
    "Chronic kidney disease": re.compile(r"^N18"),
}


def find_col(columns, *name_options):
    for name in name_options:
        for c in columns:
            if c.upper() == name.upper():
                return c
    for name in name_options:
        for c in columns:
            if name.upper() in c.upper():
                return c
    return None


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def load_demographics(pat_mrn_ids):
    path = os.path.join(RAW_DIR, PATIENTS_FILE)
    df = pd.read_csv(path, sep=sniff_delimiter(path), dtype=str, on_bad_lines="skip", engine="python", quoting=3)
    print(f"Patients.txt columns found: {list(df.columns)}")

    id_col = find_col(df.columns, "PAT_MRN_ID")
    dob_col = find_col(df.columns, "BIRTH_DATE", "DOB", "DATE_OF_BIRTH")
    sex_col = find_col(df.columns, "SEX", "GENDER")
    race_col = find_col(df.columns, "RACE")
    eth_col = find_col(df.columns, "ETHNICITY", "ETHNIC_GROUP")
    death_col = find_col(df.columns, "DEATH_DATE")

    print(f"  Detected: PAT_MRN_ID={id_col}, DOB={dob_col}, SEX={sex_col}, RACE={race_col}, "
          f"ETHNICITY={eth_col}, DEATH_DATE={death_col}")
    if not id_col:
        raise SystemExit("[ERROR] Could not find a PAT_MRN_ID column in Patients.txt — check manually.")

    df = df[df[id_col].isin(pat_mrn_ids)].copy()

    if dob_col:
        dob = pd.to_datetime(df[dob_col], errors="coerce")
        ref_date = pd.Timestamp("2026-07-23")  # data extract date, per the project's data folder name
        df["_age"] = (ref_date - dob).dt.days / 365.25

    return df, id_col, sex_col, race_col, eth_col


def scan_comorbidities(pat_ids):
    hits = {name: set() for name in COMORBIDITY_ICD_PATTERNS}
    for fname in DX_FILES:
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] Missing file: {path}")
            continue
        print(f"  Scanning {fname} for comorbidity codes...")
        delim = sniff_delimiter(path)
        for chunk in pd.read_csv(path, sep=delim, dtype=str, chunksize=500_000,
                                  on_bad_lines="skip", engine="python", quoting=3):
            id_col = find_col(chunk.columns, "PAT_ID")
            icd_col = find_col(chunk.columns, "CURRENT_ICD10_LIST", "ICD10_LIST", "DX_CODE")
            if not id_col or not icd_col:
                continue
            sub = chunk[chunk[id_col].isin(pat_ids)]
            for _, row in sub.iterrows():
                codes = str(row.get(icd_col, ""))
                pat = row.get(id_col, "")
                for name, pattern in COMORBIDITY_ICD_PATTERNS.items():
                    for code in re.split(r"[;,]\s*", codes):
                        if pattern.match(code.strip()):
                            hits[name].add(pat)
                            break
    return hits


def main():
    ap = argparse.ArgumentParser(description="Table 1 demographics + comorbidities")
    ap.add_argument("--classification-file", required=True)
    ap.add_argument("--tier", default="CONFIRMED_POSITIVE",
                     help="Which final_classification value to summarize (default: CONFIRMED_POSITIVE). "
                          "Use 'ALL' to summarize the entire file regardless of tier.")
    args = ap.parse_args()

    df = pd.read_csv(args.classification_file, dtype=str)
    if args.tier != "ALL":
        df = df[df["final_classification"] == args.tier]
    pat_mrn_col = next((c for c in df.columns if c.upper() == "PAT_MRN_ID"), None)
    if not pat_mrn_col:
        raise SystemExit("[ERROR] No PAT_MRN_ID column found in the classification file.")
    # Notes-only patients (found via NLP with zero structured signal) have PAT_ID
    # as an empty string, not NaN, so .dropna() alone would not remove them from
    # PAT_ID-based matching — but they DO have a real PAT_MRN_ID, which is why
    # this switched to using PAT_MRN_ID throughout instead.
    pat_mrn_ids = set(df[pat_mrn_col].dropna()) - {""}
    print(f"Summarizing {len(pat_mrn_ids)} patients (tier: {args.tier})\n")

    demo_df, id_col, sex_col, race_col, eth_col = load_demographics(pat_mrn_ids)
    n = len(demo_df)

    print(f"\n{'='*60}\nTABLE 1 — DEMOGRAPHICS (n={n})\n{'='*60}")
    if "_age" in demo_df.columns:
        print(f"Mean age: {demo_df['_age'].mean():.1f} (SD ± {demo_df['_age'].std():.1f})")
    if sex_col:
        print(f"\nSex:\n{demo_df[sex_col].value_counts().to_string()}")
    if race_col:
        print(f"\nRace:\n{demo_df[race_col].value_counts().to_string()}")
    if eth_col:
        print(f"\nEthnicity:\n{demo_df[eth_col].value_counts().to_string()}")

    print(f"\n{'='*60}\nCOMORBIDITIES\n{'='*60}")
    # Use PAT_ID from demo_df, not the original classification file — demo_df's
    # join via PAT_MRN_ID against Patients.txt recovers the real PAT_ID even for
    # notes-only patients who never had one in the classification file itself.
    if id_col_pat := next((c for c in demo_df.columns if c.upper() == "PAT_ID"), None):
        comorbidity_pat_ids = set(demo_df[id_col_pat].dropna()) - {""}
    else:
        print("[WARNING] No PAT_ID column found in Patients.txt — cannot scan comorbidities.")
        comorbidity_pat_ids = set()
    hits = scan_comorbidities(comorbidity_pat_ids)
    for name, pat_set in hits.items():
        count = len(pat_set)
        pct = 100 * count / n if n else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    out_path = "table1_summary.txt"
    with open(out_path, "w") as f:
        f.write(f"n={n} (tier: {args.tier})\n")
        if "_age" in demo_df.columns:
            f.write(f"Mean age: {demo_df['_age'].mean():.1f} (SD ± {demo_df['_age'].std():.1f})\n")
        for name, pat_set in hits.items():
            count = len(pat_set)
            pct = 100 * count / n if n else 0
            f.write(f"{name}: {count} ({pct:.1f}%)\n")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()