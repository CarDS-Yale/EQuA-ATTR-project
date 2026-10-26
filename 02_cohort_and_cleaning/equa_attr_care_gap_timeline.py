






#!/usr/bin/env python3
"""
ATTR-CM Care Gap Timeline Script
===================================
Identifies the SPECIFIC care-gap pattern: a patient had a real ATTR-CM signal
(diagnosis code, medication, PYP order, or genetic test order), continued to be
SEEN by cardiology afterward, but the workup was NEVER RESOLVED (no genetically
confirmed TTR status, and no definitively-read PYP scan result).

This is a diagnostic-PROCESS metric (was the workup completed), not a health
outcome — squarely diagnostic-completeness territory, not overlapping with
treatment-response/outcomes work being done elsewhere on this project.

Requires the outputs of THREE other scripts to already exist:
  1. equa_attr_identify_cohort.py   -> the candidate cohort CSV (PAT_ID list)
  2. equa_attr_summarize_results.py -> the genetics patient-level rollup CSV
  3. equa_attr_extract_pyp_results.py -> the PYP scan results CSV

Usage:
  python3 equa_attr_care_gap_timeline.py \\
      --cohort-file attr_cm_candidate_cohort.csv \\
      --genetics-rollup attr_genetics_patient_rollup.csv \\
      --pyp-results attr_pyp_results_latest.csv
"""

import os
import argparse
import pandas as pd

RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
OUTPUT_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results")
CHUNK_SIZE = 200_000

# Files + the date column that best represents when that signal occurred
DATE_SOURCES = [
    ("CarDS_2837199_Hosp_Enc_DX.txt", "CALC_DX_DATE"),
    ("CarDS_2837199_Outpatient_Enc_DX.txt", "CALC_DX_DATE"),
    ("CarDS_2837199_Medical_Hx.txt", "MED_HX_START_DT"),
    ("CarDS_2837199_Meds.txt", "ORDER_INST"),
    ("CarDS_2837199_External_Meds.txt", "WRITTEN_DATE"),
    ("CarDS_2837199_Hosp_Enc_Med_Admin.txt", "ORDER_INST"),
    ("CarDS_2837199_Outpatient_Enc_Med_Admin.txt", "ORDER_INST"),
    ("CarDS_2837199_Genetic_Testing.txt", "CONTACT_DATE"),
    ("CarDS_2837199_Imaging_1.txt", "ORDERING_DTTM"),
    ("CarDS_2837199_Imaging_2.txt", "ORDERING_DTTM"),
    ("CarDS_2837199_Hosp_Enc_ICD_PX.txt", "PROC_DATE"),
    ("CarDS_2837199_Outpatient_Enc_ICD_PX.txt", "PROC_DATE"),
]


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def iter_chunks(file_path):
    delim = sniff_delimiter(file_path)
    reader = pd.read_csv(
        file_path, sep=delim, dtype=str, on_bad_lines="skip",
        engine="python", quoting=3, chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        yield chunk


def find_col(columns, name):
    for c in columns:
        if c.upper() == name.upper():
            return c
    return None


def get_unique_path(path):
    """If path already exists, append _2, _3, etc. before the extension until an
    unused filename is found — so re-running this script (e.g. once per genetics
    rollup, once for Ollama-covered patients and once for vLLM-covered patients)
    never silently overwrites a previous run's output."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def compute_first_signal_dates(cohort_pat_ids):
    """For each cohort patient, find the EARLIEST date across all their signal-
    bearing records — their 'first signal date', the starting point for measuring
    whether care continued afterward without ever resolving."""
    earliest = {}

    def update(pat_id, date_str):
        if not date_str or str(date_str).lower() == "nan":
            return
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return
        if pat_id not in earliest or dt < earliest[pat_id]:
            earliest[pat_id] = dt

    for file_name, date_col_name in DATE_SOURCES:
        file_path = os.path.join(RAW_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning dates from {file_name} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            date_col = find_col(chunk.columns, date_col_name)
            if id_col is None or date_col is None:
                continue
            chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
            for pat_id, date_val in zip(chunk[id_col], chunk[date_col]):
                update(pat_id, date_val)

    return earliest


def count_cardiology_visits_after(cohort_pat_ids, first_signal_dates):
    """Counts OUTPATIENT cardiology visits per patient that occurred AFTER their
    own first signal date — this is the 'still being seen but never resolved'
    evidence. See count_inpatient_cardiology_after() for the inpatient equivalent
    — a patient can have real cardiology follow-up captured ONLY as an inpatient
    admission (e.g. a cardiology consult during a hospital stay), which this
    function alone would miss entirely."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Outpatient_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Scanning post-signal OUTPATIENT cardiology visits from CarDS_2837199_Outpatient_Enc.txt ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        dept_col = find_col(chunk.columns, "DEPARTMENT_NAME")
        date_col = find_col(chunk.columns, "CONTACT_DATE")
        if id_col is None or dept_col is None or date_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, dept, date_val in zip(chunk[id_col], chunk[dept_col], chunk[date_col]):
            if not isinstance(dept, str) or "cardiology" not in dept.lower():
                continue
            if pat_id not in first_signal_dates:
                continue
            visit_dt = pd.to_datetime(date_val, errors="coerce")
            if pd.isna(visit_dt):
                continue
            if visit_dt > first_signal_dates[pat_id]:
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def count_inpatient_cardiology_after(cohort_pat_ids, first_signal_dates):
    """Counts INPATIENT admissions per patient, after their first signal date,
    where cardiology appears in any of the department fields or the primary
    service field — real follow-up (e.g. a cardiology consult during a hospital
    stay) that the outpatient-only check above would silently miss. This gap was
    found only after the original run had already completed (~2.5 hours) — see
    findings log Section 28 — so this function did not exist during that run."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Hosp_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Scanning post-signal INPATIENT cardiology involvement from CarDS_2837199_Hosp_Enc.txt ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        date_col = find_col(chunk.columns, "HOSP_ADMSN_DATE")
        dept_cols = [c for c in ("FIRST_DEPARTMENT_NAME", "SECOND_DEPARTMENT_NAME", "LAST_DEPARTMENT_NAME")
                     if find_col(chunk.columns, c)]
        svc_col = find_col(chunk.columns, "PRIM_SVC_HA")
        if id_col is None or date_col is None or (not dept_cols and svc_col is None):
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for _, row in chunk.iterrows():
            pat_id = row[id_col]
            if pat_id not in first_signal_dates:
                continue
            text_fields = [str(row.get(c, "")) for c in dept_cols]
            if svc_col:
                text_fields.append(str(row.get(svc_col, "")))
            if not any("cardiology" in t.lower() for t in text_fields):
                continue
            visit_dt = pd.to_datetime(row.get(date_col), errors="coerce")
            if pd.isna(visit_dt):
                continue
            if visit_dt > first_signal_dates[pat_id]:
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Compute ATTR-CM care-gap timeline")
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--genetics-rollup", required=True)
    parser.add_argument("--pyp-results", required=True)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("ATTR-CM CARE GAP TIMELINE")
    print("=" * 70)

    cohort_df = pd.read_csv(args.cohort_file, dtype=str)
    cohort_pat_ids = set(cohort_df["PAT_ID"])
    print(f"\nLoaded {len(cohort_pat_ids)} candidate patients.")

    print("\n[1/4] Computing first-signal date per patient...")
    first_signal_dates = compute_first_signal_dates(cohort_pat_ids)
    print(f"  -> First-signal date found for {len(first_signal_dates)} patients.")

    print("\n[2/4] Counting cardiology visits AFTER first-signal date (outpatient + inpatient)...")
    outpatient_visits = count_cardiology_visits_after(cohort_pat_ids, first_signal_dates)
    print(f"  -> {len(outpatient_visits)} patients had at least one post-signal OUTPATIENT cardiology visit.")
    inpatient_visits = count_inpatient_cardiology_after(cohort_pat_ids, first_signal_dates)
    print(f"  -> {len(inpatient_visits)} patients had at least one post-signal INPATIENT cardiology admission.")
    post_signal_visits = {}
    for pat_id in set(outpatient_visits) | set(inpatient_visits):
        post_signal_visits[pat_id] = outpatient_visits.get(pat_id, 0) + inpatient_visits.get(pat_id, 0)
    inpatient_only = set(inpatient_visits) - set(outpatient_visits)
    print(f"  -> {len(post_signal_visits)} patients total had at least one post-signal cardiology visit "
          f"(outpatient and/or inpatient). {len(inpatient_only)} of these were found ONLY via the "
          f"inpatient check — invisible to the original outpatient-only version of this script.")

    print("\n[3/4] Determining resolution status (genetics or PYP)...")
    genetics_df = pd.read_csv(args.genetics_rollup, dtype=str)
    gen_mrn_col = find_col(genetics_df.columns, "PAT_MRN_ID")
    status = genetics_df["overall_genetic_status"].astype(str)
    is_positive_status = status.str.startswith(("variant", "wild_type"))
    # A patient only counts as RESOLVED via genetics if the result is genuinely a TTR
    # genetic result. Two things must NOT count as resolution:
    #   - wrong-gene conflation (TTN/BRCA1/etc. misattributed to TTR) — when the rollup
    #     has the ttr_language_confirmed column, require it to be True
    #   - "clinically presumed, NOT genetically confirmed" — a clinical label with no
    #     TTR sequencing is exactly the gap this project measures, not a resolution
    resolved_mask = is_positive_status & ~status.str.contains("clinically presumed", na=False)
    if "ttr_language_confirmed" in genetics_df.columns:
        ttr_ok = genetics_df["ttr_language_confirmed"].astype(str).str.lower() == "true"
        resolved_mask = resolved_mask & ttr_ok
    else:
        print("  [WARNING] rollup has no ttr_language_confirmed column (older summarize output) — "
              "wrong-gene conflation cannot be excluded from 'resolved'. Re-run the summarize "
              "script and use its newest rollup file.")
    # Note: "basis unknown" rows (old-script output, no type_basis) with confirmed TTR
    # language ARE counted as resolved — their basis can't be told apart, so this may
    # slightly over-count resolution for that older portion; state it as a limitation.
    resolved_genetics_mrns = set(genetics_df.loc[resolved_mask, gen_mrn_col])

    pyp_df = pd.read_csv(args.pyp_results, dtype=str)
    resolved_pyp_mrns = set(
        pyp_df.loc[
            pyp_df["pyp_result"].isin(["positive", "negative", "equivocal_borderline"]),
            "PAT_MRN_ID"
        ]
    )
    print(f"  Genetics-resolved patients: {len(resolved_genetics_mrns)}")
    print(f"  PYP-resolved patients: {len(resolved_pyp_mrns)}")

    # ── Assemble final per-patient result ──
    rows = []
    for _, r in cohort_df.iterrows():
        pat_id = r["PAT_ID"]
        mrn = r["PAT_MRN_ID"]
        fsd = first_signal_dates.get(pat_id)
        visits_after = post_signal_visits.get(pat_id, 0)
        resolved = (mrn in resolved_genetics_mrns) or (mrn in resolved_pyp_mrns)

        care_gap = (fsd is not None) and (visits_after > 0) and (not resolved)

        rows.append({
            "PAT_ID": pat_id,
            "PAT_MRN_ID": mrn,
            "first_signal_date": fsd.strftime("%Y-%m-%d") if fsd is not None else "",
            "cardiology_visits_after_first_signal": visits_after,
            "resolved_via_genetics": mrn in resolved_genetics_mrns,
            "resolved_via_pyp": mrn in resolved_pyp_mrns,
            "CARE_GAP_seen_but_unresolved": care_gap,
        })

    result_df = pd.DataFrame(rows)
    out_path = get_unique_path(os.path.join(args.output_dir, "attr_care_gap_timeline.csv"))
    result_df.to_csv(out_path, index=False)

    n_gap = result_df["CARE_GAP_seen_but_unresolved"].sum()
    n_resolved = result_df[["resolved_via_genetics", "resolved_via_pyp"]].any(axis=1).sum()
    n_no_fsd = result_df["first_signal_date"].eq("").sum()

    print(f"\n{'='*70}")
    print("CARE GAP TIMELINE COMPLETE")
    print(f"Total candidate patients: {len(result_df)}")
    print(f"  No first-signal date found (couldn't compute timeline): {n_no_fsd}")
    print(f"  Resolved (genetics OR PYP): {n_resolved}")
    print(f"\n  >>> CARE GAP — seen by cardiology after first signal, STILL UNRESOLVED: {n_gap} <<<")
    print(f"  (This is patients who had care continue, but the ATTR-CM workup never completed —")
    print(f"   likely your headline diagnostic-completeness finding.)")
    print(f"\nSaved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()










'''


#!/usr/bin/env python3
"""
ATTR-CM Care Gap Timeline Script
===================================
Identifies the SPECIFIC care-gap pattern: a patient had a real ATTR-CM signal
(diagnosis code, medication, PYP order, or genetic test order), continued to be
SEEN by cardiology afterward, but the workup was NEVER RESOLVED (no genetically
confirmed TTR status, and no definitively-read PYP scan result).

This is a diagnostic-PROCESS metric (was the workup completed), not a health
outcome — squarely diagnostic-completeness territory, not overlapping with
treatment-response/outcomes work being done elsewhere on this project.

Requires the outputs of THREE other scripts to already exist:
  1. equa_attr_identify_cohort.py   -> the candidate cohort CSV (PAT_ID list)
  2. equa_attr_summarize_results.py -> the genetics patient-level rollup CSV
  3. equa_attr_extract_pyp_results.py -> the PYP scan results CSV

Usage:
  python3 equa_attr_care_gap_timeline.py \\
      --cohort-file attr_cm_candidate_cohort.csv \\
      --genetics-rollup attr_genetics_patient_rollup.csv \\
      --pyp-results attr_pyp_results_latest.csv
"""

import os
import argparse
import pandas as pd

RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
OUTPUT_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results")
CHUNK_SIZE = 200_000

# Files + the date column that best represents when that signal occurred
DATE_SOURCES = [
    ("CarDS_2837199_Hosp_Enc_DX.txt", "CALC_DX_DATE"),
    ("CarDS_2837199_Outpatient_Enc_DX.txt", "CALC_DX_DATE"),
    ("CarDS_2837199_Medical_Hx.txt", "MED_HX_START_DT"),
    ("CarDS_2837199_Meds.txt", "ORDER_INST"),
    ("CarDS_2837199_External_Meds.txt", "WRITTEN_DATE"),
    ("CarDS_2837199_Hosp_Enc_Med_Admin.txt", "ORDER_INST"),
    ("CarDS_2837199_Outpatient_Enc_Med_Admin.txt", "ORDER_INST"),
    ("CarDS_2837199_Genetic_Testing.txt", "CONTACT_DATE"),
    ("CarDS_2837199_Imaging_1.txt", "ORDERING_DTTM"),
    ("CarDS_2837199_Imaging_2.txt", "ORDERING_DTTM"),
    ("CarDS_2837199_Hosp_Enc_ICD_PX.txt", "PROC_DATE"),
    ("CarDS_2837199_Outpatient_Enc_ICD_PX.txt", "PROC_DATE"),
]


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def iter_chunks(file_path):
    delim = sniff_delimiter(file_path)
    reader = pd.read_csv(
        file_path, sep=delim, dtype=str, on_bad_lines="skip",
        engine="python", quoting=3, chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        yield chunk


def find_col(columns, name):
    for c in columns:
        if c.upper() == name.upper():
            return c
    return None


def get_unique_path(path):
    """If path already exists, append _2, _3, etc. before the extension until an
    unused filename is found — so re-running this script (e.g. once per genetics
    rollup, once for Ollama-covered patients and once for vLLM-covered patients)
    never silently overwrites a previous run's output."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def compute_first_signal_dates(cohort_pat_ids):
    """For each cohort patient, find the EARLIEST date across all their signal-
    bearing records — their 'first signal date', the starting point for measuring
    whether care continued afterward without ever resolving."""
    earliest = {}

    def update(pat_id, date_str):
        if not date_str or str(date_str).lower() == "nan":
            return
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return
        if pat_id not in earliest or dt < earliest[pat_id]:
            earliest[pat_id] = dt

    for file_name, date_col_name in DATE_SOURCES:
        file_path = os.path.join(RAW_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning dates from {file_name} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            date_col = find_col(chunk.columns, date_col_name)
            if id_col is None or date_col is None:
                continue
            chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
            for pat_id, date_val in zip(chunk[id_col], chunk[date_col]):
                update(pat_id, date_val)

    return earliest


def count_cardiology_visits_after(cohort_pat_ids, first_signal_dates):
    """Counts cardiology visits per patient that occurred AFTER their own first
    signal date — this is the 'still being seen but never resolved' evidence."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Outpatient_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Scanning post-signal cardiology visits from CarDS_2837199_Outpatient_Enc.txt ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        dept_col = find_col(chunk.columns, "DEPARTMENT_NAME")
        date_col = find_col(chunk.columns, "CONTACT_DATE")
        if id_col is None or dept_col is None or date_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, dept, date_val in zip(chunk[id_col], chunk[dept_col], chunk[date_col]):
            if not isinstance(dept, str) or "cardiology" not in dept.lower():
                continue
            if pat_id not in first_signal_dates:
                continue
            visit_dt = pd.to_datetime(date_val, errors="coerce")
            if pd.isna(visit_dt):
                continue
            if visit_dt > first_signal_dates[pat_id]:
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser(description="Compute ATTR-CM care-gap timeline")
    parser.add_argument("--cohort-file", required=True)
    parser.add_argument("--genetics-rollup", required=True)
    parser.add_argument("--pyp-results", required=True)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    print("=" * 70)
    print("ATTR-CM CARE GAP TIMELINE")
    print("=" * 70)

    cohort_df = pd.read_csv(args.cohort_file, dtype=str)
    cohort_pat_ids = set(cohort_df["PAT_ID"])
    print(f"\nLoaded {len(cohort_pat_ids)} candidate patients.")

    print("\n[1/3] Computing first-signal date per patient...")
    first_signal_dates = compute_first_signal_dates(cohort_pat_ids)
    print(f"  -> First-signal date found for {len(first_signal_dates)} patients.")

    print("\n[2/3] Counting cardiology visits AFTER first-signal date...")
    post_signal_visits = count_cardiology_visits_after(cohort_pat_ids, first_signal_dates)
    print(f"  -> {len(post_signal_visits)} patients had at least one post-signal cardiology visit.")

    print("\n[3/3] Determining resolution status (genetics or PYP)...")
    genetics_df = pd.read_csv(args.genetics_rollup, dtype=str)
    gen_mrn_col = find_col(genetics_df.columns, "PAT_MRN_ID")
    status = genetics_df["overall_genetic_status"].astype(str)
    is_positive_status = status.str.startswith(("variant", "wild_type"))
    # A patient only counts as RESOLVED via genetics if the result is genuinely a TTR
    # genetic result. Two things must NOT count as resolution:
    #   - wrong-gene conflation (TTN/BRCA1/etc. misattributed to TTR) — when the rollup
    #     has the ttr_language_confirmed column, require it to be True
    #   - "clinically presumed, NOT genetically confirmed" — a clinical label with no
    #     TTR sequencing is exactly the gap this project measures, not a resolution
    resolved_mask = is_positive_status & ~status.str.contains("clinically presumed", na=False)
    if "ttr_language_confirmed" in genetics_df.columns:
        ttr_ok = genetics_df["ttr_language_confirmed"].astype(str).str.lower() == "true"
        resolved_mask = resolved_mask & ttr_ok
    else:
        print("  [WARNING] rollup has no ttr_language_confirmed column (older summarize output) — "
              "wrong-gene conflation cannot be excluded from 'resolved'. Re-run the summarize "
              "script and use its newest rollup file.")
    # Note: "basis unknown" rows (old-script output, no type_basis) with confirmed TTR
    # language ARE counted as resolved — their basis can't be told apart, so this may
    # slightly over-count resolution for that older portion; state it as a limitation.
    resolved_genetics_mrns = set(genetics_df.loc[resolved_mask, gen_mrn_col])

    pyp_df = pd.read_csv(args.pyp_results, dtype=str)
    resolved_pyp_mrns = set(
        pyp_df.loc[
            pyp_df["pyp_result"].isin(["positive", "negative", "equivocal_borderline"]),
            "PAT_MRN_ID"
        ]
    )
    print(f"  Genetics-resolved patients: {len(resolved_genetics_mrns)}")
    print(f"  PYP-resolved patients: {len(resolved_pyp_mrns)}")

    # ── Assemble final per-patient result ──
    rows = []
    for _, r in cohort_df.iterrows():
        pat_id = r["PAT_ID"]
        mrn = r["PAT_MRN_ID"]
        fsd = first_signal_dates.get(pat_id)
        visits_after = post_signal_visits.get(pat_id, 0)
        resolved = (mrn in resolved_genetics_mrns) or (mrn in resolved_pyp_mrns)

        care_gap = (fsd is not None) and (visits_after > 0) and (not resolved)

        rows.append({
            "PAT_ID": pat_id,
            "PAT_MRN_ID": mrn,
            "first_signal_date": fsd.strftime("%Y-%m-%d") if fsd is not None else "",
            "cardiology_visits_after_first_signal": visits_after,
            "resolved_via_genetics": mrn in resolved_genetics_mrns,
            "resolved_via_pyp": mrn in resolved_pyp_mrns,
            "CARE_GAP_seen_but_unresolved": care_gap,
        })

    result_df = pd.DataFrame(rows)
    out_path = get_unique_path(os.path.join(args.output_dir, "attr_care_gap_timeline.csv"))
    result_df.to_csv(out_path, index=False)

    n_gap = result_df["CARE_GAP_seen_but_unresolved"].sum()
    n_resolved = result_df[["resolved_via_genetics", "resolved_via_pyp"]].any(axis=1).sum()
    n_no_fsd = result_df["first_signal_date"].eq("").sum()

    print(f"\n{'='*70}")
    print("CARE GAP TIMELINE COMPLETE")
    print(f"Total candidate patients: {len(result_df)}")
    print(f"  No first-signal date found (couldn't compute timeline): {n_no_fsd}")
    print(f"  Resolved (genetics OR PYP): {n_resolved}")
    print(f"\n  >>> CARE GAP — seen by cardiology after first signal, STILL UNRESOLVED: {n_gap} <<<")
    print(f"  (This is patients who had care continue, but the ATTR-CM workup never completed —")
    print(f"   likely your headline diagnostic-completeness finding.)")
    print(f"\nSaved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

'''