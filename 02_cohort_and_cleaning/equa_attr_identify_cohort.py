



#!/usr/bin/env python3
"""
ATTR-CM Cohort Identification Script
=====================================
Identifies candidate ATTR-CM patients from STRUCTURED data only (diagnosis codes,
medications, genetic testing orders, PYP/amyloid imaging orders) across an entire
health system's data extract. This is meant to run BEFORE the notes-based genetics
NLP script (equa_attr_extract_genetics.py) — the output cohort's PAT_IDs are what
you should filter the note files down to, rather than scanning every patient's notes
in the health system.

Signals used (each is independent; a patient needs to match only ONE to be included):
  1. Diagnosis codes  - E85.81 / E85.82 (ATTR-specific, 2023 ICD-10-CM) and the
                         broader E85.x amyloidosis family / legacy ICD-9 277.3x,
                         from Hosp_Enc_DX.txt + Outpatient_Enc_DX.txt
  2. Medications       - tafamidis, acoramidis, patisiran, vutrisiran, inotersen,
                         diflunisal (brand + generic names), from Meds.txt +
                         External_Meds.txt
  3. Genetic testing   - TTR / transthyretin genetic test orders, from
                         Genetic_Testing.txt
  4. PYP/amyloid imaging - pyrophosphate scan orders, from Imaging_1.txt + Imaging_2.txt
                         (matched on PROC_NAME text, not CPT code, since PYP CPT
                         coding varies by institution)

Output: one CSV with one row per candidate patient, a boolean flag per signal, and
the specific matched value(s) as evidence for each signal that fired, plus a summary
of source and overlap counts (a natural cohort-flow / attrition table for your
methods section).

IMPORTANT: The keyword/code lists below (DRUG_KEYWORDS, ICD10_ATTR_SPECIFIC,
ICD10_AMYLOID_FAMILY, IMAGING_KEYWORDS) are a reasonable starting point but you
should review them against your actual data before trusting the cohort — e.g. check
a sample of DX_NAME values that got matched, and check whether your institution's PYP
orders actually contain "pyrophosphate" in PROC_NAME (run the --inspect mode first).

Usage:
  python3 equa_attr_identify_cohort.py --inspect       # preview what would match, no output file
  python3 equa_attr_identify_cohort.py                 # full run, writes cohort CSV

================================================================================
TODO — DOCUMENTED FOLLOW-UPS, NOT YET IMPLEMENTED (see game plan doc for detail):
================================================================================
1. FAMILY HISTORY SIGNAL (Family_Hx.txt): confirmed real MEDICAL_HX text values
   exist in this file (hypertension, cancer, heart disease, etc. dominate by volume
   since amyloidosis is rare). Still needs a direct grep/search for
   amyloid|cardiomyopathy|transthyretin|TTR|sudden cardiac death before deciding
   whether to add as a signal — caveat: free-text family history usually can't
   distinguish wild-type (non-heritable) from hereditary ATTR in a relative, so a
   positive match should be treated as "should have prompted evaluation," not proof
   of hereditary risk.
2. PYP CPT CODE (Hosp_Enc_CPT.txt / Outpatient_Enc_CPT.txt): these files have ONLY
   PAT_ID/CPT_DATE/CPT_CODE — no description field, so the real institutional PYP
   CPT code is currently unknown and NOT guessed at (guessing wrong would silently
   return zero/wrong matches). To determine it safely: join these CPT files against
   the PAT_ENC_CSN_ID-equivalent from imaging_hits' matched encounters — NOTE the
   imaging files (Imaging_1.txt/Imaging_2.txt) do NOT have a column literally named
   PAT_ENC_CSN_ID; use PERFORMING_CSN_ID instead (the encounter the scan was
   actually performed during, which is what generates the CPT billing code). Once
   the real code is identified from your own data, add it as a supplementary
   imaging signal alongside the existing PROC_NAME text match.
================================================================================
"""

import os
import re
import argparse
import pandas as pd

# ── Directories (reuse the same raw data directory as the genetics script) ──
RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
OUTPUT_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results")
CHUNK_SIZE = 200_000  # rows per chunk when reading large structured files

FILES = {
    "hosp_dx": "CarDS_2837199_Hosp_Enc_DX.txt",
    "outpt_dx": "CarDS_2837199_Outpatient_Enc_DX.txt",
    "meds": "CarDS_2837199_Meds.txt",
    "external_meds": "CarDS_2837199_External_Meds.txt",
    "genetic_testing": "CarDS_2837199_Genetic_Testing.txt",
    "imaging_1": "CarDS_2837199_Imaging_1.txt",
    "imaging_2": "CarDS_2837199_Imaging_2.txt",
    "hosp_med_admin": "CarDS_2837199_Hosp_Enc_Med_Admin.txt",
    "outpt_med_admin": "CarDS_2837199_Outpatient_Enc_Med_Admin.txt",
    "hosp_icd_px": "CarDS_2837199_Hosp_Enc_ICD_PX.txt",
    "outpt_icd_px": "CarDS_2837199_Outpatient_Enc_ICD_PX.txt",
    "patients": "CarDS_2837199_Patients.txt",
    "hosp_labs_1": "CarDS_2837199_Hosp_Enc_Labs_1.txt",
    "hosp_labs_2": "CarDS_2837199_Hosp_Enc_Labs_2.txt",
    "hosp_labs_3": "CarDS_2837199_Hosp_Enc_Labs_3.txt",
    "hosp_labs_4": "CarDS_2837199_Hosp_Enc_Labs_4.txt",
    "hosp_labs_5": "CarDS_2837199_Hosp_Enc_Labs_5.txt",
    "hosp_labs_6": "CarDS_2837199_Hosp_Enc_Labs_6.txt",
    "outpt_labs_1": "CarDS_2837199_Outpatient_Enc_Labs_1.txt",
    "outpt_labs_2": "CarDS_2837199_Outpatient_Enc_Labs_2.txt",
    "outpt_labs_3": "CarDS_2837199_Outpatient_Enc_Labs_3.txt",
    "medical_hx": "CarDS_2837199_Medical_Hx.txt",
    "family_hx": "CarDS_2837199_Family_Hx.txt",
}

# ── Matching criteria — REVIEW THESE against your actual data (see --inspect) ──

# ATTR-specific ICD-10-CM codes (added ~2023): hereditary vs wild-type
# ATTR-specific ICD-10-CM codes, VERIFIED against the actual 2023+ ICD-10-CM tabular:
#   E85.82 = Wild-type transthyretin-related (ATTR) amyloidosis (ATTRwt) — unambiguous
#   E85.4  = Organ-limited amyloidosis — officially INCLUDES "Transthyretin-related
#            (ATTR) familial amyloid cardiomyopathy" as a named inclusion term, so
#            this is where hereditary ATTR-CM (ATTRv) actually gets coded, confirmed
#            by real DX_NAME values in this dataset ("Cardiac amyloidosis", "Amyloid
#            heart muscle disease" all appear coded under E85.4)
# CRITICAL: E85.81 is Light chain (AL) amyloidosis — a DIFFERENT DISEASE (plasma cell
# disorder, unrelated to TTR), and must NEVER be treated as ATTR-specific or even as
# generic "amyloid family" — it needs its own EXCLUDE flag so those patients can be
# identified and kept out of the ATTR-CM cohort, not silently folded in.
ICD10_ATTR_SPECIFIC = ["E85.4", "E85.82"]

# Codes for AL (light chain) amyloidosis — a DIFFERENT disease from ATTR. Patients
# matching ONLY this should be flagged and excluded/reviewed, not counted as ATTR.
ICD10_AL_AMYLOID_EXCLUDE = ["E85.81"]

# Broader amyloidosis family (non-AL, non-specific-ATTR) — catches ATTR-CM coded less
# specifically, plus legacy pre-2023 records. Note: "E85.8" as a bare prefix is
# deliberately AVOIDED here, since it would silently match E85.81 (AL) via substring
# prefix matching — E85.89 (Other amyloidosis, not AL) is used explicitly instead.
ICD10_AMYLOID_FAMILY = ["E85.0", "E85.1", "E85.2", "E85.3", "E85.89", "E85.9"]
ICD9_AMYLOID_FAMILY = ["277.3"]

# DX_NAME text backstop, in case coded lists are incomplete/miscoded
DX_NAME_KEYWORDS = re.compile(r"amyloid", re.IGNORECASE)

# Explicit AL (light chain) amyloidosis name pattern — used to EXCLUDE these from
# counting toward the generic "amyloid family" DX_NAME backstop above, since AL is a
# different disease and the word "amyloid" alone in its name would otherwise sweep it
# in as a false ATTR-CM match.
AL_AMYLOID_NAME_PATTERN = re.compile(
    r"light chain.{0,10}amyloidosis|\bAL amyloidosis\b|immunoglobulin light chain amyloidosis",
    re.IGNORECASE,
)

# AL CONFIRMATORY LAB WORKUP: DX codes/names can be presumptive or reflect a
# rule-out that was later reversed. These are the actual confirmatory tests for AL
# (light chain) amyloidosis — free light chain assay, serum/urine protein
# electrophoresis with immunofixation, bone marrow biopsy for plasma cell clonality.
# Presence of this workup is a STRONGER AL signal than a DX code alone, and is used
# to catch real AL cases that might lack a specific ICD code, complementing (not
# replacing) the DX-based AL_AMYLOID_NAME_PATTERN check above.
AL_LAB_KEYWORDS = re.compile(
    r"(free light chain|serum free light|kappa.{0,20}lambda|"
    r"urine protein electrophoresis|\bUPEP\b|"
    r"immunofixation|bone marrow biopsy|bone marrow aspirate|plasma cell)",
    re.IGNORECASE,
)
# "SPEP"/"serum protein electrophoresis" alone is too nonspecific (ordered broadly,
# for many reasons unrelated to suspected amyloidosis) — only counts as AL evidence
# when the result was actually flagged abnormal, matched separately below.
AL_LAB_NONSPECIFIC_KEYWORDS = re.compile(
    r"(serum protein electrophoresis|\bSPEP\b|protein,?\s*total)", re.IGNORECASE
)

# Disease-modifying therapy names (brand + generic) — extend this list if your
# formulary/EHR uses additional trade names or abbreviations
DRUG_KEYWORDS = re.compile(
    r"(tafamidis|vyndaqel|vyndamax|acoramidis|attruby|"
    r"patisiran|onpattro|vutrisiran|amvuttra|inotersen|tegsedi)",
    re.IGNORECASE,
)
# Diflunisal is a generic NSAID with only occasional off-label amyloid-stabilizer
# use — too nonspecific to count as a primary ATTR-CM medication signal (it
# dominated and diluted the primary signal when included directly). Tracked
# separately at low confidence instead of discarded entirely.
DIFLUNISAL_PATTERN = re.compile(r"diflunisal", re.IGNORECASE)

# Genetic testing order text
GENETIC_TEST_KEYWORDS = re.compile(r"(\bTTR\b|transthyretin)", re.IGNORECASE)

# PYP / amyloid imaging order text — matched on procedure name/narrative, not CPT,
# since PYP scintigraphy CPT coding is not standardized across institutions.
# CRITICAL: "amyloid" alone also matches BRAIN amyloid PET (Alzheimer's/dementia
# workup) — a completely different clinical context. Require cardiac-specific
# language (pyrophosphate/PYP always is; a bare "amyloid" match must also mention
# cardiac/heart) and explicitly exclude brain/neuro scans even if worded oddly.
IMAGING_KEYWORDS = re.compile(
    r"(pyrophosphate|\bPYP\b|(cardiac|heart).{0,20}amyloid|amyloid.{0,20}(cardiac|heart|scintigraphy))",
    re.IGNORECASE,
)
IMAGING_EXCLUDE_PATTERN = re.compile(r"\bbrain\b|\bneuro\b|\balzheimer", re.IGNORECASE)


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def iter_chunks(file_path, usecols=None):
    """Yield chunks of a large tab/pipe-delimited file as string-typed DataFrames."""
    delim = sniff_delimiter(file_path)
    reader = pd.read_csv(
        file_path, sep=delim, dtype=str, on_bad_lines="skip",
        engine="python", quoting=3, usecols=usecols, chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        yield chunk


def code_list_matches(code_list_str, target_codes):
    """Check whether any code in a comma/semicolon-delimited code list string
    starts with any of the target code prefixes."""
    if not isinstance(code_list_str, str) or not code_list_str:
        return False
    codes = re.split(r"[;,]", code_list_str)
    codes = [c.strip() for c in codes]
    return any(any(c.startswith(t) for t in target_codes) for c in codes)


def find_col(columns, *name_fragments):
    """Find the first column whose name contains ALL given fragments (case-insensitive)."""
    for c in columns:
        cu = c.upper()
        if all(frag.upper() in cu for frag in name_fragments):
            return c
    return None


def scan_dx_files(inspect=False):
    """Signal 1: diagnosis codes. Scans BOTH encounter-level DX files (which only
    capture diagnoses tied to a specific coded visit) AND Medical_Hx.txt (the
    master problem list, which can carry diagnoses entered directly without a
    linked coded encounter) — this closes a real ascertainment gap: a diagnosis
    on the problem list but never re-coded at a later visit would otherwise be
    invisible to this scan. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_dx", "outpt_dx", "medical_hx"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            icd10_col = find_col(chunk.columns, "ICD10")
            icd9_col = find_col(chunk.columns, "ICD9")
            name_col = find_col(chunk.columns, "DX_NAME")

            for _, row in chunk.iterrows():
                icd10 = row.get(icd10_col, "") if icd10_col else ""
                icd9 = row.get(icd9_col, "") if icd9_col else ""
                dx_name = row.get(name_col, "") if name_col else ""

                is_attr_specific = code_list_matches(icd10, ICD10_ATTR_SPECIFIC)
                is_al_amyloid = (
                    code_list_matches(icd10, ICD10_AL_AMYLOID_EXCLUDE)
                    or bool(AL_AMYLOID_NAME_PATTERN.search(str(dx_name)))
                )
                is_amyloid_family = (
                    code_list_matches(icd10, ICD10_AMYLOID_FAMILY)
                    or code_list_matches(icd9, ICD9_AMYLOID_FAMILY)
                )
                # The generic "amyloid" DX_NAME backstop must NOT count an AL-named
                # diagnosis as a match on its own — AL is a different disease.
                is_name_match = (
                    bool(DX_NAME_KEYWORDS.search(str(dx_name)))
                    and not AL_AMYLOID_NAME_PATTERN.search(str(dx_name))
                )

                if not (is_attr_specific or is_amyloid_family or is_name_match or is_al_amyloid):
                    continue

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  ICD10={icd10!r}  ICD9={icd9!r}  DX_NAME={dx_name!r}  "
                          f"[{'AL-EXCLUDE' if is_al_amyloid and not is_attr_specific else 'ATTR' if is_attr_specific else 'family/name'}]")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "dx_attr_specific": False,
                    "dx_amyloid_family": False,
                    "dx_name_match": False,
                    "dx_al_amyloid_only": False,
                    "dx_evidence": set(),
                })
                entry["dx_attr_specific"] |= is_attr_specific
                entry["dx_amyloid_family"] |= is_amyloid_family
                entry["dx_name_match"] |= is_name_match
                # Track AL separately; only meaningful as "AL-only" if this patient has
                # NO other ATTR-qualifying signal from this row
                entry["dx_al_amyloid_only"] |= (is_al_amyloid and not (is_attr_specific or is_amyloid_family or is_name_match))
                if is_attr_specific or is_amyloid_family:
                    entry["dx_evidence"].add(str(icd10) if icd10 else str(icd9))
                elif is_name_match:
                    entry["dx_evidence"].add(str(dx_name))
    print(f"  -> {len(hits)} unique patients matched on diagnosis codes/names.")
    return hits


def scan_med_files(inspect=False):
    """Signal 2: disease-modifying therapy. Returns (hits, diflunisal_hits) — the
    latter tracked SEPARATELY as low-confidence, non-qualifying evidence, since
    diflunisal is a generic NSAID with only occasional off-label amyloid use and
    was found to dominate/dilute the primary signal when included directly."""
    hits = {}
    diflunisal_hits = {}
    sample_shown = 0
    for key in ("meds", "external_meds"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_cols = [c for c in chunk.columns if "MEDICATION_NAME" in c.upper()
                         or "GENERIC_NAME" in c.upper() or "SIMPLE_GENERIC" == c.upper()]

            for _, row in chunk.iterrows():
                matched_name = None
                diflunisal_name = None
                for nc in name_cols:
                    val = row.get(nc, "")
                    if not isinstance(val, str):
                        continue
                    if DRUG_KEYWORDS.search(val):
                        matched_name = val
                        break
                    if DIFLUNISAL_PATTERN.search(val):
                        diflunisal_name = val

                pat_id = row.get(id_col, "")

                if matched_name:
                    if inspect and sample_shown < 15:
                        print(f"    MATCH  MED={matched_name!r}")
                        sample_shown += 1
                    entry = hits.setdefault(pat_id, {
                        "PAT_MRN_ID": row.get(mrn_col, ""),
                        "med_evidence": set(),
                    })
                    entry["med_evidence"].add(matched_name)
                elif diflunisal_name:
                    entry = diflunisal_hits.setdefault(pat_id, {
                        "PAT_MRN_ID": row.get(mrn_col, ""),
                        "diflunisal_evidence": set(),
                    })
                    entry["diflunisal_evidence"].add(diflunisal_name)
    print(f"  -> {len(hits)} unique patients matched on ATTR-CM medications.")
    print(f"  -> {len(diflunisal_hits)} additional patients matched on diflunisal ONLY "
          f"(tracked separately, low-confidence — NOT counted in the primary medication signal).")
    return hits, diflunisal_hits


def scan_med_admin_files(inspect=False):
    """Signal: administered (not just ordered) ATTR-CM medications. Administration
    records are stronger evidence than an order alone, since they confirm the drug
    was actually given rather than just prescribed/discontinued."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_med_admin", "outpt_med_admin"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_cols = [c for c in chunk.columns if "MEDICATION_NAME" in c.upper()
                         or "GENERIC_NAME" in c.upper() or "SIMPLE_GENERIC" == c.upper()]
            action_col = find_col(chunk.columns, "MAR_ACTION") 

            for _, row in chunk.iterrows():
                matched_name = None
                for nc in name_cols:
                    val = row.get(nc, "")
                    if isinstance(val, str) and DRUG_KEYWORDS.search(val):
                        matched_name = val
                        break
                if not matched_name:
                    continue

                pat_id = row.get(id_col, "")
                action = row.get(action_col, "") if action_col else ""
                evidence = f"{matched_name} ({action})" if action else matched_name

                if inspect and sample_shown < 15:
                    print(f"    MATCH  MED_ADMIN={evidence!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "med_admin_evidence": set(),
                })
                entry["med_admin_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients matched on administered ATTR-CM medications.")
    return hits


def scan_icd_px_files(inspect=False):
    """Signal: procedure codes whose NAME text suggests PYP/amyloid imaging — a
    second, independent check on the imaging signal beyond Imaging_1/2.txt PROC_NAME,
    in case a PYP scan is captured here (as a billed procedure) but not there."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_icd_px", "outpt_icd_px"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_col = find_col(chunk.columns, "ICD_PX_NAME")

            for _, row in chunk.iterrows():
                px_name = row.get(name_col, "") if name_col else ""
                if not isinstance(px_name, str) or not IMAGING_KEYWORDS.search(px_name):
                    continue

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  ICD_PX_NAME={px_name!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "icd_px_evidence": set(),
                })
                entry["icd_px_evidence"].add(px_name)
    print(f"  -> {len(hits)} unique patients matched on PYP/amyloid procedure codes.")
    return hits


def count_cardiology_visits(cohort_pat_ids):
    """Counts outpatient visits to a cardiology department per patient in the FINAL
    cohort — a diagnostic-PROCESS metric (was the patient actually seen by the right
    specialty), not a health outcome. This is Domain A/B (diagnostic completeness),
    distinct from outcomes comparisons that are out of scope here. Applied to the
    WHOLE candidate pool deliberately — restricting to confirmed-only patients would
    exclude exactly the incomplete-workup population this metric is meant to find.

    NOTE (simple version): this counts total cardiology visits per patient, not yet
    "visits AFTER first ATTR signal with no resolution since" — that more precise
    version needs a normalized first-signal date tracked across DX/med/PYP/genetic
    signals, which isn't computed yet. Logged as a follow-up refinement."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Outpatient_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Counting cardiology department visits from CarDS_2837199_Outpatient_Enc.txt ...")
    CARDIOLOGY_PATTERN = re.compile(r"cardiology", re.IGNORECASE)
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        dept_col = find_col(chunk.columns, "DEPARTMENT_NAME")
        if id_col is None or dept_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, dept in zip(chunk[id_col], chunk[dept_col]):
            if isinstance(dept, str) and CARDIOLOGY_PATTERN.search(dept):
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def count_true_hospitalizations(cohort_pat_ids):
    """DESCRIPTIVE ONLY — counts TRUE inpatient hospitalizations (INP_YN='Y', not
    ED-only visits) per patient in the FINAL cohort, for basic cohort-characteristics
    reporting (a standard 'Table 1' style statistic — total cohort burden). This is
    NOT a comparison by care-gap status, diagnostic completeness, or treatment
    status — that kind of outcomes analysis is explicitly out of scope here."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Hosp_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Counting true hospitalizations (INP_YN=Y) from CarDS_2837199_Hosp_Enc.txt ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        inp_col = find_col(chunk.columns, "INP_YN")
        if id_col is None or inp_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, is_inp in zip(chunk[id_col], chunk[inp_col]):
            if str(is_inp).strip().upper() == "Y":
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def scan_family_hx_red_flags(inspect=False):
    """SUPPLEMENTARY, NON-QUALIFYING signal: family history of cardiomyopathy or
    sudden cardiac death — a recognized red flag for hereditary ATTR-CM (many older
    relatives with undiagnosed ATTR died before amyloidosis was ever suspected,
    showing up in family history only as vague 'cardiomyopathy'/'sudden death').

    DELIBERATELY NOT added to all_pat_ids / the qualifying-signal union: these terms
    have many causes unrelated to TTR (sarcomeric HCM genes, arrhythmia syndromes,
    coronary disease), so using this alone to ADD patients to the cohort would repeat
    the exact nonspecificity mistake already caught and fixed twice tonight (the AL
    ICD code, the TTN/MYH7/BRCA gene-panel conflation). This is descriptive context
    for patients who ALREADY qualify via a real ATTR-specific signal, not a
    qualifying signal on its own. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    file_path = os.path.join(RAW_DIR, FILES.get("family_hx", "CarDS_2837199_Family_Hx.txt"))
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return hits
    print(f"  Scanning CarDS_2837199_Family_Hx.txt ...")
    FAMILY_HX_PATTERN = re.compile(r"sudden death|sudden cardiac death|cardiomyopathy", re.IGNORECASE)
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        hx_col = find_col(chunk.columns, "MEDICAL_HX")
        relation_col = find_col(chunk.columns, "RELATION")
        age_col = find_col(chunk.columns, "AGE_OF_ONSET")

        for _, row in chunk.iterrows():
            hx_val = row.get(hx_col, "") if hx_col else ""
            if not isinstance(hx_val, str) or not FAMILY_HX_PATTERN.search(hx_val):
                continue

            pat_id = row.get(id_col, "")
            relation = row.get(relation_col, "") if relation_col else ""
            age = row.get(age_col, "") if age_col else ""
            evidence = f"{hx_val} (relation={relation}, age_of_onset={age})"

            if inspect and sample_shown < 15:
                print(f"    MATCH  FAMILY_HX={evidence!r}")
                sample_shown += 1

            entry = hits.setdefault(pat_id, {"family_hx_evidence": set()})
            entry["family_hx_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients with a family history cardiac red flag "
          f"(supplementary only — not used to qualify new cohort members).")
    return hits


def scan_al_confirmatory_labs(inspect=False):
    """Signal: confirmatory AL (light chain) amyloidosis lab workup — free light
    chain assay, SPEP/UPEP with immunofixation, bone marrow biopsy. Stronger evidence
    of true AL than a DX code alone, which can reflect a presumptive or later-reversed
    diagnosis. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    lab_keys = ("hosp_labs_1", "hosp_labs_2", "hosp_labs_3", "hosp_labs_4",
                "hosp_labs_5", "hosp_labs_6", "outpt_labs_1", "outpt_labs_2", "outpt_labs_3")
    for key in lab_keys:
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            text_cols = [c for c in chunk.columns if c.upper() in
                         ("PROC_NAME", "COMPONENT_NAME", "BASE_NAME", "DESCRIPTION", "DISPLAY_NAME")]
            abnormal_col = find_col(chunk.columns, "ABNORMAL_YN")

            for _, row in chunk.iterrows():
                matched_text = None
                is_nonspecific = False
                for tc in text_cols:
                    val = row.get(tc, "")
                    if not isinstance(val, str):
                        continue
                    if AL_LAB_KEYWORDS.search(val):
                        matched_text = val
                        is_nonspecific = False
                        break
                    if AL_LAB_NONSPECIFIC_KEYWORDS.search(val):
                        matched_text = val
                        is_nonspecific = True
                        # keep looking in case a more specific field also matches
                if not matched_text:
                    continue

                abnormal = row.get(abnormal_col, "") if abnormal_col else ""

                # Nonspecific tests (bare SPEP/total protein) only count as AL evidence
                # when actually flagged abnormal — a normal routine SPEP is not
                # meaningful AL evidence and was previously dominating this signal.
                if is_nonspecific and str(abnormal).strip().upper() != "Y":
                    continue

                pat_id = row.get(id_col, "")
                evidence = f"{matched_text} (abnormal={abnormal})" if abnormal else matched_text

                if inspect and sample_shown < 15:
                    print(f"    MATCH  AL_LAB={evidence!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "al_lab_evidence": set(),
                })
                entry["al_lab_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients matched on AL confirmatory lab workup.")
    return hits


def scan_genetic_testing_file(inspect=False):
    """Signal 3: TTR genetic testing orders. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    file_path = os.path.join(RAW_DIR, FILES["genetic_testing"])
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return hits
    print(f"  Scanning {FILES['genetic_testing']} ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
        text_cols = [c for c in chunk.columns if c.upper() in
                     ("PROC_NAME", "ORDER_DISPLAY_NAME", "DESCRIPTION", "DISPLAY_NAME")]

        for _, row in chunk.iterrows():
            matched_text = None
            for tc in text_cols:
                val = row.get(tc, "")
                if isinstance(val, str) and GENETIC_TEST_KEYWORDS.search(val):
                    matched_text = val
                    break
            if not matched_text:
                continue

            pat_id = row.get(id_col, "")
            if inspect and sample_shown < 15:
                print(f"    MATCH  PROC={matched_text!r}")
                sample_shown += 1

            entry = hits.setdefault(pat_id, {
                "PAT_MRN_ID": row.get(mrn_col, ""),
                "genetic_order_evidence": set(),
            })
            entry["genetic_order_evidence"].add(matched_text)
    print(f"  -> {len(hits)} unique patients matched on TTR genetic testing orders.")
    return hits


def scan_imaging_files(inspect=False):
    """Signal 4: PYP/amyloid imaging orders. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    for key in ("imaging_1", "imaging_2"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            proc_col = find_col(chunk.columns, "PROC_NAME")

            for _, row in chunk.iterrows():
                proc_name = row.get(proc_col, "") if proc_col else ""
                if not isinstance(proc_name, str) or not IMAGING_KEYWORDS.search(proc_name):
                    continue
                if IMAGING_EXCLUDE_PATTERN.search(proc_name):
                    continue  # brain/neuro amyloid PET — different clinical context entirely

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  PROC_NAME={proc_name!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "imaging_evidence": set(),
                })
                entry["imaging_evidence"].add(proc_name)
    print(f"  -> {len(hits)} unique patients matched on PYP/amyloid imaging orders.")
    return hits


def main():
    parser = argparse.ArgumentParser(description="Identify ATTR-CM candidate cohort from structured data")
    parser.add_argument("--inspect", action="store_true",
                         help="Print sample matches per signal without writing an output file")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ATTR-CM COHORT IDENTIFICATION [STRUCTURED DATA]")
    print(f"Directory: {RAW_DIR}")
    print("=" * 70)

    print("\n[1/7] Diagnosis codes")
    dx_hits = scan_dx_files(inspect=args.inspect)

    print("\n[2/7] Medication orders")
    med_hits, diflunisal_hits = scan_med_files(inspect=args.inspect)

    print("\n[3/7] Medication administration records")
    med_admin_hits = scan_med_admin_files(inspect=args.inspect)

    print("\n[4/7] Genetic testing orders")
    genetic_hits = scan_genetic_testing_file(inspect=args.inspect)

    print("\n[5/7] PYP/amyloid imaging orders")
    imaging_hits = scan_imaging_files(inspect=args.inspect)

    print("\n[6/7] PYP/amyloid procedure codes (ICD_PX)")
    icd_px_hits = scan_icd_px_files(inspect=args.inspect)

    print("\n[7/7] AL confirmatory lab workup (free light chains, SPEP/UPEP, bone marrow biopsy)")
    al_lab_hits = scan_al_confirmatory_labs(inspect=args.inspect)

    print("\n[bonus] Family history cardiac red flags (supplementary, non-qualifying)")
    family_hx_hits = scan_family_hx_red_flags(inspect=args.inspect)

    if args.inspect:
        print("\n[INSPECT MODE] No output file written. Review the sample matches above,")
        print("then adjust the keyword/code lists near the top of this script if needed.")
        return

    # ── Merge all signals into one cohort table ──
    all_pat_ids = (set(dx_hits) | set(med_hits) | set(med_admin_hits) | set(genetic_hits)
                   | set(imaging_hits) | set(icd_px_hits))
    print(f"\nMerging signals: {len(all_pat_ids)} unique candidate patients total.")

    rows = []
    for pat_id in all_pat_ids:
        dx = dx_hits.get(pat_id, {})
        med = med_hits.get(pat_id, {})
        med_admin = med_admin_hits.get(pat_id, {})
        gen = genetic_hits.get(pat_id, {})
        img = imaging_hits.get(pat_id, {})
        icd_px = icd_px_hits.get(pat_id, {})
        al_lab = al_lab_hits.get(pat_id, {})  # lookup only — doesn't expand candidate pool,
                                                # since AL lab evidence is evidence AGAINST
                                                # ATTR-CM, not a qualifying signal for it
        family_hx = family_hx_hits.get(pat_id, {})  # lookup only — supplementary, non-qualifying
        diflunisal = diflunisal_hits.get(pat_id, {})  # lookup only — low-confidence, non-qualifying

        mrn = (dx.get("PAT_MRN_ID") or med.get("PAT_MRN_ID") or med_admin.get("PAT_MRN_ID")
               or gen.get("PAT_MRN_ID") or img.get("PAT_MRN_ID") or icd_px.get("PAT_MRN_ID") or "")

        rows.append({
            "PAT_ID": pat_id,
            "PAT_MRN_ID": mrn,
            "flag_dx_attr_specific": dx.get("dx_attr_specific", False),
            "flag_dx_al_amyloid_only": dx.get("dx_al_amyloid_only", False),
            "flag_al_lab_workup": bool(al_lab),
            "flag_dx_amyloid_family": dx.get("dx_amyloid_family", False),
            "flag_dx_name_match": dx.get("dx_name_match", False),
            "flag_medication_order": bool(med),
            "flag_medication_admin": bool(med_admin),
            "flag_genetic_test_order": bool(gen),
            "flag_pyp_imaging": bool(img),
            "flag_icd_px": bool(icd_px),
            "dx_evidence": "; ".join(dx.get("dx_evidence", [])),
            "med_order_evidence": "; ".join(med.get("med_evidence", [])),
            "med_admin_evidence": "; ".join(med_admin.get("med_admin_evidence", [])),
            "genetic_order_evidence": "; ".join(gen.get("genetic_order_evidence", [])),
            "imaging_evidence": "; ".join(img.get("imaging_evidence", [])),
            "icd_px_evidence": "; ".join(icd_px.get("icd_px_evidence", [])),
            "al_lab_evidence": "; ".join(al_lab.get("al_lab_evidence", [])),
            "flag_family_hx_cardiac_red_flag": bool(family_hx),
            "family_hx_evidence": "; ".join(family_hx.get("family_hx_evidence", [])),
            "flag_diflunisal_only": bool(diflunisal),
            "diflunisal_evidence": "; ".join(diflunisal.get("diflunisal_evidence", [])),
        })

    cohort_df = pd.DataFrame(rows)

    # Optionally merge demographics
    patients_path = os.path.join(RAW_DIR, FILES["patients"])
    if os.path.exists(patients_path):
        print("Merging demographics from Patients.txt ...")
        pat_df = pd.read_csv(
            patients_path, sep=sniff_delimiter(patients_path), dtype=str,
            on_bad_lines="skip", engine="python", quoting=3,
        )
        cohort_df = cohort_df.merge(pat_df, on=["PAT_ID", "PAT_MRN_ID"], how="left")

    # Descriptive-only cohort characteristics (mortality, hospitalization burden) —
    # NOT split by care-gap/diagnostic-completeness status. That kind of comparative
    # outcomes analysis sits in a different project domain and is out of scope here.
    print("\nComputing descriptive cohort characteristics (mortality, hospitalizations)...")
    print("NOTE: whole-cohort description only — NOT compared by care-gap status.")
    hosp_counts = count_true_hospitalizations(set(cohort_df["PAT_ID"]))
    cohort_df["num_true_hospitalizations"] = cohort_df["PAT_ID"].map(hosp_counts).fillna(0).astype(int)

    cardiology_counts = count_cardiology_visits(set(cohort_df["PAT_ID"]))
    cohort_df["num_cardiology_visits"] = cohort_df["PAT_ID"].map(cardiology_counts).fillna(0).astype(int)

    out_path = os.path.join(OUTPUT_DIR, "attr_cm_candidate_cohort.csv")
    cohort_df.to_csv(out_path, index=False)

    signal_cols = ["flag_dx_attr_specific", "flag_dx_amyloid_family", "flag_dx_name_match",
                   "flag_medication_order", "flag_medication_admin", "flag_genetic_test_order",
                   "flag_pyp_imaging", "flag_icd_px"]

    # AL-flagged-at-all vs AL-only-with-NOTHING-else — these are different numbers and
    # were a real source of confusion before: a patient can have one AL-coded encounter
    # AND independent real ATTR evidence elsewhere, and should stay in the cohort.
    # Combined AL evidence: DX code/name OR confirmatory lab workup — either is
    # sufficient to flag AL, since DX alone can be presumptive/later-reversed and
    # lab workup alone can occur without a corresponding DX code being entered.
    cohort_df["flag_al_any_evidence"] = cohort_df["flag_dx_al_amyloid_only"] | cohort_df["flag_al_lab_workup"]
    al_flagged_total = cohort_df["flag_al_any_evidence"].sum()
    al_only_pure = cohort_df[
        (cohort_df[signal_cols].sum(axis=1) == 0) & (cohort_df["flag_al_any_evidence"] == True)
    ]
    n_al_only_pure = len(al_only_pure)
    true_cohort_size = len(cohort_df) - n_al_only_pure

    # ── Summary / cohort-flow numbers for your methods section ──
    print(f"\n{'='*70}")
    print("COHORT IDENTIFICATION COMPLETE")
    print(f"Total candidate patients (raw, before AL exclusion): {len(cohort_df)}")
    print(f"  Matched ATTR-specific DX (E85.4/E85.82): {cohort_df['flag_dx_attr_specific'].sum()}")
    print(f"  AL flagged by DX code/name: {cohort_df['flag_dx_al_amyloid_only'].sum()}")
    print(f"  AL flagged by confirmatory LAB workup (free light chains/SPEP/UPEP/bone marrow): {cohort_df['flag_al_lab_workup'].sum()}")
    print(f"  AL flagged by EITHER signal (combined): {al_flagged_total}")
    print(f"  AL-ONLY patients (no other ATTR signal anywhere) — TRUE EXCLUSIONS: {n_al_only_pure}")
    print(f"  ({al_flagged_total - n_al_only_pure} AL-flagged patients ALSO have independent ATTR "
          f"evidence elsewhere and correctly remain in the cohort — mixed/miscoded records, not exclusions)")
    print(f"  Matched broader amyloid family DX/DX_NAME: {(cohort_df['flag_dx_amyloid_family'] | cohort_df['flag_dx_name_match']).sum()}")
    print(f"  Matched ATTR-CM medication order: {cohort_df['flag_medication_order'].sum()}")
    print(f"  Matched ATTR-CM medication administration: {cohort_df['flag_medication_admin'].sum()}")
    print(f"  Matched TTR genetic testing order: {cohort_df['flag_genetic_test_order'].sum()}")
    print(f"  Matched PYP/amyloid imaging order: {cohort_df['flag_pyp_imaging'].sum()}")
    print(f"  Matched PYP/amyloid procedure code (ICD_PX): {cohort_df['flag_icd_px'].sum()}")
    print(f"  ⏳ PENDING: PYP CPT code (Hosp_Enc_CPT.txt/Outpatient_Enc_CPT.txt) — NOT scanned "
          f"yet, real institutional code unconfirmed. See TODO block at top of this script.")
    only_one_signal = (cohort_df[signal_cols].sum(axis=1) == 1).sum()
    print(f"  Patients matched on exactly ONE signal (worth spot-checking): {only_one_signal}")
    n_family_hx = cohort_df["flag_family_hx_cardiac_red_flag"].sum()
    print(f"  (Supplementary, non-qualifying) Family history of cardiomyopathy/sudden "
          f"cardiac death, among cohort members: {n_family_hx}")
    print(f"\n  >>> TRUE ATTR-CM CANDIDATE COHORT SIZE (after AL exclusion): {true_cohort_size} <<<")

    if "DEATH_DATE" in cohort_df.columns:
        n_deceased = cohort_df["DEATH_DATE"].notna().sum()
        pct_deceased = (n_deceased / len(cohort_df) * 100) if len(cohort_df) > 0 else 0
        print(f"\n--- Descriptive cohort characteristics (whole cohort, NOT split by care-gap status) ---")
        print(f"  Deceased (DEATH_DATE present): {n_deceased} ({pct_deceased:.1f}%)")
    if "num_true_hospitalizations" in cohort_df.columns:
        hosp_series = cohort_df["num_true_hospitalizations"]
        print(f"  True inpatient hospitalizations per patient — "
              f"median: {hosp_series.median():.0f}, mean: {hosp_series.mean():.2f}, "
              f"0 hospitalizations: {(hosp_series == 0).sum()} patients")
    if "num_cardiology_visits" in cohort_df.columns:
        card_series = cohort_df["num_cardiology_visits"]
        n_zero_cardiology = (card_series == 0).sum()
        print(f"  Cardiology visits per patient — median: {card_series.median():.0f}, "
              f"mean: {card_series.mean():.2f}")
        print(f"  ⚠ ZERO cardiology visits despite an ATTR-CM signal: {n_zero_cardiology} patients "
              f"— worth checking whether these were seen elsewhere or never evaluated at all")

    print(f"Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()






#newest to get correct family bhistory:

'''

#!/usr/bin/env python3
"""
ATTR-CM Cohort Identification Script
=====================================
Identifies candidate ATTR-CM patients from STRUCTURED data only (diagnosis codes,
medications, genetic testing orders, PYP/amyloid imaging orders) across an entire
health system's data extract. This is meant to run BEFORE the notes-based genetics
NLP script (equa_attr_extract_genetics.py) — the output cohort's PAT_IDs are what
you should filter the note files down to, rather than scanning every patient's notes
in the health system.

Signals used (each is independent; a patient needs to match only ONE to be included):
  1. Diagnosis codes  - E85.81 / E85.82 (ATTR-specific, 2023 ICD-10-CM) and the
                         broader E85.x amyloidosis family / legacy ICD-9 277.3x,
                         from Hosp_Enc_DX.txt + Outpatient_Enc_DX.txt
  2. Medications       - tafamidis, acoramidis, patisiran, vutrisiran, inotersen,
                         diflunisal (brand + generic names), from Meds.txt +
                         External_Meds.txt
  3. Genetic testing   - TTR / transthyretin genetic test orders, from
                         Genetic_Testing.txt
  4. PYP/amyloid imaging - pyrophosphate scan orders, from Imaging_1.txt + Imaging_2.txt
                         (matched on PROC_NAME text, not CPT code, since PYP CPT
                         coding varies by institution)

Output: one CSV with one row per candidate patient, a boolean flag per signal, and
the specific matched value(s) as evidence for each signal that fired, plus a summary
of source and overlap counts (a natural cohort-flow / attrition table for your
methods section).

IMPORTANT: The keyword/code lists below (DRUG_KEYWORDS, ICD10_ATTR_SPECIFIC,
ICD10_AMYLOID_FAMILY, IMAGING_KEYWORDS) are a reasonable starting point but you
should review them against your actual data before trusting the cohort — e.g. check
a sample of DX_NAME values that got matched, and check whether your institution's PYP
orders actually contain "pyrophosphate" in PROC_NAME (run the --inspect mode first).

Usage:
  python3 equa_attr_identify_cohort.py --inspect       # preview what would match, no output file
  python3 equa_attr_identify_cohort.py                 # full run, writes cohort CSV

================================================================================
TODO — DOCUMENTED FOLLOW-UPS, NOT YET IMPLEMENTED (see game plan doc for detail):
================================================================================
1. FAMILY HISTORY SIGNAL (Family_Hx.txt): confirmed real MEDICAL_HX text values
   exist in this file (hypertension, cancer, heart disease, etc. dominate by volume
   since amyloidosis is rare). Still needs a direct grep/search for
   amyloid|cardiomyopathy|transthyretin|TTR|sudden cardiac death before deciding
   whether to add as a signal — caveat: free-text family history usually can't
   distinguish wild-type (non-heritable) from hereditary ATTR in a relative, so a
   positive match should be treated as "should have prompted evaluation," not proof
   of hereditary risk.
2. PYP CPT CODE (Hosp_Enc_CPT.txt / Outpatient_Enc_CPT.txt): these files have ONLY
   PAT_ID/CPT_DATE/CPT_CODE — no description field, so the real institutional PYP
   CPT code is currently unknown and NOT guessed at (guessing wrong would silently
   return zero/wrong matches). To determine it safely: join these CPT files against
   the PAT_ENC_CSN_ID-equivalent from imaging_hits' matched encounters — NOTE the
   imaging files (Imaging_1.txt/Imaging_2.txt) do NOT have a column literally named
   PAT_ENC_CSN_ID; use PERFORMING_CSN_ID instead (the encounter the scan was
   actually performed during, which is what generates the CPT billing code). Once
   the real code is identified from your own data, add it as a supplementary
   imaging signal alongside the existing PROC_NAME text match.
================================================================================
"""

import os
import re
import argparse
import pandas as pd

# ── Directories (reuse the same raw data directory as the genetics script) ──
RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
OUTPUT_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results")
CHUNK_SIZE = 200_000  # rows per chunk when reading large structured files

FILES = {
    "hosp_dx": "CarDS_2837199_Hosp_Enc_DX.txt",
    "outpt_dx": "CarDS_2837199_Outpatient_Enc_DX.txt",
    "meds": "CarDS_2837199_Meds.txt",
    "external_meds": "CarDS_2837199_External_Meds.txt",
    "genetic_testing": "CarDS_2837199_Genetic_Testing.txt",
    "imaging_1": "CarDS_2837199_Imaging_1.txt",
    "imaging_2": "CarDS_2837199_Imaging_2.txt",
    "hosp_med_admin": "CarDS_2837199_Hosp_Enc_Med_Admin.txt",
    "outpt_med_admin": "CarDS_2837199_Outpatient_Enc_Med_Admin.txt",
    "hosp_icd_px": "CarDS_2837199_Hosp_Enc_ICD_PX.txt",
    "outpt_icd_px": "CarDS_2837199_Outpatient_Enc_ICD_PX.txt",
    "patients": "CarDS_2837199_Patients.txt",
    "hosp_labs_1": "CarDS_2837199_Hosp_Enc_Labs_1.txt",
    "hosp_labs_2": "CarDS_2837199_Hosp_Enc_Labs_2.txt",
    "hosp_labs_3": "CarDS_2837199_Hosp_Enc_Labs_3.txt",
    "hosp_labs_4": "CarDS_2837199_Hosp_Enc_Labs_4.txt",
    "hosp_labs_5": "CarDS_2837199_Hosp_Enc_Labs_5.txt",
    "hosp_labs_6": "CarDS_2837199_Hosp_Enc_Labs_6.txt",
    "outpt_labs_1": "CarDS_2837199_Outpatient_Enc_Labs_1.txt",
    "outpt_labs_2": "CarDS_2837199_Outpatient_Enc_Labs_2.txt",
    "outpt_labs_3": "CarDS_2837199_Outpatient_Enc_Labs_3.txt",
    "medical_hx": "CarDS_2837199_Medical_Hx.txt",
    "family_hx": "CarDS_2837199_Family_Hx.txt",
}

# ── Matching criteria — REVIEW THESE against your actual data (see --inspect) ──

# ATTR-specific ICD-10-CM codes (added ~2023): hereditary vs wild-type
# ATTR-specific ICD-10-CM codes, VERIFIED against the actual 2023+ ICD-10-CM tabular:
#   E85.82 = Wild-type transthyretin-related (ATTR) amyloidosis (ATTRwt) — unambiguous
#   E85.4  = Organ-limited amyloidosis — officially INCLUDES "Transthyretin-related
#            (ATTR) familial amyloid cardiomyopathy" as a named inclusion term, so
#            this is where hereditary ATTR-CM (ATTRv) actually gets coded, confirmed
#            by real DX_NAME values in this dataset ("Cardiac amyloidosis", "Amyloid
#            heart muscle disease" all appear coded under E85.4)
# CRITICAL: E85.81 is Light chain (AL) amyloidosis — a DIFFERENT DISEASE (plasma cell
# disorder, unrelated to TTR), and must NEVER be treated as ATTR-specific or even as
# generic "amyloid family" — it needs its own EXCLUDE flag so those patients can be
# identified and kept out of the ATTR-CM cohort, not silently folded in.
ICD10_ATTR_SPECIFIC = ["E85.4", "E85.82"]

# Codes for AL (light chain) amyloidosis — a DIFFERENT disease from ATTR. Patients
# matching ONLY this should be flagged and excluded/reviewed, not counted as ATTR.
ICD10_AL_AMYLOID_EXCLUDE = ["E85.81"]

# Broader amyloidosis family (non-AL, non-specific-ATTR) — catches ATTR-CM coded less
# specifically, plus legacy pre-2023 records. Note: "E85.8" as a bare prefix is
# deliberately AVOIDED here, since it would silently match E85.81 (AL) via substring
# prefix matching — E85.89 (Other amyloidosis, not AL) is used explicitly instead.
ICD10_AMYLOID_FAMILY = ["E85.0", "E85.1", "E85.2", "E85.3", "E85.89", "E85.9"]
ICD9_AMYLOID_FAMILY = ["277.3"]

# DX_NAME text backstop, in case coded lists are incomplete/miscoded
DX_NAME_KEYWORDS = re.compile(r"amyloid", re.IGNORECASE)

# Explicit AL (light chain) amyloidosis name pattern — used to EXCLUDE these from
# counting toward the generic "amyloid family" DX_NAME backstop above, since AL is a
# different disease and the word "amyloid" alone in its name would otherwise sweep it
# in as a false ATTR-CM match.
AL_AMYLOID_NAME_PATTERN = re.compile(
    r"light chain.{0,10}amyloidosis|\bAL amyloidosis\b|immunoglobulin light chain amyloidosis",
    re.IGNORECASE,
)

# AL CONFIRMATORY LAB WORKUP: DX codes/names can be presumptive or reflect a
# rule-out that was later reversed. These are the actual confirmatory tests for AL
# (light chain) amyloidosis — free light chain assay, serum/urine protein
# electrophoresis with immunofixation, bone marrow biopsy for plasma cell clonality.
# Presence of this workup is a STRONGER AL signal than a DX code alone, and is used
# to catch real AL cases that might lack a specific ICD code, complementing (not
# replacing) the DX-based AL_AMYLOID_NAME_PATTERN check above.
AL_LAB_KEYWORDS = re.compile(
    r"(free light chain|serum free light|kappa.{0,20}lambda|"
    r"urine protein electrophoresis|\bUPEP\b|"
    r"immunofixation|bone marrow biopsy|bone marrow aspirate|plasma cell)",
    re.IGNORECASE,
)
# "SPEP"/"serum protein electrophoresis" alone is too nonspecific (ordered broadly,
# for many reasons unrelated to suspected amyloidosis) — only counts as AL evidence
# when the result was actually flagged abnormal, matched separately below.
AL_LAB_NONSPECIFIC_KEYWORDS = re.compile(
    r"(serum protein electrophoresis|\bSPEP\b|protein,?\s*total)", re.IGNORECASE
)

# Disease-modifying therapy names (brand + generic) — extend this list if your
# formulary/EHR uses additional trade names or abbreviations
DRUG_KEYWORDS = re.compile(
    r"(tafamidis|vyndaqel|vyndamax|acoramidis|attruby|"
    r"patisiran|onpattro|vutrisiran|amvuttra|inotersen|tegsedi)",
    re.IGNORECASE,
)
# Diflunisal is a generic NSAID with only occasional off-label amyloid-stabilizer
# use — too nonspecific to count as a primary ATTR-CM medication signal (it
# dominated and diluted the primary signal when included directly). Tracked
# separately at low confidence instead of discarded entirely.
DIFLUNISAL_PATTERN = re.compile(r"diflunisal", re.IGNORECASE)

# Genetic testing order text
GENETIC_TEST_KEYWORDS = re.compile(r"(\bTTR\b|transthyretin)", re.IGNORECASE)

# PYP / amyloid imaging order text — matched on procedure name/narrative, not CPT,
# since PYP scintigraphy CPT coding is not standardized across institutions.
# CRITICAL: "amyloid" alone also matches BRAIN amyloid PET (Alzheimer's/dementia
# workup) — a completely different clinical context. Require cardiac-specific
# language (pyrophosphate/PYP always is; a bare "amyloid" match must also mention
# cardiac/heart) and explicitly exclude brain/neuro scans even if worded oddly.
IMAGING_KEYWORDS = re.compile(
    r"(pyrophosphate|\bPYP\b|(cardiac|heart).{0,20}amyloid|amyloid.{0,20}(cardiac|heart|scintigraphy))",
    re.IGNORECASE,
)
IMAGING_EXCLUDE_PATTERN = re.compile(r"\bbrain\b|\bneuro\b|\balzheimer", re.IGNORECASE)


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def iter_chunks(file_path, usecols=None):
    """Yield chunks of a large tab/pipe-delimited file as string-typed DataFrames."""
    delim = sniff_delimiter(file_path)
    reader = pd.read_csv(
        file_path, sep=delim, dtype=str, on_bad_lines="skip",
        engine="python", quoting=3, usecols=usecols, chunksize=CHUNK_SIZE,
    )
    for chunk in reader:
        yield chunk


def code_list_matches(code_list_str, target_codes):
    """Check whether any code in a comma/semicolon-delimited code list string
    starts with any of the target code prefixes."""
    if not isinstance(code_list_str, str) or not code_list_str:
        return False
    codes = re.split(r"[;,]", code_list_str)
    codes = [c.strip() for c in codes]
    return any(any(c.startswith(t) for t in target_codes) for c in codes)


def find_col(columns, *name_fragments):
    """Find the first column whose name contains ALL given fragments (case-insensitive)."""
    for c in columns:
        cu = c.upper()
        if all(frag.upper() in cu for frag in name_fragments):
            return c
    return None


def scan_dx_files(inspect=False):
    """Signal 1: diagnosis codes. Scans BOTH encounter-level DX files (which only
    capture diagnoses tied to a specific coded visit) AND Medical_Hx.txt (the
    master problem list, which can carry diagnoses entered directly without a
    linked coded encounter) — this closes a real ascertainment gap: a diagnosis
    on the problem list but never re-coded at a later visit would otherwise be
    invisible to this scan. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_dx", "outpt_dx", "medical_hx"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            icd10_col = find_col(chunk.columns, "ICD10")
            icd9_col = find_col(chunk.columns, "ICD9")
            name_col = find_col(chunk.columns, "DX_NAME")

            for _, row in chunk.iterrows():
                icd10 = row.get(icd10_col, "") if icd10_col else ""
                icd9 = row.get(icd9_col, "") if icd9_col else ""
                dx_name = row.get(name_col, "") if name_col else ""

                is_attr_specific = code_list_matches(icd10, ICD10_ATTR_SPECIFIC)
                is_al_amyloid = (
                    code_list_matches(icd10, ICD10_AL_AMYLOID_EXCLUDE)
                    or bool(AL_AMYLOID_NAME_PATTERN.search(str(dx_name)))
                )
                is_amyloid_family = (
                    code_list_matches(icd10, ICD10_AMYLOID_FAMILY)
                    or code_list_matches(icd9, ICD9_AMYLOID_FAMILY)
                )
                # The generic "amyloid" DX_NAME backstop must NOT count an AL-named
                # diagnosis as a match on its own — AL is a different disease.
                is_name_match = (
                    bool(DX_NAME_KEYWORDS.search(str(dx_name)))
                    and not AL_AMYLOID_NAME_PATTERN.search(str(dx_name))
                )

                if not (is_attr_specific or is_amyloid_family or is_name_match or is_al_amyloid):
                    continue

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  ICD10={icd10!r}  ICD9={icd9!r}  DX_NAME={dx_name!r}  "
                          f"[{'AL-EXCLUDE' if is_al_amyloid and not is_attr_specific else 'ATTR' if is_attr_specific else 'family/name'}]")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "dx_attr_specific": False,
                    "dx_amyloid_family": False,
                    "dx_name_match": False,
                    "dx_al_amyloid_only": False,
                    "dx_evidence": set(),
                })
                entry["dx_attr_specific"] |= is_attr_specific
                entry["dx_amyloid_family"] |= is_amyloid_family
                entry["dx_name_match"] |= is_name_match
                # Track AL separately; only meaningful as "AL-only" if this patient has
                # NO other ATTR-qualifying signal from this row
                entry["dx_al_amyloid_only"] |= (is_al_amyloid and not (is_attr_specific or is_amyloid_family or is_name_match))
                if is_attr_specific or is_amyloid_family:
                    entry["dx_evidence"].add(str(icd10) if icd10 else str(icd9))
                elif is_name_match:
                    entry["dx_evidence"].add(str(dx_name))
    print(f"  -> {len(hits)} unique patients matched on diagnosis codes/names.")
    return hits


def scan_med_files(inspect=False):
    """Signal 2: disease-modifying therapy. Returns (hits, diflunisal_hits) — the
    latter tracked SEPARATELY as low-confidence, non-qualifying evidence, since
    diflunisal is a generic NSAID with only occasional off-label amyloid use and
    was found to dominate/dilute the primary signal when included directly."""
    hits = {}
    diflunisal_hits = {}
    sample_shown = 0
    for key in ("meds", "external_meds"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_cols = [c for c in chunk.columns if "MEDICATION_NAME" in c.upper()
                         or "GENERIC_NAME" in c.upper() or "SIMPLE_GENERIC" == c.upper()]

            for _, row in chunk.iterrows():
                matched_name = None
                diflunisal_name = None
                for nc in name_cols:
                    val = row.get(nc, "")
                    if not isinstance(val, str):
                        continue
                    if DRUG_KEYWORDS.search(val):
                        matched_name = val
                        break
                    if DIFLUNISAL_PATTERN.search(val):
                        diflunisal_name = val

                pat_id = row.get(id_col, "")

                if matched_name:
                    if inspect and sample_shown < 15:
                        print(f"    MATCH  MED={matched_name!r}")
                        sample_shown += 1
                    entry = hits.setdefault(pat_id, {
                        "PAT_MRN_ID": row.get(mrn_col, ""),
                        "med_evidence": set(),
                    })
                    entry["med_evidence"].add(matched_name)
                elif diflunisal_name:
                    entry = diflunisal_hits.setdefault(pat_id, {
                        "PAT_MRN_ID": row.get(mrn_col, ""),
                        "diflunisal_evidence": set(),
                    })
                    entry["diflunisal_evidence"].add(diflunisal_name)
    print(f"  -> {len(hits)} unique patients matched on ATTR-CM medications.")
    print(f"  -> {len(diflunisal_hits)} additional patients matched on diflunisal ONLY "
          f"(tracked separately, low-confidence — NOT counted in the primary medication signal).")
    return hits, diflunisal_hits


def scan_med_admin_files(inspect=False):
    """Signal: administered (not just ordered) ATTR-CM medications. Administration
    records are stronger evidence than an order alone, since they confirm the drug
    was actually given rather than just prescribed/discontinued."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_med_admin", "outpt_med_admin"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_cols = [c for c in chunk.columns if "MEDICATION_NAME" in c.upper()
                         or "GENERIC_NAME" in c.upper() or "SIMPLE_GENERIC" == c.upper()]
            action_col = find_col(chunk.columns, "MAR_ACTION") 

            for _, row in chunk.iterrows():
                matched_name = None
                for nc in name_cols:
                    val = row.get(nc, "")
                    if isinstance(val, str) and DRUG_KEYWORDS.search(val):
                        matched_name = val
                        break
                if not matched_name:
                    continue

                pat_id = row.get(id_col, "")
                action = row.get(action_col, "") if action_col else ""
                evidence = f"{matched_name} ({action})" if action else matched_name

                if inspect and sample_shown < 15:
                    print(f"    MATCH  MED_ADMIN={evidence!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "med_admin_evidence": set(),
                })
                entry["med_admin_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients matched on administered ATTR-CM medications.")
    return hits


def scan_icd_px_files(inspect=False):
    """Signal: procedure codes whose NAME text suggests PYP/amyloid imaging — a
    second, independent check on the imaging signal beyond Imaging_1/2.txt PROC_NAME,
    in case a PYP scan is captured here (as a billed procedure) but not there."""
    hits = {}
    sample_shown = 0
    for key in ("hosp_icd_px", "outpt_icd_px"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            name_col = find_col(chunk.columns, "ICD_PX_NAME")

            for _, row in chunk.iterrows():
                px_name = row.get(name_col, "") if name_col else ""
                if not isinstance(px_name, str) or not IMAGING_KEYWORDS.search(px_name):
                    continue

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  ICD_PX_NAME={px_name!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "icd_px_evidence": set(),
                })
                entry["icd_px_evidence"].add(px_name)
    print(f"  -> {len(hits)} unique patients matched on PYP/amyloid procedure codes.")
    return hits


def count_cardiology_visits(cohort_pat_ids):
    """Counts outpatient visits to a cardiology department per patient in the FINAL
    cohort — a diagnostic-PROCESS metric (was the patient actually seen by the right
    specialty), not a health outcome. This is Domain A/B (diagnostic completeness),
    distinct from outcomes comparisons that are out of scope here. Applied to the
    WHOLE candidate pool deliberately — restricting to confirmed-only patients would
    exclude exactly the incomplete-workup population this metric is meant to find.

    NOTE (simple version): this counts total cardiology visits per patient, not yet
    "visits AFTER first ATTR signal with no resolution since" — that more precise
    version needs a normalized first-signal date tracked across DX/med/PYP/genetic
    signals, which isn't computed yet. Logged as a follow-up refinement."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Outpatient_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Counting cardiology department visits from CarDS_2837199_Outpatient_Enc.txt ...")
    CARDIOLOGY_PATTERN = re.compile(r"cardiology", re.IGNORECASE)
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        dept_col = find_col(chunk.columns, "DEPARTMENT_NAME")
        if id_col is None or dept_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, dept in zip(chunk[id_col], chunk[dept_col]):
            if isinstance(dept, str) and CARDIOLOGY_PATTERN.search(dept):
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def count_true_hospitalizations(cohort_pat_ids):
    """DESCRIPTIVE ONLY — counts TRUE inpatient hospitalizations (INP_YN='Y', not
    ED-only visits) per patient in the FINAL cohort, for basic cohort-characteristics
    reporting (a standard 'Table 1' style statistic — total cohort burden). This is
    NOT a comparison by care-gap status, diagnostic completeness, or treatment
    status — that kind of outcomes analysis is explicitly out of scope here."""
    counts = {}
    file_path = os.path.join(RAW_DIR, "CarDS_2837199_Hosp_Enc.txt")
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return counts
    print(f"  Counting true hospitalizations (INP_YN=1/Y) from CarDS_2837199_Hosp_Enc.txt ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        inp_col = find_col(chunk.columns, "INP_YN")
        if id_col is None or inp_col is None:
            continue
        chunk = chunk[chunk[id_col].isin(cohort_pat_ids)]
        for pat_id, is_inp in zip(chunk[id_col], chunk[inp_col]):
            # Real data uses '1'/'0' (confirmed via direct inspection), not 'Y'/'N'
            # as originally assumed — handle both formats defensively.
            if str(is_inp).strip().upper() in ("Y", "1", "1.0", "TRUE"):
                counts[pat_id] = counts.get(pat_id, 0) + 1
    return counts


def scan_family_hx_red_flags(inspect=False):
    """SUPPLEMENTARY, NON-QUALIFYING signal: family history of cardiomyopathy or
    sudden cardiac death — a recognized red flag for hereditary ATTR-CM (many older
    relatives with undiagnosed ATTR died before amyloidosis was ever suspected,
    showing up in family history only as vague 'cardiomyopathy'/'sudden death').

    DELIBERATELY NOT added to all_pat_ids / the qualifying-signal union: these terms
    have many causes unrelated to TTR (sarcomeric HCM genes, arrhythmia syndromes,
    coronary disease), so using this alone to ADD patients to the cohort would repeat
    the exact nonspecificity mistake already caught and fixed twice tonight (the AL
    ICD code, the TTN/MYH7/BRCA gene-panel conflation). This is descriptive context
    for patients who ALREADY qualify via a real ATTR-specific signal, not a
    qualifying signal on its own. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    file_path = os.path.join(RAW_DIR, FILES.get("family_hx", "CarDS_2837199_Family_Hx.txt"))
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return hits
    print(f"  Scanning CarDS_2837199_Family_Hx.txt ...")
    FAMILY_HX_PATTERN = re.compile(r"sudden death|sudden cardiac death|cardiomyopathy", re.IGNORECASE)
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        hx_col = find_col(chunk.columns, "MEDICAL_HX")
        relation_col = find_col(chunk.columns, "RELATION")
        age_col = find_col(chunk.columns, "AGE_OF_ONSET")

        for _, row in chunk.iterrows():
            hx_val = row.get(hx_col, "") if hx_col else ""
            if not isinstance(hx_val, str) or not FAMILY_HX_PATTERN.search(hx_val):
                continue

            pat_id = row.get(id_col, "")
            relation = row.get(relation_col, "") if relation_col else ""
            age = row.get(age_col, "") if age_col else ""
            evidence = f"{hx_val} (relation={relation}, age_of_onset={age})"

            if inspect and sample_shown < 15:
                print(f"    MATCH  FAMILY_HX={evidence!r}")
                sample_shown += 1

            entry = hits.setdefault(pat_id, {"family_hx_evidence": set()})
            entry["family_hx_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients with a family history cardiac red flag "
          f"(supplementary only — not used to qualify new cohort members).")
    return hits


def scan_al_confirmatory_labs(inspect=False):
    """Signal: confirmatory AL (light chain) amyloidosis lab workup — free light
    chain assay, SPEP/UPEP with immunofixation, bone marrow biopsy. Stronger evidence
    of true AL than a DX code alone, which can reflect a presumptive or later-reversed
    diagnosis. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    lab_keys = ("hosp_labs_1", "hosp_labs_2", "hosp_labs_3", "hosp_labs_4",
                "hosp_labs_5", "hosp_labs_6", "outpt_labs_1", "outpt_labs_2", "outpt_labs_3")
    for key in lab_keys:
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            text_cols = [c for c in chunk.columns if c.upper() in
                         ("PROC_NAME", "COMPONENT_NAME", "BASE_NAME", "DESCRIPTION", "DISPLAY_NAME")]
            abnormal_col = find_col(chunk.columns, "ABNORMAL_YN")

            for _, row in chunk.iterrows():
                matched_text = None
                is_nonspecific = False
                for tc in text_cols:
                    val = row.get(tc, "")
                    if not isinstance(val, str):
                        continue
                    if AL_LAB_KEYWORDS.search(val):
                        matched_text = val
                        is_nonspecific = False
                        break
                    if AL_LAB_NONSPECIFIC_KEYWORDS.search(val):
                        matched_text = val
                        is_nonspecific = True
                        # keep looking in case a more specific field also matches
                if not matched_text:
                    continue

                abnormal = row.get(abnormal_col, "") if abnormal_col else ""

                # Nonspecific tests (bare SPEP/total protein) only count as AL evidence
                # when actually flagged abnormal — a normal routine SPEP is not
                # meaningful AL evidence and was previously dominating this signal.
                if is_nonspecific and str(abnormal).strip().upper() != "Y":
                    continue

                pat_id = row.get(id_col, "")
                evidence = f"{matched_text} (abnormal={abnormal})" if abnormal else matched_text

                if inspect and sample_shown < 15:
                    print(f"    MATCH  AL_LAB={evidence!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "al_lab_evidence": set(),
                })
                entry["al_lab_evidence"].add(evidence)
    print(f"  -> {len(hits)} unique patients matched on AL confirmatory lab workup.")
    return hits


def scan_genetic_testing_file(inspect=False):
    """Signal 3: TTR genetic testing orders. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    file_path = os.path.join(RAW_DIR, FILES["genetic_testing"])
    if not os.path.exists(file_path):
        print(f"  [SKIP] Missing file: {file_path}")
        return hits
    print(f"  Scanning {FILES['genetic_testing']} ...")
    for chunk in iter_chunks(file_path):
        id_col = find_col(chunk.columns, "PAT_ID")
        mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
        text_cols = [c for c in chunk.columns if c.upper() in
                     ("PROC_NAME", "ORDER_DISPLAY_NAME", "DESCRIPTION", "DISPLAY_NAME")]

        for _, row in chunk.iterrows():
            matched_text = None
            for tc in text_cols:
                val = row.get(tc, "")
                if isinstance(val, str) and GENETIC_TEST_KEYWORDS.search(val):
                    matched_text = val
                    break
            if not matched_text:
                continue

            pat_id = row.get(id_col, "")
            if inspect and sample_shown < 15:
                print(f"    MATCH  PROC={matched_text!r}")
                sample_shown += 1

            entry = hits.setdefault(pat_id, {
                "PAT_MRN_ID": row.get(mrn_col, ""),
                "genetic_order_evidence": set(),
            })
            entry["genetic_order_evidence"].add(matched_text)
    print(f"  -> {len(hits)} unique patients matched on TTR genetic testing orders.")
    return hits


def scan_imaging_files(inspect=False):
    """Signal 4: PYP/amyloid imaging orders. Returns dict PAT_ID -> evidence dict."""
    hits = {}
    sample_shown = 0
    for key in ("imaging_1", "imaging_2"):
        file_path = os.path.join(RAW_DIR, FILES[key])
        if not os.path.exists(file_path):
            print(f"  [SKIP] Missing file: {file_path}")
            continue
        print(f"  Scanning {FILES[key]} ...")
        for chunk in iter_chunks(file_path):
            id_col = find_col(chunk.columns, "PAT_ID")
            mrn_col = find_col(chunk.columns, "PAT_MRN_ID") or id_col
            proc_col = find_col(chunk.columns, "PROC_NAME")

            for _, row in chunk.iterrows():
                proc_name = row.get(proc_col, "") if proc_col else ""
                if not isinstance(proc_name, str) or not IMAGING_KEYWORDS.search(proc_name):
                    continue
                if IMAGING_EXCLUDE_PATTERN.search(proc_name):
                    continue  # brain/neuro amyloid PET — different clinical context entirely

                pat_id = row.get(id_col, "")
                if inspect and sample_shown < 15:
                    print(f"    MATCH  PROC_NAME={proc_name!r}")
                    sample_shown += 1

                entry = hits.setdefault(pat_id, {
                    "PAT_MRN_ID": row.get(mrn_col, ""),
                    "imaging_evidence": set(),
                })
                entry["imaging_evidence"].add(proc_name)
    print(f"  -> {len(hits)} unique patients matched on PYP/amyloid imaging orders.")
    return hits


def main():
    parser = argparse.ArgumentParser(description="Identify ATTR-CM candidate cohort from structured data")
    parser.add_argument("--inspect", action="store_true",
                         help="Print sample matches per signal without writing an output file")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("ATTR-CM COHORT IDENTIFICATION [STRUCTURED DATA]")
    print(f"Directory: {RAW_DIR}")
    print("=" * 70)

    print("\n[1/7] Diagnosis codes")
    dx_hits = scan_dx_files(inspect=args.inspect)

    print("\n[2/7] Medication orders")
    med_hits, diflunisal_hits = scan_med_files(inspect=args.inspect)

    print("\n[3/7] Medication administration records")
    med_admin_hits = scan_med_admin_files(inspect=args.inspect)

    print("\n[4/7] Genetic testing orders")
    genetic_hits = scan_genetic_testing_file(inspect=args.inspect)

    print("\n[5/7] PYP/amyloid imaging orders")
    imaging_hits = scan_imaging_files(inspect=args.inspect)

    print("\n[6/7] PYP/amyloid procedure codes (ICD_PX)")
    icd_px_hits = scan_icd_px_files(inspect=args.inspect)

    print("\n[7/7] AL confirmatory lab workup (free light chains, SPEP/UPEP, bone marrow biopsy)")
    al_lab_hits = scan_al_confirmatory_labs(inspect=args.inspect)

    print("\n[bonus] Family history cardiac red flags (supplementary, non-qualifying)")
    family_hx_hits = scan_family_hx_red_flags(inspect=args.inspect)

    if args.inspect:
        print("\n[INSPECT MODE] No output file written. Review the sample matches above,")
        print("then adjust the keyword/code lists near the top of this script if needed.")
        return

    # ── Merge all signals into one cohort table ──
    all_pat_ids = (set(dx_hits) | set(med_hits) | set(med_admin_hits) | set(genetic_hits)
                   | set(imaging_hits) | set(icd_px_hits))
    print(f"\nMerging signals: {len(all_pat_ids)} unique candidate patients total.")

    rows = []
    for pat_id in all_pat_ids:
        dx = dx_hits.get(pat_id, {})
        med = med_hits.get(pat_id, {})
        med_admin = med_admin_hits.get(pat_id, {})
        gen = genetic_hits.get(pat_id, {})
        img = imaging_hits.get(pat_id, {})
        icd_px = icd_px_hits.get(pat_id, {})
        al_lab = al_lab_hits.get(pat_id, {})  # lookup only — doesn't expand candidate pool,
                                                # since AL lab evidence is evidence AGAINST
                                                # ATTR-CM, not a qualifying signal for it
        family_hx = family_hx_hits.get(pat_id, {})  # lookup only — supplementary, non-qualifying
        diflunisal = diflunisal_hits.get(pat_id, {})  # lookup only — low-confidence, non-qualifying

        mrn = (dx.get("PAT_MRN_ID") or med.get("PAT_MRN_ID") or med_admin.get("PAT_MRN_ID")
               or gen.get("PAT_MRN_ID") or img.get("PAT_MRN_ID") or icd_px.get("PAT_MRN_ID") or "")

        rows.append({
            "PAT_ID": pat_id,
            "PAT_MRN_ID": mrn,
            "flag_dx_attr_specific": dx.get("dx_attr_specific", False),
            "flag_dx_al_amyloid_only": dx.get("dx_al_amyloid_only", False),
            "flag_al_lab_workup": bool(al_lab),
            "flag_dx_amyloid_family": dx.get("dx_amyloid_family", False),
            "flag_dx_name_match": dx.get("dx_name_match", False),
            "flag_medication_order": bool(med),
            "flag_medication_admin": bool(med_admin),
            "flag_genetic_test_order": bool(gen),
            "flag_pyp_imaging": bool(img),
            "flag_icd_px": bool(icd_px),
            "dx_evidence": "; ".join(dx.get("dx_evidence", [])),
            "med_order_evidence": "; ".join(med.get("med_evidence", [])),
            "med_admin_evidence": "; ".join(med_admin.get("med_admin_evidence", [])),
            "genetic_order_evidence": "; ".join(gen.get("genetic_order_evidence", [])),
            "imaging_evidence": "; ".join(img.get("imaging_evidence", [])),
            "icd_px_evidence": "; ".join(icd_px.get("icd_px_evidence", [])),
            "al_lab_evidence": "; ".join(al_lab.get("al_lab_evidence", [])),
            "flag_family_hx_cardiac_red_flag": bool(family_hx),
            "family_hx_evidence": "; ".join(family_hx.get("family_hx_evidence", [])),
            "flag_diflunisal_only": bool(diflunisal),
            "diflunisal_evidence": "; ".join(diflunisal.get("diflunisal_evidence", [])),
        })

    cohort_df = pd.DataFrame(rows)

    # Optionally merge demographics
    patients_path = os.path.join(RAW_DIR, FILES["patients"])
    if os.path.exists(patients_path):
        print("Merging demographics from Patients.txt ...")
        pat_df = pd.read_csv(
            patients_path, sep=sniff_delimiter(patients_path), dtype=str,
            on_bad_lines="skip", engine="python", quoting=3,
        )
        cohort_df = cohort_df.merge(pat_df, on=["PAT_ID", "PAT_MRN_ID"], how="left")

    # Descriptive-only cohort characteristics (mortality, hospitalization burden) —
    # NOT split by care-gap/diagnostic-completeness status. That kind of comparative
    # outcomes analysis sits in a different project domain and is out of scope here.
    print("\nComputing descriptive cohort characteristics (mortality, hospitalizations)...")
    print("NOTE: whole-cohort description only — NOT compared by care-gap status.")
    hosp_counts = count_true_hospitalizations(set(cohort_df["PAT_ID"]))
    cohort_df["num_true_hospitalizations"] = cohort_df["PAT_ID"].map(hosp_counts).fillna(0).astype(int)

    cardiology_counts = count_cardiology_visits(set(cohort_df["PAT_ID"]))
    cohort_df["num_cardiology_visits"] = cohort_df["PAT_ID"].map(cardiology_counts).fillna(0).astype(int)

    out_path = os.path.join(OUTPUT_DIR, "attr_cm_candidate_cohort.csv")
    cohort_df.to_csv(out_path, index=False)

    signal_cols = ["flag_dx_attr_specific", "flag_dx_amyloid_family", "flag_dx_name_match",
                   "flag_medication_order", "flag_medication_admin", "flag_genetic_test_order",
                   "flag_pyp_imaging", "flag_icd_px"]

    # AL-flagged-at-all vs AL-only-with-NOTHING-else — these are different numbers and
    # were a real source of confusion before: a patient can have one AL-coded encounter
    # AND independent real ATTR evidence elsewhere, and should stay in the cohort.
    # Combined AL evidence: DX code/name OR confirmatory lab workup — either is
    # sufficient to flag AL, since DX alone can be presumptive/later-reversed and
    # lab workup alone can occur without a corresponding DX code being entered.
    cohort_df["flag_al_any_evidence"] = cohort_df["flag_dx_al_amyloid_only"] | cohort_df["flag_al_lab_workup"]
    al_flagged_total = cohort_df["flag_al_any_evidence"].sum()
    al_only_pure = cohort_df[
        (cohort_df[signal_cols].sum(axis=1) == 0) & (cohort_df["flag_al_any_evidence"] == True)
    ]
    n_al_only_pure = len(al_only_pure)
    true_cohort_size = len(cohort_df) - n_al_only_pure

    # ── Summary / cohort-flow numbers for your methods section ──
    print(f"\n{'='*70}")
    print("COHORT IDENTIFICATION COMPLETE")
    print(f"Total candidate patients (raw, before AL exclusion): {len(cohort_df)}")
    print(f"  Matched ATTR-specific DX (E85.4/E85.82): {cohort_df['flag_dx_attr_specific'].sum()}")
    print(f"  AL flagged by DX code/name: {cohort_df['flag_dx_al_amyloid_only'].sum()}")
    print(f"  AL flagged by confirmatory LAB workup (free light chains/SPEP/UPEP/bone marrow): {cohort_df['flag_al_lab_workup'].sum()}")
    print(f"  AL flagged by EITHER signal (combined): {al_flagged_total}")
    print(f"  AL-ONLY patients (no other ATTR signal anywhere) — TRUE EXCLUSIONS: {n_al_only_pure}")
    print(f"  ({al_flagged_total - n_al_only_pure} AL-flagged patients ALSO have independent ATTR "
          f"evidence elsewhere and correctly remain in the cohort — mixed/miscoded records, not exclusions)")
    print(f"  Matched broader amyloid family DX/DX_NAME: {(cohort_df['flag_dx_amyloid_family'] | cohort_df['flag_dx_name_match']).sum()}")
    print(f"  Matched ATTR-CM medication order: {cohort_df['flag_medication_order'].sum()}")
    print(f"  Matched ATTR-CM medication administration: {cohort_df['flag_medication_admin'].sum()}")
    print(f"  Matched TTR genetic testing order: {cohort_df['flag_genetic_test_order'].sum()}")
    print(f"  Matched PYP/amyloid imaging order: {cohort_df['flag_pyp_imaging'].sum()}")
    print(f"  Matched PYP/amyloid procedure code (ICD_PX): {cohort_df['flag_icd_px'].sum()}")
    print(f"  ⏳ PENDING: PYP CPT code (Hosp_Enc_CPT.txt/Outpatient_Enc_CPT.txt) — NOT scanned "
          f"yet, real institutional code unconfirmed. See TODO block at top of this script.")
    only_one_signal = (cohort_df[signal_cols].sum(axis=1) == 1).sum()
    print(f"  Patients matched on exactly ONE signal (worth spot-checking): {only_one_signal}")
    n_family_hx = cohort_df["flag_family_hx_cardiac_red_flag"].sum()
    print(f"  (Supplementary, non-qualifying) Family history of cardiomyopathy/sudden "
          f"cardiac death, among cohort members: {n_family_hx}")
    print(f"\n  >>> TRUE ATTR-CM CANDIDATE COHORT SIZE (after AL exclusion): {true_cohort_size} <<<")

    if "DEATH_DATE" in cohort_df.columns:
        n_deceased = cohort_df["DEATH_DATE"].notna().sum()
        pct_deceased = (n_deceased / len(cohort_df) * 100) if len(cohort_df) > 0 else 0
        print(f"\n--- Descriptive cohort characteristics (whole cohort, NOT split by care-gap status) ---")
        print(f"  Deceased (DEATH_DATE present): {n_deceased} ({pct_deceased:.1f}%)")
    if "num_true_hospitalizations" in cohort_df.columns:
        hosp_series = cohort_df["num_true_hospitalizations"]
        print(f"  True inpatient hospitalizations per patient — "
              f"median: {hosp_series.median():.0f}, mean: {hosp_series.mean():.2f}, "
              f"0 hospitalizations: {(hosp_series == 0).sum()} patients")
    if "num_cardiology_visits" in cohort_df.columns:
        card_series = cohort_df["num_cardiology_visits"]
        n_zero_cardiology = (card_series == 0).sum()
        print(f"  Cardiology visits per patient — median: {card_series.median():.0f}, "
              f"mean: {card_series.mean():.2f}")
        print(f"  ⚠ ZERO cardiology visits despite an ATTR-CM signal: {n_zero_cardiology} patients "
              f"— worth checking whether these were seen elsewhere or never evaluated at all")

    print(f"Saved: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


'''