



#!/usr/bin/env python3
"""
ATTR-CM Genetics Extraction Summary Script
============================================
Reads the LLM extraction output CSV (whatever has been produced SO FAR — this works
fine on a partial/in-progress run, you don't need to wait for the full corpus to
finish) and produces patient-level counts:

  - How many unique patients had at least one note trigger a keyword hit
  - Of those, how many had genetic testing documented (has_genetic_testing)
  - Of those, how many were positive (variant) vs negative (wild-type) vs unclear
  - Breakdown of specific variants found
  - Patients with CONFLICTING signals across their notes (flagged for manual review —
    e.g. one note says wild-type, another says variant for the same patient)
  - Patients where extraction errored (worth re-running just those, later)

This can be re-run at any time against the in-progress output file — run it now,
run it again tomorrow after more of the corpus has processed, the numbers just
update to reflect whatever's been done so far.

Usage:
  python3 equa_attr_summarize_results.py
  python3 equa_attr_summarize_results.py --input /path/to/attr_genetics_raw_latest.csv
"""

import os
import argparse
import pandas as pd


def get_unique_path(path):
    """If path already exists, append _2, _3, etc. before the extension until an
    unused filename is found — so re-running this script never silently overwrites
    a previous run's rollup output."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1

DEFAULT_INPUT = os.path.expanduser(
    "~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_genetics_raw_latest.csv"
)
DEFAULT_OUTPUT_DIR = os.path.expanduser(
    "~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
)


import re

# Genuine TTR-specific language — used to flag variant_mutation/evidence_text values
# that DON'T match this pattern, which strongly suggests the extraction picked up an
# unrelated gene (TTN, MYH7, BRCA, PKP2, DSP, MUTYH, etc. have all been observed
# contaminating results despite the gene-specificity prompt fix). This is a
# deterministic text check on already-extracted data — no LLM re-run needed.
TTR_SPECIFIC_PATTERN = re.compile(
    r"\bTTR\b|transthyretin|\bV142I\b|Val142Ile|\bV122I\b|Val122Ile|\bV30M\b|Val30Met|"
    r"\bT60A\b|Thr60Ala|hATTR|wtATTR|ATTRv\b|ATTRwt\b|p\.Val142|p\.V142",
    re.IGNORECASE,
)

# Scheduling/pending language — used together with TTR-specific or general genetic-test
# language to catch patients whose TTR genetic testing was referred/ordered but has no
# result yet in the notes. Current attr_type/has_genetic_testing logic correctly
# reports these as "unclear" or "not_documented" (no result exists), but that hides a
# real, meaningful care-gap signal: someone was suspected enough to order testing.
SCHEDULING_PATTERN = re.compile(
    r"\b(scheduled|ordered|pending|referred for|referral for|to r/o|to rule out|"
    r"plan(?:ned)? for|will undergo|awaiting|upcoming|to be (?:scheduled|arranged))\b",
    re.IGNORECASE,
)
GENERAL_GENETIC_TEST_PATTERN = re.compile(r"genetic test|genetic screen|genetic counsel", re.IGNORECASE)

# Known non-TTR genes commonly seen in cardiomyopathy/neuromuscular gene PANELS
# (e.g. Invitae cardiomyopathy panel tests dozens of these simultaneously) — real
# examples observed contaminating extraction results despite the gene-specificity
# prompt fix. An explicit match here is STRONGER confirmation of conflation than
# merely "no TTR language found" — it names the actual unrelated gene responsible.
NON_TTR_GENE_PATTERN = re.compile(
    r"\b(TTN|MYH6|MYH7|LMNA|DSP|PKP2|RYR1|RYR2|FLNC|MYBPC3|SCN5A|SCN9A|ANO5|CAPN3|"
    r"DMD|VCP|FBN1|GLA|COL5A1|COL6A1|COL6A2|COL6A3|MYLK2|NEBL|FKTN|TMEM43|PRKAG2|"
    r"SMAD3|HCN4|MVK|ATM|MUTYH|BRCA1|BRCA2|C9orf72|MTHFR|TNNT2|TNNI3|ACTA2|LDLR|"
    r"CDKN2A|PALB2|CHEK2|HOXB13|NRAS|DICER1|NF1|PTEN|MLH1|MSH2|MSH6|PMS2|APC|"
    r"BARD1|BRIP1|BMPR1A|CDH1|CDK4|EPCAM|GREM1|MRE11A|NBN|POLD1|POLE|RAD50|"
    r"RAD51C|RAD51D|SMARCA4|STK11|TP53)\b",
    re.IGNORECASE,
)


def is_ttr_specific(variant_text, evidence_text):
    """Check whether the variant/evidence text actually mentions TTR-specific
    language, rather than an unrelated gene that got conflated during extraction."""
    combined = f"{variant_text or ''} {evidence_text or ''}"
    return bool(TTR_SPECIFIC_PATTERN.search(combined))


def find_non_ttr_gene(variant_text, evidence_text):
    """Returns the specific non-TTR gene name found (or None) — stronger, named
    confirmation of conflation than just 'no TTR language found'."""
    combined = f"{variant_text or ''} {evidence_text or ''}"
    match = NON_TTR_GENE_PATTERN.search(combined)
    return match.group(0).upper() if match else None


# Catches a specific, real error type: the model linking two DIFFERENT variant
# positions as if equivalent (e.g. "Val122Ile (also known as p.V142I)" — these are
# two distinct real ATTR variants, position 122 vs 142, not the same finding).
VARIANT_POSITION_PATTERN = re.compile(
    r"(?:val|v)\.?\s?(\d{1,4})|p\.(?:val)?(\d{1,4})", re.IGNORECASE
)
EQUIVALENCE_LANGUAGE_PATTERN = re.compile(r"also known as|\baka\b|i\.e\.|same as|equivalent to", re.IGNORECASE)


def check_variant_position_inconsistency(variant_text):
    """Returns (has_inconsistency, distinct_positions) — True if the text links
    multiple DIFFERENT variant position numbers together with equivalence language,
    suggesting the model conflated two distinct variants as if they were one."""
    variant_text = "" if pd.isna(variant_text) else str(variant_text)
    if not variant_text or not EQUIVALENCE_LANGUAGE_PATTERN.search(variant_text):
        return False, []
    positions = set()
    for match in VARIANT_POSITION_PATTERN.finditer(variant_text):
        pos = match.group(1) or match.group(2)
        if pos:
            positions.add(pos)
    if len(positions) > 1:
        return True, sorted(positions)
    return False, []


def is_testing_pending(evidence_text, clinical_summary):
    """Check whether a note describes TTR/genetic testing as scheduled/ordered but
    NOT YET RESULTED — a real care-gap signal that current has_genetic_testing/
    attr_type logic correctly reports as unclear/not_documented (since no result
    exists yet), but which should still be surfaced separately rather than treated
    the same as a patient with no genetics workup happening at all."""
    combined = f"{evidence_text or ''} {clinical_summary or ''}"
    has_scheduling_language = bool(SCHEDULING_PATTERN.search(combined))
    has_genetic_language = bool(TTR_SPECIFIC_PATTERN.search(combined)) or bool(GENERAL_GENETIC_TEST_PATTERN.search(combined))
    return has_scheduling_language and has_genetic_language


def classify_patient(sub_df):
    """Given all extracted rows (notes) for ONE patient, decide their overall
    genetic status. Distinguishes GENETICALLY CONFIRMED status from CLINICALLY
    PRESUMED status (e.g. a patient labeled "wild-type ATTR" from a PYP scan alone,
    who was never actually TTR gene-tested) — these are different, both-important
    categories for a care-gap analysis, not the same thing.
    Returns (overall_status, variant_list, has_conflict, has_error)."""
    attr_types = sub_df["attr_type"].dropna().astype(str).str.strip().str.lower()

    # type_basis may not exist in older output (before this field was added) — treat
    # missing as unknown basis rather than erroring
    if "type_basis" in sub_df.columns:
        basis = sub_df["type_basis"].fillna("").astype(str).str.strip().str.lower()
    else:
        basis = pd.Series([""] * len(sub_df), index=sub_df.index)

    has_genetic_flags = sub_df["has_genetic_testing"].astype(str).str.lower()

    has_error = (attr_types == "error").any()

    def type_with_basis(type_name, basis_name):
        mask = (attr_types == type_name) & (basis == basis_name)
        return mask.any()

    variant_genetic = type_with_basis("variant", "genetic")
    variant_clinical = type_with_basis("variant", "clinical_or_imaging")
    wild_genetic = type_with_basis("wild_type", "genetic")
    wild_clinical = type_with_basis("wild_type", "clinical_or_imaging")
    has_variant_any = (attr_types == "variant").any()
    has_wild_any = (attr_types == "wild_type").any()
    has_unclear = (attr_types == "unclear").any()
    has_ruled_out = (attr_types == "ruled_out").any()
    any_tested = has_genetic_flags.isin(["true", "1", "yes"]).any()

    # Conflict: patient has BOTH a wild-type note AND a variant note — needs a human look
    has_conflict = has_variant_any and has_wild_any

    if has_conflict:
        overall = "CONFLICTING (needs review)"
    elif variant_genetic:
        overall = "variant — GENETICALLY CONFIRMED"
    elif variant_clinical:
        overall = "variant — clinically presumed, NOT genetically confirmed"
    elif has_variant_any:
        overall = "variant — basis unknown"
    elif wild_genetic:
        overall = "wild_type — GENETICALLY CONFIRMED"
    elif wild_clinical:
        overall = "wild_type — clinically presumed, NOT genetically confirmed"
    elif has_wild_any:
        overall = "wild_type — basis unknown"
    elif has_ruled_out:
        overall = "ruled_out — TTR CONFIRMED NEGATIVE (via imaging rule-out)"
    elif has_unclear:
        overall = "unclear"
    elif any_tested:
        overall = "tested_status_unclear"
    else:
        overall = "no_type_documented_in_flagged_notes"

    variants = sorted(set(
        str(v).strip() for v in sub_df["variant_mutation"].dropna()
        if str(v).strip().lower() not in ("", "nan", "not specified", "error")
    ))

    return overall, variants, has_conflict, has_error


def main():
    parser = argparse.ArgumentParser(description="Summarize ATTR-CM genetics extraction results so far")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT,
                         help="Path to the extraction output CSV (attr_genetics_raw_latest.csv)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                         help="Where to write the patient-level rollup CSV")
    parser.add_argument("--cohort-file", type=str, default=None,
                         help="Path to attr_cm_candidate_cohort.csv from equa_attr_identify_cohort.py. "
                              "When given, cross-checks each genetic-testing-positive patient against "
                              "independent structured signals (DX code, PYP scan, meds). A patient with "
                              "has_genetic_testing=true but ZERO corroborating cohort signal is flagged "
                              "for manual review — this is exactly the pattern seen in known conflation "
                              "errors (e.g. an unrelated gene panel like MUTYH or ThyroSeq being "
                              "misattributed to TTR status).")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        print("Nothing has been written yet, or the path is wrong — check the extraction run's log.")
        return

    df = pd.read_csv(args.input, dtype=str)
    print("=" * 70)
    print("ATTR-CM GENETICS EXTRACTION — SUMMARY (based on what's processed so far)")
    print(f"Input: {args.input}")
    print("=" * 70)

    print(f"\nTotal extracted note-rows so far: {len(df)}")

    mrn_col = "PAT_MRN_ID" if "PAT_MRN_ID" in df.columns else df.columns[0]
    unique_patients = df[mrn_col].nunique()
    print(f"Unique patients with at least one keyword-hit note processed: {unique_patients}")

    # Load cohort file for cross-referencing, if given
    corroborated_ids = None
    if args.cohort_file:
        if not os.path.exists(args.cohort_file):
            print(f"[WARNING] --cohort-file {args.cohort_file} not found; skipping cross-check.")
        else:
            cohort_df = pd.read_csv(args.cohort_file, dtype=str)
            signal_cols = [c for c in cohort_df.columns if c.startswith("flag_") and c != "flag_dx_al_amyloid_only"]
            # A patient "corroborates" if they match ANY real ATTR signal (not the AL-exclude flag)
            has_any_signal = cohort_df[signal_cols].apply(
                lambda col: col.astype(str).str.lower().isin(["true", "1"])
            ).any(axis=1)
            corroborated_ids = set(cohort_df.loc[has_any_signal, "PAT_MRN_ID"].dropna())
            print(f"Loaded cohort file: {len(corroborated_ids)} patients with an independent corroborating signal.")

    # Per-patient rollup
    rollup_rows = []
    for mrn, sub in df.groupby(mrn_col):
        overall, variants, has_conflict, has_error = classify_patient(sub)
        is_genetic_positive_status = overall.startswith("variant") or overall.startswith("wild_type")

        # Check whether ANY of this patient's positive-status rows actually mention
        # TTR-specific language — if none do, this is very likely a gene-conflation
        # error (TTN/MYH7/BRCA/etc. mistakenly attributed to ATTR-CM status), not a
        # real TTR finding.
        ttr_language_confirmed = False
        detected_non_ttr_gene = None
        if is_genetic_positive_status:
            positive_rows = sub[sub["attr_type"].astype(str).str.lower().isin(["variant", "wild_type"])]
            for _, r in positive_rows.iterrows():
                if is_ttr_specific(r.get("variant_mutation", ""), r.get("evidence_text", "")):
                    ttr_language_confirmed = True
                    break
            if not ttr_language_confirmed:
                for _, r in positive_rows.iterrows():
                    gene = find_non_ttr_gene(r.get("variant_mutation", ""), r.get("evidence_text", ""))
                    if gene:
                        detected_non_ttr_gene = gene
                        break

        # Even among TTR-confirmed rows, check for a specific error pattern: two
        # DIFFERENT variant positions linked as if equivalent (e.g. "Val122Ile
        # (also known as p.V142I)" — these are different real ATTR variants).
        position_inconsistency = False
        inconsistent_positions = []
        if ttr_language_confirmed:
            for _, r in positive_rows.iterrows():
                has_issue, positions = check_variant_position_inconsistency(r.get("variant_mutation", ""))
                if has_issue:
                    position_inconsistency = True
                    inconsistent_positions = positions
                    break

        # For patients WITHOUT a positive genetic status, check whether any of their
        # notes describe TTR testing as scheduled/ordered but not yet resulted — a
        # real care-gap signal currently hidden inside "unclear"/"no_type_documented".
        testing_pending = False
        if not is_genetic_positive_status:
            for _, r in sub.iterrows():
                if is_testing_pending(r.get("evidence_text", ""), r.get("clinical_summary", "")):
                    testing_pending = True
                    break
            if testing_pending:
                overall = f"{overall} — TESTING SCHEDULED/PENDING, no result yet"

        row = {
            "PAT_MRN_ID": mrn,
            "num_flagged_notes": len(sub),
            "overall_genetic_status": overall,
            "variants_found": "; ".join(variants) if variants else "",
            "has_conflicting_notes": has_conflict,
            "had_extraction_error": has_error,
            "genetic_testing_pending_no_result": testing_pending,
            # Always initialize these with safe, neutral defaults — regardless of
            # is_genetic_positive_status — so no key is ever missing across rows.
            # A previous version only set these INSIDE the positive-status branch,
            # which left the key absent for non-positive patients; pandas then
            # filled those with NaN (not empty string), and a later `!= ""` check
            # incorrectly counted every NaN as "matched", producing a nonsensical
            # conflation count far exceeding the total patient count.
            "ttr_language_confirmed": False,
            "likely_wrong_gene_conflation": False,
            "confirmed_conflation_gene": "",
            "conflation_confidence": "n/a",
            "variant_position_inconsistency_flag": False,
            "inconsistent_positions": "",
        }

        if is_genetic_positive_status:
            row["ttr_language_confirmed"] = ttr_language_confirmed
            row["likely_wrong_gene_conflation"] = not ttr_language_confirmed
            row["confirmed_conflation_gene"] = detected_non_ttr_gene or ""
            row["conflation_confidence"] = (
                "n/a" if ttr_language_confirmed
                else ("CONFIRMED — named gene: " + detected_non_ttr_gene if detected_non_ttr_gene
                      else "AMBIGUOUS — no TTR language, no recognized non-TTR gene either")
            )
            row["variant_position_inconsistency_flag"] = position_inconsistency
            row["inconsistent_positions"] = "; ".join(inconsistent_positions) if inconsistent_positions else ""

        if corroborated_ids is not None:
            corroborated = mrn in corroborated_ids
            row["corroborated_by_cohort_signal"] = corroborated
            row["needs_manual_review_possible_false_positive"] = (
                is_genetic_positive_status and not corroborated and ttr_language_confirmed
            )

        rollup_rows.append(row)
    rollup_df = pd.DataFrame(rollup_rows)

    out_path = get_unique_path(os.path.join(args.output_dir, "attr_genetics_patient_rollup.csv"))
    rollup_df.to_csv(out_path, index=False)

    # ── Console summary — the actual numbers for your draft ──
    status_counts = rollup_df["overall_genetic_status"].value_counts()
    print("\n--- Patient-level genetic status breakdown ---")
    print(status_counts.to_string())

    n_conflict = rollup_df["has_conflicting_notes"].sum()
    n_error = rollup_df["had_extraction_error"].sum()
    print(f"\nPatients with conflicting notes (wild-type AND variant both documented): {n_conflict}")
    print(f"Patients with at least one extraction error (LLM call failed, worth re-running): {n_error}")

    if "likely_wrong_gene_conflation" in rollup_df.columns:
        n_conflation = rollup_df["likely_wrong_gene_conflation"].sum()
        n_ttr_confirmed = rollup_df["ttr_language_confirmed"].sum()
        n_named_conflation = (rollup_df["confirmed_conflation_gene"] != "").sum()
        n_ambiguous = n_conflation - n_named_conflation
        print(f"\n🧬 TTR-specificity check (text-based, no LLM re-run needed):")
        print(f"  Genetic-positive patients with CONFIRMED TTR-specific language: {n_ttr_confirmed}")
        print(f"  Genetic-positive patients, NO TTR language found: {n_conflation}")
        print(f"    -> CONFIRMED conflation (specific non-TTR gene named, e.g. TTN/MYH7/LMNA): {n_named_conflation}")
        print(f"    -> AMBIGUOUS (no TTR language AND no recognized non-TTR gene either): {n_ambiguous}")
        if n_named_conflation > 0:
            print(f"\n  Most common conflated genes:")
            gene_counts = rollup_df.loc[rollup_df["confirmed_conflation_gene"] != "", "confirmed_conflation_gene"].value_counts()
            print("  " + gene_counts.head(10).to_string().replace("\n", "\n  "))

    if "variant_position_inconsistency_flag" in rollup_df.columns:
        n_inconsistent = rollup_df["variant_position_inconsistency_flag"].sum()
        if n_inconsistent > 0:
            print(f"\n⚠ Patients with a variant-position naming inconsistency "
                  f"(two DIFFERENT positions linked as equivalent, e.g. "
                  f"'Val122Ile also known as p.V142I') — verify manually: {n_inconsistent}")
        print("  Treat the conflation group as excluded/needs-review by default; only the")
        print("  TTR-confirmed group should be trusted as real ATTR-CM genetic findings.")

    n_pending = rollup_df["genetic_testing_pending_no_result"].sum()
    print(f"\n📋 TTR genetic testing SCHEDULED/ORDERED but not yet resulted "
          f"(hidden inside unclear/not_documented until now): {n_pending}")
    print("  This is a real care-gap signal on its own — clinical suspicion was documented")
    print("  enough to order testing. Worth tracking separately from 'no evidence at all'.")

    if corroborated_ids is not None:
        n_flagged = rollup_df["needs_manual_review_possible_false_positive"].sum()
        print(f"\n⚠ Of the TTR-confirmed patients, those with NO corroborating cohort signal "
              f"(no DX code, no PYP scan, no meds) — review these too, could be asymptomatic "
              f"carriers OR still-uncaught errors: {n_flagged}")

    # Variant breakdown across all genetically-positive patients (any basis)
    all_variants = []
    for v in rollup_df.loc[rollup_df["overall_genetic_status"].str.startswith("variant"), "variants_found"]:
        if v:
            all_variants.extend([x.strip() for x in v.split(";") if x.strip()])
    if all_variants:
        print("\n--- Specific variants found (counts) ---")
        print(pd.Series(all_variants).value_counts().to_string())
    else:
        print("\nNo specific variant names extracted yet among positive patients.")

    print(f"\nPatient-level rollup saved to: {out_path}")
    print("Re-run this script anytime — it just re-reads whatever's in the input CSV,")
    print("so the numbers update as more of the corpus finishes processing.")
    print("=" * 70)


if __name__ == "__main__":
    main()