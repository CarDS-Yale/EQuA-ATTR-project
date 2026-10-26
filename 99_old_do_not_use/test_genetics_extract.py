#!/usr/bin/env python3
"""
FH Genetic Testing Result Extraction Agent
===========================================
Extracts structured genetic testing results from clinical notes
authored by Arya Mani using a local LLM via Ollama.

Input:  arya_mani_notes.csv (tab-delimited, all 1128 notes)
Output: genetic_results_extracted.csv — one row per gene-test-result per patient

Usage:
    python3 extract_genetic_results.py
    python3 extract_genetic_results.py --fh-only        # only FH-relevant notes
    python3 extract_genetic_results.py --test 5          # test on first 5 notes
"""

import os
import sys
import re
import json
import time
import argparse
import pandas as pd
import requests
from datetime import datetime

# ── Configuration ──
OLLAMA_URL = "http://localhost:11435"
DEFAULT_MODEL = "llama3.1"

INPUT_FILE = "/home/sa2496/saabucket/FH March 23 2026 Data/arya_mani_notes.csv"
OUTPUT_DIR = "/home/sa2496/FH agents"





MAX_NOTE_CHARS = 8000

# ── Negation / Hypothetical Patterns ──
NEGATION_WINDOW = 80  # characters before the gene mention to scan

NEGATION_PATTERNS = re.compile(
    r"(not\s+tested|no\s+testing|no\s+mutation|no\s+variant|"
    r"declined\s+testing|deferred\s+testing|did\s+not\s+undergo|"
    r"was\s+not\s+tested|has\s+not\s+been\s+tested|"
    r"not\s+sent|not\s+ordered|pending|"
    r"consider\s+testing|would\s+recommend\s+testing|"
    r"should\s+be\s+tested|plan\s+to\s+test|will\s+order|"
    r"recommend\s+genetic|offered\s+genetic|"
    r"if\s+positive|if\s+negative|if\s+found)",
    re.IGNORECASE
)

# ── Extraction Prompt ──
SYSTEM_PROMPT = """You are a clinical data extraction assistant. Your job is to extract 
genetic testing results from clinical notes written by a cardiovascular genetics expert.

You must return ONLY valid JSON — no explanation, no markdown, no extra text.

EXTRACTION METHOD — follow these steps IN ORDER for each genetic test result:
1. FIRST, find the exact sentence(s) in the note that describe a genetic test result.
2. THEN, extract the structured fields ONLY from what is explicitly stated in those sentences.
3. If a field's value is not explicitly stated in the evidence text, use "not specified".

For each genetic test mentioned in the note, extract these fields IN THIS ORDER:
- evidence_text: FIRST, copy the EXACT sentence(s) from the note that contain the genetic result. This must be a verbatim quote. Include enough context to verify the extraction. All subsequent fields must be derivable from this text.
- gene: the gene name (e.g., LDLR, APOB, PCSK9, LDLRAP1, SCN5A, etc.)
- tested_individual: "patient", "mother", "father", "son", "daughter", "sibling", "grandfather", "grandmother", or "other relative"
- variant: the specific variant if mentioned in the evidence text. Look for protein-level variants like G465E, R3527Q, D1771N, N730fs, and rs numbers like rs5742904. Only use "not specified" if truly no variant is stated.
- notation: HGVS or transcript notation if given (e.g., "NM_000384:c.G9811A:p.G3271S") or "not specified"
- zygosity: "heterozygous", "homozygous", "compound heterozygous", or "not specified"
- classification: MUST be exactly one of: "pathogenic", "likely pathogenic", "VUS", "likely benign", "benign", "negative", or "not specified".
- result_summary: one sentence summarizing the finding in plain English
- fh_relevant: true if this gene is relevant to familial hypercholesterolemia (LDLR, APOB, PCSK9, LDLRAP1, LPA), false otherwise

CLASSIFICATION RULES — READ CAREFULLY:
- ONLY assign "pathogenic" if the note EXPLICITLY uses the word "pathogenic" OR says the variant is "disease-causing", "disease causing", "causative", or "known to cause disease" for this specific variant.- ONLY assign "likely pathogenic" if the note EXPLICITLY says "likely pathogenic" for this specific variant.
- ONLY assign "VUS" if the note EXPLICITLY says "VUS", "variant of uncertain significance", or "VUSES".
- ONLY assign "likely benign" or "benign" if the note EXPLICITLY uses those terms.
- ONLY assign "negative" if the note says testing was negative or no mutation was found.
- If the note simply mentions a variant by name WITHOUT stating its classification, you MUST use "not specified". 
- DO NOT infer pathogenicity from clinical context, family history, or the clinician's tone.
- DO NOT assume a variant is pathogenic just because it is being discussed in a genetics clinic.
- "Referred for evaluation of [variant]" does NOT imply the variant is pathogenic — use "not specified".
- "Found to have variants in [gene]" does NOT imply pathogenic — use "not specified".
- When in doubt, ALWAYS use "not specified". It is better to under-classify than to over-classify.

GENE vs LAB/DRUG RULES:
- "ApoB" or "APOB" followed by terms like "elevated", "level", "mg/dl", "slightly elevated", "normal" refers to the APOLIPOPROTEIN B LAB TEST, NOT the APOB gene. DO NOT extract this as a genetic result.
- "PCSK9 inhibitor", "on PCSK9", "PCSK9i", "Repatha", "evolocumab", "alirocumab" refers to MEDICATION, NOT the PCSK9 gene. DO NOT extract this as a genetic result.
- Only extract APOB or PCSK9 as genetic results when there is a clear genetic testing context (variant name, mutation, sequencing, panel, genetic test result).

If no genetic testing results are found in the note, return:
{"genetic_tests": [], "has_genetic_results": false}

If genetic testing results ARE found, return:
{"genetic_tests": [<list of results>], "has_genetic_results": true, "clinical_context": "<brief description of why testing was done>"}

CRITICAL RULES:
- ONLY extract genetic results that are explicitly mentioned in the note text.
- DO NOT extract results that are planned, pending, recommended, or hypothetical.
- DO NOT infer, guess, or fabricate any results.
- Every gene you report MUST appear by name in the note.
- Every variant you report MUST appear in the note.
- Every classification you report MUST be explicitly stated in the note, or else use "not specified".
- If you are unsure about a result, DO NOT include it.
- It is better to miss a result than to fabricate one.
- The evidence_text MUST be copied verbatim from the note. Do not reword it.

Return ONLY the JSON object. No preamble, no markdown backticks, no explanation."""

USER_PROMPT_TEMPLATE = """Extract all genetic testing results from this clinical note:

---NOTE START---
{note_text}
---NOTE END---

Return ONLY valid JSON."""


def call_ollama(note_text, model, max_retries=3):
    """Send note to Ollama and get structured extraction."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                            note_text=note_text
                        )},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.05,
                        "num_predict": 2048,
                    }
                },
                timeout=180
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"]

            # Clean up common LLM output issues
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            result = json.loads(content)
            return result

        except json.JSONDecodeError as e:
            print(f"\n    JSON parse error (attempt {attempt+1}): {e}")
            print(f"    Raw response: {content[:500]}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"\n    Request error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
        except Exception as e:
            print(f"\n    Unexpected error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)

    return {"genetic_tests": [], "has_genetic_results": False, "error": "extraction_failed"}


def check_negation_context(gene, note_text):
    """
    Check whether a gene mention appears in a negated or hypothetical context.
    Returns True if the mention looks negated/hypothetical (i.e. should be BLOCKED).
    Returns False if the mention looks like a real result (i.e. should be KEPT).
    """
    note_upper = note_text.upper()
    gene_upper = gene.upper()

    # Find all positions where the gene is mentioned
    positions = []
    start = 0
    while True:
        idx = note_upper.find(gene_upper, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        return True  # gene not found at all — block

    # For each mention, check the surrounding window for negation
    negated_count = 0
    for pos in positions:
        window_start = max(0, pos - NEGATION_WINDOW)
        window_end = min(len(note_text), pos + len(gene) + NEGATION_WINDOW)
        window = note_text[window_start:window_end]

        if NEGATION_PATTERNS.search(window):
            negated_count += 1

    # If ALL mentions are negated/hypothetical, block the result
    # If at least one mention is NOT negated, keep it
    return negated_count == len(positions)


def verify_evidence_text(evidence_text, note_text, threshold=0.6):
    """
    Check whether the evidence_text actually appears in the note.
    Uses substring matching first, then falls back to word-overlap ratio.
    Returns (is_verified, match_score).
    """
    if not evidence_text or evidence_text.strip() == "":
        return False, 0.0

    evidence_clean = evidence_text.strip()
    note_clean = note_text.strip()

    # Exact substring match (case-insensitive)
    if evidence_clean.lower() in note_clean.lower():
        return True, 1.0

    # Word overlap ratio as fallback
    # (LLM sometimes adds/drops a word or changes punctuation)
    evidence_words = set(evidence_clean.lower().split())
    note_words = set(note_clean.lower().split())

    if not evidence_words:
        return False, 0.0

    overlap = len(evidence_words & note_words)
    score = overlap / len(evidence_words)

    return score >= threshold, score


def filter_fh_relevant(df):
    """Filter to FH-relevant notes."""
    upper = df["NOTE_TEXT"].fillna("").str.upper()
    fh_genes = upper.str.contains("LDLR|APOB|PCSK9|LDLRAP1", na=False)
    fh_context = upper.str.contains(
        "HYPERCHOLESTEROL|FAMILIAL|HIGH CHOLESTEROL|LDL|LIPID CLINIC", na=False
    )
    result_terms = upper.str.contains(
        "PATHOGENIC|VUS|VARIANT|BENIGN|NEGATIVE|POSITIVE|MUTATION", na=False
    )
    return df.loc[fh_genes | (fh_context & result_terms)].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Extract genetic results from notes")
    parser.add_argument("--fh-only", action="store_true",
                        help="Only process FH-relevant notes (~195)")
    parser.add_argument("--test", type=int, default=0,
                        help="Test on first N notes only")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--input", type=str, default=INPUT_FILE,
                        help="Input file path")
    args = parser.parse_args()

    model = args.model

    # ── Load notes ──
    print("=" * 70)
    print("FH GENETIC TESTING RESULT EXTRACTION AGENT")
    print(f"Model: {model} | Ollama: {OLLAMA_URL}")
    print("=" * 70)

    print(f"\nLoading notes from: {args.input}")

    # Count raw lines first to detect dropped rows
    with open(args.input, "r", errors="replace") as f:
        raw_line_count = sum(1 for _ in f) - 1  # minus header

    df = pd.read_csv(args.input, sep="\t", dtype=str, on_bad_lines="skip")
    skipped_lines = raw_line_count - len(df)
    print(f"  Raw lines in file: {raw_line_count}")
    print(f"  Successfully loaded: {len(df)}")
    if skipped_lines > 0:
        print(f"  *** WARNING: {skipped_lines} lines skipped due to parsing errors ***")
    print(f"  Unique patients: {df['PAT_MRN_ID'].nunique()}")

    if args.fh_only:
        df = filter_fh_relevant(df)
        print(f"  After FH filter: {len(df)} notes, {df['PAT_MRN_ID'].nunique()} patients")

    if args.test > 0:
        df = df.head(args.test)
        print(f"  Test mode: processing first {args.test} notes")

    # ── Check Ollama connectivity ──
    print(f"\nChecking Ollama at {OLLAMA_URL}...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in r.json().get("models", [])]
        print(f"  Connected. Available models: {models}")
        if not any(model in m for m in models):
            print(f"  WARNING: {model} not found in available models!")
    except Exception as e:
        print(f"  ERROR: Cannot connect to Ollama: {e}")
        print("  Make sure Ollama is running. Exiting.")
        sys.exit(1)

    # ── Process notes ──
    print(f"\nProcessing {len(df)} notes...")
    all_results = []
    notes_with_results = 0
    truncated_notes = 0
    negation_blocked = 0
    evidence_failed = 0
    hallucination_blocked = 0
    pcsk9_drug_blocked = 0
    apob_lab_blocked = 0
    t0 = time.time()

    for i, row in df.iterrows():
        note_text = row.get("NOTE_TEXT", "")
        if not note_text or pd.isna(note_text):
            continue

        pat_mrn = row.get("PAT_MRN_ID", "")
        note_id = row.get("NOTE_ID", "")
        note_date = row.get("NOTE_CREATED_TIME", "")
        note_type = row.get("NOTE_TYPE", "")

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = (len(df) - i - 1) / rate / 60 if rate > 0 else 0
        print(f"  [{i+1}/{len(df)}] MRN={pat_mrn} | {note_type} | "
              f"{elapsed:.0f}s elapsed, ~{remaining:.1f}min remaining", end="")

        # Handle truncation with logging
        if len(note_text) > MAX_NOTE_CHARS:
            truncated_notes += 1
            print(f" [TRUNCATED {len(note_text)}→{MAX_NOTE_CHARS} chars]", end="")
            note_text_for_llm = note_text[:MAX_NOTE_CHARS]
        else:
            note_text_for_llm = note_text

        result = call_ollama(note_text_for_llm, model=model)

        # ── Post-extraction verification pipeline ──
        if result.get("has_genetic_results"):
            note_upper = note_text.upper()
            verified = []

            for test in result.get("genetic_tests", []):
                gene = test.get("gene", "").upper()

                # CHECK 1: Gene name actually in note
                if not gene or gene not in note_upper:
                    hallucination_blocked += 1
                    print(f" [HALLUCINATION: {gene}]", end="")
                    continue

                # CHECK 2: PCSK9 drug vs gene
                if gene == "PCSK9":
                    pcsk9_drug_terms = [
                        "PCSK9 INHIBIT", "PCSK9I", "ON PCSK9",
                        "REPATHA", "EVOLOCUMAB", "ALIROCUMAB", "PRALUENT"
                    ]
                    has_drug_mention = any(t in note_upper for t in pcsk9_drug_terms)
                    has_gene_mention = any(t in note_upper for t in [
                        "PCSK9 GENE", "PCSK9 MUTATION", "PCSK9 VARIANT",
                        "PCSK9 TEST", "PCSK9 PATHOGENIC", "PCSK9 VUS",
                        "PCSK9 BENIGN", "PCSK9 LIKELY"
                    ])
                    if has_drug_mention and not has_gene_mention:
                        pcsk9_drug_blocked += 1
                        print(f" [PCSK9 DRUG: BLOCKED]", end="")
                        continue

                # CHECK 2b: APOB lab value vs gene
                if gene == "APOB":
                    apob_lab_terms = [
                        "APOB LEVEL", "APOB IS", "APOB WAS",
                        "APOB ELEVATED", "APOB SLIGHTLY", "APOB NORMAL",
                        "APOB OF ", "APOB MG", "APOLIPOPROTEIN B",
                        "APOB HIGH", "APOB LOW"
                    ]
                    has_lab_mention = any(t in note_upper for t in apob_lab_terms)
                    has_gene_mention = any(t in note_upper for t in [
                        "APOB GENE", "APOB MUTATION", "APOB VARIANT",
                        "APOB TEST", "APOB PATHOGENIC", "APOB VUS",
                        "APOB BENIGN", "APOB LIKELY", "APOB P.", "APOB C.",
                        "APOB R3527", "APOB HETEROZYGOUS", "APOB HOMOZYGOUS"
                    ])
                    if has_lab_mention and not has_gene_mention:
                        apob_lab_blocked += 1
                        print(f" [APOB LAB: BLOCKED]", end="")
                        continue

                # CHECK 3: Negation / hypothetical context
                if check_negation_context(gene, note_text):
                    negation_blocked += 1
                    print(f" [NEGATED: {gene}]", end="")
                    continue

                # CHECK 4: Evidence text verification
                evidence = test.get("evidence_text", "")
                is_verified, match_score = verify_evidence_text(evidence, note_text)
                if not is_verified:
                    evidence_failed += 1
                    print(f" [EVIDENCE MISMATCH {gene} score={match_score:.2f}]", end="")
                    # Don't hard-block — flag it but keep the result
                    test["evidence_verified"] = False
                    test["evidence_match_score"] = round(match_score, 2)
                else:
                    test["evidence_verified"] = True
                    test["evidence_match_score"] = round(match_score, 2)

                verified.append(test)

            result["genetic_tests"] = verified
            if not verified:
                result["has_genetic_results"] = False

        if result.get("has_genetic_results"):
            n_tests = len(result.get("genetic_tests", []))
            notes_with_results += 1
            print(f" → {n_tests} result(s) found")

            for test in result.get("genetic_tests", []):
                fh_genes = ["LDLR", "APOB", "PCSK9", "LDLRAP1", "LPA"]
                gene_upper = test.get("gene", "").upper()
                test["fh_relevant"] = gene_upper in fh_genes


                all_results.append({
                    "PAT_MRN_ID": pat_mrn,
                    "NOTE_ID": note_id,
                    "NOTE_DATE": note_date,
                    "NOTE_TYPE": note_type,
                    "gene": test.get("gene", ""),
                    "tested_individual": test.get("tested_individual", "patient"),
                    "variant": test.get("variant", ""),
                    "notation": test.get("notation", ""),
                    "zygosity": test.get("zygosity", ""),
                    "classification": test.get("classification", ""),
                    "result_summary": test.get("result_summary", ""),
                    "evidence_text": test.get("evidence_text", ""),
                    "evidence_verified": test.get("evidence_verified", False),
                    "evidence_match_score": test.get("evidence_match_score", 0.0),
                    "fh_relevant": test.get("fh_relevant", False),
                    "clinical_context": result.get("clinical_context", ""),
                })
        else:
            print(f" → no genetic results")

    # ── Save ──
    elapsed_total = time.time() - t0
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"  Time: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"  Notes processed: {len(df)}")
    print(f"  Notes with genetic results: {notes_with_results}")
    print(f"  Total gene-level results: {len(all_results)}")
    print(f"\n  Verification stats:")
    print(f"    Notes truncated: {truncated_notes}")
    print(f"    Hallucinations blocked (gene not in note): {hallucination_blocked}")
    print(f"    PCSK9 drug mentions blocked: {pcsk9_drug_blocked}")
    print(f"    Negation/hypothetical blocked: {negation_blocked}")
    print(f"    Evidence text mismatches (flagged, not blocked): {evidence_failed}")
    print(f"    APOB lab mentions blocked: {apob_lab_blocked}")

    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save full results
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(OUTPUT_DIR, f"genetic_results_extracted_{ts}.csv")
        results_df.to_csv(out_csv, index=False)
        print(f"\n  Full results: {out_csv}")

        # Also save a stable filename for easy access
        out_latest = os.path.join(OUTPUT_DIR, "genetic_results_extracted.csv")
        results_df.to_csv(out_latest, index=False)
        print(f"  Latest copy:  {out_latest}")

        # Summary
        print(f"\n  Results summary:")
        print(f"    Unique patients with results: {results_df['PAT_MRN_ID'].nunique()}")
        print(f"    FH-relevant results: {results_df['fh_relevant'].sum()}")
        print(f"    Evidence verified: {results_df['evidence_verified'].sum()}/{len(results_df)}")
        print(f"\n    Genes found:")
        print(results_df["gene"].value_counts().to_string(header=False))
        print(f"\n    Classifications:")
        print(results_df["classification"].value_counts().to_string(header=False))
    else:
        print("\n  No genetic results extracted.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()