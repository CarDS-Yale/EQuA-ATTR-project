


#!/usr/bin/env python3
"""
PYP Scan Result Interpretation Script
========================================
Reads Imaging_1.txt / Imaging_2.txt, isolates actual PYP/amyloid cardiac
scintigraphy studies (most rows in these files are unrelated imaging studies), and
uses an LLM to interpret the scan RESULT from ORDER_NARRATIVE / ORDER_IMPRESSION
text — positive / negative / equivocal / not-yet-read — plus the Perugini visual
grade (0-3) and heart-to-contralateral (H/CL) ratio when reported.

This is the piece that lets you classify your candidate pool into confirmed-positive,
confirmed-negative (fully ruled out), borderline/equivocal, and incomplete-workup —
symmetric diagnostic-completeness assessment regardless of the eventual finding.

Clinical interpretation criteria used (standard, published cardiac amyloid imaging
criteria — Perugini grading; Gillmore et al. 2016 positivity criteria):
  - Grade 2 or 3 myocardial uptake, OR H/CL ratio > 1.5  -> positive
  - Grade 0 uptake, explicit "no amyloid"                -> negative
  - Grade 1 uptake (mild, present but < bone)             -> equivocal/borderline
    (Grade 1 is explicitly NONDIAGNOSTIC per Gillmore criteria — NOT negative,
    despite being low uptake. Confirmed against Gillmore et al. 2016 Circulation
    and multiple subsequent validation studies.)
  - Cancelled, not performed, report not finalized         -> not_yet_read

Usage:
  python3 equa_attr_extract_pyp_results.py --test 5
  python3 equa_attr_extract_pyp_results.py --workers 6
  python3 equa_attr_extract_pyp_results.py --resume
"""

import os
import re
import sys
import json
import time
import argparse
import threading
import pandas as pd
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Configuration ──
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"

RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)
OUTPUT_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results")

IMAGING_FILES = ["CarDS_2837199_Imaging_1.txt", "CarDS_2837199_Imaging_2.txt"]

MAX_TEXT_CHARS = 4000
CHECKPOINT_EVERY = 500

# ── Pre-filter: isolate actual PYP/amyloid scans from all other imaging studies ──
PYP_SCAN_FILTER = re.compile(
    r"(pyrophosphate|\bPYP\b|technetium.{0,20}pyrophosphate|\bTc-?99m\b.{0,20}PYP|"
    r"cardiac amyloid|amyloid.{0,20}scintigraphy|amyloid.{0,20}imaging)",
    re.IGNORECASE,
)
# Brain/neuro amyloid PET (Alzheimer's/dementia workup) is a different clinical
# context entirely — exclude it even if the order name otherwise matches above.
PYP_EXCLUDE_PATTERN = re.compile(r"\bbrain\b|\bneuro\b|\balzheimer", re.IGNORECASE)

SYSTEM_PROMPT = """You are a clinical data extraction assistant specializing in cardiac amyloidosis nuclear imaging (technetium pyrophosphate / PYP scintigraphy). Your job is to interpret the RESULT of a PYP scan from the report text.

You must return ONLY valid JSON — no explanation, no markdown backticks, no extra text.

OUTPUT SCHEMA:
{
  "pyp_result": "positive" / "negative" / "equivocal_borderline" / "indeterminate_technical" / "not_yet_read",
  "perugini_grade": "0" / "1" / "2" / "3" / "not specified",
  "h_cl_ratio": "numeric value if reported (e.g. 1.6) or not specified",
  "determination_basis": "explicit_statement" / "visual_grade" / "h_cl_ratio_only" / "combined",
  "evidence_text": "EXACT verbatim sentence(s) from the report supporting the result",
  "clinical_summary": "one sentence summary of the finding"
}

DETERMINATION_BASIS — this matters for review, be honest about it:
- "explicit_statement": the report directly states positive/negative/equivocal in words (most reliable)
- "visual_grade": your result is based on a stated Perugini grade (0-3) with no explicit positive/negative wording
- "h_cl_ratio_only": your result is based SOLELY on a numeric H/CL ratio, with no visual grade and no explicit statement — ratio-only determinations are the LEAST reliable of the three (technique/scanner-dependent, no universal cutoff), flag this basis honestly so these get prioritized for manual review
- "combined": more than one of the above agree

CLINICAL INTERPRETATION CRITERIA (standard published criteria — Gillmore et al. 2016, Circulation — apply these consistently):
- Grade 2 or 3 myocardial uptake (visually equal to or greater than adjacent rib/bone) → "positive"
- Grade 0 uptake (no myocardial uptake, normal bone uptake only), or an explicit statement of "no evidence of cardiac amyloidosis" / "no significant myocardial uptake" → "negative"
- Grade 1 uptake (mild, present but LESS than bone) → "equivocal_borderline" — this is NOT negative. Grade 1 is explicitly nondiagnostic per the standard criteria and requires further evaluation; do not classify it as negative even though uptake is low.
- H/CL (heart-to-contralateral) ratio: interpretation is time-point-dependent (1-hour vs 3-hour imaging use different equivocal ranges) and less reliable than visual grading alone — report the ratio value regardless of your interpretation, but do not treat a ratio alone as strong evidence without corroborating visual grade or explicit statement
- Any other equivocal, indeterminate, or conflicting language in the report → "equivocal_borderline"
- Study cancelled, not performed, or report not yet finalized → "not_yet_read"
- Do NOT confuse normal background blood pool activity (nonspecific, expected) with actual myocardial radiotracer uptake (the finding that matters)
- If a specific Perugini grade (0-3) is stated, extract it exactly as given
- The evidence_text MUST be copied verbatim from the report

Return ONLY the JSON object."""

USER_PROMPT_TEMPLATE = """Interpret this PYP/cardiac amyloid scan report:

---REPORT START---
{report_text}
---REPORT END---

Return ONLY valid JSON."""

write_lock = threading.Lock()


def call_ollama(report_text, model, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(report_text=report_text)},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 512},
                },
                timeout=120
            )
            resp.raise_for_status()
            content = resp.json()["message"]["content"].strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            return json.loads(content)
        except (json.JSONDecodeError, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(2)

    return {
        "pyp_result": "error",
        "perugini_grade": "error",
        "h_cl_ratio": "error",
        "evidence_text": "",
        "clinical_summary": "Extraction failed",
    }


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def load_dataframe(file_path):
    delim = sniff_delimiter(file_path)
    return pd.read_csv(file_path, sep=delim, dtype=str, on_bad_lines="skip", engine="python", quoting=3)


def get_unique_path(path):
    """If path already exists, append _2, _3, etc. before the extension until an
    unused filename is found — used only in NON-resume runs, since --resume
    specifically means 'keep appending to the existing file'."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    n = 2
    while True:
        candidate = f"{base}_{n}{ext}"
        if not os.path.exists(candidate):
            return candidate
        n += 1


def load_done_ids(done_path):
    if not os.path.exists(done_path):
        return set()
    with open(done_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def append_done_id(done_path, order_id):
    with open(done_path, "a") as f:
        f.write(f"{order_id}\n")


def flush_results(buffer, out_path):
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)


def process_scan(args_tuple, model):
    pat_mrn, order_id, source_file, report_text = args_tuple
    if not report_text or report_text == "nan":
        return None
    text_for_llm = report_text[:MAX_TEXT_CHARS] if len(report_text) > MAX_TEXT_CHARS else report_text
    res = call_ollama(text_for_llm, model=model)
    return {
        "PAT_MRN_ID": pat_mrn,
        "ORDER_ID": order_id,
        "SOURCE_FILE": source_file,
        "pyp_result": res.get("pyp_result", "error"),
        "perugini_grade": res.get("perugini_grade", "not specified"),
        "h_cl_ratio": res.get("h_cl_ratio", "not specified"),
        "determination_basis": res.get("determination_basis", "not specified"),
        "evidence_text": res.get("evidence_text", ""),
        "clinical_summary": res.get("clinical_summary", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Interpret PYP/cardiac amyloid scan results")
    parser.add_argument("--test", type=int, default=0, help="Test on first N matched scans")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--files", type=str, default=None,
                         help="Comma-separated exact filenames to restrict to")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_files = IMAGING_FILES
    if args.files:
        requested = [f.strip() for f in args.files.split(",") if f.strip()]
        target_files = [f for f in target_files if f in requested]

    out_latest = os.path.join(OUTPUT_DIR, "attr_pyp_results_latest.csv")
    done_path = os.path.join(OUTPUT_DIR, "attr_pyp_results.done")
    if not args.resume:
        # Fresh run (not continuing a previous one) — don't silently overwrite an
        # existing output/ledger pair from an earlier run.
        out_latest = get_unique_path(out_latest)
        done_path = get_unique_path(done_path)
    done_ids = load_done_ids(done_path) if args.resume else set()

    print("=" * 70)
    print("PYP SCAN RESULT INTERPRETATION")
    print(f"Directory: {RAW_DIR}")
    print(f"Model: {args.model} | Ollama URL: {OLLAMA_URL} | Workers: {args.workers}")
    if args.resume:
        print(f"Resume mode: {len(done_ids)} scans already completed, will be skipped")
    print("=" * 70)

    total_seen = 0
    total_matched = 0
    total_sent = 0
    result_buffer = []
    t0 = time.time()

    for file_name in target_files:
        file_path = os.path.join(RAW_DIR, file_name)
        if not os.path.exists(file_path):
            print(f"\n[SKIP] Missing file: {file_path}")
            continue

        print(f"\nProcessing File: {file_name}")
        df = load_dataframe(file_path)
        print(f"  Loaded {len(df)} imaging orders (all study types).")

        id_col = next((c for c in df.columns if c.upper() == "PAT_MRN_ID"), df.columns[0])
        order_id_col = next((c for c in df.columns if c.upper() == "ORDER_ID"), "ORDER_ID")
        narrative_col = next((c for c in df.columns if c.upper() == "ORDER_NARRATIVE"), None)
        impression_col = next((c for c in df.columns if c.upper() == "ORDER_IMPRESSION"), None)
        proc_col = next((c for c in df.columns if c.upper() == "PROC_NAME"), None)

        work_items = []
        for i, row in df.iterrows():
            total_seen += 1
            proc_name = str(row.get(proc_col, "")) if proc_col else ""
            if not PYP_SCAN_FILTER.search(proc_name):
                continue  # not an actual PYP/amyloid scan — skip, don't waste LLM time
            if PYP_EXCLUDE_PATTERN.search(proc_name):
                continue  # brain/neuro amyloid PET — different clinical context
            total_matched += 1

            narrative = str(row.get(narrative_col, "")) if narrative_col else ""
            impression = str(row.get(impression_col, "")) if impression_col else ""
            combined_text = " ".join(t for t in [narrative, impression] if t and t != "nan")

            pat_mrn = row.get(id_col, "")
            order_id = str(row.get(order_id_col, f"row_{i}"))

            if args.resume and order_id in done_ids:
                continue

            work_items.append((pat_mrn, order_id, file_name, combined_text))

        if args.test > 0:
            work_items = work_items[:args.test]
            print(f"  [TEST MODE] Processing first {args.test} matched scans.")

        print(f"  {total_matched} rows matched PYP/amyloid scan filter (out of {len(df)} total imaging orders).")
        print(f"  {len(work_items)} scans queued for LLM interpretation.")

        processed_count = 0
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_scan, item, args.model): item[1] for item in work_items}
                for future in as_completed(futures):
                    order_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  [ERROR] order {order_id}: {e}")
                        result = None
                    with write_lock:
                        processed_count += 1
                        total_sent += 1
                        if result is not None:
                            result_buffer.append(result)
                        append_done_id(done_path, order_id)
                        if len(result_buffer) >= CHECKPOINT_EVERY:
                            flush_results(result_buffer, out_latest)
                            result_buffer = []
                        if processed_count % 50 == 0 or processed_count == len(work_items):
                            print(f"  Processed {processed_count}/{len(work_items)} scans...")
        else:
            for item in work_items:
                order_id = item[1]
                result = process_scan(item, args.model)
                processed_count += 1
                total_sent += 1
                if result is not None:
                    result_buffer.append(result)
                append_done_id(done_path, order_id)
                if len(result_buffer) >= CHECKPOINT_EVERY:
                    flush_results(result_buffer, out_latest)
                    result_buffer = []
                if processed_count % 50 == 0 or processed_count == len(work_items):
                    print(f"  Processed {processed_count}/{len(work_items)} scans...")

    flush_results(result_buffer, out_latest)

    print(f"\n{'='*70}")
    print("PYP INTERPRETATION COMPLETE")
    print(f"Total imaging orders seen (all types): {total_seen}")
    print(f"Matched as real PYP/amyloid scans: {total_matched}")
    print(f"Sent to LLM: {total_sent}")
    print(f"Time Taken: {(time.time()-t0)/60:.2f} minutes")
    print(f"Saved: {out_latest}")
    if os.path.exists(out_latest):
        try:
            results_df = pd.read_csv(out_latest, dtype=str)
            print("\nSummary Breakdown (pyp_result):")
            print(results_df["pyp_result"].value_counts().to_string())
        except Exception as e:
            print(f"  (could not summarize output: {e})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()