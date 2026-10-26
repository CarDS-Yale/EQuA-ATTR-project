

#This is older version of script but it processed file 1 in the mnt files - CarDS_2837199_Hosp_Enc_Notes.txt. New script is in
#same dir: new_equa_attr_extract_genetics.py. It is more efficient and has more features. This script is kept for reference only.









#!/usr/bin/env python3
"""
Dual-Source ATTR-CM Genetic Status Extraction Script (v2)
===========================================================
Processes either regex-filtered CSVs or raw source .txt notes files using Qwen2.5 via Ollama.

New in this version:
  - Keyword pre-filter: skips the LLM entirely for notes with no TTR/genetic-testing
    language, so you're not burning LLM time on millions of irrelevant rows.
  - Checkpointing: results are appended to disk every CHECKPOINT_EVERY notes, and a
    "done" ledger of processed NOTE_IDs lets a restarted run skip completed work
    instead of reprocessing everything from scratch.
  - Parallel Ollama calls: a thread pool sends concurrent requests to Ollama. This only
    helps if Ollama itself is configured to serve requests in parallel (see NOTES below).

Usage:
  python3 equa_attr_extract_genetics.py --source regex --test 5
  python3 equa_attr_extract_genetics.py --source raw --test 5
  python3 equa_attr_extract_genetics.py --source raw --workers 4
  python3 equa_attr_extract_genetics.py --source raw --resume

NOTES ON PARALLELISM:
  A single Ollama server processes requests sequentially by default. To actually get
  concurrency benefit from --workers > 1, set these environment variables BEFORE
  starting the Ollama server (not in this script):
      export OLLAMA_NUM_PARALLEL=4
      export OLLAMA_MAX_LOADED_MODELS=1
  Then restart `ollama serve`. Without this, extra worker threads will just queue up
  behind Ollama's own sequential processing and you won't see a speedup. The real
  bottleneck is almost always GPU/CPU capacity for the model, not Python.
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

# Directories
REGEX_DIR = os.path.expanduser("~/work_tmp/equa_attr_preprocessing_results")
RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")
OUTPUT_DIR = os.path.join(REGEX_DIR, "equa_attr_ollama_results")

# File Mapping
REGEX_FILES = [
    "isolated_regex_CarDS_2837199_Hosp_Enc_Notes.csv",
    "isolated_regex_CarDS_2837199_Telephone_Enc_Notes.csv",
    "isolated_regex_CarDS_2837199_Outpatient_Enc_Notes_1.csv",
    "isolated_regex_CarDS_2837199_Outpatient_Enc_Notes_2.csv",
    "isolated_regex_CarDS_2837199_Outpatient_Enc_Notes_3.csv",
]

RAW_FILES = [
    "CarDS_2837199_Hosp_Enc_Notes.txt",
    "CarDS_2837199_Telephone_Enc_Notes.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_1.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_2.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_3.txt",
]

MAX_NOTE_CHARS = 8000
CHECKPOINT_EVERY = 500  # rows buffered before flushing to disk

# ── Keyword pre-filter ──
# Skip the LLM entirely for notes that don't even mention TTR / genetic testing /
# variant nomenclature. This is the single biggest speed win on a millions-of-rows
# corpus, since most notes won't be relevant at all.
# ── Keyword pre-filter ──
# Skip the LLM entirely for notes that don't even mention TTR / genetic testing /
# variant nomenclature. This is the single biggest speed win on a millions-of-rows
# corpus, since most notes won't be relevant at all.
#
# IMPORTANT: "amyloid" and drug names ALONE are deliberately excluded from the
# default trigger below. A patient already diagnosed with ATTR-CM has "amyloid" and
# their medication name in nearly every routine follow-up note, regardless of whether
# that specific note discusses genetic testing — including those as standalone
# triggers balloons the queue with notes that have almost no chance of containing new
# genetic information. This default requires GENETICS-SPECIFIC language instead,
# targeting the actual research question directly. Use --broad-filter to fall back to
# the old inclusive behavior (amyloid/drug names alone qualify) for a later,
# lower-priority exhaustive pass if you want maximum sensitivity at the cost of volume.
GENETIC_KEYWORDS = re.compile(
    r"("
    r"\bTTR\b|transthyretin|genetic test|genetic screen|genetic counsel|"
    r"wild.?type|wtATTR|hATTR|"
    r"hereditary amyloid|familial amyloid|familial amyloid polyneuropathy|\bFAP\b|"
    r"next.?generation sequencing|\bNGS\b|gene panel|sequencing (?:confirmed|revealed|showed)|"
    r"\bInvitae\b|\bAmbry\b|\bGeneDx\b|"
    r"\bp\.[A-Za-z]{3}\d{1,4}[A-Za-z]{3}\b|"          # p.Val142Ile style
    r"\b[A-Za-z]{3}\d{1,4}[A-Za-z]{3}\b|"              # Val142Ile style
    r"\b[A-Z]\d{1,4}[A-Z]\b"                           # V142I style
    r")",
    re.IGNORECASE,
)

# Old, broader filter: matches on generic "amyloid"/"ATTR-CM" mentions and DMT drug
# names ALONE too, with no genetics-specific language required. High recall, but
# floods the queue with routine follow-up notes. Only used with --broad-filter.
BROAD_GENETIC_KEYWORDS = re.compile(
    r"("
    r"\bTTR\b|transthyretin|genetic test|genetic screen|genetic counsel|"
    r"amyloid(?:osis)?|cardiac amyloidosis|ATTR(?:wt|v|-CM)?|wild.?type|wtATTR|hATTR|"
    r"hereditary amyloid|familial amyloid|familial amyloid polyneuropathy|\bFAP\b|"
    r"next.?generation sequencing|\bNGS\b|gene panel|sequencing (?:confirmed|revealed|showed)|"
    r"\bInvitae\b|\bAmbry\b|\bGeneDx\b|"
    r"tafamidis|vyndaqel|vyndamax|acoramidis|attruby|"
    r"patisiran|onpattro|vutrisiran|amvuttra|inotersen|tegsedi|diflunisal|"
    r"\bp\.[A-Za-z]{3}\d{1,4}[A-Za-z]{3}\b|"
    r"\b[A-Za-z]{3}\d{1,4}[A-Za-z]{3}\b|"
    r"\b[A-Z]\d{1,4}[A-Z]\b"
    r")",
    re.IGNORECASE,
)

# ── Prompt Schema ──
SYSTEM_PROMPT = """You are a clinical data extraction assistant specializing in cardiovascular disease and transthyretin amyloidosis (ATTR-CM). Your job is to analyze clinical notes and extract genetic testing results for TTR (transthyretin) mutations.

You must return ONLY valid JSON — no explanation, no markdown backticks, no extra text.

EXTRACTION RULES:
1. Determine if the note explicitly documents that GENETIC TESTING for TTR / ATTR-CM was performed.
2. If genetic testing was performed, determine if the disease is:
   - "wild_type" (wtATTR, non-hereditary, age-related, negative for TTR mutation, wild type)
   - "variant" (vATTR, hereditary, hATTR, mutant, positive for TTR mutation)
   - "unclear" (testing mentioned, but result not clearly stated)
3. If a specific variant or amino acid substitution is mentioned (e.g., V122I, Val122Ile, T60A, Thr60Ala, V30M, Val30Met, Ile122Phe, p.V142I), extract it.
4. Extract the exact sentence(s) as evidence_text.

OUTPUT SCHEMA:
Return a single JSON object formatted as follows:

{
  "has_genetic_testing": true / false / "unclear",
  "attr_type": "wild_type" / "variant" / "not_tested" / "unclear",
  "variant_mutation": "string (e.g. V122I, T60A) or not specified",
  "zygosity": "heterozygous" / "homozygous" / "not specified",
  "evidence_text": "EXACT verbatim sentence(s) from the note documenting genetic testing status or type",
  "clinical_summary": "one sentence summary of genetic status"
}

CRITICAL RULES:
- DO NOT confuse Pyrophosphate (PYP) scan results or Technetium scans with GENETIC testing. PYP scans confirm amyloid deposits, NOT genetic mutations.
- DO NOT classify drug treatment (e.g., Vyndaqel, Tafamidis, Patisiran, Inotersen) as genetic testing.
- If genetic testing was NOT performed, set "has_genetic_testing": false, "attr_type": "not_tested", "variant_mutation": "not specified".
- If the note mentions "wild type" or "wtATTR", the patient DOES NOT have a pathogenic TTR mutation.
- The evidence_text MUST be copied verbatim from the note.

Return ONLY the JSON object."""

USER_PROMPT_TEMPLATE = """Extract ATTR-CM genetic testing status from this clinical note:

---NOTE START---
{note_text}
---NOTE END---

Return ONLY valid JSON."""

# Thread-safe lock for file writes / print statements
write_lock = threading.Lock()


def call_ollama(note_text, model, max_retries=3):
    """Send note to Ollama API and return parsed JSON."""
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(note_text=note_text)},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 1024,
                    }
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
        "has_genetic_testing": "error",
        "attr_type": "error",
        "variant_mutation": "error",
        "evidence_text": "",
        "clinical_summary": "Extraction failed"
    }


def sniff_delimiter(file_path):
    """Detect the real delimiter by counting candidates on the header line.
    A try/except on read_csv is NOT reliable here: pd.read_csv(sep='|') on a
    tab-delimited file does not raise an error, it just silently returns a
    single mangled column. Counting characters on the raw header line first
    is the only safe way to pick the right delimiter."""
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def load_dataframe(file_path):
    """Load CSV or TXT file, auto-detecting the real delimiter."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, dtype=str, on_bad_lines="skip")
    else:
        delim = sniff_delimiter(file_path)
        # quoting=3 (QUOTE_NONE): these are raw Epic/Clarity-style exports, not
        # properly quoted CSVs, so stray quote characters in note text shouldn't
        # be treated as field quoting.
        return pd.read_csv(
            file_path, sep=delim, dtype=str, on_bad_lines="skip",
            engine="python", quoting=3
        )


def reconstruct_notes_from_lines(df):
    """Some raw note exports store ONE ROW PER LINE of a note (a 'LINE' column
    plus NOTE_ID), rather than one row per note. If not reconstructed, the
    keyword filter and LLM see disconnected line fragments instead of full
    notes, which silently tanks match rates. Detects that pattern and joins
    fragments back into whole notes, ordered by LINE, grouped by NOTE_ID.
    If the file doesn't have this shape, returns df unchanged."""
    cols_upper = {c.upper(): c for c in df.columns}
    id_col = cols_upper.get("NOTE_ID")
    line_col = cols_upper.get("LINE")
    text_col = cols_upper.get("NOTE_TEXT")

    if not (id_col and line_col and text_col):
        return df  # not a line-fragmented note file — nothing to do

    print(f"  Detected line-fragmented notes (LINE column present). "
          f"Reconstructing full notes by NOTE_ID...")

    work = df.copy()
    work[line_col] = pd.to_numeric(work[line_col], errors="coerce").fillna(0)
    work[text_col] = work[text_col].fillna("")
    work = work.sort_values([id_col, line_col])

    # Join note text in line order; keep the first value of every other
    # metadata column (PAT_MRN_ID, NOTE_TYPE, dates, etc.) per NOTE_ID.
    agg_map = {text_col: lambda s: "\n".join(s)}
    for c in work.columns:
        if c not in (id_col, line_col, text_col):
            agg_map[c] = "first"

    reconstructed = work.groupby(id_col, sort=False).agg(agg_map).reset_index()
    print(f"  Reconstructed {len(reconstructed)} full notes from {len(df)} line fragments.")
    return reconstructed


def load_done_ids(done_path):
    """Load the set of already-processed NOTE_IDs for resume support."""
    if not os.path.exists(done_path):
        return set()
    with open(done_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def append_done_id(done_path, note_id):
    with open(done_path, "a") as f:
        f.write(f"{note_id}\n")


def flush_results(buffer, out_path):
    """Append buffered rows to the output CSV (creating header only if new)."""
    if not buffer:
        return
    pd.DataFrame(buffer).to_csv(
        out_path, mode="a", header=not os.path.exists(out_path), index=False
    )


def process_note(args_tuple, model, keyword_pattern):
    """Worker function: runs keyword filter + LLM call for a single note.
    Returns None if the note is skipped (empty or no keyword match)."""
    pat_mrn, note_id, source, file_name, note_text = args_tuple

    if not note_text or note_text == "nan":
        return None

    if not keyword_pattern.search(note_text):
        return None  # pre-filter: not worth an LLM call

    note_text_llm = note_text[:MAX_NOTE_CHARS] if len(note_text) > MAX_NOTE_CHARS else note_text
    res = call_ollama(note_text_llm, model=model)

    return {
        "PAT_MRN_ID": pat_mrn,
        "NOTE_ID": note_id,
        "SOURCE_DATASET": source,
        "SOURCE_FILE": file_name,
        "has_genetic_testing": res.get("has_genetic_testing", False),
        "attr_type": res.get("attr_type", "not_tested"),
        "variant_mutation": res.get("variant_mutation", "not specified"),
        "zygosity": res.get("zygosity", "not specified"),
        "evidence_text": res.get("evidence_text", ""),
        "clinical_summary": res.get("clinical_summary", "")
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ATTR-CM Genetic Status")
    parser.add_argument("--source", type=str, choices=["regex", "raw"], default="regex",
                        help="Select dataset source: 'regex' or 'raw'")
    parser.add_argument("--test", type=int, default=0, help="Test on first N notes per file")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of concurrent Ollama requests (see NOTES in file header)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip NOTE_IDs already recorded in the .done ledger")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable the keyword pre-filter and send every note to the LLM")
    parser.add_argument("--cohort-file", type=str, default=None,
                        help="Path to the cohort CSV from equa_attr_identify_cohort.py. "
                             "When given, notes are restricted to only these PAT_IDs "
                             "before any other processing — use this instead of scanning "
                             "the whole health system's notes.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Randomly sample N queued (post-keyword-filter) notes PER FILE "
                             "instead of processing all of them. Use this to get fast, "
                             "representative preliminary results ahead of a deadline, while "
                             "a full unsampled run continues separately.")
    parser.add_argument("--broad-filter", action="store_true",
                        help="Use the old broad keyword filter (generic 'amyloid' or "
                             "drug names ALONE qualify a note) instead of the default, "
                             "tighter genetics-specific filter. Much higher note volume "
                             "and runtime — use for a later exhaustive pass, not your "
                             "first/time-pressured run.")
    args = parser.parse_args()

    active_keywords = BROAD_GENETIC_KEYWORDS if args.broad_filter else GENETIC_KEYWORDS

    cohort_ids = None
    if args.cohort_file:
        cohort_df_ids = pd.read_csv(args.cohort_file, dtype=str)
        id_col_name = next((c for c in cohort_df_ids.columns if c.upper() == "PAT_ID"), None)
        if id_col_name is None:
            print(f"[ERROR] --cohort-file {args.cohort_file} has no PAT_ID column. Aborting.")
            sys.exit(1)
        cohort_ids = set(cohort_df_ids[id_col_name].dropna())
        print(f"Cohort restriction ACTIVE: {len(cohort_ids)} patients loaded from {args.cohort_file}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target_files = REGEX_FILES if args.source == "regex" else RAW_FILES
    target_dir = REGEX_DIR if args.source == "regex" else RAW_DIR

    out_latest = os.path.join(OUTPUT_DIR, f"attr_genetics_{args.source}_latest.csv")
    done_path = os.path.join(OUTPUT_DIR, f"attr_genetics_{args.source}.done")

    done_ids = load_done_ids(done_path) if args.resume else set()

    print("=" * 70)
    print(f"ATTR-CM GENETIC TESTING EXTRACTION AGENT [{args.source.upper()} SOURCE]")
    print(f"Directory: {target_dir}")
    print(f"Model: {args.model} | Ollama URL: {OLLAMA_URL} | Workers: {args.workers}")
    filter_mode = "OFF" if args.no_filter else ("BROAD" if args.broad_filter else "TIGHT (genetics-specific)")
    print(f"Keyword pre-filter: {filter_mode}")
    if args.resume:
        print(f"Resume mode: {len(done_ids)} notes already completed, will be skipped")
    print("=" * 70)

    total_notes_seen = 0
    total_notes_sent_to_llm = 0
    total_notes_skipped_filter = 0
    result_buffer = []
    t0 = time.time()

    for file_name in target_files:
        file_path = os.path.join(target_dir, file_name)
        if not os.path.exists(file_path):
            print(f"\n[SKIP] Missing file: {file_path}")
            continue

        print(f"\nProcessing File: {file_name}")
        df = load_dataframe(file_path)
        print(f"  Loaded {len(df)} rows.")
        df = reconstruct_notes_from_lines(df)
        print(f"  {len(df)} notes available after reconstruction.")

        if cohort_ids is not None:
            pat_id_col = next((c for c in df.columns if c.upper() == "PAT_ID"), None)
            if pat_id_col is None:
                print(f"  [WARNING] No PAT_ID column found in {file_name}; cannot apply cohort restriction to this file.")
            else:
                before = len(df)
                df = df[df[pat_id_col].isin(cohort_ids)]
                print(f"  Cohort restriction: {before} -> {len(df)} notes (only cohort patients).")

        if args.test > 0:
            df = df.head(args.test)
            print(f"  [TEST MODE] Processing first {args.test} notes.")

        mrn_col = next((c for c in df.columns if c.upper() == "PAT_MRN_ID"), df.columns[0])
        note_id_col = next((c for c in df.columns if c.upper() == "NOTE_ID"), "NOTE_ID")
        text_col = next((c for c in df.columns if c.upper() == "NOTE_TEXT"), None) \
            or next((c for c in df.columns if "TEXT" in c.upper()), df.columns[-1])

        # Build the work list for this file, applying resume + keyword filter up front
        work_items = []
        for i, row in df.iterrows():
            total_notes_seen += 1
            note_text = str(row.get(text_col, ""))
            pat_mrn = row.get(mrn_col, "")
            note_id = str(row.get(note_id_col, f"row_{i}"))

            if args.resume and note_id in done_ids:
                continue

            if not args.no_filter:
                if not note_text or note_text == "nan" or not active_keywords.search(note_text):
                    total_notes_skipped_filter += 1
                    continue

            work_items.append((pat_mrn, note_id, args.source, file_name, note_text))

        print(f"  {len(work_items)} notes queued for LLM extraction "
              f"({total_notes_skipped_filter} skipped by keyword filter so far).")

        if args.sample and len(work_items) > args.sample:
            import random
            random.seed(42)  # reproducible sample
            work_items = random.sample(work_items, args.sample)
            print(f"  [SAMPLE MODE] Randomly sampled down to {len(work_items)} notes for this file.")

        # Process work_items, optionally in parallel
        processed_count = 0
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_note, item, args.model, active_keywords): item[1]
                    for item in work_items
                }
                for future in as_completed(futures):
                    note_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"  [ERROR] note {note_id}: {e}")
                        result = None

                    with write_lock:
                        processed_count += 1
                        total_notes_sent_to_llm += 1
                        if result is not None:
                            result_buffer.append(result)
                        append_done_id(done_path, note_id)

                        if len(result_buffer) >= CHECKPOINT_EVERY:
                            flush_results(result_buffer, out_latest)
                            result_buffer = []

                        if processed_count % 50 == 0 or processed_count == len(work_items):
                            print(f"  Processed {processed_count}/{len(work_items)} queued notes...")
        else:
            for item in work_items:
                note_id = item[1]
                result = process_note(item, args.model, active_keywords)
                processed_count += 1
                total_notes_sent_to_llm += 1
                if result is not None:
                    result_buffer.append(result)
                append_done_id(done_path, note_id)

                if len(result_buffer) >= CHECKPOINT_EVERY:
                    flush_results(result_buffer, out_latest)
                    result_buffer = []

                if processed_count % 50 == 0 or processed_count == len(work_items):
                    print(f"  Processed {processed_count}/{len(work_items)} queued notes...")

    # Final flush of any remaining buffered rows
    flush_results(result_buffer, out_latest)

    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE [{args.source.upper()}]")
    print(f"Total Notes Seen: {total_notes_seen}")
    print(f"Skipped by Keyword Filter: {total_notes_skipped_filter}")
    print(f"Sent to LLM: {total_notes_sent_to_llm}")
    print(f"Time Taken: {(time.time() - t0)/60:.2f} minutes")
    print(f"Saved (appended): {out_latest}")

    if os.path.exists(out_latest):
        try:
            results_df = pd.read_csv(out_latest, dtype=str)
            print("\nSummary Breakdown (attr_type):")
            print(results_df["attr_type"].value_counts().to_string())
        except Exception as e:
            print(f"  (could not summarize output: {e})")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()