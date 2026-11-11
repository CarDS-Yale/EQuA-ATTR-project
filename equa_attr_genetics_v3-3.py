#!/usr/bin/env python3
"""
EQuA-ATTR  -  TTR genetic-testing extraction, v3 (Nov 2026)
============================================================
ONE script for BOTH servers. Pick the server with --backend:
    --backend ollama   (analysis server, Ollama on port 11434)
    --backend vllm     (vLLM server, OpenAI-compatible API on port 8000)
The prompt, categories, keyword filter and settings are defined ONCE in this
file, so the two servers cannot drift apart. Copy the SAME file to both.

What changed from new_equa_attr_extract_genetics.py / vllm_equa_attr_extract_genetics.py
  1. New categories (one task only: TTR genetic testing):
        positive | vus | negative | tested_no_result | not_tested | not_mentioned | other_gene_only
     plus about_patient (False when the statement is about a relative) and an
     evidence quote that is CHECKED in code to appear verbatim in the note.
  2. Context window fixed:
        - Ollama: num_ctx is now set explicitly (was never set -> small default,
          silent truncation that could drop the instructions).
        - Notes are no longer chopped at 8,000 characters. Notes up to MAX_NOTE_CHARS
          go in whole; longer notes are reduced to the windows around keyword hits,
          and the row is flagged (note_handling = keyword_windows).
  3. Dry-run mode: keyword filter only, no LLM. Prints notes/patients queued and
     note-length stats, so you can size the run before starting it.
  4. Author filter (--author-col / --author-pattern / --exclude-author-pattern) to
     pick the development notes (e.g. Gallegos/Miller) and a held-out test set.
  5. --tag names every output, so dev samples and the full run never mix.
  6. Review mode (--review): writes one readable .txt per sampled note
     (model answer on top, full note below) plus a review_sheet CSV to fill in.
     These files contain note text: they stay on the server. Paste only counts.
  7. Failed notes are NOT written to the .done ledger, so --resume retries them.

Typical use (see RUNBOOK_genetics_v3.md for the full order):
  dry run   : python3 equa_attr_genetics_v3.py --backend ollama --dry-run --files <f1,f2>
  dev sample: python3 equa_attr_genetics_v3.py --backend vllm --author-col <COL> \
                 --author-pattern "GALLEGOS|MILLER" --sample 25 --review --tag dev1
  full run  : nohup python3 equa_attr_genetics_v3.py --backend vllm --files <f3,f4,f5> \
                 --cohort-file <ids.csv> --workers 8 --resume --tag full > full_vllm.log 2>&1 &
"""

import os
import re
import sys
import json
import time
import random
import argparse
import threading
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPT_VERSION = "genetics_v3.2 (2026-11-04)"  # code rev h: tighter filter (case-sensitive variants, no A1c/gene/sequencing), cohort filter before rebuild, --cohort-tier

# ── Server settings (one place) ─────────────────────────────────────────────
BACKENDS = {
    "ollama": {
        "url": "http://127.0.0.1:11434/api/chat",
        "model": "qwen2.5:7b",
        "raw_dir": "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23",
        "out_dir": "~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/genetics_v3",
    },
    "vllm": {
        "url": "http://127.0.0.1:8000/v1/chat/completions",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "raw_dir": "~/mnt_3/equa_input_dir/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23",
        "out_dir": "~/my_work_tmp/equa_results/vllm_ollama_equa_results/genetics_v3",
    },
}

RAW_FILES = [
    "CarDS_2837199_Hosp_Enc_Notes.txt",
    "CarDS_2837199_Telephone_Enc_Notes.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_1.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_2.txt",
    "CarDS_2837199_Outpatient_Enc_Notes_3.txt",
]

# Context window. Qwen2.5-7B supports 32,768 tokens. 16,384 leaves room for the
# prompt (~1.5k tokens) + a long note (~10k tokens) + the answer, while keeping GPU
# memory reasonable when several requests run in parallel.
NUM_CTX = 16384
MAX_NOTE_CHARS = 36000      # ~9-10k tokens of clinical text; longer notes -> keyword windows
WINDOW_CHARS = 1500         # characters kept on each side of a keyword hit in a long note
MAX_ANSWER_TOKENS = 600
TEMPERATURE = 0.0
TIMEOUT_S = 300
CHECKPOINT_EVERY = 200

# ── Keyword pre-filter (recall first: a note the filter drops is never seen) ──
AA = "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val"
_WORDS = re.compile(
    r"(?:"
    r"\bTTR\b|transthyretin|"
    r"genetic (?:test|testing|tested|screen|counsel|evaluation|work.?up|result|analysis|panel)|"
    r"\bgenetics\b|genotyp|gene panel|"
    r"wild.?type|\bwt.?ATTR|\bATTRwt\b|\bATTRv\b|\bhATTR|\bvATTR|"
    r"hereditary (?:ATTR|amyloid|transthyretin)|familial amyloid|"
    r"pathogenic|uncertain significance|\bVUS\b|heterozygo|homozygo|"
    r"\bInvitae\b|\bAmbry\b|\bGeneDx\b|Prevention ?Genetics"
    r")",
    re.IGNORECASE,
)
_VARIANTS = re.compile(            # case-sensitive: V142I yes, "a1c" no
    r"(?:"
    rf"\b(?:p\.)?(?:{AA})\d{{1,4}}(?:{AA})\b|"
    r"\bp\.[A-Z][a-z]{2}\d{1,4}|"
    r"\bc\.\d{2,4}[ACGT]>[ACGT]\b|"
    r"\b(?!A1C\b)[ACDEFGHIKLMNPQRSTVWY]\d{1,3}[ACDEFGHIKLMNPQRSTVWY]\b"
    r")"
)


class _Either:
    """Acts like one compiled regex: matches if the word OR the variant pattern matches."""
    def search(self, text):
        return _WORDS.search(text) or _VARIANTS.search(text)

    def finditer(self, text):
        spans = sorted(list(_WORDS.finditer(text)) + list(_VARIANTS.finditer(text)), key=lambda m: m.start())
        return iter(spans)


GENETIC_KEYWORDS = _Either()


# ── Prompt (one task: TTR genetic testing) ─────────────────────────────────
SYSTEM_PROMPT = """You extract ONE thing from a clinical note: what the note says about TTR (transthyretin) GENETIC TESTING for the patient. Return ONLY one JSON object, no other text.

Choose ttr_genetic_status from exactly these values:
- "positive": the patient has a pathogenic or likely pathogenic TTR variant. This includes a NAMED TTR variant stated for the patient even if the test itself is not described, for example "hATTR with Phe64Ser (F64S)", "V142I ATTR-CM", "Val122Ile", "heterozygous for TTR p.Val142Ile", "asymptomatic carrier of the V142I TTR variant", "hereditary ATTR, TTR mutation positive".
- "vus": a TTR variant of uncertain significance was found.
- "negative": a TTR genetic test found NO TTR variant (e.g. "TTR gene sequencing negative", "negative genetic testing", "no pathogenic TTR variant", "genetic testing confirmed wild-type").
- "tested_no_result": a TTR genetic test was actually sent, drawn, or completed, but the note does not state the result.
- "not_tested": the note EXPLICITLY states TTR genetic testing was not done, was declined, has not been done yet, or is only planned: "will refer for genetic testing", "genetic counselor will contact the patient", "genetic testing will be performed", "recommend genetic testing", "declined genetic testing", "no genetic testing done". A referral or plan is NOT a test that was sent: use "not_tested", not "tested_no_result". You MUST quote that sentence in evidence_text.
- "other_gene_only": the only genetic testing mentioned is for genes other than TTR (BRCA1/2, APC, MUTYH, Lynch, TTN, MYH7, cancer or cardiomyopathy panels without a TTR result). Never report a non-TTR result as positive or negative.
- "not_mentioned": the note does not discuss TTR genetic testing. This is the answer for notes that mention amyloidosis, ATTR-CM, wild-type ATTR (wtATTR), PYP scans, tafamidis, AL work-up or biopsy but say nothing about a TTR genetic test. Silence about testing is "not_mentioned", never "not_tested".

Rules:
1. A PYP / technetium pyrophosphate / bone scintigraphy / DPD / HMDP scan, a biopsy, mass spectrometry, an AL work-up (SPEP, IFE, free light chains), or a medication is NOT genetic testing.
2. "wild-type ATTR", "wtATTR" or "ATTRwt" stated as a diagnosis WITHOUT a mention of a genetic test is NOT "negative": use "not_mentioned" and set clinical_label to "wild_type".
3. about_patient = false if the genetic result belongs to a relative (father, sister, son, daughter, "family member", or another named person), not this patient.
4. Negated statements ("no TTR testing done", "genetic testing not performed") are "not_tested", never "negative".
5. evidence_text must be copied EXACTLY, word for word, from the note (one or two sentences). Never write a sentence that is not in the note. Use "" when ttr_genetic_status is "not_mentioned".
6. If several statements conflict, report the most definitive TTR result about the patient and quote it.

JSON format:
{"ttr_genetic_status": "...", "variant": "variant as written, or \\"\\"", "zygosity": "heterozygous|homozygous|not_stated", "about_patient": true, "clinical_label": "wild_type|hereditary|none", "evidence_text": "..."}"""

USER_TEMPLATE = """Clinical note:
---NOTE START---
{note_text}
---NOTE END---
Return only the JSON object."""

VALID_STATUS = {"positive", "vus", "negative", "tested_no_result", "not_tested",
                "other_gene_only", "not_mentioned"}

write_lock = threading.Lock()

# Local model server: use 127.0.0.1 and ignore any proxy settings in the environment
# (on the vLLM server, 'localhost' + proxy variables made connections hang).
HTTP = requests.Session()
HTTP.trust_env = False


# ── LLM calls ──────────────────────────────────────────────────────────────
def _parse_json(content):
    content = content.strip()
    content = re.sub(r"^```(?:json)?", "", content).strip()
    content = re.sub(r"```$", "", content).strip()
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise json.JSONDecodeError("no JSON object", content, 0)
    return json.loads(m.group(0))


def call_llm(backend, model, note_text, max_retries=3):
    cfg = BACKENDS.get(backend)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(note_text=note_text)}]
    last_err = ""
    for attempt in range(max_retries):
        try:
            if backend == "mock":                      # offline test only
                return mock_answer(note_text), ""
            if backend == "ollama":
                body = {"model": model, "messages": messages, "stream": False, "format": "json",
                        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX,
                                    "num_predict": MAX_ANSWER_TOKENS}}
                r = HTTP.post(cfg["url"], json=body, timeout=TIMEOUT_S)
                r.raise_for_status()
                content = r.json()["message"]["content"]
            else:
                body = {"model": model, "messages": messages,
                        "temperature": TEMPERATURE, "max_tokens": MAX_ANSWER_TOKENS}
                r = HTTP.post(cfg["url"], json=body, timeout=TIMEOUT_S)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
            return _parse_json(content), ""
        except (json.JSONDecodeError, requests.exceptions.RequestException, KeyError, ValueError) as e:
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
            if attempt < max_retries - 1:
                time.sleep(3)
    return None, last_err


def mock_answer(note_text):
    """Offline stand-in for the LLM, used only to test the pipeline (--backend mock)."""
    t = note_text.lower()
    if "uncertain significance" in t:
        s = "vus"
    elif "no pathogenic" in t or "sequencing negative" in t:
        s = "negative"
    elif "not done" in t or "not performed" in t:
        s = "not_tested"
    elif "pending" in t or "sent" in t:
        s = "tested_no_result"
    elif "brca" in t:
        s = "other_gene_only"
    elif "v142i" in t or "val142ile" in t or "v122i" in t:
        s = "positive"
    else:
        s = "not_mentioned"
    sent = next((x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", note_text)
                 if GENETIC_KEYWORDS.search(x)), "")
    if "hypertension" in t:
        s, sent = "not_tested", ""
    return {"ttr_genetic_status": s, "variant": "", "zygosity": "not_stated",
            "about_patient": "sister" not in t, "clinical_label": "none",
            "evidence_text": sent if s not in ("not_mentioned",) else ""}


# ── Note loading (same logic as the earlier scripts) ───────────────────────
def sniff_delimiter(path):
    with open(path, "r", errors="replace") as f:
        first = f.readline()
    c = {"\t": first.count("\t"), "|": first.count("|"), ",": first.count(",")}
    best = max(c, key=c.get)
    return best if c[best] > 0 else "\t"


NEEDED_COLS = {"PAT_ID", "PAT_MRN_ID", "NOTE_ID", "LINE", "NOTE_TEXT"}


def load_notes(path, author_col=None, author_pattern=None, exclude_author_pattern=None, cohort=None):
    """Load only the columns we need (saves memory on multi-million-note files),
    apply the author filter BEFORE rebuilding notes, then rebuild line fragments."""
    wanted = set(NEEDED_COLS)
    if author_col:
        wanted.add(author_col.upper())
    usecols = lambda c: c.replace("\ufeff", "").strip().upper() in wanted
    df = pd.read_csv(path, sep=sniff_delimiter(path), dtype=str, on_bad_lines="skip",
                     engine="python", quoting=3, usecols=usecols)
    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
    if cohort:
        cc = next((c for c in df.columns if c.upper() == cohort[0]), None)
        if cc is None:
            print(f"  [WARNING] no {cohort[0]} column; cohort filter NOT applied")
        else:
            df = df[df[cc].isin(cohort[1])]
            print(f"  cohort filter applied to line rows: {len(df):,} rows kept")
    if author_col:
        ac = next((c for c in df.columns if c.upper() == author_col.upper()), None)
        if ac is None:
            print(f"  [ERROR] author column '{author_col}' not found; columns are {list(df.columns)}")
            return None
        auth = df[ac].fillna("")
        if author_pattern:
            df = df[auth.str.contains(author_pattern, case=False, regex=True)]
            auth = df[ac].fillna("")
        if exclude_author_pattern:
            df = df[~auth.str.contains(exclude_author_pattern, case=False, regex=True)]
        print(f"  author filter applied to line rows: {len(df):,} rows kept")
    cu = {c.upper(): c for c in df.columns}
    idc, linec, textc = cu.get("NOTE_ID"), cu.get("LINE"), cu.get("NOTE_TEXT")
    if idc and linec and textc:
        n_lines = len(df)
        df = df.copy()
        df[linec] = pd.to_numeric(df[linec], errors="coerce").fillna(0)
        df[textc] = df[textc].fillna("")
        df = df.sort_values([idc, linec])
        agg = {textc: lambda s: "\n".join(s)}
        for c in df.columns:
            if c not in (idc, linec, textc):
                agg[c] = "first"
        df = df.groupby(idc, sort=False).agg(agg).reset_index()
        print(f"  Rebuilt {len(df):,} notes from {n_lines:,} line fragments.")
    return df


def find_col(df, *names):
    cu = {c.upper(): c for c in df.columns}
    for n in names:
        if n in cu:
            return cu[n]
    return None


def prepare_text(note_text):
    """Return (text_for_llm, handling). Long notes -> windows around keyword hits."""
    if len(note_text) <= MAX_NOTE_CHARS:
        return note_text, "full"
    spans = []
    for m in GENETIC_KEYWORDS.finditer(note_text):
        a, b = max(0, m.start() - WINDOW_CHARS), min(len(note_text), m.end() + WINDOW_CHARS)
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    pieces, total = [], 0
    for a, b in spans:
        if total + (b - a) > MAX_NOTE_CHARS:
            break
        pieces.append(note_text[a:b])
        total += b - a
    return "\n[...]\n".join(pieces), "keyword_windows"


def norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="EQuA-ATTR TTR genetic-testing extraction v3")
    ap.add_argument("--backend", required=True, choices=["ollama", "vllm", "mock"])
    ap.add_argument("--model", default=None, help="override the backend's default model name")
    ap.add_argument("--raw-dir", default=None, help="override the backend's input folder")
    ap.add_argument("--out-dir", default=None, help="override the backend's output folder")
    ap.add_argument("--files", default=None, help="comma-separated note files for THIS server")
    ap.add_argument("--tag", default="dev", help="name for this run's outputs (dev1, test1, full ...)")
    ap.add_argument("--dry-run", action="store_true", help="keyword filter + counts only, no LLM")
    ap.add_argument("--sample", type=int, default=None, help="random N queued notes PER FILE")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--review", action="store_true", help="write readable review files (sample runs)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cohort-file", default=None,
                    help="CSV of patients to keep (matched on PAT_MRN_ID, else PAT_ID)")
    ap.add_argument("--cohort-tier", default=None,
                    help="keep only cohort rows with this final_classification (e.g. CONFIRMED_POSITIVE)")
    ap.add_argument("--author-col", default=None, help="column holding the note author/provider")
    ap.add_argument("--author-pattern", default=None, help="keep notes whose author matches (regex)")
    ap.add_argument("--exclude-author-pattern", default=None, help="drop notes whose author matches")
    ap.add_argument("--exclude-notes-file", default=None,
                    help="CSV with a NOTE_ID column; those notes are skipped (keeps test != dev)")
    args = ap.parse_args()

    cfg = BACKENDS.get(args.backend, BACKENDS["ollama"])
    model = args.model or cfg["model"]
    raw_dir = os.path.expanduser(args.raw_dir or cfg["raw_dir"])
    out_dir = os.path.expanduser(args.out_dir or cfg["out_dir"])
    os.makedirs(out_dir, exist_ok=True)

    files = RAW_FILES
    if args.files:
        req = [f.strip() for f in args.files.split(",") if f.strip()]
        files = [f for f in req]
    out_csv = os.path.join(out_dir, f"genetics_v3_{args.tag}_{args.backend}.csv")
    done_path = os.path.join(out_dir, f"genetics_v3_{args.tag}_{args.backend}.done")
    done_ids = set()
    if args.resume and os.path.exists(done_path):
        done_ids = set(l.strip() for l in open(done_path) if l.strip())

    cohort = None
    if args.cohort_file:
        cdf = pd.read_csv(args.cohort_file, dtype=str)
        if args.cohort_tier:
            fc = find_col(cdf, "FINAL_CLASSIFICATION")
            if fc is None:
                sys.exit("[ERROR] --cohort-tier given but the cohort file has no final_classification column")
            cdf = cdf[cdf[fc] == args.cohort_tier]
        ccol = find_col(cdf, "PAT_MRN_ID", "PAT_ID")
        if ccol is None:
            sys.exit(f"[ERROR] {args.cohort_file} has no PAT_MRN_ID or PAT_ID column")
        cohort = (ccol.upper(), set(cdf[ccol].dropna()))
    exclude_notes = set()
    if args.exclude_notes_file:
        edf = pd.read_csv(args.exclude_notes_file, dtype=str)
        exclude_notes = set(edf[find_col(edf, "NOTE_ID")].dropna())

    print("=" * 70)
    print(f"TTR GENETIC TESTING v3 | prompt {PROMPT_VERSION}")
    print(f"backend {args.backend} | model {model} | num_ctx {NUM_CTX} | max note chars {MAX_NOTE_CHARS}")
    print(f"input {raw_dir}")
    print(f"tag {args.tag} | {'DRY RUN (no LLM)' if args.dry_run else 'LLM run'} | workers {args.workers}")
    if cohort:
        print(f"cohort filter: {len(cohort[1]):,} patients on {cohort[0]}")
    if args.resume:
        print(f"resume: {len(done_ids):,} notes already done")
    print("=" * 70)

    all_items, seen_total, length_list = [], 0, []
    for fn in files:
        path = os.path.join(raw_dir, fn)
        if not os.path.exists(path):
            print(f"[SKIP] missing {path}")
            continue
        print(f"\nFile {fn}")
        df = load_notes(path, args.author_col, args.author_pattern, args.exclude_author_pattern, cohort)
        if df is None:
            continue
        seen_total += len(df)
        idc, textc = find_col(df, "NOTE_ID"), find_col(df, "NOTE_TEXT")
        mrnc, patc = find_col(df, "PAT_MRN_ID"), find_col(df, "PAT_ID")
        if not (idc and textc):
            print(f"  [ERROR] no NOTE_ID/NOTE_TEXT column; columns are {list(df.columns)}")
            continue
        texts = df[textc].fillna("")
        hit = texts.map(lambda t: bool(GENETIC_KEYWORDS.search(t)))
        q = df[hit]
        if exclude_notes:
            q = q[~q[idc].isin(exclude_notes)]
        if done_ids:
            q = q[~q[idc].astype(str).isin(done_ids)]
        print(f"  keyword filter: {len(df):,} notes -> {int(hit.sum()):,} queued"
              f" ({len(q):,} after exclusions/resume)")
        if args.sample and len(q) > args.sample:
            q = q.sample(n=args.sample, random_state=args.seed)
            print(f"  sample: {len(q)} notes (seed {args.seed})")
        for _, r in q.iterrows():
            t = str(r[textc])
            length_list.append(len(t))
            all_items.append({
                "PAT_MRN_ID": r[mrnc] if mrnc else "", "PAT_ID": r[patc] if patc else "",
                "NOTE_ID": str(r[idc]), "SOURCE_FILE": fn,
                "AUTHOR": r[next((c for c in q.columns if c.upper() == args.author_col.upper()), q.columns[0])] if args.author_col else "", "text": t})

    n = len(all_items)
    pats = len({(i["PAT_MRN_ID"] or i["PAT_ID"]) for i in all_items})
    print(f"\nQUEUED: {n:,} notes from {pats:,} patients (of {seen_total:,} notes read)")
    if length_list:
        s = pd.Series(length_list)
        long_n = int((s > MAX_NOTE_CHARS).sum())
        print(f"note length (characters): median {int(s.median()):,} | 95th pct {int(s.quantile(.95)):,}"
              f" | max {int(s.max()):,} | longer than {MAX_NOTE_CHARS:,}: {long_n:,} (keyword windows)")
        print(f"approx tokens (chars/4): median {int(s.median()/4):,} | 95th pct {int(s.quantile(.95)/4):,}"
              f" | context window {NUM_CTX:,}")
    if args.dry_run or n == 0:
        print("\nDry run: nothing sent to the LLM." if args.dry_run else "\nNothing to do.")
        return

    buffer, results_for_review = [], []
    counters = {"done": 0, "failed": 0}
    from collections import Counter, defaultdict
    run_status = Counter()
    run_patients = defaultdict(set)
    run_variants = Counter()

    def running_summary(label):
        print(f"  ---- running summary ({label}) ----")
        for k in ["positive", "vus", "negative", "tested_no_result", "not_tested",
                  "other_gene_only", "not_mentioned"]:
            print(f"    {k:17s} notes {run_status.get(k, 0):7,} | patients {len(run_patients.get(k, ())):6,}")
        if run_variants:
            print("    variants named (notes): " + ", ".join(f"{v} {n}" for v, n in run_variants.most_common(8)))
        print("  (note-level, this server only; the morning roll-up gives one row per patient)", flush=True)
    t0 = time.time()

    def work(item):
        txt, handling = prepare_text(item["text"])
        res, err = call_llm(args.backend, model, txt)
        row = {k: item[k] for k in ("PAT_MRN_ID", "PAT_ID", "NOTE_ID", "SOURCE_FILE", "AUTHOR")}
        row.update({"backend": args.backend, "model": model, "prompt_version": PROMPT_VERSION,
                    "note_chars": len(item["text"]), "note_handling": handling})
        if res is None:
            row.update({"ttr_genetic_status": "error", "error": err})
            return row, item, False
        status = str(res.get("ttr_genetic_status", "")).strip().lower()
        ev = str(res.get("evidence_text", "") or "")
        rule_note = ""
        verbatim = bool(ev.strip()) and norm(ev) in norm(item["text"])
        if status == "not_tested" and not ev.strip():
            status, rule_note = "not_mentioned", "not_tested_without_quote->not_mentioned"
        elif status == "not_tested" and not verbatim:
            status, rule_note = "not_mentioned", "not_tested_quote_not_in_note->not_mentioned"
        elif status in VALID_STATUS - {"not_mentioned"} and not ev.strip():
            rule_note = "no_quote"
        elif status in VALID_STATUS - {"not_mentioned"} and not verbatim:
            rule_note = "quote_not_in_note"
        row.update({
            "ttr_genetic_status": status if status in VALID_STATUS else f"invalid:{status}",
            "variant": str(res.get("variant", "") or ""),
            "zygosity": str(res.get("zygosity", "") or ""),
            "about_patient": str(res.get("about_patient", "")),
            "clinical_label": str(res.get("clinical_label", "") or ""),
            "evidence_text": ev,
            "evidence_verbatim": bool(ev) and norm(ev) in norm(item["text"]),
            "rule_applied": rule_note,
            "error": ""})
        return row, item, True

    def record(row, item, ok):
        with write_lock:
            buffer.append(row)
            if args.review:
                results_for_review.append((row, item["text"]))
            if ok:
                counters["done"] += 1
                st = row.get("ttr_genetic_status", "")
                run_status[st] += 1
                if str(row.get("about_patient", "")).lower() != "false":
                    run_patients[st].add(row.get("PAT_MRN_ID") or row.get("PAT_ID"))
                if st in ("positive", "vus") and row.get("variant"):
                    run_variants[str(row["variant"]).strip()[:40]] += 1
                with open(done_path, "a") as f:
                    f.write(row["NOTE_ID"] + "\n")
            else:
                counters["failed"] += 1
            k = counters["done"] + counters["failed"]
            if len(buffer) >= CHECKPOINT_EVERY:
                flush(buffer)
            if k % 500 == 0:
                running_summary(f"{k:,} notes")
            if k % 25 == 0 or k == n:
                el = time.time() - t0
                rate = k / el * 60 if el else 0
                eta = (n - k) / (k / el) / 60 if k else 0
                print(f"  {k:,}/{n:,} notes | {rate:.1f} notes/min | ~{eta:.0f} min left"
                      f" | failed {counters['failed']}", flush=True)

    def flush(buf):
        if buf:
            pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not os.path.exists(out_csv), index=False)
            buf.clear()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for fut in as_completed([ex.submit(work, it) for it in all_items]):
                record(*fut.result())
    else:
        for it in all_items:
            record(*work(it))
    with write_lock:
        flush(buffer)

    running_summary("final")
    mins = (time.time() - t0) / 60
    print(f"\nDONE in {mins:.1f} min | {counters['done']:,} ok | {counters['failed']:,} failed "
          f"(re-run with --resume to retry failures)")
    print(f"output: {out_csv}")
    out = pd.read_csv(out_csv, dtype=str)
    print("\nttr_genetic_status (notes):")
    print(out["ttr_genetic_status"].value_counts().to_string())
    if "evidence_verbatim" in out:
        e = out[out["evidence_text"].fillna("") != ""]
        if len(e):
            v = (e["evidence_verbatim"] == "True").sum()
            print(f"evidence quote found verbatim in note: {v}/{len(e)}")

    if args.review:
        rdir = os.path.join(out_dir, f"review_{args.tag}_{args.backend}")
        os.makedirs(rdir, exist_ok=True)
        sheet = []
        for i, (row, text) in enumerate(sorted(results_for_review, key=lambda x: x[0]["NOTE_ID"]), 1):
            rid = f"R{i:03d}"
            with open(os.path.join(rdir, f"{rid}.txt"), "w") as f:
                f.write(f"REVIEW {rid} | file {row['SOURCE_FILE']} | handling {row['note_handling']}\n")
                for k in ("ttr_genetic_status", "variant", "zygosity", "about_patient",
                          "clinical_label", "evidence_verbatim", "evidence_text"):
                    f.write(f"MODEL {k}: {row.get(k, '')}\n")
                f.write("=" * 70 + "\nNOTE\n" + "=" * 70 + "\n" + text + "\n")
            sheet.append({"review_id": rid, "NOTE_ID": row["NOTE_ID"],
                          "model_status": row["ttr_genetic_status"],
                          "true_status": "", "about_patient_ok (Y/N)": "", "evidence_ok (Y/N)": "",
                          "comment": ""})
        sp = os.path.join(out_dir, f"review_sheet_{args.tag}_{args.backend}.csv")
        pd.DataFrame(sheet).to_csv(sp, index=False)
        print(f"\nreview files: {rdir}/R001.txt ...  ({len(sheet)} notes)")
        print(f"review sheet to fill in: {sp}")
        print("Fill true_status with one of: " + ", ".join(sorted(VALID_STATUS)))


if __name__ == "__main__":
    main()
