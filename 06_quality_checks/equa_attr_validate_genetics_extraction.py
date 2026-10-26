



#!/usr/bin/env python3
"""
Genetics Extraction Validation Tool
======================================
Generates a manual-review sample (pulling the ORIGINAL full note text, not the
model's own evidence_text, to avoid circular validation), and later scores your
manual labels against the model's output (sensitivity/specificity/PPV/NPV).

STEP 1 — generate a sample to review (each source note file is read ONCE, not once
per note, so this takes minutes rather than hours):
  python3 equa_attr_validate_extraction.py --generate \
      --extraction-csv attr_genetics_ALL_combined.csv --n 120 \
      --out validation_sample_general.csv

  Refuses to overwrite an existing --out file (you may already have labels in it).
  Use a different --out name, or --force to overwrite deliberately.

  Fill in these columns by hand:
      manual_has_genetic_testing   (true / false / unclear)
      manual_attr_type             (wild_type / variant / not_documented / unclear)
      reviewer_notes                (optional)

STEP 2 — after labeling:
  python3 equa_attr_validate_extraction.py --score --reviewed-csv validation_sample_general.csv
  (writes validation_sample_general_results.txt next to it)
"""

import os
import argparse
from collections import defaultdict
import pandas as pd

RAW_DIR = os.path.expanduser(
    "~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
)


def sniff_delimiter(file_path):
    with open(file_path, "r", errors="replace") as f:
        first_line = f.readline()
    candidates = {"\t": first_line.count("\t"), "|": first_line.count("|"), ",": first_line.count(",")}
    best = max(candidates, key=candidates.get)
    return best if candidates[best] > 0 else "\t"


def load_note_texts(needed):
    """needed: {source_file: set(note_ids)} -> {(source_file, note_id): full note text}.
    Reads each source file ONCE (chunked) and reassembles fragmented notes by LINE."""
    texts = {}
    for source_file, ids in needed.items():
        path = os.path.join(RAW_DIR, source_file)
        if not os.path.exists(path):
            for nid in ids:
                texts[(source_file, nid)] = "[SOURCE FILE NOT FOUND]"
            continue
        print(f"  Reading {source_file} once for {len(ids)} sampled notes ...")
        fragments = defaultdict(list)
        reader = pd.read_csv(
            path, sep=sniff_delimiter(path), dtype=str, on_bad_lines="skip",
            engine="python", quoting=3, chunksize=500_000,
        )
        for chunk in reader:
            id_col = next((c for c in chunk.columns if c.upper() == "NOTE_ID"), None)
            line_col = next((c for c in chunk.columns if c.upper() == "LINE"), None)
            text_col = next((c for c in chunk.columns if c.upper() == "NOTE_TEXT"), None)
            if id_col is None or text_col is None:
                continue
            sub = chunk[chunk[id_col].isin(ids)]
            if sub.empty:
                continue
            if line_col:
                lines = pd.to_numeric(sub[line_col], errors="coerce").fillna(0)
            else:
                lines = pd.Series(0, index=sub.index)
            for nid, ln, txt in zip(sub[id_col], lines, sub[text_col].fillna("")):
                fragments[nid].append((ln, str(txt)))
        for nid in ids:
            if nid in fragments:
                ordered = sorted(fragments[nid], key=lambda x: x[0])
                texts[(source_file, nid)] = "\n".join(t for _, t in ordered)
            else:
                texts[(source_file, nid)] = "[NOTE_ID NOT FOUND IN SOURCE FILE]"
    return texts


def generate_sample(args):
    if os.path.exists(args.out) and not args.force:
        print(f"[STOP] {args.out} already exists — not overwriting (it may contain your labels).")
        print("       Use a different --out name, or --force to overwrite deliberately.")
        return

    df = pd.read_csv(args.extraction_csv, dtype=str)
    print(f"Loaded {len(df)} extracted rows from {args.extraction_csv}")

    # Stratify by attr_type so rare positive/variant cases aren't drowned out by the
    # common "no signal" category
    categories = df["attr_type"].fillna("not_documented").astype(str)
    groups = df.groupby(categories)
    per_group = max(1, args.n // max(1, groups.ngroups))
    parts = [g.sample(n=min(len(g), per_group), random_state=42) for _, g in groups]
    sample_df = pd.concat(parts)

    if len(sample_df) < args.n:
        remaining = df.drop(sample_df.index)
        extra = remaining.sample(n=min(len(remaining), args.n - len(sample_df)), random_state=42)
        sample_df = pd.concat([sample_df, extra])

    sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sample_df)} rows (stratified across attr_type categories).")

    needed = defaultdict(set)
    for sf, nid in zip(sample_df["SOURCE_FILE"], sample_df["NOTE_ID"]):
        needed[sf].add(str(nid))
    print("Pulling FULL original note text (one pass per source file)...")
    texts = load_note_texts(needed)
    sample_df["FULL_ORIGINAL_NOTE_TEXT"] = [
        texts.get((sf, str(nid)), "[NOTE_ID NOT FOUND IN SOURCE FILE]")
        for sf, nid in zip(sample_df["SOURCE_FILE"], sample_df["NOTE_ID"])
    ]

    sample_df["manual_has_genetic_testing"] = ""
    sample_df["manual_attr_type"] = ""
    sample_df["reviewer_notes"] = ""

    out_cols = [
        "PAT_MRN_ID", "NOTE_ID", "SOURCE_FILE", "FULL_ORIGINAL_NOTE_TEXT",
        "has_genetic_testing", "attr_type", "variant_mutation",  # model answers: look AFTER you label
        "manual_has_genetic_testing", "manual_attr_type", "reviewer_notes",
    ]
    out_cols = [c for c in out_cols if c in sample_df.columns]
    sample_df[out_cols].to_csv(args.out, index=False)

    print(f"\n{'='*70}")
    print(f"Saved: {args.out}")
    print("For each row: read FULL_ORIGINAL_NOTE_TEXT first, IGNORING the model's columns,")
    print("then fill manual_has_genetic_testing and manual_attr_type. Then run --score.")
    print(f"{'='*70}")


def score_sample(args):
    df = pd.read_csv(args.reviewed_csv, dtype=str)
    df = df[df["manual_has_genetic_testing"].notna() & (df["manual_has_genetic_testing"] != "")]
    print(f"Scoring {len(df)} manually-labeled rows...")

    def normalize(s):
        return str(s).strip().lower()

    pred_pos = df["has_genetic_testing"].apply(normalize) == "true"
    true_pos = df["manual_has_genetic_testing"].apply(normalize) == "true"
    tp = int((pred_pos & true_pos).sum())
    fp = int((pred_pos & ~true_pos).sum())
    fn = int((~pred_pos & true_pos).sum())
    tn = int((~pred_pos & ~true_pos).sum())

    def ratio(a, b):
        return a / b if b > 0 else float("nan")

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    ppv = ratio(tp, tp + fp)
    npv = ratio(tn, tn + fn)
    accuracy = ratio(tp + tn, len(df))

    print(f"\n{'='*70}\nHAS_GENETIC_TESTING validation (true vs false/unclear)\n{'='*70}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Sensitivity: {sensitivity:.3f}   Specificity: {specificity:.3f}")
    print(f"  PPV:         {ppv:.3f}   NPV:         {npv:.3f}   Accuracy: {accuracy:.3f}")
    print("  (Note: a stratified sample over-represents rare categories, so PPV/NPV here")
    print("   describe THIS sample, not the whole corpus — say so in your methods.)")

    print(f"\n{'='*70}\nATTR_TYPE confusion matrix (rows = model, columns = manual)\n{'='*70}")
    conf = pd.crosstab(
        df["attr_type"].apply(normalize), df["manual_attr_type"].apply(normalize),
        rownames=["model"], colnames=["manual"],
    )
    print(conf.to_string())
    agreement = (df["attr_type"].apply(normalize) == df["manual_attr_type"].apply(normalize)).mean()
    print(f"\n  attr_type agreement rate: {agreement:.3f}")

    base, _ = os.path.splitext(args.reviewed_csv)
    out_path = f"{base}_results.txt"
    with open(out_path, "w") as f:
        f.write(f"n={len(df)}\n")
        f.write(f"has_genetic_testing: TP={tp} FP={fp} FN={fn} TN={tn}\n")
        f.write(f"sensitivity={sensitivity:.3f} specificity={specificity:.3f} "
                f"PPV={ppv:.3f} NPV={npv:.3f} accuracy={accuracy:.3f}\n")
        f.write(f"attr_type agreement: {agreement:.3f}\n\nattr_type confusion matrix:\n{conf.to_string()}\n")
    print(f"\nSaved summary to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and score a manual validation sample")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--extraction-csv", type=str)
    parser.add_argument("--reviewed-csv", type=str, default="validation_sample_TO_REVIEW.csv")
    parser.add_argument("--out", type=str, default="validation_sample_TO_REVIEW.csv")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing --out file")
    parser.add_argument("--n", type=int, default=120)
    args = parser.parse_args()

    if args.generate:
        if not args.extraction_csv:
            raise SystemExit("--generate needs --extraction-csv")
        generate_sample(args)
    elif args.score:
        score_sample(args)
    else:
        print("Specify --generate or --score.")


if __name__ == "__main__":
    main()