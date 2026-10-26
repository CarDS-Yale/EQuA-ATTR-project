



#!/usr/bin/env python3
"""
Fixes a systematic mislabeling found by manual review: notes where a PYP scan
explicitly RULES OUT TTR/cardiac amyloidosis ("not consistent with TTR", "ruled
out TTR") were labeled attr_type=wild_type instead of a genuine rule-out. This
matters beyond cosmetics: the care-gap timeline and final classification scripts
treat wild_type as CONFIRMED_POSITIVE evidence, so a ruled-out patient mislabeled
this way would be wrongly counted as ATTR-CM positive.

The distinguishing pattern (validated against real examples from manual review):
  BUG (rule-out language):    "not consistent with TTR", "ruled out TTR"
  LEGITIMATE wild_type:       "consistent with TTR" (no "not"), "presumed to have
                               wild-type... unless proven", "confirmed wild-type
                               based on genetic testing"

Adds a new attr_type value: "ruled_out" — ATTR-CM explicitly excluded. This value
does NOT exist in the original 4-value schema (wild_type/variant/not_documented/
unclear); it's a genuine gap in the original prompt design, not just a labeling
slip. Future extraction runs should add this as a 5th schema value.

DRY RUN by default. Add --apply to modify the file (a backup is written first).

Usage:
  python3 equa_attr_fix_pyp_ruleout.py --csv attr_genetics_raw_vllm_latest.csv
  python3 equa_attr_fix_pyp_ruleout.py --csv attr_genetics_ALL_combined.csv --apply
"""

import argparse
import shutil
import re
from datetime import datetime
import pandas as pd

RULE_OUT_PATTERN = re.compile(
    r"(?:not consistent with|rule(?:d)? out|does not support|no evidence of)\s+"
    r"(?:TTR(?!\s*mutation)|cardiac amyloidosis)",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser(description="Fix PYP-rule-out-mislabeled-as-wild_type rows")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    for col in ("attr_type", "evidence_text"):
        if col not in df.columns:
            raise SystemExit(f"[ERROR] column {col} not found in {args.csv}")

    is_wild_type = df["attr_type"].fillna("").str.lower() == "wild_type"
    has_rule_out_language = df["evidence_text"].fillna("").str.contains(RULE_OUT_PATTERN, regex=True)
    mask = is_wild_type & has_rule_out_language

    print("=" * 60)
    print(f"Rows currently labeled wild_type: {int(is_wild_type.sum())}")
    print(f"Of those, containing explicit rule-out language: {int(mask.sum())}")
    print("=" * 60)
    if mask.sum() > 0:
        print("\nSample rows that would be relabeled wild_type -> ruled_out:")
        for _, r in df.loc[mask].head(5).iterrows():
            print(f"  NOTE_ID={r.get('NOTE_ID','?')}: {str(r['evidence_text'])[:100]}")

    if not args.apply:
        print("\nDry run only — nothing changed. Re-run with --apply to fix.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.csv, f"{args.csv}.bak_{ts}")
    df.loc[mask, "attr_type"] = "ruled_out"
    df.to_csv(args.csv, index=False)
    print(f"\nApplied. {int(mask.sum())} rows relabeled. Backup: {args.csv}.bak_{ts}")
    print("NOTE: your summarize script's classify_patient() logic doesn't yet know about")
    print("'ruled_out' as a category — it will currently fall into an 'else' bucket.")
    print("Re-run the summarize script and check how ruled_out rows get classified")
    print("before trusting the patient-level rollup.")


if __name__ == "__main__":
    main()