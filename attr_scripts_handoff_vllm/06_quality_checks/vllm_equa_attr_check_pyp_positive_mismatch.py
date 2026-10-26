

#!/usr/bin/env python3
"""
Fixes two mislabeling patterns found by manual review, both stemming from the same
principle: a POSITIVE PYP scan confirms ATTR-CM disease presence, but does NOT by
itself confirm genetic subtype. Absent real genetic testing, the correct label is
'wild_type' (clinically presumed) — never 'variant' (which implies hereditary
confirmation), and never left as 'not_documented'/'unclear' (which loses a real
positive finding).

PATTERN A: attr_type == 'variant', with positive PYP imaging language present, but
           NO real genetic confirmation (no named variant, no "hereditary", no
           explicit genetic-test-positive wording) -> relabel to 'wild_type'.
PATTERN B: attr_type in ('not_documented', 'unclear'), with positive PYP imaging
           language present -> relabel to 'wild_type' (a missed positive).

If a type_basis column exists, it is force-set to 'clinical_or_imaging' for every
row this tool touches — these are, by definition, imaging-confirmed, not
genetically-confirmed, findings.

DRY RUN by default. Add --apply to modify the file (a backup is written first).

Usage:
  python3 equa_attr_fix_pyp_positive_mismatch.py --csv attr_genetics_raw_latest.csv
  python3 equa_attr_fix_pyp_positive_mismatch.py --csv attr_genetics_raw_latest.csv --apply
"""

import argparse
import re
import shutil
from datetime import datetime
import pandas as pd

POSITIVE_PYP_PATTERN = re.compile(
    r"consistent with TTR|consistent with (?:a )?TTR cardiac amyloidosis|"
    r"grade\s*[23]\b|uptake score (?:of|was)\s*[23]\b",
    re.IGNORECASE,
)
TTR_GENETIC_PATTERN = re.compile(
    r"\bV142I\b|Val142Ile|\bV122I\b|Val122Ile|\bV30M\b|Val30Met|\bT60A\b|Thr60Ala|"
    r"hereditary|mutation|genetic testing (?:results|confirms|showed|revealed)",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser(description="Fix positive-PYP mismatches (auto-apply)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    for col in ("attr_type", "evidence_text"):
        if col not in df.columns:
            raise SystemExit(f"[ERROR] column {col} not found in {args.csv}")

    evidence = df["evidence_text"].fillna("")
    has_positive_pyp = evidence.str.contains(POSITIVE_PYP_PATTERN)
    has_genetic_confirm = evidence.str.contains(TTR_GENETIC_PATTERN)
    attr_lower = df["attr_type"].fillna("").str.lower()

    pattern_a = (attr_lower == "variant") & has_positive_pyp & ~has_genetic_confirm
    pattern_b = attr_lower.isin(["not_documented", "unclear"]) & has_positive_pyp
    mask = pattern_a | pattern_b

    print("=" * 60)
    print(f"PATTERN A (variant -> wild_type, imaging-only): {int(pattern_a.sum())}")
    print(f"PATTERN B (missed positive -> wild_type): {int(pattern_b.sum())}")
    print(f"TOTAL rows that would be relabeled: {int(mask.sum())}")
    print("=" * 60)

    if not args.apply:
        print("\nDry run only — nothing changed. Re-run with --apply to fix.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.csv, f"{args.csv}.bak_{ts}")
    df.loc[mask, "attr_type"] = "wild_type"
    if "type_basis" in df.columns:
        df.loc[mask, "type_basis"] = "clinical_or_imaging"
    df.to_csv(args.csv, index=False)
    print(f"\nApplied. {int(mask.sum())} rows relabeled to wild_type (clinically presumed). "
          f"Backup: {args.csv}.bak_{ts}")


if __name__ == "__main__":
    main()