

#!/usr/bin/env python3
"""
Repairs damage from a real bug in an earlier version of equa_attr_fix_pyp_ruleout.py.
The old rule-out pattern incorrectly matched "no evidence of TTR MUTATION" (a real
genetic test result — meaning GENUINELY CONFIRMED wild-type ATTR-CM) as if it were
"no evidence of TTR CARDIAC AMYLOIDOSIS" (a PYP/imaging rule-out, meaning disease
excluded entirely). These are opposite findings. Any row the old pattern relabeled
to 'ruled_out' needs to be checked: if it was ONLY caught because of "TTR mutation"
language (not genuine rule-out language), it must be restored — specifically to
wild_type with type_basis=genetic, since "no evidence of TTR mutation" from a real
genetic test (e.g., "Sent to Admera") IS a genuine, confirmed genetic finding.

Method: re-check every row currently labeled 'ruled_out' against BOTH the old
(buggy) and new (fixed) patterns. Rows the OLD pattern matched but the NEW pattern
does NOT match are exactly the false corrections — restore those. Rows both
patterns agree on are genuine rule-outs — leave those alone.

DRY RUN by default. Add --apply to modify the file (a backup is written first).

Usage:
  python3 equa_attr_repair_ruleout_mutation_bug.py --csv attr_genetics_raw_latest.csv
  python3 equa_attr_repair_ruleout_mutation_bug.py --csv attr_genetics_raw_latest.csv --apply
"""

import argparse
import re
import shutil
from datetime import datetime
import pandas as pd

OLD_BUGGY_PATTERN = re.compile(
    r"not consistent with (?:TTR|cardiac amyloidosis)|"
    r"rule(?:d)? out (?:TTR|cardiac amyloidosis)|"
    r"does not support (?:TTR|cardiac amyloidosis)|"
    r"no evidence of (?:TTR|cardiac amyloidosis)",
    re.IGNORECASE,
)
NEW_FIXED_PATTERN = re.compile(
    r"(?:not consistent with|rule(?:d)? out|does not support|no evidence of)\s+"
    r"(?:TTR(?!\s*mutation)|cardiac amyloidosis)",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser(description="Repair the TTR-mutation rule-out mislabeling bug")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, dtype=str)
    for col in ("attr_type", "evidence_text"):
        if col not in df.columns:
            raise SystemExit(f"[ERROR] column {col} not found in {args.csv}")

    is_ruled_out = df["attr_type"].fillna("").str.lower() == "ruled_out"
    evidence = df["evidence_text"].fillna("")
    old_matched = evidence.str.contains(OLD_BUGGY_PATTERN)
    new_matches = evidence.str.contains(NEW_FIXED_PATTERN)

    # False corrections: currently 'ruled_out', the OLD pattern is why (it must have
    # matched to have been relabeled), but the NEW pattern does NOT match — meaning
    # it was only caught via "TTR mutation" language, a genetic-test-negative result.
    false_correction = is_ruled_out & old_matched & ~new_matches
    # Genuine rule-outs: currently 'ruled_out' AND the new pattern still agrees.
    genuine = is_ruled_out & new_matches

    print("=" * 60)
    print(f"Rows currently labeled 'ruled_out': {int(is_ruled_out.sum())}")
    print(f"  Genuine PYP/imaging rule-outs (correct, leave alone): {int(genuine.sum())}")
    print(f"  FALSE corrections (real genetic-test-negative, wrongly relabeled): {int(false_correction.sum())}")
    print("=" * 60)

    if false_correction.sum() > 0:
        print("\nRows to be RESTORED to wild_type (genetic — real confirmed TTR-negative genetic test):")
        for _, r in df.loc[false_correction].head(10).iterrows():
            print(f"  NOTE_ID={r.get('NOTE_ID','?')}: {str(r['evidence_text'])[:110]}")

    if not args.apply:
        print("\nDry run only — nothing changed. Re-run with --apply to fix.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.csv, f"{args.csv}.bak_{ts}")
    df.loc[false_correction, "attr_type"] = "wild_type"
    if "type_basis" in df.columns:
        df.loc[false_correction, "type_basis"] = "genetic"
    df.to_csv(args.csv, index=False)
    print(f"\nApplied. {int(false_correction.sum())} rows restored to genetically-confirmed wild_type. "
          f"Backup: {args.csv}.bak_{ts}")


if __name__ == "__main__":
    main()