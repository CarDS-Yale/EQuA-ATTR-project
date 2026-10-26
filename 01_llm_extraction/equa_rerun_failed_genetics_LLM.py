


#equa_attr_reopen_failed.py

#!/usr/bin/env python3
"""
Reopen notes for reprocessing via the extraction script's --resume flag.

The extraction scripts mark a note as "done" in the .done ledger immediately, but
only flush results to the CSV every 500 notes. So:
  (a) notes whose LLM call failed are recorded as done but have an 'error' row, and
  (b) if a run was killed/shut down mid-batch, up to ~500 notes can be in the ledger
      with NO row in the CSV at all (lost buffer).
--resume skips anything in the ledger, so neither kind gets retried on its own.

This tool finds both kinds (and, optionally, old-script positives) and removes them
from the CSV and the ledger, so the next --resume run reprocesses exactly those notes.

DRY RUN by default (prints counts only, changes nothing). Add --apply to modify files.
Backups of both files are written next to them before any change.
Only run this while no extraction job is running against these files.

Usage:
  python3 equa_attr_reopen_failed.py --csv attr_genetics_raw_latest.csv --done attr_genetics_raw.done
  python3 equa_attr_reopen_failed.py --csv ... --done ... --include-old-positive
  python3 equa_attr_reopen_failed.py --csv ... --done ... --apply
"""

import argparse
import shutil
from datetime import datetime
import pandas as pd


def norm(series):
    return series.fillna("").astype(str).str.strip().str.lower()


def main():
    p = argparse.ArgumentParser(description="Reopen failed/missing notes for --resume")
    p.add_argument("--csv", required=True, help="Extraction output CSV")
    p.add_argument("--done", required=True, help="The matching .done ledger")
    p.add_argument("--include-old-positive", action="store_true",
                   help="Also reopen rows written by the OLD script (empty type_basis) whose "
                        "attr_type is variant/wild_type, so they get re-extracted with the "
                        "current (gene-specificity-fixed) prompt")
    p.add_argument("--apply", action="store_true", help="Actually modify files (default: dry run)")
    a = p.parse_args()

    df = pd.read_csv(a.csv, dtype=str)
    for col in ("NOTE_ID", "has_genetic_testing", "attr_type"):
        if col not in df.columns:
            raise SystemExit(f"[ERROR] column {col} not found in {a.csv}")

    err_mask = (norm(df["has_genetic_testing"]) == "error") | (norm(df["attr_type"]) == "error")

    with open(a.done) as f:
        done_ids = {line.strip() for line in f if line.strip()}
    csv_ids = set(df["NOTE_ID"].dropna().astype(str))
    missing_ids = done_ids - csv_ids

    old_pos_mask = pd.Series(False, index=df.index)
    if a.include_old_positive and "type_basis" in df.columns:
        basis_empty = df["type_basis"].fillna("").astype(str).str.strip() == ""
        old_pos_mask = basis_empty & norm(df["attr_type"]).isin(["variant", "wild_type"])

    drop_mask = err_mask | old_pos_mask
    reopen_ids = set(df.loc[drop_mask, "NOTE_ID"].astype(str)) | missing_ids

    print("=" * 60)
    print(f"CSV rows: {len(df)} | ledger entries: {len(done_ids)}")
    print(f"  Error rows (LLM call failed):                      {int(err_mask.sum())}")
    print(f"  In ledger but NO CSV row (lost buffer / unwritten): {len(missing_ids)}")
    if a.include_old_positive:
        print(f"  Old-script variant/wild_type rows (re-extract):    {int(old_pos_mask.sum())}")
    print(f"  TOTAL notes that would be reopened:                {len(reopen_ids)}")
    print("=" * 60)

    if not a.apply:
        print("Dry run only — nothing changed. Re-run with --apply to reopen these notes.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(a.csv, f"{a.csv}.bak_{ts}")
    shutil.copy2(a.done, f"{a.done}.bak_{ts}")
    df[~drop_mask].to_csv(a.csv, index=False)
    with open(a.done, "w") as f:
        for nid in sorted(done_ids - reopen_ids):
            f.write(nid + "\n")
    print(f"Applied. Backups: {a.csv}.bak_{ts} and {a.done}.bak_{ts}")
    print("Now run the extraction script with --resume (and --files if you restrict files).")


if __name__ == "__main__":
    main()