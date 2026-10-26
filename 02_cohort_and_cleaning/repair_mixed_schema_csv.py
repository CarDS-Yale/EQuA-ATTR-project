



#!/usr/bin/env python3
"""
Repair a genetics extraction output CSV that has a MIXED schema: some rows written
by the old script (10 columns, no type_basis) and some by the new script (11 columns,
type_basis inserted as the 7th field) — all sitting under a single stale 10-column
header. Reading this file naively (pandas.read_csv or otherwise) would silently
misalign every field from type_basis onward for every new-format row.

This script uses Python's csv module, which correctly respects CSV quoting (unlike a
naive awk/comma split), reads each row's ACTUAL field count, and rebuilds ONE clean,
consistent 11-column output:
  - New-format (11-field) rows: written as-is, correct columns.
  - Old-format (10-field) rows: type_basis is set to "" (empty) — the summarize
    script already treats a missing/empty type_basis as "basis unknown", so this is
    the correct, safe placeholder, not a guess.
  - Any row with an unexpected field count is set aside into a separate
    _needs_manual_review.csv file rather than guessed at.

The original file is never modified — a fully separate, corrected copy is written.

Usage:
  python3 repair_mixed_schema_csv.py --input attr_genetics_raw_latest.csv
"""

import os
import csv
import argparse

OLD_HEADER = [
    "PAT_MRN_ID", "NOTE_ID", "SOURCE_DATASET", "SOURCE_FILE",
    "has_genetic_testing", "attr_type", "variant_mutation", "zygosity",
    "evidence_text", "clinical_summary"
]
NEW_HEADER = [
    "PAT_MRN_ID", "NOTE_ID", "SOURCE_DATASET", "SOURCE_FILE",
    "has_genetic_testing", "attr_type", "type_basis", "variant_mutation",
    "zygosity", "evidence_text", "clinical_summary"
]


def main():
    parser = argparse.ArgumentParser(description="Repair a mixed-schema extraction output CSV")
    parser.add_argument("--input", required=True, help="Path to the CSV to repair")
    args = parser.parse_args()

    base, ext = os.path.splitext(args.input)
    fixed_path = f"{base}_FIXED{ext}"
    review_path = f"{base}_needs_manual_review{ext}"

    n_old_format = 0
    n_new_format = 0
    n_other = 0

    with open(args.input, "r", newline="", encoding="utf-8", errors="replace") as infile, \
         open(fixed_path, "w", newline="", encoding="utf-8") as outfile, \
         open(review_path, "w", newline="", encoding="utf-8") as reviewfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        review_writer = csv.writer(reviewfile)

        header = next(reader)
        print(f"Original header ({len(header)} columns): {header}")

        writer.writerow(NEW_HEADER)
        review_writer.writerow(["ORIGINAL_LINE_NUMBER"] + [f"field_{i}" for i in range(20)])

        for line_num, row in enumerate(reader, start=2):  # start=2: line 1 was header
            if len(row) == 11:
                # Already correct new-format row — write as-is
                writer.writerow(row)
                n_new_format += 1
            elif len(row) == 10:
                # Old-format row — insert empty type_basis at position 6 (0-indexed)
                fixed_row = row[:6] + [""] + row[6:]
                writer.writerow(fixed_row)
                n_old_format += 1
            else:
                # Unexpected shape — don't guess, flag for manual review
                review_writer.writerow([line_num] + row)
                n_other += 1

    print(f"\n{'='*70}")
    print("REPAIR COMPLETE")
    print(f"Old-format rows (10 fields) repaired, type_basis set to empty: {n_old_format}")
    print(f"New-format rows (11 fields) written as-is: {n_new_format}")
    print(f"Rows with unexpected field count, sent to review file: {n_other}")
    print(f"\nFixed file written to: {fixed_path}")
    if n_other > 0:
        print(f"Rows needing manual review: {review_path}")
    print(f"\nOriginal file was NOT modified: {args.input}")
    print("Once you've spot-checked the fixed file, replace the original with it, e.g.:")
    print(f"  mv {args.input} {base}_ORIGINAL_BACKUP{ext}")
    print(f"  mv {fixed_path} {args.input}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()