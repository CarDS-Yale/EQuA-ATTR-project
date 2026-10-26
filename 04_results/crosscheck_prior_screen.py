

#!/usr/bin/env python3
"""
Cross-checks the 978 CONFIRMED_POSITIVE patients against a prior manual TTR
screening file (Sheet1 + Sheet2 combined), bucketing matches into Yes/No based
on the prior screen's own TTR CA (Y/N) determination.

Usage:
  python3 crosscheck_prior_screen.py \
      --classification-file attr_cm_final_classification.csv \
      --screening-file 03132026-Favour-screening-TTR-2.xlsx
"""
import argparse
import pandas as pd

def find_col(columns, *names):
    for name in names:
        for c in columns:
            if c.strip().upper() == name.upper():
                return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classification-file", required=True)
    ap.add_argument("--screening-file", required=True)
    ap.add_argument("--out", default="crosscheck_no_bucket_for_review.csv")
    args = ap.parse_args()

    final = pd.read_csv(args.classification_file, dtype=str)
    positive = final[final["final_classification"] == "CONFIRMED_POSITIVE"].copy()
    print(f"Total CONFIRMED_POSITIVE patients: {len(positive)}")

    sheets = pd.read_excel(args.screening_file, sheet_name=None, dtype=str)
    print(f"Sheets found in screening file: {list(sheets.keys())}")
    screen = pd.concat(sheets.values(), ignore_index=True)
    print(f"Total rows across all sheets (before dedup): {len(screen)}")

    mrn_col_screen = find_col(screen.columns, "PAT_MRN_ID")
    ttr_ca_col = find_col(screen.columns, "TTR CA (Y/N)", "TTR CA")
    cardiac_amyloid_col = find_col(screen.columns, "Cardac Amyloidosis (Y/N)", "Cardiac Amyloidosis (Y/N)")
    print(f"Detected columns: PAT_MRN_ID={mrn_col_screen}, TTR CA={ttr_ca_col}, Cardiac Amyloidosis={cardiac_amyloid_col}\n")

    if not mrn_col_screen:
        raise SystemExit("[ERROR] Could not find PAT_MRN_ID column in screening file — check column names manually.")

    screen = screen.dropna(subset=[mrn_col_screen])
    screen_dedup = screen.drop_duplicates(subset=[mrn_col_screen], keep="first")
    print(f"Unique patients in prior screen (after dedup): {len(screen_dedup)}")

    merged = positive.merge(
        screen_dedup[[mrn_col_screen, ttr_ca_col, cardiac_amyloid_col]].rename(
            columns={mrn_col_screen: "PAT_MRN_ID"}
        ),
        on="PAT_MRN_ID", how="left"
    )

    found_in_screen = merged[ttr_ca_col].notna()
    print(f"\n{'='*60}\nOVERLAP\n{'='*60}")
    print(f"Of 978 CONFIRMED_POSITIVE, found in prior screen: {found_in_screen.sum()} ({100*found_in_screen.sum()/len(merged):.1f}%)")
    print(f"NOT found in prior screen (never screened before — new NLP finding or outside original scope): {(~found_in_screen).sum()}")

    matched = merged[found_in_screen]
    print(f"\n{'='*60}\nBUCKET BREAKDOWN (of the {len(matched)} found in prior screen)\n{'='*60}")
    print(f"TTR CA (Y/N) breakdown:")
    print(matched[ttr_ca_col].value_counts(dropna=False).to_string())

    ttr_ca_normalized = matched[ttr_ca_col].astype(str).str.strip().str.upper()
    yes_mask = ttr_ca_normalized.isin(["Y", "YES"])
    no_mask = ttr_ca_normalized.isin(["N", "NO"])
    unclear_mask = ~yes_mask & ~no_mask
    print(f"\nAgreement: {yes_mask.sum()} ({100*yes_mask.sum()/len(matched):.1f}%) prior screen says YES (consistent)")
    print(f"Disagreement: {no_mask.sum()} ({100*no_mask.sum()/len(matched):.1f}%) prior screen says NO (needs manual chart review)")
    if unclear_mask.sum() > 0:
        print(f"Unclear/unexpected value in TTR CA column (verify manually): {unclear_mask.sum()}")
        print(f"  Distinct unexpected values found: {matched.loc[unclear_mask, ttr_ca_col].unique().tolist()}")

    no_bucket = matched[no_mask]
    if len(no_bucket) > 0:
        sample = no_bucket.sample(n=min(20, len(no_bucket)), random_state=42)
        sample.to_csv(args.out, index=False)
        print(f"\nSaved up to 20 random 'NO' cases for manual chart review to: {args.out}")

if __name__ == "__main__":
    main()