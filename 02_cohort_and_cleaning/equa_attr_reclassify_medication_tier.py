


#!/usr/bin/env python3
"""
Splits the cohort script's medication evidence into two tiers, without re-scanning
raw source files (the drug name is already captured in the evidence text columns).

Verified current FDA indications (checked, not assumed):
  CM-ONLY (approved SOLELY for ATTR-CM — safe as standalone positive evidence):
    tafamidis (Vyndaqel/Vyndamax, approved 2019), acoramidis (Attruby, 2025)
  DUAL-INDICATION OR PN-PRIMARY (NOT sufficient standalone ATTR-CM evidence):
    vutrisiran (Amvuttra) — approved for BOTH ATTR-CM (June 2025) AND ATTR-PN;
      being on it alone does not tell you which condition it's treating, unlike
      tafamidis/acoramidis which have no PN indication at all
    patisiran (Onpattro), inotersen (Tegsedi) — PN-primary, no CM indication
    A patient on any of these alone may have ATTR-PN with no cardiomyopathy —
    requires corroboration (DX code, PYP result, etc.) to count as CM-positive.

Adds two new columns to the cohort CSV:
  flag_medication_cm_specific  — TRUE only if CM-specific drug evidence present
  flag_medication_pn_primary_only — TRUE if ONLY PN-primary drug evidence present
                                     (no CM-specific drug also present)

Usage:
  python3 equa_attr_reclassify_medication_tier.py --cohort-file attr_cm_candidate_cohort.csv
  python3 equa_attr_reclassify_medication_tier.py --cohort-file attr_cm_candidate_cohort.csv --apply
"""

import argparse
import re
import shutil
from datetime import datetime
import pandas as pd

CM_SPECIFIC_PATTERN = re.compile(
    r"tafamidis|vyndaqel|vyndamax|acoramidis|attruby",
    re.IGNORECASE,
)
PN_PRIMARY_PATTERN = re.compile(
    r"patisiran|onpattro|inotersen|tegsedi|vutrisiran|amvuttra",
    re.IGNORECASE,
)


def main():
    ap = argparse.ArgumentParser(description="Reclassify medication evidence into CM-specific vs PN-primary tiers")
    ap.add_argument("--cohort-file", required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort_file, dtype=str)
    ev_cols = [c for c in ("med_order_evidence", "med_admin_evidence") if c in df.columns]
    if not ev_cols:
        raise SystemExit("[ERROR] No med_order_evidence/med_admin_evidence columns found in this file.")

    combined_evidence = df[ev_cols].fillna("").agg(" ".join, axis=1)
    has_cm = combined_evidence.str.contains(CM_SPECIFIC_PATTERN)
    has_pn = combined_evidence.str.contains(PN_PRIMARY_PATTERN)

    flag_cm_specific = has_cm
    flag_pn_only = has_pn & ~has_cm

    old_med_positive = pd.Series(False, index=df.index)
    for c in ("flag_medication_order", "flag_medication_admin"):
        if c in df.columns:
            old_med_positive = old_med_positive | df[c].astype(str).str.lower().isin(["true", "1"])

    n_pn_only_currently_flagged = int((old_med_positive & flag_pn_only).sum())

    print("=" * 60)
    print(f"Patients with CM-ONLY med evidence (tafamidis/acoramidis): {int(flag_cm_specific.sum())}")
    print(f"Patients with ONLY dual-indication/PN-primary med evidence "
          f"(vutrisiran/patisiran/inotersen, no CM-only drug): {int(flag_pn_only.sum())}")
    print(f"  Of those, currently flagged medication-positive under the OLD (unsplit) logic: {n_pn_only_currently_flagged}")
    print("  These should NOT count as automatic ATTR-CM-positive evidence on their own.")
    print("=" * 60)

    if not args.apply:
        print("\nDry run only — nothing changed. Re-run with --apply to write the new columns.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.cohort_file, f"{args.cohort_file}.bak_{ts}")
    df["flag_medication_cm_specific"] = flag_cm_specific
    df["flag_medication_pn_primary_only"] = flag_pn_only
    df.to_csv(args.cohort_file, index=False)
    print(f"\nApplied. Added flag_medication_cm_specific and flag_medication_pn_primary_only columns. "
          f"Backup: {args.cohort_file}.bak_{ts}")
    print("Use flag_medication_cm_specific (not flag_medication_order/admin) as the standalone")
    print("positive-evidence trigger in the final classification script.")


if __name__ == "__main__":
    main()