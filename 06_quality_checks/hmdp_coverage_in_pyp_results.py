

import pandas as pd
import re
import os

RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")

PYP_SCAN_FILTER = re.compile(
    r"(pyrophosphate|\bPYP\b|technetium.{0,20}pyrophosphate|\bTc-?99m\b.{0,20}PYP|"
    r"cardiac amyloid|amyloid.{0,20}scintigraphy|amyloid.{0,20}imaging)",
    re.IGNORECASE,
)
# All real alternative bone-avid tracers used for cardiac amyloid imaging, not just PYP
ALT_TRACER_PATTERN = re.compile(r"\bHMDP\b|\bHDP\b|\bDPD\b|hydroxymethylene", re.IGNORECASE)
# Require actual cardiac/amyloid context nearby -- not a bare tracer mention, which
# could just be a routine unrelated bone scan (HDP/HMDP's primary general use)
CARDIAC_CONTEXT = re.compile(r"cardiac|heart|amyloid", re.IGNORECASE)

for fname in ["CarDS_2837199_Imaging_1.txt", "CarDS_2837199_Imaging_2.txt"]:
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        continue
    print(f"Scanning {fname}...")
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000,
                              on_bad_lines="skip", engine="python", quoting=3):
        proc_col = next((c for c in chunk.columns if "PROC_NAME" in c.upper()), None)
        if not proc_col:
            continue
        text = chunk[proc_col].fillna('')
        has_alt_tracer = text.str.contains(ALT_TRACER_PATTERN)
        has_cardiac_context = text.str.contains(CARDIAC_CONTEXT)
        relevant = chunk[has_alt_tracer & has_cardiac_context]  # HMDP/HDP/DPD AND cardiac/amyloid context

        print(f"  Alt-tracer (HMDP/HDP/DPD) orders WITH cardiac/amyloid context: {len(relevant)}")
        if len(relevant) > 0:
            caught = relevant[proc_col].fillna('').str.contains(PYP_SCAN_FILTER)
            print(f"    Already caught by current filter: {caught.sum()}")
            print(f"    MISSED (genuinely relevant, not currently captured): {(~caught).sum()}")
            if (~caught).sum() > 0:
                print(f"    Sample missed PROC_NAME values:")
                for v in relevant.loc[~caught, proc_col].unique()[:5]:
                    print(f"      {v}")

        # Bonus: how many alt-tracer orders are bare mentions with NO cardiac context
        # (i.e. routine bone scans, correctly irrelevant -- not a gap)
        irrelevant = (has_alt_tracer & ~has_cardiac_context).sum()
        if irrelevant > 0:
            print(f"  (For reference: {irrelevant} HMDP/HDP/DPD orders with NO cardiac context -- correctly irrelevant, routine bone scans)")