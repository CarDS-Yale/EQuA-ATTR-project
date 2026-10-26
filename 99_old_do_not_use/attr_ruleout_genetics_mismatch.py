


import pandas as pd
import sys

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
raw = pd.read_csv(f'{BASE}/attr_genetics_ALL_combined.csv', dtype=str)
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)

overridden = final[(final['final_classification'] == 'CONFIRMED_POSITIVE') &
                    (final['positive_evidence_source'] == 'NOTES_GENETICS')]
mrn = overridden.iloc[0]['PAT_MRN_ID']
print(f"Diagnosing PAT_MRN_ID={mrn!r} (type: {type(mrn)}, repr shows exact string incl. whitespace)\n")

# 1. Does this exact MRN string exist AT ALL in the raw file?
exact_match = raw[raw['PAT_MRN_ID'] == mrn]
print(f"Exact-match rows in attr_genetics_ALL_combined.csv: {len(exact_match)}")

# 2. Fuzzy check: stripped/case-insensitive match, in case of whitespace/case mismatch
raw_mrns_stripped = raw['PAT_MRN_ID'].astype(str).str.strip()
fuzzy_match = raw[raw_mrns_stripped == str(mrn).strip()]
print(f"Whitespace-stripped match: {len(fuzzy_match)}")

if len(exact_match) > 0:
    print(f"\nALL attr_type values found for this patient (unfiltered):")
    print(exact_match['attr_type'].value_counts(dropna=False).to_string())
    print(f"\nFull rows:")
    for _, r in exact_match.iterrows():
        print(f"  NOTE_ID={r.get('NOTE_ID','?')} attr_type={r.get('attr_type','?')!r} type_basis={r.get('type_basis','?')!r}")
        print(f"    evidence: {str(r.get('evidence_text',''))[:150]}")

# 3. Cross-check rollup's own status for this patient
rollup_row = rollup[rollup['PAT_MRN_ID'] == mrn]
print(f"\nRollup status for this patient: {rollup_row['overall_genetic_status'].values if len(rollup_row)>0 else 'NOT FOUND IN ROLLUP'}")
print(f"ttr_language_confirmed: {rollup_row['ttr_language_confirmed'].values if len(rollup_row)>0 else 'N/A'}")

# 4. File freshness check -- do row counts/mtimes suggest a stale file somewhere?
import os
print(f"\n--- FILE TIMESTAMPS (to check for stale/out-of-sync files) ---")
for fname in ['attr_genetics_ALL_combined.csv', 'attr_genetics_patient_rollup.csv', 'attr_cm_final_classification.csv']:
    path = f'{BASE}/{fname}'
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        import datetime
        print(f"  {fname}: modified {datetime.datetime.fromtimestamp(mtime)}")