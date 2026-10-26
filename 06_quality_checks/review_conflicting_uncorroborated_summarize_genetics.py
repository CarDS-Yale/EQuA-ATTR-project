

import pandas as pd

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)
raw = pd.read_csv(f'{BASE}/attr_genetics_ALL_combined.csv', dtype=str)

# --- 259 conflicting patients (both wild-type AND variant documented) ---
conflicting_mrns = rollup.loc[rollup['has_conflicting_notes'].astype(str).str.lower() == 'true', 'PAT_MRN_ID']
print(f"=== CONFLICTING PATIENTS: {len(conflicting_mrns)} total, showing 5 samples ===\n")
for mrn in conflicting_mrns.head(5):
    print(f"--- PAT_MRN_ID: {mrn} ---")
    pat_rows = raw[(raw['PAT_MRN_ID'] == mrn) & (raw['attr_type'].astype(str).str.lower().isin(['variant','wild_type']))]
    for _, r in pat_rows.iterrows():
        print(f"  attr_type={r.get('attr_type','')}: {str(r.get('evidence_text',''))[:150]}")
    print()

print(f"\n{'='*70}")
# --- 139 TTR-confirmed, no corroborating cohort signal ---
no_corrob_mrns = rollup.loc[rollup.get('needs_manual_review_possible_false_positive', pd.Series(dtype=str)).astype(str).str.lower() == 'true', 'PAT_MRN_ID']
if len(no_corrob_mrns) == 0:
    print("(Column name for the 139 may differ -- check rollup columns below if this is empty)")
    print(list(rollup.columns))
else:
    print(f"=== TTR-CONFIRMED, NO COHORT SIGNAL: {len(no_corrob_mrns)} total, showing 5 samples ===\n")
    for mrn in no_corrob_mrns.head(5):
        print(f"--- PAT_MRN_ID: {mrn} ---")
        pat_rows = raw[(raw['PAT_MRN_ID'] == mrn) & (raw['attr_type'].astype(str).str.lower().isin(['variant','wild_type']))]
        for _, r in pat_rows.iterrows():
            print(f"  attr_type={r.get('attr_type','')}: {str(r.get('evidence_text',''))[:200]}")
        print()