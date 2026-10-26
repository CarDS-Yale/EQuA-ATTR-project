


import pandas as pd

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)
raw = pd.read_csv(f'{BASE}/attr_genetics_ALL_combined.csv', dtype=str)

ruled_out_mrns = set(rollup.loc[rollup['overall_genetic_status'].str.contains('ruled_out', case=False, na=False), 'PAT_MRN_ID'])
print(f"Total patients with ruled_out status somewhere in their genetics data: {len(ruled_out_mrns)}")

affected = final[final['PAT_MRN_ID'].isin(ruled_out_mrns)]
print(f"\nOf those, final classification breakdown:")
print(affected['final_classification'].value_counts().to_string())

overridden = affected[affected['final_classification'] == 'CONFIRMED_POSITIVE']
print(f"\n{len(overridden)} were overridden to CONFIRMED_POSITIVE despite having a ruled_out note. Via which source:")
print(overridden['positive_evidence_source'].value_counts().to_string())

med_only_override = overridden[overridden['positive_evidence_source'] == 'MEDICATION+DX']
if len(med_only_override) > 0:
    print(f"\n!!! {len(med_only_override)} patients overridden via MEDICATION+DX ONLY (weakest justification) — worth manual review:")
    for _, r in med_only_override.head(10).iterrows():
        print(f"  {r['PAT_MRN_ID']}")

# --- 10 real examples, side by side: the ruled-out note vs the note that overrode it ---
sample_mrns = overridden['PAT_MRN_ID'].head(10).tolist()
print(f"\n{'='*80}\n10 REAL EXAMPLES — ruled-out note vs. the overriding positive note\n{'='*80}")
for mrn in sample_mrns:
    print("=" * 80)
    print(f"PAT_MRN_ID: {mrn}")
    print("=" * 80)
    pat_rows = raw[raw['PAT_MRN_ID'] == mrn]

    ruleout_rows = pat_rows[pat_rows['attr_type'].astype(str).str.lower() == 'ruled_out']
    print(f"\n--- RULED_OUT note(s) ({len(ruleout_rows)}) ---")
    for _, r in ruleout_rows.iterrows():
        print(f"  NOTE_ID={r.get('NOTE_ID','?')}: {r.get('evidence_text','')[:200]}")

    positive_rows = pat_rows[pat_rows['attr_type'].astype(str).str.lower().isin(['variant','wild_type'])]
    print(f"\n--- POSITIVE genetics note(s) that caused the override ({len(positive_rows)}) ---")
    for _, r in positive_rows.iterrows():
        print(f"  NOTE_ID={r.get('NOTE_ID','?')} (attr_type={r.get('attr_type','')}): {r.get('evidence_text','')[:200]}")
    print()





