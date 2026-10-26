


import pandas as pd

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
care_gap = pd.read_csv(f'{BASE}/attr_care_gap_timeline.csv', dtype=str)

non_med_positive = final[
    (final['final_classification'] == 'CONFIRMED_POSITIVE') &
    (final['positive_evidence_source'] != 'MEDICATION+DX')
]
merged = non_med_positive.merge(care_gap[['PAT_MRN_ID', 'first_signal_date']], on='PAT_MRN_ID', how='left')

TAFAMIDIS_APPROVAL = pd.Timestamp('2019-05-01')
first_signal = pd.to_datetime(merged['first_signal_date'], errors='coerce')
diagnosed_after_approval = first_signal >= TAFAMIDIS_APPROVAL
alive = merged['DEATH_DATE'].isna() if 'DEATH_DATE' in merged.columns else pd.Series(True, index=merged.index)

print(f"Total identified independent of medication: {len(merged)}")
print(f"  Diagnosed after tafamidis approval (May 2019): {diagnosed_after_approval.sum()}")
print(f"  Of those, alive at last data capture: {(diagnosed_after_approval & alive).sum()}")
print("  NOTE: hospice status is NOT available in current data sources -- not applied, flag as limitation")

eligible = merged[diagnosed_after_approval & alive]
on_therapy = eligible['flag_medication_cm_specific'].astype(str).str.lower().eq('true').sum() if 'flag_medication_cm_specific' in eligible.columns else None
if on_therapy is not None:
    print(f"\nRESTRICTED, plausibly-eligible population: {len(eligible)}")
    print(f"  On CM-specific therapy: {on_therapy} ({100*on_therapy/len(eligible):.1f}%)")
    print(f"  No documented therapy: {len(eligible)-on_therapy} ({100*(len(eligible)-on_therapy)/len(eligible):.1f}%)")