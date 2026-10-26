


import pandas as pd

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)

status = rollup['overall_genetic_status'].astype(str)
is_clinical_presumed = status.str.startswith(('variant','wild_type')) & status.str.contains('clinically presumed', na=False)
ttr_confirmed = rollup['ttr_language_confirmed'].astype(str).str.lower() == 'true'

# These patients: model categorized them as a real TTR-type finding, clinically
# presumed (no genetic test), BUT the post-hoc text check could not find literal
# "TTR"/"transthyretin" anywhere -- EXCLUDED from CONFIRMED_POSITIVE as a result.
silently_excluded = rollup[is_clinical_presumed & ~ttr_confirmed]

print(f"Total 'clinically presumed' positive-status patients: {is_clinical_presumed.sum()}")
print(f"  Of those, ttr_language_confirmed=True (correctly counted as CONFIRMED_POSITIVE): {(is_clinical_presumed & ttr_confirmed).sum()}")
print(f"  Of those, ttr_language_confirmed=False (SILENTLY EXCLUDED despite model categorizing them positive): {len(silently_excluded)}")

if len(silently_excluded) > 0:
    print(f"\nSample of excluded patients (model said positive, but no literal TTR/transthyretin found in evidence):")
    for _, r in silently_excluded.head(10).iterrows():
        print(f"  {r['PAT_MRN_ID']}: status={r['overall_genetic_status']}, variants_found={str(r.get('variants_found','') or '')[:80]}")