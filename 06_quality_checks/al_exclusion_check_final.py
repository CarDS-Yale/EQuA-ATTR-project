

import pandas as pd
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
pyp = pd.read_csv(f'{BASE}/attr_pyp_results_latest.csv', dtype=str)

pyp_positive_mrns = set(pyp.loc[pyp['pyp_result']=='positive', 'PAT_MRN_ID'])
patients = final[final['PAT_MRN_ID'].isin(pyp_positive_mrns) & (final['final_classification']=='CONFIRMED_POSITIVE')]

SPECIFIC_PATTERN = re.compile(
    r"free light chain|immunofixation|bone marrow|kappa.{0,20}lambda",
    re.IGNORECASE,
)
evidence = patients['al_lab_evidence'].fillna('')
has_specific = evidence.str.contains(SPECIFIC_PATTERN)
has_specific_abnormal = has_specific & evidence.str.contains('abnormal=Y')
nonspecific_only = evidence.str.contains('protein total|protein, total', case=False, regex=True) & ~has_specific

print("=" * 70)
print("PART 1 — SUMMARY")
print("=" * 70)
print(f"Total PYP-positive CONFIRMED_POSITIVE patients: {len(patients)}")
print(f"  Have a SPECIFIC AL-confirmatory test on record: {has_specific.sum()}")
print(f"    Of those, flagged ABNORMAL specifically (genuine AL concern): {has_specific_abnormal.sum()}")
print(f"  ONLY nonspecific protein-total evidence: {nonspecific_only.sum()}")
print(f"  No AL-lab evidence of any kind: {(~evidence.astype(bool)).sum()}")

print("\n" + "=" * 70)
print("PART 2 — FULL VERIFICATION: every distinct al_lab_evidence value, with counts")
print("=" * 70)
display_evidence = patients['al_lab_evidence'].fillna('[NONE]')
counts = display_evidence.value_counts()
for val, n in counts.items():
    print(f"  [{n}] {val}")

print("\n" + "=" * 70)
print("PART 3 — REAL PATIENT ROWS (full context, for manual eyes-on verification)")
print("=" * 70)
context_cols = ['PAT_MRN_ID', 'al_lab_evidence', 'dx_evidence', 'imaging_evidence',
                'positive_evidence_source', 'subtype_basis']
context_cols = [c for c in context_cols if c in patients.columns]
for _, row in patients.head(20).iterrows():
    for c in context_cols:
        val = row.get(c, '')
        val = str(val)[:150] if val else '[EMPTY]'
        print(f"  {c}: {val}")
    print("-" * 70)




