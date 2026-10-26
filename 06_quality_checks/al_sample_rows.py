

import pandas as pd
final = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_final_classification.csv', dtype=str)
pyp = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_pyp_results_latest.csv', dtype=str)

pyp_positive_mrns = set(pyp.loc[pyp['pyp_result']=='positive', 'PAT_MRN_ID'])
patients = final[final['PAT_MRN_ID'].isin(pyp_positive_mrns) & (final['final_classification']=='CONFIRMED_POSITIVE')]
has_al_lab = patients[patients['flag_al_lab_workup'].astype(str).str.lower().isin(['true','1'])]

print(f"Showing 10 of {len(has_al_lab)} sample rows:\n")
for _, row in has_al_lab.head(10).iterrows():
    print(f"PAT_MRN_ID: {row['PAT_MRN_ID']}")
    print(f"al_lab_evidence: {row.get('al_lab_evidence', 'MISSING')}")
    print("-" * 80)