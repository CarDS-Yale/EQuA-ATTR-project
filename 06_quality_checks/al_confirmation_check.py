

import pandas as pd
final = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_final_classification.csv', dtype=str)
pyp = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_pyp_results_latest.csv', dtype=str)

pyp_positive_mrns = set(pyp.loc[pyp['pyp_result']=='positive', 'PAT_MRN_ID'])
patients = final[final['PAT_MRN_ID'].isin(pyp_positive_mrns) & (final['final_classification']=='CONFIRMED_POSITIVE')]
has_al_lab = patients[patients['flag_al_lab_workup'].astype(str).str.lower().isin(['true','1'])]

abnormal_flagged = has_al_lab['al_lab_evidence'].fillna('').str.contains('abnormal=Y', case=False)
print(f"Of {len(has_al_lab)} with AL-lab evidence:")
print(f"  Explicitly flagged ABNORMAL (genuinely concerning, worth manual review): {abnormal_flagged.sum()}")
print(f"  Specific AL-related test present, no abnormal flag shown (likely normal/ruled-out, but verify): {(~abnormal_flagged).sum()}")
