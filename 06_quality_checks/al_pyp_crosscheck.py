


import pandas as pd
final = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_final_classification.csv', dtype=str)
pyp = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_pyp_results_latest.csv', dtype=str)

pyp_positive_mrns = set(pyp.loc[pyp['pyp_result']=='positive', 'PAT_MRN_ID'])
pyp_pos_patients = final[final['PAT_MRN_ID'].isin(pyp_positive_mrns) & (final['final_classification']=='CONFIRMED_POSITIVE')]

al_lab_col = 'flag_al_lab_workup' if 'flag_al_lab_workup' in pyp_pos_patients.columns else None
print(f"PYP-positive, CONFIRMED_POSITIVE patients: {len(pyp_pos_patients)}")
if al_lab_col:
    has_al_lab = pyp_pos_patients[al_lab_col].astype(str).str.lower().isin(['true','1']).sum()
    print(f"  Of those, ALSO have AL-lab evidence on record: {has_al_lab}")
    print(f"  No AL-lab evidence at all on record: {len(pyp_pos_patients) - has_al_lab}")
