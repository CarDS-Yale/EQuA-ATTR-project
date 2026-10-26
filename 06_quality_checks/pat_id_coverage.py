

import pandas as pd
import os

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")

final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
notes_only = final[(final['PAT_ID'].isna()) | (final['PAT_ID'].astype(str).str.strip() == '')]
print(f"Total patients with blank PAT_ID (notes-only discoveries): {len(notes_only)}")
print(f"  Of those, CONFIRMED_POSITIVE: {(notes_only['final_classification']=='CONFIRMED_POSITIVE').sum()}")

patients_file = pd.read_csv(f'{RAW_DIR}/CarDS_2837199_Patients.txt', sep='\t', dtype=str,
                             on_bad_lines='skip', engine='python', quoting=3)
patients_mrns = set(patients_file['PAT_MRN_ID'].dropna())

notes_only_mrns = set(notes_only['PAT_MRN_ID'].dropna())
found_in_patients_txt = notes_only_mrns & patients_mrns
missing_entirely = notes_only_mrns - patients_mrns

print(f"\nFound in Patients.txt (PAT_ID recoverable): {len(found_in_patients_txt)} ({100*len(found_in_patients_txt)/len(notes_only_mrns):.1f}%)")
print(f"NOT found anywhere in Patients.txt (genuinely missing): {len(missing_entirely)}")
if missing_entirely:
    print(f"  These PAT_MRN_IDs are not recoverable by any means: {list(missing_entirely)[:10]}")