

import pandas as pd
import os

RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")
final = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
positive_ids = set(final.loc[final['final_classification']=='CONFIRMED_POSITIVE', 'PAT_MRN_ID'])

patients = pd.read_csv(f'{RAW_DIR}/CarDS_2837199_Patients.txt', sep='\t', dtype=str, on_bad_lines='skip', engine='python', quoting=3)
pos_patients = patients[patients['PAT_MRN_ID'].isin(positive_ids)]

print(f"Matched {len(pos_patients)} of {len(positive_ids)} confirmed-positive patients in Patients.txt")
deceased = pos_patients['DEATH_DATE'].notna().sum()
print(f'{deceased}/{len(pos_patients)} ({100*deceased/len(pos_patients):.1f}%) of confirmed positives are deceased')
