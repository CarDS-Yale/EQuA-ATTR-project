

import pandas as pd
import sys
import os

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")

if len(sys.argv) < 2:
    print("Usage: python3 al_raw_source_lookup.py <REAL_PAT_MRN_ID>")
    sys.exit(1)
target_mrn = sys.argv[1]

final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
match = final[final['PAT_MRN_ID'] == target_mrn]
if match.empty:
    print(f"No patient found with PAT_MRN_ID={target_mrn}")
    sys.exit(1)
pat_id = match.iloc[0]['PAT_ID']
print(f"PAT_MRN_ID={target_mrn} -> PAT_ID={pat_id}\n")

LAB_FILES = [
    "CarDS_2837199_Hosp_Enc_Labs_1.txt", "CarDS_2837199_Hosp_Enc_Labs_2.txt",
    "CarDS_2837199_Hosp_Enc_Labs_3.txt", "CarDS_2837199_Hosp_Enc_Labs_4.txt",
    "CarDS_2837199_Hosp_Enc_Labs_5.txt", "CarDS_2837199_Hosp_Enc_Labs_6.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_1.txt", "CarDS_2837199_Outpatient_Enc_Labs_2.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_3.txt",
]

for fname in LAB_FILES:
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        continue
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000,
                              on_bad_lines="skip", engine="python", quoting=3):
        if "PAT_ID" not in chunk.columns:
            continue
        rows = chunk[chunk["PAT_ID"] == pat_id]
        if not rows.empty:
            print(f"=== RAW ROWS from {fname} ===")
            print(rows.to_string())
            print()