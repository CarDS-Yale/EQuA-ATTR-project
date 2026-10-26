

import pandas as pd
import os
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")

final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
positive = final[final['final_classification'] == 'CONFIRMED_POSITIVE'].copy()

# Recover PAT_ID for notes-only patients (blank PAT_ID) via Patients.txt lookup by
# PAT_MRN_ID -- same technique as the Table 1 demographics fix. The genetics/PYP
# extraction files never carried PAT_ID at all, so these patients only ever had
# PAT_MRN_ID populated here -- NOT because they lack a real PAT_ID in the health
# system, just because nothing had looked it up until now.
missing_pat_id_mask = positive['PAT_ID'].isna() | (positive['PAT_ID'].astype(str).str.strip() == '')
n_missing_before = missing_pat_id_mask.sum()
if n_missing_before > 0:
    print(f"Recovering PAT_ID for {n_missing_before} notes-only patients via Patients.txt lookup...")
    patients_file = pd.read_csv(f'{RAW_DIR}/CarDS_2837199_Patients.txt', sep='\t', dtype=str,
                                 on_bad_lines='skip', engine='python', quoting=3)
    mrn_to_patid = dict(zip(patients_file['PAT_MRN_ID'], patients_file['PAT_ID']))
    recovered = positive.loc[missing_pat_id_mask, 'PAT_MRN_ID'].map(mrn_to_patid)
    positive.loc[missing_pat_id_mask, 'PAT_ID'] = recovered
    n_still_missing = (positive['PAT_ID'].isna() | (positive['PAT_ID'].astype(str).str.strip() == '')).sum()
    print(f"  Recovered: {n_missing_before - n_still_missing} / {n_missing_before}. "
          f"Genuinely unrecoverable (not even in Patients.txt): {n_still_missing}")

pat_id_to_mrn = dict(zip(positive['PAT_ID'], positive['PAT_MRN_ID']))
target_pat_ids = set(positive['PAT_ID'].dropna())
print(f"Checking raw lab files directly for {len(target_pat_ids)} CONFIRMED_POSITIVE patients "
      f"(of {len(positive)} total)")

SPECIFIC_PATTERN = re.compile(r"free light chain|immunofixation|bone marrow|kappa.{0,20}lambda", re.IGNORECASE)

LAB_FILES = [
    "CarDS_2837199_Hosp_Enc_Labs_1.txt", "CarDS_2837199_Hosp_Enc_Labs_2.txt",
    "CarDS_2837199_Hosp_Enc_Labs_3.txt", "CarDS_2837199_Hosp_Enc_Labs_4.txt",
    "CarDS_2837199_Hosp_Enc_Labs_5.txt", "CarDS_2837199_Hosp_Enc_Labs_6.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_1.txt", "CarDS_2837199_Outpatient_Enc_Labs_2.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_3.txt",
]

findings = {}  # pat_id -> list of (component, abnormal_flag)
for fname in LAB_FILES:
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        continue
    print(f"  Scanning {fname} ...")
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000,
                              on_bad_lines="skip", engine="python", quoting=3):
        id_col = next((c for c in chunk.columns if c.upper() == "PAT_ID"), None)
        comp_col = next((c for c in chunk.columns if "COMPONENT_NAME" in c.upper()), None)
        abn_col = next((c for c in chunk.columns if "ABNORMAL" in c.upper()), None)
        if not id_col or not comp_col:
            continue
        sub = chunk[chunk[id_col].isin(target_pat_ids)]
        if sub.empty:
            continue
        for pat_id, comp, abn in zip(sub[id_col], sub[comp_col], sub.get(abn_col, "")):
            if isinstance(comp, str) and SPECIFIC_PATTERN.search(comp):
                findings.setdefault(pat_id, []).append((comp, abn))

print(f"\n{'='*70}")
print(f"RESULT: {len(findings)} of {len(target_pat_ids)} CONFIRMED_POSITIVE patients have a "
      f"SPECIFIC AL-confirmatory lab test (free light chain/immunofixation/bone marrow) anywhere in their raw lab records")
print(f"{'='*70}")

abnormal_patients = {}
for pat_id, tests in findings.items():
    abnormal = [t for t in tests if str(t[1]).strip().upper() == "Y"]
    if abnormal:
        abnormal_patients[pat_id] = abnormal

print(f"Of those, {len(abnormal_patients)} have at least one flagged ABNORMAL -- genuine AL concern, needs manual review:\n")
for pat_id, tests in list(abnormal_patients.items())[:20]:
    mrn = pat_id_to_mrn.get(pat_id, "?")
    print(f"  PAT_MRN_ID={mrn} (PAT_ID={pat_id}):")
    for comp, abn in tests:
        print(f"    {comp} (abnormal={abn})")

















'''


import pandas as pd
import os
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
RAW_DIR = os.path.expanduser("~/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23")

final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
positive = final[final['final_classification'] == 'CONFIRMED_POSITIVE'].copy()

# Recover PAT_ID for notes-only patients (blank PAT_ID) via Patients.txt lookup by
# PAT_MRN_ID -- same technique as the Table 1 demographics fix. The genetics/PYP
# extraction files never carried PAT_ID at all, so these patients only ever had
# PAT_MRN_ID populated here -- NOT because they lack a real PAT_ID in the health
# system, just because nothing had looked it up until now.
missing_pat_id_mask = positive['PAT_ID'].isna() | (positive['PAT_ID'].astype(str).str.strip() == '')
n_missing_before = missing_pat_id_mask.sum()
if n_missing_before > 0:
    print(f"Recovering PAT_ID for {n_missing_before} notes-only patients via Patients.txt lookup...")
    patients_file = pd.read_csv(f'{RAW_DIR}/CarDS_2837199_Patients.txt', sep='\t', dtype=str,
                                 on_bad_lines='skip', engine='python', quoting=3)
    mrn_to_patid = dict(zip(patients_file['PAT_MRN_ID'], patients_file['PAT_ID']))
    recovered = positive.loc[missing_pat_id_mask, 'PAT_MRN_ID'].map(mrn_to_patid)
    positive.loc[missing_pat_id_mask, 'PAT_ID'] = recovered
    n_still_missing = (positive['PAT_ID'].isna() | (positive['PAT_ID'].astype(str).str.strip() == '')).sum()
    print(f"  Recovered: {n_missing_before - n_still_missing} / {n_missing_before}. "
          f"Genuinely unrecoverable (not even in Patients.txt): {n_still_missing}")

pat_id_to_mrn = dict(zip(positive['PAT_ID'], positive['PAT_MRN_ID']))
target_pat_ids = set(positive['PAT_ID'].dropna())
print(f"Checking raw lab files directly for {len(target_pat_ids)} CONFIRMED_POSITIVE patients "
      f"(of {len(positive)} total)")

SPECIFIC_PATTERN = re.compile(r"free light chain|immunofixation|bone marrow|kappa.{0,20}lambda", re.IGNORECASE)

LAB_FILES = [
    "CarDS_2837199_Hosp_Enc_Labs_1.txt", "CarDS_2837199_Hosp_Enc_Labs_2.txt",
    "CarDS_2837199_Hosp_Enc_Labs_3.txt", "CarDS_2837199_Hosp_Enc_Labs_4.txt",
    "CarDS_2837199_Hosp_Enc_Labs_5.txt", "CarDS_2837199_Hosp_Enc_Labs_6.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_1.txt", "CarDS_2837199_Outpatient_Enc_Labs_2.txt",
    "CarDS_2837199_Outpatient_Enc_Labs_3.txt",
]

findings = {}  # pat_id -> list of (component, abnormal_flag)
for fname in LAB_FILES:
    path = os.path.join(RAW_DIR, fname)
    if not os.path.exists(path):
        continue
    print(f"  Scanning {fname} ...")
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=500_000,
                              on_bad_lines="skip", engine="python", quoting=3):
        id_col = next((c for c in chunk.columns if c.upper() == "PAT_ID"), None)
        comp_col = next((c for c in chunk.columns if "COMPONENT_NAME" in c.upper()), None)
        abn_col = next((c for c in chunk.columns if "ABNORMAL" in c.upper()), None)
        if not id_col or not comp_col:
            continue
        sub = chunk[chunk[id_col].isin(target_pat_ids)]
        if sub.empty:
            continue
        for pat_id, comp, abn in zip(sub[id_col], sub[comp_col], sub.get(abn_col, "")):
            if isinstance(comp, str) and SPECIFIC_PATTERN.search(comp):
                findings.setdefault(pat_id, []).append((comp, abn))

print(f"\n{'='*70}")
print(f"RESULT: {len(findings)} of {len(target_pat_ids)} CONFIRMED_POSITIVE patients have a "
      f"SPECIFIC AL-confirmatory lab test (free light chain/immunofixation/bone marrow) anywhere in their raw lab records")
print(f"{'='*70}")

abnormal_patients = {}
for pat_id, tests in findings.items():
    abnormal = [t for t in tests if str(t[1]).strip().upper() == "Y"]
    if abnormal:
        abnormal_patients[pat_id] = abnormal

print(f"Of those, {len(abnormal_patients)} have at least one flagged ABNORMAL -- genuine AL concern, needs manual review:\n")
for pat_id, tests in list(abnormal_patients.items())[:20]:
    mrn = pat_id_to_mrn.get(pat_id, "?")
    print(f"  PAT_MRN_ID={mrn} (PAT_ID={pat_id}):")
    for comp, abn in tests:
        print(f"    {comp} (abnormal={abn})")
'''