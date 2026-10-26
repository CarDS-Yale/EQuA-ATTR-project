

import pandas as pd
cohort = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_candidate_cohort.csv', dtype=str)
pat_ids = set(cohort['PAT_ID'].dropna())

counts = {}
for chunk in pd.read_csv('/home/fao9/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23/CarDS_2837199_Hosp_Enc.txt',
                          sep='\t', dtype=str, chunksize=500_000, on_bad_lines='skip', engine='python', quoting=3):
    sub = chunk[chunk['PAT_ID'].isin(pat_ids)]
    for pat_id, is_inp in zip(sub['PAT_ID'], sub['INP_YN']):
        if str(is_inp).strip().upper() in ('Y','1','1.0','TRUE'):
            counts[pat_id] = counts.get(pat_id, 0) + 1

cohort['num_true_hospitalizations'] = cohort['PAT_ID'].map(counts).fillna(0).astype(int)
cohort.to_csv('attr_cm_candidate_cohort.csv', index=False)

vals = cohort['num_true_hospitalizations']
print(f'Median: {vals.median()}, Mean: {vals.mean():.2f}, Zero-hospitalization patients: {(vals==0).sum()}')