

import pandas as pd, os
BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
if os.path.exists(f"{BASE}/validation_key.csv"):
    raise SystemExit("validation_key.csv already exists: the sample was already drawn. Do NOT redraw.")
TRUE = {'true', '1', 'yes', 'y', '1.0'}
c = pd.read_csv(f"{BASE}/attr_cm_final_classification.csv", dtype=str)
g = pd.read_csv(f"{BASE}/attr_genetics_ALL_combined.csv", dtype=str, usecols=['PAT_MRN_ID'])
has_notes = set(g['PAT_MRN_ID'].dropna())
seen = set()
for name in ['review_note_vs_neg_imaging.csv', 'review_genotype_only.csv']:
    p = f"{BASE}/{name}"
    if os.path.exists(p):
        seen |= set(pd.read_csv(p, dtype=str)['PAT_MRN_ID'].dropna())
adj = c['adjudication'].fillna('') != ''
c['llm_pos'] = c['cat_e'].fillna('').str.strip().str.lower().isin(TRUE)
pool = c[c['PAT_MRN_ID'].isin(has_notes) & ~adj & ~c['PAT_MRN_ID'].isin(seen)].drop_duplicates('PAT_MRN_ID')
print(f"Evaluated pool: {len(c)} | with >=1 LLM-processed note: {c['PAT_MRN_ID'].isin(has_notes).sum()} | excluded as already seen: {(c['PAT_MRN_ID'].isin(has_notes) & (adj | c['PAT_MRN_ID'].isin(seen))).sum()} | sampling pool: {len(pool)}")
N1, N0 = int(pool['llm_pos'].sum()), int((~pool['llm_pos']).sum())
print(f"Sampling pool by LLM call: LLM says ATTR-CM (cat_e) = {N1} | LLM does not = {N0} | sum = {N1 + N0}")
a = pool[pool['llm_pos']].sample(n=min(15, N1), random_state=20261006).assign(llm_call='LLM_POS')
b = pool[~pool['llm_pos']].sample(n=min(15, N0), random_state=20261006).assign(llm_call='LLM_NEG')
s = pd.concat([a, b]).sample(frac=1, random_state=7).reset_index(drop=True)
s['sample_id'] = [f"V{i+1:02d}" for i in range(len(s))]
keycols = [x for x in ['sample_id', 'PAT_MRN_ID', 'llm_call', 'final_classification', 'positive_evidence_source', 'genotype'] if x in s.columns]
s[keycols].to_csv(f"{BASE}/validation_key.csv", index=False)
sheet = s[['sample_id', 'PAT_MRN_ID']].copy()
sheet['human_attr_cm (Y/N/U)'] = ''
sheet['basis (diagnosis / PYP / biopsy / drug / none)'] = ''
sheet['AL_workup (done-negative / done-abnormal / not found)'] = ''
sheet['notes'] = ''
sheet.to_csv(f"{BASE}/validation_sheet.csv", index=False)
with open(f"{BASE}/validation_population.txt", 'w') as fh:
    fh.write(f"N_LLM_POS={N1}\nN_LLM_NEG={N0}\n")
print(f"Drawn: LLM_POS {len(a)} + LLM_NEG {len(b)} = {len(s)}. Wrote validation_sheet.csv (labels hidden) and validation_key.csv (DO NOT OPEN).")
