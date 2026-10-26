

import pandas as pd
d = '/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/'
key = pd.read_csv(d + 'validation_key.csv', dtype=str)
sheet = pd.read_csv(d + 'validation_sheet.csv', dtype=str)
hum = [c for c in sheet.columns if c.startswith('human_attr_cm')][0]
m = key.merge(sheet[['sample_id', hum]], on='sample_id', how='left')
m['human'] = m[hum].fillna('').str.strip().str.upper().str[:1]
m['src'] = m['positive_evidence_source'].fillna('')
print('rows merged:', len(m), '| missing human label:', int((m['human'] == '').sum()))
print('llm_call values:', m['llm_call'].value_counts().to_dict())

pos = m[m['llm_call'].str.upper().str.contains('POS', na=False)].copy()
pos['group'] = pos['src'].apply(lambda s: 'LLM-only (note statement alone)' if s == 'NOTES_DISEASE_STATEMENT' else 'corroborated (2nd source)')
print('\n=== LLM-POSITIVE stratum: human label by corroboration ===')
print(pd.crosstab(pos['group'], pos['human'], margins=True))
for g, sub in pos.groupby('group'):
    s = sub[sub['human'] != 'U']
    print(f"{g}: confirmed {int((s['human']=='Y').sum())}/{len(s)} (U excluded; U = {int((sub['human']=='U').sum())})")

print('\n=== LLM-POSITIVE stratum: evidence source by human label ===')
print(pd.crosstab(pos['src'], pos['human']))