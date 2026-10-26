

import pandas as pd
c = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_final_classification.csv', dtype=str)
pos = c['final_classification'] == 'CONFIRMED_POSITIVE'
nlp_only = pos & c['positive_evidence_source'].eq('NOTES_DISEASE_STATEMENT')
c.loc[nlp_only, 'final_classification'] = 'NLP_ONLY_UNCONFIRMED'
c.to_csv('tmp_confirmed634.csv', index=False)
conf = c[c['final_classification'] == 'CONFIRMED_POSITIVE']
print(f"positives before {int(pos.sum())} | note-only relabeled {int(nlp_only.sum())} | corroborated positives {len(conf)}")
print("evidence source:", conf['positive_evidence_source'].value_counts().to_dict())
g = conf['genotype'].value_counts()
print("genotype:", g.to_dict())
doc = int(conf['genotype'].isin(['HEREDITARY_VARIANT', 'NO_VARIANT_FOUND', 'CONFLICTING']).sum())
print(f"documented TTR genetic result: {doc} of {len(conf)} ({100*doc/len(conf):.1f}%)")
a = conf['positive_evidence_source'].fillna('').str.contains('MEDICATION\\+DX')
print(f"identifiable by drug+code: {int(a.sum())} of {len(conf)} ({100*a.sum()/len(conf):.1f}%)")