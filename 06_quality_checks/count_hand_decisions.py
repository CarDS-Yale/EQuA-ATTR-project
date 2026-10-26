

import pandas as pd
d = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
d = d[d['adjudication'].fillna('') != '']
r = d['adjudication_reason'].fillna('')
print(len(d), 'patients hand-decided:', d['adjudication'].value_counts().to_dict())
print('LLM error tags:', int(r.str.contains('LLM error').sum()), '| PYP script error tags:', int(r.str.contains('PYP script error').sum()))
for k in ['negation', 'family history', 'rule-out']:
    print(f'  LLM error subtype {k}:', int((r.str.contains('LLM error') & r.str.contains(k)).sum()))
pl = r.str.contains('prompt limitation')
print('prompt limitation tags:', int(pl.sum()), '| schema limitation tags:', int(r.str.contains('schema limitation').sum()))
print('extracardiac noted:', int(r.str.contains('extracardiac').sum()), '| decided on extracted snippets:', int(r.str.contains('snippet').sum()))
for code, g in d.groupby('adjudication'):
    rr = g['adjudication_reason'].fillna('')
    ex = int(rr.str.contains('AL excluded').sum())
    ab = int(rr.str.contains('AL abnormal').sum())
    nx = int(rr.str.contains('AL not excluded').sum())
    print(f'  {code}: n={len(g)} | AL excluded {ex} | AL abnormal {ab} | AL not excluded {nx}')




'''
import pandas as pd
d = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
d = d[d['adjudication'].fillna('') != '']
r = d['adjudication_reason'].fillna('')
print(len(d), 'patients hand-decided:', d['adjudication'].value_counts().to_dict())
print('LLM error tags:', int(r.str.contains('LLM error').sum()), '| PYP script error tags:', int(r.str.contains('PYP script error').sum()))
for k in ['negation', 'family history', 'rule-out']:
    print(f'  LLM error subtype {k}:', int((r.str.contains('LLM error') & r.str.contains(k)).sum()))
print('schema limitation tags (the LLM followed the schema; the schema conflated genotype and disease):', int(r.str.contains('schema limitation').sum()))
print('extracardiac amyloid findings noted (tag word: extracardiac):', int(r.str.contains('extracardiac').sum()))
print('AL phrases by decision (no phrase = AL not documented):')
for code, g in d.groupby('adjudication'):
    rr = g['adjudication_reason'].fillna('')
    print(f'  {code}: n={len(g)} | AL excluded {int(rr.str.contains(\"AL excluded\").sum())} | AL abnormal {int(rr.str.contains(\"AL abnormal\").sum())} | AL not excluded {int(rr.str.contains(\"AL not excluded\").sum())} | AL not documented {int((~rr.str.contains(\"AL excluded|AL abnormal|AL not excluded\")).sum())}')
'''                                                                                                                                                                                                                                                                      