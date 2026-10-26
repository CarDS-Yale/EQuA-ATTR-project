

import pandas as pd
df = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
positive = df[df['final_classification'] == 'CONFIRMED_POSITIVE']

no_structured = positive[~positive['resolution_source'].str.contains('STRUCTURED', na=False)]
notes_only_confirm = positive[positive['positive_evidence_source'] == 'NOTES_GENETICS']

print(f'{len(positive)} total confirmed positive')
print(f'{len(no_structured)} ({100*len(no_structured)/len(positive):.1f}%) had NO structured signal at all — pure discovery gap')
print(f'{len(notes_only_confirm)} ({100*len(notes_only_confirm)/len(positive):.1f}%) needed notes/genetics as the ONLY confirming evidence — confirmation gap')