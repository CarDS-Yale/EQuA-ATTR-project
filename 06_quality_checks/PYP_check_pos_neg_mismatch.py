


import pandas as pd, re
pp = pd.read_csv('attr_pyp_results_latest.csv', dtype=str)
r = pp['pyp_result'].fillna('').str.lower(); ev = pp['evidence_text'].fillna('')
POSTXT = re.compile(r"(?:consistent|compatible|suggestive|diagnostic)\s+(?:with|of|for)\s+(?:\w+\s+){0,3}?(?:TTR|transthyretin|ATTR|cardiac amyloid\w*|amyloid\w*)|(?:significant|intense|marked|avid|increased|abnormal|diffuse)\s+(?:\w+\s+){0,3}?(?:uptake|accumulation)|grade\s*[23]\b|(?:visual\s*)?score\s*(?:of\s*)?[23]\b|positive for", re.I)
NEGTXT = re.compile(r"not\s+consistent|no\s+evidence|does\s+not\s+support|no\s+(?:abnormal|myocardial|cardiac|significant)\s+(?:radiotracer\s+)?uptake|negative\s+for|\bnormal\b|\bunremarkable\b|not\s+(?:suggestive|diagnostic)", re.I)
low = r.isin(['negative', 'equivocal_borderline'])
out = pp[low & ev.str.contains(POSTXT) & ~ev.str.contains(NEGTXT)].copy()
print(f"PYP rows: {len(pp)} | labeled negative or equivocal: {int(low.sum())} | text reads POSITIVE, no negating words: {len(out)} ({out['PAT_MRN_ID'].nunique()} patients)")
c = pd.read_csv('attr_cm_final_classification.csv', dtype=str, usecols=['PAT_MRN_ID', 'final_classification'])
out = out.merge(c, on='PAT_MRN_ID', how='left'); out['final_classification'] = out['final_classification'].fillna('not in pool')
print("Their current tier:", out.drop_duplicates('PAT_MRN_ID')['final_classification'].value_counts().to_dict())
out.to_csv('pyp_negative_but_positive_text.csv', index=False)