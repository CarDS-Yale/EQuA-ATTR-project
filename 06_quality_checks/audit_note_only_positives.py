


# Block 11 audit of note-only ATTR-CM positives (six failure types). Run from the results folder, or pass the folder as argument.
import pandas as pd, re
import sys, os
BASE = sys.argv[1] if len(sys.argv) > 1 else (os.getcwd() if os.path.exists("attr_cm_final_classification.csv") else "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results")
f = pd.read_csv(f"{BASE}/attr_cm_final_classification.csv", dtype=str, usecols=['PAT_MRN_ID', 'final_classification', 'positive_evidence_source', 'genotype'])
nlp = set(f.loc[(f['final_classification'] == 'CONFIRMED_POSITIVE') & (f['positive_evidence_source'] == 'NOTES_DISEASE_STATEMENT'), 'PAT_MRN_ID'])
print(f"Positives whose ONLY evidence is a note statement (no drug, no PYP report, no biopsy statement): {len(nlp)}")
raw = pd.read_csv(f"{BASE}/attr_genetics_ALL_combined.csv", dtype=str, usecols=['PAT_MRN_ID', 'NOTE_ID', 'attr_type', 'type_basis', 'evidence_text'])
raw = raw[raw['PAT_MRN_ID'].isin(nlp)].copy()
ev = raw['evidence_text'].fillna('')
WIDE = re.compile(r"(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT)|\bscan\b)[^.;]{0,120}?(?:\b(?:negative|normal|unremarkable)\b|no\s+evidence|not\s+consistent|does\s+not\s+(?:show|support)|without\s+evidence|no\s+(?:abnormal|myocardial|cardiac)\s+(?:radiotracer\s+)?uptake)|\b(?:negative|normal)\b[^.;]{0,60}?(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT))", re.I)
POS = re.compile(r"\bpositive\b|abnormal|grade\s*[23]|score\s*(?:of|was)?\s*[23]|(?<!not )consistent with TTR|false[- ]negative|falsely[- ]negative|despite (?:a |the )?(?:negative|normal)", re.I)
FAM = re.compile(r"\b(?:brother|sister|sibling|mother|father|parent|son|daughter|aunt|uncle|cousin|grandmother|grandfather|grandparent|nephew|niece|relative|family history)\b", re.I)
WORK = re.compile(r"\bR/O\b|rule[- ]?out|evaluat\w+\s+for|work[- ]?up|concern\s+for|\bsuspect\w*|suspicion|to\s+exclude|screen\w*\s+for|\?", re.I)
def snip(text, rx=None, n=110):
    text = str(text)
    m = rx.search(text) if rx else None
    if m:
        a, b = max(0, m.start() - n), min(len(text), m.end() + n)
        return ("..." if a > 0 else "") + text[a:b] + ("..." if b < len(text) else "")
    return text[:2 * n]
a = raw[ev.str.contains(WIDE) & ~ev.str.contains(POS)]
print(f"\nA) a note-only positive with negative-scan wording in a wide window: {a['PAT_MRN_ID'].nunique()} patients")
out_a = pd.DataFrame({'PAT_MRN_ID': a['PAT_MRN_ID'], 'NOTE_ID': a['NOTE_ID'], 'attr_type': a['attr_type'], 'text': [snip(t, WIDE) for t in a['evidence_text']]})
out_a.to_csv(f"{BASE}/audit_nlp_only_negative_wording.csv", index=False)
for _, r in out_a.drop_duplicates('PAT_MRN_ID').head(25).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}): {r['text']}")
vw = raw[raw['attr_type'].isin(['wild_type', 'variant'])]
famrow = vw['evidence_text'].fillna('').str.contains(FAM)
allfam = famrow.groupby(vw['PAT_MRN_ID']).all()
b_ids = set(allfam[allfam].index)
b = vw[vw['PAT_MRN_ID'].isin(b_ids)]
print(f"\nB) a note-only positive where EVERY labeled row mentions a relative (possible family-history error): {len(b_ids)} patients")
out_b = pd.DataFrame({'PAT_MRN_ID': b['PAT_MRN_ID'], 'NOTE_ID': b['NOTE_ID'], 'attr_type': b['attr_type'], 'text': [snip(t, FAM) for t in b['evidence_text']]})
out_b.to_csv(f"{BASE}/audit_nlp_only_family_history.csv", index=False)
for _, r in out_b.drop_duplicates('PAT_MRN_ID').head(25).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}): {r['text']}")
workrow = vw['evidence_text'].fillna('').str.contains(WORK)
allwork = workrow.groupby(vw['PAT_MRN_ID']).all()
c_ids = set(allwork[allwork].index)
c = vw[vw['PAT_MRN_ID'].isin(c_ids)]
print(f"\nC) a note-only positive where EVERY labeled row is work-up / rule-out language (e.g. R/o ATTR): {len(c_ids)} patients")
out_c = pd.DataFrame({'PAT_MRN_ID': c['PAT_MRN_ID'], 'NOTE_ID': c['NOTE_ID'], 'attr_type': c['attr_type'], 'text': [snip(t, WORK) for t in c['evidence_text']]})
out_c.to_csv(f"{BASE}/audit_nlp_only_workup_language.csv", index=False)
for _, r in out_c.drop_duplicates('PAT_MRN_ID').head(25).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}): {r['text']}")
DPHRASE = re.compile(r"(?:ATTR|TTR|transthyretin)[\s-]*(?:CM\b|cardiomyopathy|cardiac amyloid|amyloid cardiomyopathy)|(?:hereditary|variant|wild[- ]?type)[\s-]+(?:ATTR|TTR|transthyretin)?[\s-]*(?:CM\b|cardiomyopathy|cardiac amyloid)", re.I)
varset = set(f.loc[f['PAT_MRN_ID'].isin(nlp) & f['genotype'].isin(['HEREDITARY_VARIANT', 'CONFLICTING']), 'PAT_MRN_ID'])
dd = vw.copy()
dd['has_phrase'] = dd['evidence_text'].fillna('').str.contains(DPHRASE)
dd['clinical_label'] = dd['type_basis'].fillna('').eq('clinical_or_imaging')
anyphrase = dd.groupby('PAT_MRN_ID')['has_phrase'].any()
sus = set(anyphrase[~anyphrase].index)
print(f"\nD) a note-only positive with NO explicit disease phrase in any labeled row (positive only through the extraction's clinical-basis label; e.g. 'V142I gene positivity', 'No TTR genetic testing was done', 'ATTR amyloidosis' with no cardiac wording): {len(sus)} patients, of which {len(sus & varset)} have a variant or conflicting genotype")
ds = dd[dd['PAT_MRN_ID'].isin(sus) & dd['clinical_label']]
out_d = pd.DataFrame({'PAT_MRN_ID': ds['PAT_MRN_ID'], 'NOTE_ID': ds['NOTE_ID'], 'attr_type': ds['attr_type'], 'text': [snip(t, None, 110) for t in ds['evidence_text']]})
out_d.to_csv(f"{BASE}/audit_nlp_only_no_disease_phrase.csv", index=False)
for _, r in out_d.groupby('PAT_MRN_ID').head(2).head(30).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}): {r['text']}")
negall = vw['evidence_text'].fillna('').str.contains(WIDE)
allneg2 = negall.groupby(vw['PAT_MRN_ID']).all()
e_ids = set(allneg2[allneg2].index)
e = vw[vw['PAT_MRN_ID'].isin(e_ids)]
print(f"\nE) a note-only positive where EVERY labeled row is a negative-scan statement, even if a positive-sounding word appears elsewhere in the same text: {len(e_ids)} patients")
out_e = pd.DataFrame({'PAT_MRN_ID': e['PAT_MRN_ID'], 'NOTE_ID': e['NOTE_ID'], 'attr_type': e['attr_type'], 'text': [snip(t, WIDE) for t in e['evidence_text']]})
out_e.to_csv(f"{BASE}/audit_nlp_only_all_rows_negative_scan.csv", index=False)
for _, r in out_e.drop_duplicates('PAT_MRN_ID').head(25).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}): {r['text']}")
SURR = re.compile(r"\bfake(?:r{1,2}|\s+name)\s+([A-Za-z]+)", re.I)
who = vw['evidence_text'].fillna('').apply(lambda t: {m.lower() for m in SURR.findall(t)})
multi = vw[who.apply(len) >= 2]
f_ids = set(multi['PAT_MRN_ID'])
print(f"\nF) a note-only positive with a labeled row that names TWO OR MORE different people (de-identification names such as FAKER SAM, fake name Tom; a family summary): {len(f_ids)} patients")
out_f = pd.DataFrame({'PAT_MRN_ID': multi['PAT_MRN_ID'], 'NOTE_ID': multi['NOTE_ID'], 'attr_type': multi['attr_type'], 'people_named': [", ".join(sorted(w)) for w in who[multi.index]], 'text': [str(t)[:230] for t in multi['evidence_text']]})
out_f.to_csv(f"{BASE}/audit_nlp_only_multiple_people.csv", index=False)
for _, r in out_f.drop_duplicates('PAT_MRN_ID').head(25).iterrows():
    print(f"  {r['PAT_MRN_ID']} ({r['attr_type']}) names: {r['people_named']} | {r['text'][:120]}")
print("\nFull lists: audit_nlp_only_negative_wording.csv, audit_nlp_only_family_history.csv, audit_nlp_only_workup_language.csv, audit_nlp_only_no_disease_phrase.csv, audit_nlp_only_all_rows_negative_scan.csv, audit_nlp_only_multiple_people.csv")