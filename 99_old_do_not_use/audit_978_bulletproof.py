

import pandas as pd
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
raw = pd.read_csv(f'{BASE}/attr_genetics_ALL_combined.csv', dtype=str)

positive_mrns = set(final.loc[final['final_classification']=='CONFIRMED_POSITIVE', 'PAT_MRN_ID'])
rows = raw[raw['PAT_MRN_ID'].isin(positive_mrns)].copy()
print(f"Total rows belonging to {len(positive_mrns)} confirmed-positive patients: {len(rows)}")

# Same patterns as the two existing fixes, re-applied fresh against evidence_text
RULEOUT_PATTERN = re.compile(
    r"(?:not consistent with|rule(?:d)? out|does not support|no evidence of)\s+"
    r"(?:TTR(?!\s*mutation)|cardiac amyloidosis)", re.IGNORECASE)
POSITIVE_PATTERN = re.compile(
    r"consistent with TTR|grade\s*[23]\b|uptake score (?:of|was)\s*[23]\b", re.IGNORECASE)
# NEW: pt 6's phrasing - "X iso/in the setting of TTR cardiac amyloidosis" stated as existing dx
PT6_STYLE_PATTERN = re.compile(
    r"\b(?:iso|in the setting of)\s+TTR cardiac amyloidosis", re.IGNORECASE)

evidence = rows['evidence_text'].fillna('')
attr_type = rows['attr_type'].fillna('').str.lower()

should_be_ruledout = evidence.str.contains(RULEOUT_PATTERN) & (attr_type != 'ruled_out')
should_be_positive_a = evidence.str.contains(POSITIVE_PATTERN) & (attr_type.isin(['not_documented','unclear']))
should_be_positive_b = evidence.str.contains(PT6_STYLE_PATTERN) & (attr_type.isin(['not_documented','unclear']))

print(f"\n=== BUG CHECK 1: rule-out language present, but NOT labeled ruled_out (fix not applied here) ===")
print(f"  {should_be_ruledout.sum()} rows")
if should_be_ruledout.sum() > 0:
    for _, r in rows[should_be_ruledout].head(10).iterrows():
        print(f"  {r['PAT_MRN_ID']} (attr_type={r['attr_type']}): {r['evidence_text'][:120]}")

print(f"\n=== BUG CHECK 2: positive language present, but labeled not_documented/unclear (fix not applied here) ===")
print(f"  {should_be_positive_a.sum()} rows")
if should_be_positive_a.sum() > 0:
    for _, r in rows[should_be_positive_a].head(10).iterrows():
        print(f"  {r['PAT_MRN_ID']} (attr_type={r['attr_type']}): {r['evidence_text'][:120]}")

print(f"\n=== BUG CHECK 3 (NEW): pt6-style 'iso/in the setting of TTR' language, mislabeled ===")
print(f"  {should_be_positive_b.sum()} rows")
if should_be_positive_b.sum() > 0:
    for _, r in rows[should_be_positive_b].head(10).iterrows():
        print(f"  {r['PAT_MRN_ID']} (attr_type={r['attr_type']}): {r['evidence_text'][:120]}")

total_flagged_patients = set(rows.loc[should_be_ruledout | should_be_positive_a | should_be_positive_b, 'PAT_MRN_ID'])
print(f"\n=== SUMMARY ===")
print(f"Unique patients among the 978 with at least one unresolved bug-pattern row: {len(total_flagged_patients)}")
print(f"({100*len(total_flagged_patients)/len(positive_mrns):.1f}% of 978)")