

import pandas as pd, re, os, csv, sys
from collections import Counter
RAW = sys.argv[1] if len(sys.argv) > 1 else "/home/fao9/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
FILES = ["CarDS_2837199_Hosp_Enc_Labs_%d.txt" % i for i in range(1, 7)] + ["CarDS_2837199_Outpatient_Enc_Labs_%d.txt" % i for i in range(1, 4)]
c = pd.read_csv('tmp_confirmed634.csv', dtype=str)
pos = c[c['final_classification'] == 'CONFIRMED_POSITIVE'].copy()
norm = lambda s: s.fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
pos['MRN'] = norm(pos['PAT_MRN_ID'])
pos['PID'] = norm(pos['PAT_ID']) if 'PAT_ID' in pos.columns else ''
mrns = set(pos['MRN']) - {''}
pid2mrn = {p: m for p, m in zip(pos['PID'], pos['MRN']) if p}
print(f"634 check: {len(pos)} positives | with MRN {len(mrns)} | with PAT_ID {len(pid2mrn)}")
CAT = {'FLC': re.compile(r"kappa|lambda|light\s*chain|\bFLC\b", re.I),
       'IFE': re.compile(r"immunofix|\bIFE\b", re.I),
       'SPEP/UPEP': re.compile(r"electrophor|\bSPEP\b|\bUPEP\b|\bPEP\b|M[- ]?spike|paraprotein|monoclonal", re.I)}
ANY = re.compile("|".join(r.pattern for r in CAT.values()), re.I)
found = {}; abn = set(); names = Counter(); rows_matched_id = 0
for fn in FILES:
    p = os.path.join(RAW, fn)
    if not os.path.exists(p):
        print("  missing:", fn); continue
    n0 = rows_matched_id
    for ch in pd.read_csv(p, sep='\t', dtype=str, chunksize=500000, quoting=csv.QUOTE_NONE, on_bad_lines='skip',
                          usecols=lambda x: x in {'PAT_ID', 'PAT_MRN_ID', 'PROC_NAME', 'COMPONENT_NAME', 'ABNORMAL_YN', 'RESULT_FLAG'}):
        mrn = norm(ch['PAT_MRN_ID']) if 'PAT_MRN_ID' in ch else pd.Series('', index=ch.index)
        pid = norm(ch['PAT_ID']) if 'PAT_ID' in ch else pd.Series('', index=ch.index)
        who = mrn.where(mrn.isin(mrns), pid.map(pid2mrn).fillna(''))
        keep = who != ''
        rows_matched_id += int(keep.sum())
        if not keep.any(): continue
        sub = ch[keep]; who = who[keep]
        txt = (sub.get('PROC_NAME', '').fillna('') + ' | ' + sub.get('COMPONENT_NAME', '').fillna('')).astype(str)
        hit = txt.str.contains(ANY)
        for w, t, a, f in zip(who[hit], txt[hit], sub.get('ABNORMAL_YN', pd.Series('', index=sub.index))[hit].fillna(''), sub.get('RESULT_FLAG', pd.Series('', index=sub.index))[hit].fillna('')):
            s = found.setdefault(w, set())
            for k, r in CAT.items():
                if r.search(t): s.add(k)
            names[t.split(' | ')[-1].strip().upper()] += 1
            if str(a).strip().upper() == 'Y' or str(f).strip().lower() in {'h', 'l', 'high', 'low', 'abnormal', 'a', 'hh', 'll'}:
                abn.add(w)
    print(f"  {fn}: rows belonging to the 634 = {rows_matched_id - n0}")
print(f"\nRows in lab files belonging to the 634: {rows_matched_id}  (if 0, it is an ID-matching problem, not a test-name problem)")
n = len(pos)
anyp = set(found)
print(f"\nANY AL test (FLC, immunofixation or SPEP/UPEP): {len(anyp)} of {n} ({100*len(anyp)/n:.1f}%)")
for k in CAT:
    print(f"   {k}: {len({w for w, s in found.items() if k in s})}")
both = {w for w, s in found.items() if 'FLC' in s and ('IFE' in s or 'SPEP/UPEP' in s)}
print(f"   complete screen (FLC + immunofixation or electrophoresis): {len(both)}")
print(f"   at least one AL test flagged abnormal: {len(abn & anyp)}")
src = pos['positive_evidence_source'].fillna('')
g = set(pos.loc[src.str.contains('PYP_IMAGING_SCRIPT') & ~src.str.contains('NOTES_BIOPSY_STATEMENT'), 'MRN'])
print(f"\nPYP-positive WITHOUT a documented biopsy (non-biopsy diagnosis, AL exclusion required): {len(g)}")
print(f"   any AL test: {len(g & anyp)} ({100*len(g & anyp)/max(len(g),1):.1f}%) | complete screen: {len(g & both)} | NO AL test found: {len(g - anyp)} ({100*len(g - anyp)/max(len(g),1):.1f}%)")
print("\nTop matched lab test names (names only):")
for k, v in names.most_common(25):
    print(f"   {v:6d}  {k[:80]}")
