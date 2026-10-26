

import pandas as pd, re, os, csv, glob, sys, random
RAW = sys.argv[1] if len(sys.argv) > 1 else "/home/fao9/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23"
c = pd.read_csv('tmp_confirmed634.csv', dtype=str)
pos = c[c['final_classification'] == 'CONFIRMED_POSITIVE'].copy()
norm = lambda s: s.fillna('').astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
pos['MRN'] = norm(pos['PAT_MRN_ID'])
mrns = set(pos['MRN'])
CAT = {
 'FLC': re.compile(r"free\s+light\s+chain|\bFLCs?\b|kappa\s*/\s*lambda|kappa[^.;]{0,25}lambda|light\s+chain\s+ratio|\bK\s*/\s*L\s+ratio", re.I),
 'IFE': re.compile(r"immuno-?fixation|\bS?IFE\b|\bUIFE\b", re.I),
 'SPEP/UPEP/M-protein': re.compile(r"\bSPEP\b|\bUPEP\b|protein\s+electrophoresis|\bM[- ]?spike\b|monoclonal\s+(?:protein|gammopathy|spike|band|component)|paraprotein|\bMGUS\b", re.I)}
ANY = re.compile("|".join(r.pattern for r in CAT.values()), re.I)
found = {}; snips = {}; rows_634 = 0; notes_634 = set(); seen = set()
files = sorted(glob.glob(os.path.join(RAW, '*Notes*.txt')))
print("note files:", [os.path.basename(f) for f in files])
for f in files:
    with open(f, 'r', errors='replace') as fh:
        head = fh.readline()
    sep = max(['|', '\t'], key=head.count)
    n0 = rows_634
    for ch in pd.read_csv(f, sep=sep, dtype=str, chunksize=500000, quoting=csv.QUOTE_NONE, on_bad_lines='skip',
                          engine='python' if sep == '|' else 'c', usecols=lambda x: x in {'PAT_MRN_ID', 'NOTE_ID', 'NOTE_TEXT'}):
        m = norm(ch['PAT_MRN_ID'])
        sub = ch[m.isin(mrns)]; who = m[m.isin(mrns)]
        rows_634 += len(sub); notes_634 |= set(sub['NOTE_ID'].dropna()); seen |= set(who)
        txt = sub['NOTE_TEXT'].fillna('')
        hit = txt.str.contains(ANY)
        for w, t in zip(who[hit], txt[hit]):
            s = found.setdefault(w, set())
            for k, r in CAT.items():
                mm = r.search(t)
                if mm:
                    s.add(k)
                    if len(snips.setdefault(w, [])) < 3:
                        a, b = max(0, mm.start() - 120), min(len(t), mm.end() + 120)
                        snips[w].append(f"[{k}] ...{t[a:b]}...")
    print(f"  {os.path.basename(f)}: note lines belonging to the 634 = {rows_634 - n0}")
n = len(pos); anyp = set(found)
print(f"\nNote lines for the 634: {rows_634} | distinct notes: {len(notes_634)} | patients with any note: {len(seen)} of {len(pos)}")
print(f"\nANY AL work-up mention in notes (FLC, immunofixation, SPEP/UPEP or M-protein): {len(anyp)} of {n} ({100*len(anyp)/n:.1f}%)")
for k in CAT:
    print(f"   {k}: {len({w for w, s in found.items() if k in s})}")
both = {w for w, s in found.items() if 'FLC' in s and ('IFE' in s or 'SPEP/UPEP/M-protein' in s)}
print(f"   complete screen documented (FLC + immunofixation or SPEP/UPEP): {len(both)} ({100*len(both)/n:.1f}%)")
src = pos['positive_evidence_source'].fillna('')
g = set(pos.loc[src.str.contains('PYP_IMAGING_SCRIPT') & ~src.str.contains('NOTES_BIOPSY_STATEMENT'), 'MRN'])
print(f"\nPYP-positive WITHOUT documented biopsy (AL exclusion required): {len(g)}")
print(f"   any AL work-up mention: {len(g & anyp)} ({100*len(g & anyp)/len(g):.1f}%) | complete screen: {len(g & both)} ({100*len(g & both)/len(g):.1f}%) | NO mention: {len(g - anyp)} ({100*len(g - anyp)/len(g):.1f}%)")
random.seed(7)
rows = []
for w in random.sample(sorted(g & anyp), min(10, len(g & anyp))):
    rows.append({'group': 'mention found', 'PAT_MRN_ID': w, 'snippets': ' || '.join(snips.get(w, []))})
for w in random.sample(sorted(g - anyp), min(5, len(g - anyp))):
    rows.append({'group': 'NO mention found', 'PAT_MRN_ID': w, 'snippets': ''})
pd.DataFrame(rows).to_csv('al_notes_spotcheck.csv', index=False)
print("\nSpot-check list (10 with a mention, 5 without) saved to al_notes_spotcheck.csv -- read on the server, do not paste.")
print(f"\nOf the {len(g - anyp)} PYP-without-biopsy patients with NO mention, {len((g - anyp) - seen)} have no notes at all in the extract")
print(f"Among PYP-without-biopsy patients WITH notes: {len(g & seen)} | no AL mention: {len((g - anyp) & seen)} ({100*len((g - anyp) & seen)/max(len(g & seen),1):.1f}%)")
