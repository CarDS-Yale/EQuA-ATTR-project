

#!/usr/bin/env python3
"""
Record hand decisions on the review lists, one keystroke per patient.
Keys: r = ATTR-CM negative | l = real positive | c = no disease evidence, genetics only | a = ask postdoc | s = skip | q = quit
Add a short reason after the key (e.g. "l biopsy-proven wtATTR-CM, on Vyndamax").
Each decision is appended to manual_adjudications_log.txt as:  MRN | CODE | reason | date
(l -> ATTRCM_POS, r -> ATTRCM_NEG, c -> TTR_CARRIER, a -> ASK). Quit any time; run it again to resume.
It changes nothing else. run_final_numbers.sh applies the decisions.
"""
import pandas as pd, re, os, datetime
from collections import Counter
BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
LOG = f"{BASE}/manual_adjudications_log.txt"
tty = open(os.environ.get('REVIEW_INPUT', '/dev/tty'))
def ask(prompt):
    print(prompt, end='', flush=True)
    line = tty.readline()
    return 'q' if line == '' else line.strip()
raw = pd.read_csv(f"{BASE}/attr_genetics_ALL_combined.csv", dtype=str,
                  usecols=['PAT_MRN_ID', 'NOTE_ID', 'attr_type', 'evidence_text'])
t = raw['attr_type'].fillna('').str.lower()
ev = raw['evidence_text'].fillna('')
NEG = re.compile(r"(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT))[^.;]{0,40}?\b(?:negative|normal|unremarkable)\b|\b(?:negative|normal)\b[^.;]{0,25}?(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT))", re.I)
POS = re.compile(r"\bpositive\b|abnormal|grade\s*[23]|score\s*(?:of|was)?\s*[23]|(?<!not )consistent with TTR|false[- ]negative|falsely[- ]negative|despite (?:a |the )?(?:negative|normal)", re.I)
def snippet(text, n=260):
    text = str(text)
    m = NEG.search(text)
    if m:
        a, b = max(0, m.start() - 110), min(len(text), m.end() + 110)
        return ("..." if a > 0 else "") + text[a:b] + ("..." if b < len(text) else "")
    return text[:n]
negrow = ev.str.contains(NEG)
clean = negrow & ~ev.where(negrow, '').str.contains(POS)
vw = t.isin(['wild_type', 'variant'])
pv = raw[vw].assign(clean_neg=clean[vw])
allneg = pv.groupby('PAT_MRN_ID')['clean_neg'].all()
flip = set(allneg[allneg].index)
has_neg = set(pv.loc[pv['clean_neg'], 'PAT_MRN_ID'])
def read_ids(name):
    f = f"{BASE}/{name}"
    return list(pd.read_csv(f, dtype=str)['PAT_MRN_ID']) if os.path.exists(f) else []
lists = [(read_ids('review_note_vs_neg_imaging.csv'), 'REVIEW tier: a note says ATTR-CM, imaging is negative'),
         (sorted(flip), 'negative-PYP wording: every labeled row is negative'),
         (sorted(has_neg - flip), 'negative-PYP wording: has other labeled rows (read them)'),
         (read_ids('review_genotype_only.csv'), 'genotype only: variant or conflicting, no disease evidence')]
why, order = {}, []
for ids, label in lists:
    for x in ids:
        if x not in why:
            why[x] = label
            order.append(x)
show = raw[vw | t.eq('ruled_out') | negrow].copy()
show['clean'] = clean[show.index]
show['neg'] = negrow[show.index]
by = {k: v for k, v in show.groupby('PAT_MRN_ID')}
pyp_by = {}
if os.path.exists(f"{BASE}/attr_pyp_results_latest.csv"):
    pp = pd.read_csv(f"{BASE}/attr_pyp_results_latest.csv", dtype=str)
    pyp_by = {k: v for k, v in pp.groupby('PAT_MRN_ID')}
rule = {}
if os.path.exists(f"{BASE}/attr_cm_final_classification.csv"):
    cc = pd.read_csv(f"{BASE}/attr_cm_final_classification.csv", dtype=str,
                     usecols=lambda x: x in {'PAT_MRN_ID', 'final_classification', 'src_pyp_report', 'src_med_dx',
                                             'src_notes_negative_imaging', 'genotype'})
    rule = cc.drop_duplicates('PAT_MRN_ID').set_index('PAT_MRN_ID').to_dict('index')
CODES = {'r': 'ATTRCM_NEG', 'l': 'ATTRCM_POS', 'c': 'TTR_CARRIER', 'a': 'ASK'}
decided = {}
if os.path.exists(LOG):
    for line in open(LOG):
        mm = re.match(r"^\s*(\S+)\s*\|\s*(ATTRCM_POS|ATTRCM_NEG|TTR_CARRIER|ASK)\s*\|", line)
        if mm: decided[mm.group(1)] = mm.group(2)
todo = [x for x in order if x not in decided]
print(f"{len(order)} patients on the review lists; {len(order) - len(todo)} already decided; {len(todo)} left.")
print("Keys: r = ATTR-CM negative | l = real positive | c = no disease evidence, genetics only | a = ask postdoc | s = skip | q = quit")
print("Add a short reason after the key, e.g.  r AL excluded.  Each decision is saved at once and OVERRIDES the rule in the next classification run.")
quit_all = False
for i, mrn in enumerate(todo, 1):
    print(f"\n[{i}/{len(todo)}] {mrn}  --  {why[mrn]}")
    rows = by.get(mrn)
    if rows is None:
        print("   (no labeled rows found for this patient)")
    else:
        for _, r in rows.head(14).iterrows():
            kind = 'NEG-ONLY' if r['clean'] else ('MIXED' if r['neg'] else ('ruled_out' if r['attr_type'] == 'ruled_out' else 'other'))
            print(f"   [{kind}] ({r['attr_type']}) NOTE_ID={r.get('NOTE_ID', '?')}: {snippet(r['evidence_text'])}")
        if len(rows) > 14:
            print(f"   ... and {len(rows) - 14} more rows")
    pg = pyp_by.get(mrn)
    if pg is None:
        print("   [PYP REPORT] none found for this patient")
    else:
        for _, r in pg.head(6).iterrows():
            print(f"   [PYP REPORT] result={r.get('pyp_result', '?')} | Perugini={r.get('perugini_grade', '?')} | {str(r.get('evidence_text', ''))[:220]}")
    if mrn in rule:
        q = rule[mrn]
        print(f"   [RULE'S VIEW] tier={q.get('final_classification')} | PYP report={q.get('src_pyp_report')} | drug+code={q.get('src_med_dx')} "
              f"| negative-scan wording in notes={q.get('src_notes_negative_imaging')} | genotype={q.get('genotype')}")
    while True:
        ans = ask("  decision > ")
        key, reason = ans[:1].lower(), ans[1:].strip()
        if key == 'q':
            quit_all = True
            break
        if key == 's':
            break
        if key in CODES:
            with open(LOG, 'a') as f:
                f.write(f"{mrn} | {CODES[key]} | {reason} | {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
            decided[mrn] = CODES[key]
            break
        print("  please type r, l, c, a, s or q (optionally followed by a reason)")
    if quit_all:
        break
print("\nDecisions so far on these lists:", dict(Counter(decided[x] for x in order if x in decided)))
print("Nothing else was changed. Run  bash run_final_numbers.sh  to apply your decisions.")