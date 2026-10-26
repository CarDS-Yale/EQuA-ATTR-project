


import pandas as pd, os, glob, csv
BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
NOTES_DIR = "/home/fao9/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23/"
OUT = f"{BASE}/validation_notes"
os.makedirs(OUT, exist_ok=True)
sheet = pd.read_csv(f"{BASE}/validation_sheet.csv", dtype=str)
g = pd.read_csv(f"{BASE}/attr_genetics_ALL_combined.csv", dtype=str, usecols=['PAT_MRN_ID', 'NOTE_ID'])
g = g[g['PAT_MRN_ID'].isin(set(sheet['PAT_MRN_ID']))].drop_duplicates()
mrn2sid = dict(zip(sheet['PAT_MRN_ID'], sheet['sample_id']))
note2sid = {nid: mrn2sid[m] for m, nid in zip(g['PAT_MRN_ID'], g['NOTE_ID'])}
want = set(note2sid)
print(f"{len(sheet)} sampled patients | {len(want)} keyword-hit notes to rebuild")
files = sorted(glob.glob(os.path.join(NOTES_DIR, '*Notes*.txt')))
print("Note files:", [os.path.basename(f) for f in files])
parts = []
for f in files:
    with open(f, 'r', errors='replace') as fh:
        head = fh.readline()
    sep = max(['|', '\t', ','], key=head.count)
    cols = {c.strip().upper(): c for c in next(csv.reader([head], delimiter=sep))}
    need = ['NOTE_ID', 'LINE', 'NOTE_TEXT']
    if not all(k in cols for k in need):
        print(f"  skip {os.path.basename(f)}: columns NOTE_ID / LINE / NOTE_TEXT not all found")
        continue
    hits = 0
    for ch in pd.read_csv(f, sep=sep, dtype=str, usecols=[cols[k] for k in need], chunksize=500000,
                          engine='python' if sep == '|' else 'c', on_bad_lines='skip',
                          quoting=csv.QUOTE_NONE if sep != ',' else csv.QUOTE_MINIMAL):
        ch.columns = need
        ch = ch[ch['NOTE_ID'].isin(want)]
        if len(ch):
            ch = ch.assign(SRC=os.path.basename(f))
            parts.append(ch)
            hits += len(ch)
    print(f"  {os.path.basename(f)}: {hits} matching line-fragments")
if not parts:
    raise SystemExit("No matching notes found. Check NOTES_DIR and the file names.")
d = pd.concat(parts)
d['LINE_N'] = pd.to_numeric(d['LINE'], errors='coerce')
d = d.sort_values(['NOTE_ID', 'LINE_N'])
notes = d.groupby('NOTE_ID').agg(SRC=('SRC', 'first'), TEXT=('NOTE_TEXT', lambda x: ' '.join(x.fillna(''))))
found = 0
for sid in sheet['sample_id']:
    ids = [n for n, s_ in note2sid.items() if s_ == sid and n in notes.index]
    with open(f"{OUT}/{sid}.txt", 'w') as fh:
        fh.write(f"{sid}: {len(ids)} keyword-hit notes (full text, rebuilt from the raw note files)\n")
        for k, nid in enumerate(ids, 1):
            fh.write(f"\n===== note {k}/{len(ids)} ({notes.loc[nid, 'SRC']}) =====\n{notes.loc[nid, 'TEXT']}\n")
    found += bool(ids)
print(f"Wrote {OUT}/V01.txt ... ({found} of {len(sheet)} patients have at least one note found)")




























'''
import pandas as pd, os, glob, csv
BASE = os.environ.get('VAL_BASE', "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results")
NOTES_DIR = '/home/fao9/mnt_2/quality_meas_attr_cm_project/cardsjdat-CC1022-MEDINT/2837199-EQuA-ATTR/data/data-2026-07-23/'
OUT = f"{BASE}/validation_notes"
os.makedirs(OUT, exist_ok=True)
sheet = pd.read_csv(f"{BASE}/validation_sheet.csv", dtype=str)
g = pd.read_csv(f"{BASE}/attr_genetics_ALL_combined.csv", dtype=str, usecols=['PAT_MRN_ID', 'NOTE_ID'])
g = g[g['PAT_MRN_ID'].isin(set(sheet['PAT_MRN_ID']))].drop_duplicates()
mrn2sid = dict(zip(sheet['PAT_MRN_ID'], sheet['sample_id']))
note2sid = {nid: mrn2sid[m] for m, nid in zip(g['PAT_MRN_ID'], g['NOTE_ID'])}
want = set(note2sid)
print(f"{len(sheet)} sampled patients | {len(want)} keyword-hit notes to rebuild")
files = sorted(glob.glob(os.path.join(NOTES_DIR, '*Notes*.txt')))
print("Note files:", [os.path.basename(f) for f in files])
parts = []
for f in files:
    with open(f, 'r', errors='replace') as fh:
        head = fh.readline()
    sep = max(['|', '\t', ','], key=head.count)
    cols = {c.strip().upper(): c for c in next(csv.reader([head], delimiter=sep))}
    need = ['NOTE_ID', 'LINE', 'NOTE_TEXT']
    if not all(k in cols for k in need):
        print(f"  skip {os.path.basename(f)}: columns NOTE_ID / LINE / NOTE_TEXT not all found"); continue
    hits = 0
    for ch in pd.read_csv(f, sep=sep, dtype=str, usecols=[cols[k] for k in need], chunksize=500000,
                          engine='python' if sep == '|' else 'c', on_bad_lines='skip', quoting=csv.QUOTE_NONE if sep != ',' else csv.QUOTE_MINIMAL):
        ch.columns = need
        ch = ch[ch['NOTE_ID'].isin(want)]
        if len(ch):
            ch = ch.assign(SRC=os.path.basename(f)); parts.append(ch); hits += len(ch)
    print(f"  {os.path.basename(f)}: {hits} matching line-fragments")
if not parts:
    raise SystemExit("No matching notes found. Check NOTES_DIR and the file names.")
d = pd.concat(parts)
d['LINE_N'] = pd.to_numeric(d['LINE'], errors='coerce')
d = d.sort_values(['NOTE_ID', 'LINE_N'])
notes = d.groupby('NOTE_ID').agg(SRC=('SRC', 'first'), TEXT=('NOTE_TEXT', lambda x: ' '.join(x.fillna(''))))
found = 0
for sid in sheet['sample_id']:
    ids = [n for n, s_ in note2sid.items() if s_ == sid and n in notes.index]
    with open(f"{OUT}/{sid}.txt", 'w') as fh:
        fh.write(f"{sid}: {len(ids)} keyword-hit notes (full text, rebuilt from the raw note files)\n")
        for k, nid in enumerate(ids, 1):
            fh.write(f"\n===== note {k}/{len(ids)} ({notes.loc[nid, 'SRC']}) =====\n{notes.loc[nid, 'TEXT']}\n")
    found += bool(ids)
print(f"Wrote {OUT}/V01.txt ... ({found} of {len(sheet)} patients have at least one note found)")
'''