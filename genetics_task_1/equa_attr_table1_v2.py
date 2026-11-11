
#!/usr/bin/env python3
"""
EQuA-ATTR Table 1 v2 (Nov 10, 2026): structured data only. Prints and saves AGGREGATE tables only.

    # 1) look at the column names first (prints headers only, no data):
    python3 equa_attr_table1_v2.py --data-dir <raw data folder> --inspect

    # 2) build the table:
    python3 equa_attr_table1_v2.py --data-dir <raw data folder> \
        --cohort tmp_confirmed634.csv --tier CONFIRMED_POSITIVE \
        --signal-file attr_cm_final_classification.csv \
        --genetics genetics_v3/genetics_v3_full2_patient_rollup.csv \
        --out-prefix table1_634

What it gives:
  * Age at 3 time points (mean ± SD, median [IQR]):
      - first ATTR signal  (first_signal_date from the classification file; UNRELIABLE, goes back decades)
      - diagnosis (proxy)  = first encounter with an ATTR ICD-10 code (E85.82 wild-type, E85.1x hereditary,
                             E85.4 organ-limited); a proxy, not a chart-confirmed diagnosis date
      - current / death    = age at death, or at the last date seen in the data for living patients
  * Sex, race, ethnicity, deceased; TTR genetic status (if --genetics given)
  * By age group (<65, 65-74, 75-84, >=85; --age-basis picks which age): top 10 ICD-10 categories and
    top 10 medications (% of patients), with ATTR therapies listed separately.
  * --min-cell 11 hides counts 1-10 (use for anything leaving the server, e.g. GitHub).
Files (found by name in --data-dir): *Patients.txt, *Hosp_Enc_DX.txt, *Outpatient_Enc_DX.txt,
*Meds.txt, *External_Meds.txt, *Med_Admin.txt. Column names are detected; --inspect shows them.
"""
import argparse
import glob
import os
import re
import sys
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--inspect", action="store_true", help="print each file's column names and exit")
ap.add_argument("--cohort")
ap.add_argument("--tier", default=None, help="keep only this final_classification in --cohort")
ap.add_argument("--signal-file", default=None, help="CSV with PAT_MRN_ID + a first_signal date column")
ap.add_argument("--class-file", default=None,
                help="attr_cm_final_classification.csv: uses cat_b / cat_e to find patients identified WITHOUT the "
                     "medication route (drug-independent therapy rate)")
ap.add_argument("--genetics", default=None, help="patient roll-up CSV (PAT_MRN_ID, genetic_v3)")
ap.add_argument("--age-basis", choices=["current", "diagnosis"], default="current",
                help="age used for the age groups (diagnosis falls back to current when missing)")
ap.add_argument("--min-cell", type=int, default=0)
ap.add_argument("--keep-z", action="store_true", help="keep Z-codes (encounters/screening) in the top ICD-10 list")
ap.add_argument("--meds-all-sources", action="store_true",
                help="top-medication list from all files incl. inpatient administration (default: outpatient "
                     "prescriptions + external meds only; ATTR therapy always uses all files)")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--chunksize", type=int, default=500_000)
ap.add_argument("--out-prefix", default="table1")
a = ap.parse_args()


def find(pattern):
    hits = sorted(glob.glob(os.path.join(a.data_dir, "**", pattern), recursive=True))
    return hits


def sep_of(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        line = f.readline()
    return max(["\t", "|", ","], key=line.count)


def header(path):
    s = sep_of(path)
    cols = pd.read_csv(path, sep=s, nrows=0, encoding="utf-8-sig", encoding_errors="replace").columns
    return s, [c.replace("﻿", "").strip() for c in cols]


def pick(cols, *cands, contains=None):
    up = {c.upper(): c for c in cols}
    for c in cands:
        if c.upper() in up:
            return up[c.upper()]
    if contains:
        for c in cols:
            if all(k in c.upper() for k in contains):
                return c
    return None


FILES = {
    "patients": find("*Patients.txt"),
    "dx": find("*Hosp_Enc_DX.txt") + find("*Outpatient_Enc_DX.txt"),
    "meds": find("*Meds.txt") + find("*Med_Admin.txt"),
}
FILES["meds"] = sorted(set(FILES["meds"]))

if a.inspect:
    for k, ps in FILES.items():
        for p in ps:
            s, cols = header(p)
            print(f"[{k}] {os.path.basename(p)}  sep={repr(s)}\n    {cols}")
    if not any(FILES.values()):
        print("no files found; check --data-dir")
    sys.exit(0)

if not a.cohort:
    sys.exit("--cohort is required (unless --inspect)")


def sup(n):
    return f"<{a.min_cell}" if 0 < n < a.min_cell else f"{n:,}"


def pct(n, d):
    return f"{sup(n)} ({100*n/max(d,1):.1f})" if not (0 < n < a.min_cell) else f"{sup(n)}"


# ── cohort ────────────────────────────────────────────────────────────────
c = pd.read_csv(a.cohort, dtype=str)
if a.tier:
    c = c[c["final_classification"] == a.tier]
ids = set(c["PAT_MRN_ID"].dropna())
N = len(ids)
print(f"cohort: {N:,} patients")

# ── Patients.txt ─────────────────────────────────────────────────────────
pf = FILES["patients"][0]
s, cols = header(pf)
col = dict(mrn=pick(cols, "PAT_MRN_ID"), dob=pick(cols, "BIRTH_DATE", contains=["BIRTH"]),
           dod=pick(cols, "DEATH_DATE", contains=["DEATH"]), sex=pick(cols, "SEX", "GENDER"),
           race=pick(cols, "PATIENT_RACE_ALL", "RACE_ALL", "RACE", contains=["RACE"]),
           eth=pick(cols, "PATIENT_ETHNICITY", "ETHNICITY", "ETHNIC_GROUP", contains=["ETHNIC"]))
print("Patients.txt columns used:", col)
P = pd.read_csv(pf, sep=s, dtype=str, usecols=[v for v in col.values() if v], encoding="utf-8-sig", encoding_errors="replace")
P.columns = [c.replace("\ufeff", "").strip() for c in P.columns]
P = P.rename(columns={v: k for k, v in col.items() if v})  # by name, not position
P = P[P["mrn"].isin(ids)].drop_duplicates("mrn").set_index("mrn")
P["dob"] = pd.to_datetime(P["dob"], errors="coerce")
P["dod"] = pd.to_datetime(P.get("dod"), errors="coerce")
print(f"  found in Patients.txt: {len(P):,}/{N:,}")


# ── scan DX files: first ATTR-code date, last date seen, ICD-10 categories ──
ATTR_CODE = re.compile(r"^E85\.(?:82|1\d*|4)$")
last_seen = pd.Series(pd.NaT, index=P.index, dtype="datetime64[ns]")
first_attr = pd.Series(pd.NaT, index=P.index, dtype="datetime64[ns]")
pt_codes = {}      # mrn -> set of ICD-10 categories (3 characters)
pt_full = {}       # mrn -> set of full ICD-10 codes
cat_name = {}      # category -> most common DX_NAME seen
name_counts = {}


def upd_min(ser, new):
    new = new.groupby(level=0).min()
    cur = ser.reindex(new.index)
    ser.loc[new.index] = cur.where(cur.notna() & (cur <= new), new)


def upd_max(ser, new):
    new = new.groupby(level=0).max()
    cur = ser.reindex(new.index)
    ser.loc[new.index] = cur.where(cur.notna() & (cur >= new), new)


for f in FILES["dx"]:
    s, cols = header(f)
    mrn = pick(cols, "PAT_MRN_ID")
    icd = pick(cols, "CURRENT_ICD10_LIST", "ICD10", "ICD10_CODE", contains=["ICD10"])
    nm = pick(cols, "DX_NAME", contains=["DX_NAME"])
    dt = pick(cols, "CALC_DX_DATE", "CONTACT_DATE", "ENC_DATE", "DX_DATE", "HOSP_ADMSN_TIME", "ADM_DATE",
              "ADMIT_DATE", contains=["DATE"])
    dt2 = pick(cols, "DX_DATE") if dt != pick(cols, "DX_DATE") else None   # fills blanks in the first date
    print(f"{os.path.basename(f)}: mrn={mrn} icd10={icd} name={nm} date={dt} (blanks filled from {dt2})")
    if not (mrn and icd):
        print("  skipped (no PAT_MRN_ID or ICD-10 column)")
        continue
    use = [x for x in (mrn, icd, nm, dt, dt2) if x]
    for ch in pd.read_csv(f, sep=s, dtype=str, usecols=use, chunksize=a.chunksize, encoding="utf-8-sig", encoding_errors="replace",
                          on_bad_lines="skip"):
        ch = ch[ch[mrn].isin(ids)]
        if ch.empty:
            continue
        ch = ch.assign(_code=ch[icd].fillna("").str.split(r"[,;]")).explode("_code")
        ch["_code"] = ch["_code"].str.strip().str.upper()
        ch = ch[ch["_code"] != ""]
        ch["_cat"] = ch["_code"].str[:3]
        for m, g in ch.groupby(mrn)["_code"]:
            pt_full.setdefault(m, set()).update(g)
        for m, g in ch.groupby(mrn)["_cat"]:
            pt_codes.setdefault(m, set()).update(g)
        if nm:
            for (cat, n), k in ch.groupby(["_cat", nm]).size().items():
                name_counts[(cat, n)] = name_counts.get((cat, n), 0) + k
        if dt:
            d = pd.to_datetime(ch[dt], errors="coerce")
            if dt2:
                d = d.fillna(pd.to_datetime(ch[dt2], errors="coerce"))
            upd_max(last_seen, pd.Series(d.values, index=ch[mrn].values).dropna())
            am = ch["_code"].str.match(ATTR_CODE)
            upd_min(first_attr, pd.Series(d[am].values, index=ch.loc[am, mrn].values).dropna())
for (cat, n), k in sorted(name_counts.items(), key=lambda x: -x[1]):
    cat_name.setdefault(cat, re.sub(r"\s*\(HC CODE\)", "", str(n))[:60])

# ── meds ──────────────────────────────────────────────────────────────────
ATTR_RX = {"tafamidis": r"tafamidis|vyndaqel|vyndamax", "acoramidis": r"acoramidis|attruby",
           "vutrisiran": r"vutrisiran|amvuttra", "patisiran": r"patisiran|onpattro",
           "inotersen": r"inotersen|tegsedi", "eplontersen": r"eplontersen|wainua", "diflunisal": r"diflunisal"}
pt_meds = {}
pt_attr = {k: set() for k in ATTR_RX}


BRAND = {"eliquis": "apixaban", "xarelto": "rivaroxaban", "coumadin": "warfarin", "jantoven": "warfarin",
         "vyndamax": "tafamidis", "vyndaqel": "tafamidis", "attruby": "acoramidis", "amvuttra": "vutrisiran",
         "onpattro": "patisiran", "tegsedi": "inotersen", "wainua": "eplontersen", "lasix": "furosemide",
         "bumex": "bumetanide", "demadex": "torsemide", "soaanz": "torsemide", "aldactone": "spironolactone",
         "entresto": "sacubitril-valsartan", "jardiance": "empagliflozin", "farxiga": "dapagliflozin",
         "lipitor": "atorvastatin", "crestor": "rosuvastatin", "zocor": "simvastatin", "toprol": "metoprolol",
         "lopressor": "metoprolol", "coreg": "carvedilol", "plavix": "clopidogrel", "synthroid": "levothyroxine",
         "protonix": "pantoprazole", "prilosec": "omeprazole", "tylenol": "acetaminophen", "zofran": "ondansetron",
         "keflex": "cephalexin", "norvasc": "amlodipine", "flomax": "tamsulosin", "pacerone": "amiodarone",
         "lantus": "insulin", "humalog": "insulin", "ozempic": "semaglutide", "kerendia": "finerenone"}


def med_key(x):
    x = str(x).lower()
    x = re.split(r"\d|\(|,|\[", x)[0]
    x = re.sub(r"\b(tablet|tab|capsule|cap|oral|solution|injection|inj|er|xl|sr|dr|hcl|sodium|potassium|meglumine|calcium|magnesium|succinate|tartrate|besylate|mesylate|maleate|fumarate|citrate|acetate|sulfate|bromide|chloride|extended|release|delayed|chewable|disintegrating|mg|mcg)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()
    first = x.split(" ")[0] if x else ""
    if first in BRAND:
        return BRAND[first]
    if x.startswith("doxycycline"):
        return "doxycycline"
    if x.startswith("insulin"):
        return "insulin"
    return x


for f in FILES["meds"]:
    s, cols = header(f)
    mrn = pick(cols, "PAT_MRN_ID")
    nm = pick(cols, "GENERIC_NAME", "MEDICATION_NAME", "MED_NAME", "DISPLAY_NAME", "DESCRIPTION",
              contains=["NAME"])
    extra = [x for x in (pick(cols, "MEDICATION_NAME"), pick(cols, "ORDER_DESCRIPTION")) if x and x != nm]
    stc = pick(cols, "ORDER_STATUS")
    mode = pick(cols, "ORDERING_MODE")
    is_admin = "MED_ADMIN" in os.path.basename(f).upper()
    print(f"{os.path.basename(f)}: mrn={mrn} name={nm} also searched for ATTR drugs={extra} status={stc}")
    if not (mrn and nm):
        print("  skipped (no PAT_MRN_ID or medication name column)")
        continue
    for ch in pd.read_csv(f, sep=s, dtype=str, usecols=[mrn, nm] + extra + ([stc] if stc else []) + ([mode] if mode else []), chunksize=a.chunksize, encoding="utf-8-sig", encoding_errors="replace",
                          on_bad_lines="skip"):
        ch = ch[ch[mrn].isin(ids) & ch[nm].notna()]
        if stc:  # drop canceled orders
            ch = ch[~ch[stc].fillna("").str.contains("cancel", case=False)]
        if ch.empty:
            continue
        ch["_k"] = ch[nm].map(med_key)
        top = ch
        if not a.meds_all_sources:
            if is_admin:
                top = ch.iloc[0:0]
            elif mode:
                top = ch[ch[mode].fillna("").str.contains("outpatient", case=False)]
        for m, g in top.groupby(mrn)["_k"]:
            pt_meds.setdefault(m, set()).update(k for k in g if len(k) > 2)
        low = ch[[nm] + extra].fillna("").agg(" ".join, axis=1).str.lower()   # brand names live in MEDICATION_NAME
        for drug, pat in ATTR_RX.items():
            pt_attr[drug].update(ch.loc[low.str.contains(pat), mrn])

# ── ages ─────────────────────────────────────────────────────────────────
end = P["dod"].where(P["dod"].notna(), last_seen.reindex(P.index))
P["age_current"] = (end - P["dob"]).dt.days / 365.25
P["age_dx"] = (first_attr.reindex(P.index) - P["dob"]).dt.days / 365.25
P["age_signal"] = float("nan")
if a.signal_file:
    sg = pd.read_csv(a.signal_file, dtype=str)
    scol = next((x for x in sg.columns if "first_signal" in x.lower() and "date" in x.lower()), None)
    if scol:
        sd = pd.to_datetime(sg.drop_duplicates("PAT_MRN_ID").set_index("PAT_MRN_ID")[scol], errors="coerce")
        P["age_signal"] = (sd.reindex(P.index) - P["dob"]).dt.days / 365.25
    else:
        print("  no first_signal date column in --signal-file; columns:", list(sg.columns))
for k in ["age_signal", "age_dx", "age_current"]:
    P.loc[(P[k] < 0) | (P[k] > 110), k] = float("nan")

basis = P["age_current"] if a.age_basis == "current" else P["age_dx"].fillna(P["age_current"])
P["age_group"] = pd.cut(basis, [0, 65, 75, 85, 200], right=False, labels=["<65", "65-74", "75-84", ">=85"])

# ── output ───────────────────────────────────────────────────────────────
L = []
add = L.append
add(f"# Table 1. {os.path.basename(a.cohort)}" + (f" [{a.tier}]" if a.tier else "") + f" (n = {N:,})\n")
add("| Characteristic | Value |\n|---|---|")


def agerow(label, k):
    v = P[k].dropna()
    if len(v):
        add(f"| {label} | {v.mean():.1f} ± {v.std():.1f}; median {v.median():.1f} "
            f"[{v.quantile(.25):.1f}-{v.quantile(.75):.1f}]; available {len(v):,}/{N:,} |")
    else:
        add(f"| {label} | not available |")


add("| **Age, years** (mean ± SD; median [IQR]) | |")
agerow("At first ATTR signal (unreliable: earliest amyloid mention in any data)", "age_signal")
agerow("At diagnosis (proxy: first ATTR ICD-10 code E85.82/E85.1/E85.4)", "age_dx")
agerow("Current, or at death", "age_current")
gap = (P["age_current"] - P["age_dx"]).dropna()
if len(gap):
    add(f"| Years from diagnosis proxy to current/death | median {gap.median():.1f} [{gap.quantile(.25):.1f}-{gap.quantile(.75):.1f}] |")
add(f"| **Age group** (by {'current/death' if a.age_basis == 'current' else 'diagnosis proxy'} age) | |")
for g, n in P["age_group"].value_counts().reindex(["<65", "65-74", "75-84", ">=85"]).fillna(0).astype(int).items():
    add(f"| {g} | {pct(n, N)} |")
for lab, k in [("Sex", "sex"), ("Race", "race"), ("Ethnicity", "eth")]:
    if k in P:
        add(f"| **{lab}** | |")
        for v, n in P[k].fillna("Unknown").value_counts().items():
            add(f"| {v} | {pct(int(n), N)} |")
add(f"| **Deceased** (death date recorded) | {pct(int(P['dod'].notna().sum()), N)} |")
add(f"| Not found in Patients.txt | {pct(N - len(P), N)} |")
if a.genetics:
    gx = pd.read_csv(a.genetics, dtype=str).drop_duplicates("PAT_MRN_ID").set_index("PAT_MRN_ID")["genetic_v3"]
    gx = gx.reindex(sorted(ids)).fillna("not_documented")
    add("| **TTR genetic testing (LLM, notes)** | |")
    for v in ["positive", "vus", "negative", "conflicting", "tested_no_result", "not_tested", "not_documented"]:
        add(f"| {v} | {pct(int((gx == v).sum()), N)} |")
add("| **ATTR therapy ever recorded** (structured meds) | |")
for drug in ATTR_RX:
    add(f"| {drug} | {pct(len(pt_attr[drug]), N)} |")

add(f"\n## Top {a.top} ICD-10 categories{'' if a.keep_z else ' (Z-codes excluded)'} and medications "
    f"({'all sources' if a.meds_all_sources else 'outpatient prescriptions + external meds'}) by age group "
    f"(% of patients in the group; age basis: {a.age_basis})\n")
for g in ["<65", "65-74", "75-84", ">=85"]:
    m = P.index[P["age_group"] == g]
    n = len(m)
    add(f"\n### {g} (n = {sup(n)})\n")
    if n == 0:
        continue
    cc = pd.Series([x for p in m for x in pt_codes.get(p, ()) if a.keep_z or not x.startswith("Z")],
                   dtype=str).value_counts().head(a.top)
    mc = pd.Series([x for p in m for x in pt_meds.get(p, ())]).value_counts().head(a.top)
    add("| # | ICD-10 | Description | Patients n (%) | Medication | Patients n (%) |\n|---|---|---|---|---|---|")
    for i in range(max(len(cc), len(mc))):
        c1 = (cc.index[i], cat_name.get(cc.index[i], ""), pct(int(cc.iloc[i]), n)) if i < len(cc) else ("", "", "")
        m1 = (mc.index[i], pct(int(mc.iloc[i]), n)) if i < len(mc) else ("", "")
        add(f"| {i+1} | {c1[0]} | {c1[1]} | {c1[2]} | {m1[0]} | {m1[1]} |")
    add("\nATTR therapy in this group: " + ", ".join(
        f"{d} {pct(len(pt_attr[d] & set(m)), n)}" for d in ATTR_RX))

add("\n**Notes.** Diagnoses: any Hosp/Outpatient encounter ICD-10 code at any time, grouped to 3-character "
    "category. Medications: any order or administration at any time (name before the dose). The diagnosis-age "
    "proxy is the first ATTR-coded encounter; earlier diagnoses made outside the system are missed. First-signal "
    "age is shown for completeness only." + (f" Counts below {a.min_cell} are hidden." if a.min_cell else ""))

md = "\n".join(L)
open(a.out_prefix + ".md", "w").write(md)
print("\n" + md)
print(f"\nsaved {a.out_prefix}.md (aggregate only; safe to share)")

# ══ Journal-style Table 1: columns = Overall + age groups ═════════════════
GROUPS = ["<65", "65-74", "75-84", ">=85"]
COLS = [("Overall", P.index)] + [(g, P.index[P["age_group"] == g]) for g in GROUPS]
COMORB = [("Heart failure", ["I50"]), ("Hypertension", ["I10", "I11", "I12", "I13"]),
          ("Atrial fibrillation or flutter", ["I48"]), ("Chronic kidney disease", ["N18"]),
          ("Diabetes mellitus", ["E10", "E11"]), ("Hyperlipidemia", ["E78"]),
          ("Coronary artery disease", ["I25"]), ("Stroke or TIA", ["I63", "G45", "Z86.73"])]
REDFLAG = [("Aortic stenosis", ["I35.0", "I35.2"]), ("AV or bundle-branch block", ["I44", "I45"]),
           ("Pacemaker or ICD", ["Z95.0", "Z95.81", "Z45.0"]), ("Carpal tunnel syndrome", ["G56.0"]),
           ("Lumbar spinal stenosis", ["M48.0"]), ("Peripheral neuropathy", ["G60", "G62", "G63"])]


def has(m, prefixes):
    return any(c.startswith(px) for c in pt_full.get(m, ()) for px in prefixes)


def npct(mask_ids, denom_ids):
    d = len(denom_ids)
    if d == 0:
        return "–"
    k = len(set(mask_ids) & set(denom_ids))
    if 0 < k < a.min_cell:
        return f"<{a.min_cell}"
    return f"{k:,} ({100*k/max(d,1):.1f})"


def meansd(series, idx):
    v = series.reindex(idx).dropna()
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else "–"


race = P["race"].fillna("Unknown") if "race" in P else pd.Series("Unknown", index=P.index)
eth = P["eth"].fillna("Unknown") if "eth" in P else pd.Series("Unknown", index=P.index)
unk = race.str.lower().isin(["unknown", "not listed", "patient declined", "declined", "", "nan"])
RACE = [("White", race == "White"), ("Black or African American", race == "Black or African American"),
        ("Asian", race == "Asian"),
        ("Other or more than one", ~unk & ~race.isin(["White", "Black or African American", "Asian"])),
        ("Unknown or not listed", unk)]
eh = eth.str.lower()
ETH = [("Hispanic or Latino", eh.str.startswith("hispanic")), ("Not Hispanic or Latino", eh.str.startswith("not hisp")),
       ("Unknown", ~eh.str.startswith("hispanic") & ~eh.str.startswith("not hisp"))]
STAB = pt_attr["tafamidis"] | pt_attr["acoramidis"]
SIL = pt_attr["vutrisiran"] | pt_attr["patisiran"] | pt_attr["inotersen"] | pt_attr["eplontersen"]
ANYRX = STAB | SIL

rows = []   # (kind, label, [values...]) kind: section | row | sub
rows.append(("n", "Patients, n", [f"{len(ix):,}" for _, ix in COLS]))
rows.append(("section", "Demographics", None))
rows.append(("row", "Age at diagnosis proxy, y, mean ± SD", [meansd(P["age_dx"], ix) for _, ix in COLS]))
rows.append(("row", "Age at last follow-up or death, y, mean ± SD", [meansd(P["age_current"], ix) for _, ix in COLS]))
rows.append(("row", "Sex", None))
rows.append(("sub", "Male", [npct(P.index[P["sex"].str.lower() == "male"], ix) for _, ix in COLS]))
rows.append(("sub", "Female", [npct(P.index[P["sex"].str.lower() == "female"], ix) for _, ix in COLS]))
_other_sex = P.index[~P["sex"].fillna("").str.lower().isin(["male", "female"])]
if len(_other_sex):
    rows.append(("sub", "Other or unknown", [npct(_other_sex, ix) for _, ix in COLS]))
rows.append(("row", "Race", None))
for lab, msk in RACE:
    rows.append(("sub", lab, [npct(P.index[msk], ix) for _, ix in COLS]))
rows.append(("row", "Ethnicity", None))
for lab, msk in ETH:
    rows.append(("sub", lab, [npct(P.index[msk], ix) for _, ix in COLS]))
rows.append(("row", "Deceased", [npct(P.index[P["dod"].notna()], ix) for _, ix in COLS]))
rows.append(("section", "Comorbidities (ICD-10)", None))
for lab, px in COMORB:
    rows.append(("row", lab, [npct([m for m in ix if has(m, px)], ix) for _, ix in COLS]))
rows.append(("section", "ATTR red flags (ICD-10)", None))
for lab, px in REDFLAG:
    rows.append(("row", lab, [npct([m for m in ix if has(m, px)], ix) for _, ix in COLS]))
if a.genetics:
    rows.append(("section", "TTR genetic testing (clinical notes, LLM)", None))
    ment = gx.index[gx != "not_documented"]
    tested = gx.index[gx.isin(["positive", "vus", "negative", "conflicting", "tested_no_result"])]
    res = gx.index[gx.isin(["positive", "vus", "negative", "conflicting"])]
    rows.append(("row", "Testing mentioned", [npct(ment, ix) for _, ix in COLS]))
    rows.append(("row", "Tested (result or ordered)", [npct(tested, ix) for _, ix in COLS]))
    rows.append(("row", "Result documented", [npct(res, ix) for _, ix in COLS]))
    for lab, k in [("Pathogenic variant", ["positive"]), ("Variant of uncertain significance", ["vus"]),
                   ("No variant (wild-type)", ["negative"]), ("Conflicting reports", ["conflicting"])]:
        rows.append(("sub", lab, [npct(gx.index[gx.isin(k)], ix) for _, ix in COLS]))
rows.append(("section", "ATTR therapy ever ordered or given", None))
rows.append(("row", "TTR stabilizer (tafamidis or acoramidis)", [npct(STAB, ix) for _, ix in COLS]))
rows.append(("sub", "Tafamidis", [npct(pt_attr["tafamidis"], ix) for _, ix in COLS]))
rows.append(("sub", "Acoramidis", [npct(pt_attr["acoramidis"], ix) for _, ix in COLS]))
rows.append(("row", "TTR silencer (vutrisiran, patisiran, inotersen, eplontersen)", [npct(SIL, ix) for _, ix in COLS]))
rows.append(("row", "Diflunisal", [npct(pt_attr["diflunisal"], ix) for _, ix in COLS]))
rows.append(("row", "Any ATTR-specific therapy", [npct(ANYRX, ix) for _, ix in COLS]))
if a.class_file:
    cf = pd.read_csv(a.class_file, dtype=str).drop_duplicates("PAT_MRN_ID").set_index("PAT_MRN_ID")
    tru = lambda c: cf[c].fillna("").str.strip().str.lower().isin(["true", "1", "yes", "y"]) if c in cf else pd.Series(False, index=cf.index)
    indep = set(cf.index[tru("cat_b") | tru("cat_e")]) & set(P.index)
    print(f"drug-independent (cat_b or cat_e): {len(indep):,} of {len(P):,}")
    vals = []
    for _, ix in COLS:
        d = indep & set(ix)
        k = len(ANYRX & d)
        vals.append("–" if not d else f"<{a.min_cell}" if 0 < k < a.min_cell else f"{k:,}/{len(d):,} ({100*k/len(d):.1f})")
    rows.append(("row", "Any ATTR therapy, patients identified without the medication criterion", vals))

hdr = ["Characteristic", "Overall", "<65", "65–74", "75–84", "≥85"]
tl = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for kind, lab, vals in rows:
    if kind == "section":
        tl.append(f"| **{lab}** |" + " |" * (len(hdr) - 1))
    elif vals is None:
        tl.append(f"| {lab} |" + " |" * (len(hdr) - 1))
    else:
        tl.append(f"| {'&nbsp;&nbsp;' if kind == 'sub' else ''}{lab} | " + " | ".join(vals) + " |")
foot = (f"Values are n (%) unless stated. Age groups by {'age at last follow-up or death' if a.age_basis == 'current' else 'diagnosis-proxy age'}. "
        "Diagnosis proxy = first encounter with ICD-10 E85.82, E85.1x or E85.4. Comorbidities and red flags: any "
        "hospital or outpatient encounter code at any time. Therapy: any order or administration; an ATTR-specific "
        "medication was one route into the cohort, so therapy rates may be inflated; the last row restricts to patients "
        "identified without that route."
        + (f" Counts below {a.min_cell} are suppressed." if a.min_cell else ""))
strat = f"# Table 1. Characteristics of {len(P):,} corroborated ATTR-CM patients, by age group\n\n" + "\n".join(tl) + "\n\n" + foot
open(a.out_prefix + "_by_age.md", "w").write(strat)
pd.DataFrame([[lab if kind != "sub" else "  " + lab] + (vals or [""] * 5) for kind, lab, vals in rows],
             columns=hdr).to_csv(a.out_prefix + "_by_age.csv", index=False)
print("\n" + strat)
print(f"\nsaved {a.out_prefix}_by_age.md and {a.out_prefix}_by_age.csv (aggregate only; safe to share)")

try:   # publication-style picture of the table, if matplotlib is installed
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow = len(rows) + 3
    fig, ax = plt.subplots(figsize=(11, 0.24 * nrow + 0.9), dpi=200)
    ax.axis("off")
    xs = [0.0, 0.47, 0.585, 0.69, 0.795, 0.90]
    y = 1.0
    dy = 1.0 / nrow
    ax.text(0, y, f"Table 1. Characteristics of {len(P):,} corroborated ATTR-CM patients, overall and by age group",
            fontsize=10, fontweight="bold", va="top", transform=ax.transAxes)
    y -= dy * 1.3
    ax.plot([0, 1], [y, y], color="black", lw=1.2, transform=ax.transAxes, clip_on=False)
    ax.text((xs[2] + 1) / 2, y - dy * 0.45, "Age group (years)", ha="center", va="center", fontsize=8.5,
            fontweight="bold", transform=ax.transAxes)
    ax.plot([xs[2] - 0.02, 1], [y - dy * 0.85, y - dy * 0.85], color="black", lw=0.6, transform=ax.transAxes)
    y -= dy * 1.35
    for j, h in enumerate(hdr):
        ax.text(xs[j] + (0 if j == 0 else 0.05), y, h, ha="left" if j == 0 else "center", va="center", fontsize=8.5,
                fontweight="bold", transform=ax.transAxes)
    y -= dy * 0.6
    ax.plot([0, 1], [y, y], color="black", lw=0.8, transform=ax.transAxes)
    y -= dy * 0.6
    for kind, lab, vals in rows:
        bold = kind in ("section", "n")
        ax.text(0.015 if kind == "sub" else 0, y, lab, fontsize=8, va="center", fontweight="bold" if bold else "normal",
                transform=ax.transAxes)
        for j, v in enumerate(vals or []):
            ax.text(xs[j + 1] + 0.05, y, v, ha="center", va="center", fontsize=8, transform=ax.transAxes)
        y -= dy
    ax.plot([0, 1], [y + dy * 0.45, y + dy * 0.45], color="black", lw=1.2, transform=ax.transAxes)
    import textwrap
    ax.text(0, y - dy * 0.1, "\n".join(textwrap.wrap(foot, 170)), fontsize=6.5, va="top", color="#444444",
            transform=ax.transAxes)
    fig.savefig(a.out_prefix + "_by_age.png", bbox_inches="tight")
    fig.savefig(a.out_prefix + "_by_age.pdf", bbox_inches="tight")
    print(f"saved {a.out_prefix}_by_age.png / .pdf")
except ImportError:
    print("matplotlib not installed: no picture; send the _by_age.csv instead")


