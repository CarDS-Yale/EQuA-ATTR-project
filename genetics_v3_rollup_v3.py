

#!/usr/bin/env python3
"""
EQuA-ATTR genetics v3: combine BOTH servers' outputs and roll up to ONE ROW PER PATIENT.
Roll-up v3 (Nov 10, 2026, after hand-checking rows). Variants whose numbering system can't be told apart are listed as OTHER for hand review. Prints COUNTS only (safe to paste); writes a patient-level CSV (keep on server).

    python3 genetics_v3_rollup.py \
        --inputs genetics_v3_full2_vllm.csv genetics_v3_full2_ollama.csv \
        --report-subset tmp_confirmed634.csv --subset-tier CONFIRMED_POSITIVE \
        --out genetics_v3_full2_patient_rollup.csv

What v2 adds:
  1. Label clean-up: off-list labels mapped (recommended/declined -> not_tested; pending -> tested_no_result;
     invalid labels naming a non-TTR gene -> other_gene_only).
  2. GENE CHECK: a positive/VUS row that names a non-TTR gene (BRCA, APC, MUTYH, factor V Leiden ...)
     and has NO TTR wording is moved to other_gene_only (counted and flagged).
  3. Variant names normalized for the common TTR variants (legacy and precursor numbering merged):
     V122I = V142I = Val122Ile = Val142Ile = p.V142I = c.424G>A  ->  "V142I (V122I)", etc.
     Other variants are kept as written and flagged for hand review (numbering can be ambiguous).
  4. Quote check by status (is each answer supported by a sentence really in the note?) and a
     verbatim-only sensitivity analysis.
What v3 adds (from Favour's row checks):
  5. GENE CHECK now covers negative / tested_no_result / not_tested too (pheo, MTHFR, BRCA/MSH3 ... with no
     TTR wording -> other_gene_only), and positive/VUS whose variant position is impossible for TTR
     (TTR has 147 amino acids: p.Met458Ile or c.1374G>A cannot be TTR) -> other_gene_only.
  6. OFF-TOPIC QUOTE: a TTR status whose quote has no genetic wording at all (e.g. a smoking-history
     sentence) -> not_mentioned.
  7. GENE UNSTATED: quote mentions genetic testing but names no gene ("Declined genetic testing") -> kept,
     flagged; --ttr-named-only drops them (sensitivity analysis).
  8. not_tested split by reason from the quote: declined / not_indicated / not_done / planned_or_recommended / unclear.
 10. ABOUT-PATIENT CHECK: about_patient=False is kept only if the quote is a statement about a relative
     ("his mother had", "father has", "family history"); a relative acting as caregiver/spokesperson does not count;
     otherwise it is set back to the patient. Reports patients whose only TTR statements were about relatives.
 11. ATTR / ATTR-CM / wtATTR / wild-type wording counts as TTR wording (never flagged "gene unstated").
  9. Counts rows the extractor demoted not_tested -> not_mentioned because the quote did not match the note.
Patient rule (statements about the PATIENT, TTR only):
    positive and negative both present -> conflicting
    else positive > vus > negative > tested_no_result > not_tested ; no TTR statement -> not_documented
"""
import re
import argparse
import pandas as pd

ORDER = ["positive", "vus", "negative", "tested_no_result", "not_tested"]
RESULT = {"positive", "vus", "negative", "conflicting"}

NON_TTR = re.compile(
    r"\b(?:BRCA ?1|BRCA ?2|BRCA|APC|MUTYH|BRIP1|CHEK2|ATM|PALB2|MLH1|MSH2|MSH6|PMS2|EPCAM|Lynch|TP53|CDH1|PTEN|STK11|"
    r"RAD51[CD]?|BARD1|NBN|CDKN2A|TTN|MYH7|MYBPC3|LMNA|DSP|PKP2|FLNC|TNNT2|SCN5A|RBM20|GLA|Fabry|HFE|CFTR|"
    r"factor V|Leiden|prothrombin|G20210A|MTHFR|JAK2|CALR|MPL|BCR.?ABL|FLT3|NPM1|KRAS|EGFR|ALK|BRAF|HLA|"
    r"MSH3|SDH[ABCD]?|VHL|NF1|pheos?|pheochromocytomas?|paragangliomas?|hereditary cancer|cancer panel|"
    r"thrombophilia|hemochromatosis|Huntington|cystic fibrosis|sickle|thalassemia|Myriad)\b",
    re.IGNORECASE)
TTR_WORDS = re.compile(
    r"\bTTR\b|transthyretin|\bhATTR|\bATTRv|\bvATTR|V1[24]2I|Val1[24]2Ile|V[35]0M|Val[35]0Met|T[68]0A|Thr[68]0Ala|"
    r"c\.424G>A|amyloid|\b(?:wt|h|v|a)?-?ATTR(?:-?(?:CM|PN|wt|v))?\b|wild[- ]?type", re.IGNORECASE)

GENETIC_WORDS = re.compile(
    r"genetic|\bgenes?\b|genotyp|mutation|variant|\bcarrier|sequenc|\btest|\bTTR\W{0,3}(?:neg|pos|wild|wt)|"
    r"hATTR|ATTRv|vATTR|Invitae|Ambry|GeneDx|counsel|heterozyg|homozyg|V1[24]2I|Val1[24]2Ile|\bVUS\b|p\.[A-Z]|c\.\d",
    re.IGNORECASE)
AA3 = "Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val"
P_POS = re.compile(rf"\b(?:p\.\(?(?:{AA3}|[A-Z])|(?:{AA3}))(\d+)")
C_POS = re.compile(r"\bc\.(\d+)")


def impossible_for_ttr(t):
    """TTR precursor = 147 amino acids (coding c.1-444). A higher position cannot be TTR."""
    t = str(t)
    return any(int(x) > 147 for x in P_POS.findall(t)) or any(int(x) > 444 for x in C_POS.findall(t))


NT_DECLINED = re.compile(r"declin|refus|did not want|does not want|not interested|deferred by patient", re.I)
NT_NOTIND = re.compile(r"not indicated|not necessary|not needed|not warranted|no indication|not recommended", re.I)
NT_NOTDONE = re.compile(r"not (?:been |yet )?(?:performed|done|completed|obtained|sent|tested)|never|no genetic test|"
                        r"has not had|hasn't had|was not tested|not undergone", re.I)
NT_PLANNED = re.compile(r"recommend|refer|will |plan|consider|should|\bif\b|would|counsel|order|discuss|offer|need", re.I)


def nt_reason(q):
    q = str(q)
    if NT_DECLINED.search(q):
        return "declined"
    if NT_NOTIND.search(q):
        return "not_indicated"
    if NT_NOTDONE.search(q):
        return "not_done"
    if NT_PLANNED.search(q):
        return "planned_or_recommended"
    return "unclear"


# Common TTR variants: (canonical label, patterns for legacy + precursor/HGVS spellings)
CANON = [
    ("V142I (V122I)", r"V1[24]2I|Val1[24]2Ile|c\.424G>A|c\.G424A|Val-?1[24]2-?Ile"),
    ("V50M (V30M)",   r"V[35]0M|Val[35]0Met|c\.148G>A"),
    ("T80A (T60A)",   r"T[68]0A|Thr[68]0Ala|c\.238A>G"),
    ("F64S (F44S)",   r"p\.(?:Phe64Ser|F64S)|c\.191T>C|\bF44S\b|Phe44Ser"),      # bare "F64S" is ambiguous -> hand review
    ("A140S (A120S)", r"p\.(?:Ala140Ser|A140S)|c\.418G>T|\bA120S\b|Ala120Ser"),  # bare "A140S" is ambiguous -> hand review
    ("E109Q (E89Q)",  r"E(?:89|109)Q|Glu(?:89|109)Gln"),
    ("I88L (I68L)",   r"I[68]8L|Ile[68]8Leu"),
    ("L131M (L111M)", r"L1[13]1M|Leu1[13]1Met"),
    ("D58A (D38A)",   r"D[35]8A|Asp[35]8Ala"),
]
CANON = [(n, re.compile(p, re.IGNORECASE)) for n, p in CANON]
JUNK = {"", "vus", "not mentioned", "not specified", "none", "unknown", "n/a", "na", "not_stated", "not stated"}


def canon_variant(v):
    v = str(v).strip().strip('"').strip("'").strip()
    if v.lower() in JUNK or not re.search(r"\d", v):
        return ""
    for name, pat in CANON:
        if pat.search(v):
            return name
    return "OTHER: " + v[:60]


ap = argparse.ArgumentParser()
ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--cohort", default=None, help="CSV of all evaluated patients (PAT_MRN_ID); missing -> not_documented")
ap.add_argument("--report-subset", default=None, help="CSV to report on, e.g. tmp_confirmed634.csv")
ap.add_argument("--subset-tier", default=None, help="keep only this final_classification in the subset")
ap.add_argument("--verbatim-only", action="store_true", help="sensitivity analysis: ignore rows whose quote is not in the note")
ap.add_argument("--ttr-named-only", action="store_true",
                help="sensitivity analysis: ignore rows whose quote names no gene (\"declined genetic testing\")")
ap.add_argument("--out", default="genetics_v3_patient_rollup.csv")
a = ap.parse_args()

frames = []
for p in a.inputs:
    d = pd.read_csv(p, dtype=str).fillna("")
    frames.append(d)
    print(f"{p.split('/')[-1]}: {len(d):,} rows, {d['PAT_MRN_ID'].nunique():,} patients")
df = pd.concat(frames, ignore_index=True)

print("\nCHECKS")
n0 = len(df)
dup = df[df["NOTE_ID"] != ""].duplicated("NOTE_ID", keep=False)
print(f"  rows {n0:,} | duplicate NOTE_IDs across/within files: {int(dup.sum()):,}")
df = df.drop_duplicates("NOTE_ID", keep="first")
print(f"  blank PAT_MRN_ID: {int((df['PAT_MRN_ID']=='').sum()):,} | errors: {int((df['ttr_genetic_status']=='error').sum()):,}")
both = df.groupby("PAT_MRN_ID")["backend"].nunique()
print(f"  patients with notes from BOTH servers: {int((both>1).sum()):,}")

# 1. label clean-up
st = df["ttr_genetic_status"].str.strip().str.lower()
syn = {"invalid:recommended": "not_tested", "invalid:planned": "not_tested", "invalid:declined": "not_tested",
       "invalid:referred": "not_tested", "invalid:pending": "tested_no_result", "invalid:ordered": "tested_no_result"}
st = st.replace(syn)
inv = st.str.startswith("invalid:")
inv_gene = inv & st.str.contains(NON_TTR) & ~st.str.contains(TTR_WORDS)
st = st.where(~inv_gene, "other_gene_only")
inv_left = st.str.startswith("invalid:")
print(f"  off-list labels: {int(inv.sum()):,} (non-TTR gene -> other_gene_only: {int(inv_gene.sum()):,}; "
      f"unresolved, excluded: {int(inv_left.sum()):,})")

# 2. gene check on positive / vus
text = df["variant"].fillna("") + " | " + df["evidence_text"].fillna("")
pv = st.isin(["positive", "vus"])
gene_bad = pv & text.str.contains(NON_TTR) & ~text.str.contains(TTR_WORDS)
st = st.where(~gene_bad, "other_gene_only")
pos_bad = pv & ~gene_bad & text.map(impossible_for_ttr) & ~df["variant"].fillna("").map(lambda v: canon_variant(v) != "" and not canon_variant(v).startswith("OTHER")) & ~text.str.contains(TTR_WORDS)
st = st.where(~pos_bad, "other_gene_only")
rest = st.isin(["negative", "tested_no_result", "not_tested"])
gene_bad2 = rest & text.str.contains(NON_TTR) & ~text.str.contains(TTR_WORDS)
st = st.where(~gene_bad2, "other_gene_only")
orig = df["ttr_genetic_status"].str.strip().str.lower()
print(f"  GENE CHECK (-> other_gene_only, no TTR wording in quote):")
print(f"    positive/VUS naming a non-TTR gene:        {int(gene_bad.sum()):,} "
      f"(positive {int((gene_bad & (orig=='positive')).sum()):,}, vus {int((gene_bad & (orig=='vus')).sum()):,})")
print(f"    positive/VUS with a position impossible for TTR (>147 aa / >c.444): {int(pos_bad.sum()):,}")
print(f"    negative/tested_no_result/not_tested naming a non-TTR gene: {int(gene_bad2.sum()):,} "
      + ", ".join(f"{k} {int((gene_bad2 & (orig==k)).sum()):,}" for k in ["negative", "tested_no_result", "not_tested"]))

# off-topic quote: a TTR status whose quote has no genetic wording at all
q = df["evidence_text"].fillna("")
off = st.isin(ORDER) & (q.str.strip() != "") & ~q.str.contains(GENETIC_WORDS)
print(f"  OFF-TOPIC QUOTE (no genetic wording, e.g. a smoking sentence) -> not_mentioned: {int(off.sum()):,} "
      + ", ".join(f"{k} {int((off & (st==k)).sum()):,}" for k in ORDER))
st = st.where(~off, "not_mentioned")
df["status_clean"] = st
df["gene_check_moved"] = gene_bad | pos_bad | gene_bad2
df["quote_off_topic"] = off
df["gene_unstated"] = st.isin(ORDER) & ~text.str.contains(TTR_WORDS)
print(f"  GENE UNSTATED (quote is about genetic testing but names no gene; kept, flagged):")
for k in ORDER:
    m = st == k
    if m.any():
        print(f"    {k:17s} {int((m & df['gene_unstated']).sum()):7,} / {int(m.sum()):7,} "
              f"({100*(m & df['gene_unstated']).sum()/m.sum():.0f}%)")
dem = df.get("rule_applied", pd.Series("", index=df.index)) == "not_tested_quote_not_in_note->not_mentioned"
qq = q[dem].str.strip()
print(f"  DEMOTED by the extractor (not_tested, quote not found in note -> not_mentioned): {int(dem.sum()):,} rows, "
      f"{df.loc[dem, 'PAT_MRN_ID'].nunique():,} patients; quote wrapped in quote marks: "
      f"{int(qq.str.match(r'^[\"\u201c\u2018\']').sum()):,}  (these may be real; see the rescue check)")

# 4. quote check by status
ver = df["evidence_verbatim"].astype(str).str.lower() == "true"
has_q = df["evidence_text"] != ""
print("\nQUOTE CHECK by status (rows with a quote: verbatim / total)")
for k in ORDER + ["other_gene_only"]:
    m = (df["status_clean"] == k) & has_q
    if m.any():
        print(f"  {k:17s} {int((m & ver).sum()):7,} / {int(m.sum()):7,} ({100*(m & ver).sum()/m.sum():.0f}%)")
if a.ttr_named_only:
    drop = df["gene_unstated"]
    print(f"\n--ttr-named-only: dropping {int(drop.sum()):,} rows whose quote names no gene")
    df = df[~drop]
if a.verbatim_only:
    keep = ~has_q | ver | ~df["status_clean"].isin(ORDER)
    print(f"\n--verbatim-only: dropping {int((~keep).sum()):,} rows whose quote is not in the note")
    df = df[keep]

# 3. variants
df["variant_canon"] = ""
m = df["status_clean"].isin(["positive", "vus"])
df.loc[m, "variant_canon"] = df.loc[m, "variant"].map(canon_variant)

# about-patient check
_R = (r"(?:mother|mom|father|dad|sisters?|brothers?|siblings?|sons?|daughters?|child|children|kids?|cousins?|aunts?|"
      r"uncles?|nieces?|nephews?|grand\w*|relatives?|parents?|twin|offspring|proband)")
# a statement ABOUT a relative ("his mother had", "father has V122I", "family history"), not a relative acting
# as caregiver ("discussed with son; patient declined")
REL = re.compile(rf"\b(?:his|her|their|patient'?s|pt'?s)\s+{_R}\b|\b{_R}\s+(?:had|has|was|is|with|carries|carrying|"
                 rf"tested|testing|who|also|diagnosed|positive|negative)\b|family (?:history|hx)|\bFHx?\b", re.I)
ap_false = df["about_patient"].str.lower() == "false"
flip = ap_false & ~df["evidence_text"].fillna("").str.contains(REL)
df.loc[flip, "about_patient"] = "True"
print(f"\nABOUT-PATIENT CHECK: about_patient=False rows {int(ap_false.sum()):,}; no relative named in quote -> "
      f"set back to patient: {int(flip.sum()):,}")
ttr_rows = df["status_clean"].isin(ORDER)
rel_only = set(df.loc[ttr_rows & (df["about_patient"].str.lower() == "false"), "PAT_MRN_ID"]) - \
    set(df.loc[ttr_rows & (df["about_patient"].str.lower() != "false"), "PAT_MRN_ID"])
print(f"  patients whose ONLY TTR statements are about a relative (not counted; check some by hand): {len(rel_only):,}")

# patient roll-up
usable = df[df["status_clean"].isin(ORDER) & (df["about_patient"].str.lower() != "false")]
rows = []
for pid, g in df.groupby("PAT_MRN_ID"):
    u = usable[usable["PAT_MRN_ID"] == pid]
    s = set(u["status_clean"])
    cat = "conflicting" if ("positive" in s and "negative" in s) else next((o for o in ORDER if o in s), "not_documented")
    vs = sorted({v for v in u.loc[u["status_clean"].isin(["positive", "vus"]), "variant_canon"] if v})
    pick = u[u["status_clean"] == ("positive" if cat == "conflicting" else cat)]
    ntr = ""
    if cat == "not_tested":
        rs = set(u.loc[u["status_clean"] == "not_tested", "evidence_text"].map(nt_reason))
        ntr = next((r for r in ["declined", "not_indicated", "not_done", "planned_or_recommended", "unclear"] if r in rs), "")
    rows.append({"PAT_MRN_ID": pid, "genetic_v3": cat, "variants": "; ".join(vs),
                 "clinical_label_wild_type": "wild_type" in set(g["clinical_label"]),
                 "not_tested_reason": ntr,
                 "ttr_named_in_deciding_quotes": bool(len(pick)) and not pick["gene_unstated"].all(),
                 "n_notes": len(g), "servers": ",".join(sorted(set(g["backend"]))),
                 "other_gene_only_notes": int((g["status_clean"] == "other_gene_only").sum()),
                 "deciding_NOTE_ID": pick.iloc[0]["NOTE_ID"] if len(pick) else "",
                 "deciding_evidence": pick.iloc[0]["evidence_text"] if len(pick) else ""})
pt = pd.DataFrame(rows)
if a.cohort:
    ids = set(pd.read_csv(a.cohort, dtype=str)["PAT_MRN_ID"].dropna())
    miss = ids - set(pt["PAT_MRN_ID"])
    pt = pd.concat([pt, pd.DataFrame({"PAT_MRN_ID": sorted(miss), "genetic_v3": "not_documented", "n_notes": 0})])
    # keep EVERY patient with notes; cohort patients without notes are added as not_documented
pt.to_csv(a.out, index=False)
print(f"\nsaved {a.out} ({len(pt):,} patients; one row each: {pt['PAT_MRN_ID'].is_unique}) -- patient-level, keep on server")


def report(sub, label):
    n = len(sub)
    vc = sub["genetic_v3"].value_counts()
    print(f"\n{label}: n = {n:,}")
    for k in ["positive", "vus", "negative", "conflicting", "tested_no_result", "not_tested", "not_documented"]:
        v = int(vc.get(k, 0))
        print(f"  {k:18s} {v:7,} ({100*v/max(n,1):5.1f}%)")
    nt = sub[sub["genetic_v3"] == "not_tested"]
    if len(nt):
        print("    not_tested by reason: " + ", ".join(f"{k} {v:,}" for k, v in nt["not_tested_reason"].value_counts().items()))
    dec = sub[sub["genetic_v3"].isin(ORDER + ["conflicting"])]
    if len(dec):
        tn = dec["ttr_named_in_deciding_quotes"].fillna(False).astype(bool)
        print("    status backed by a quote that names TTR/amyloid/variant: " + ", ".join(
            f"{k} {int(tn[dec['genetic_v3']==k].sum()):,}/{int((dec['genetic_v3']==k).sum()):,}"
            for k in ["positive", "vus", "negative", "conflicting", "tested_no_result", "not_tested"]
            if (dec['genetic_v3']==k).any()))
    doc = int(sub["genetic_v3"].isin(RESULT).sum())
    print(f"  -> documented TTR result: {doc:,} ({100*doc/max(n,1):.1f}%) | none: {n-doc:,} ({100*(n-doc)/max(n,1):.1f}%)")
    pos = sub[sub["genetic_v3"].isin(["positive", "conflicting"])]
    if len(pos):
        named = (pos["variants"].fillna("") != "").sum()
        print(f"  positive patients with a variant named: {int(named):,}/{len(pos):,} ({100*named/len(pos):.0f}%)")
        vv = pos["variants"].fillna("").str.split("; ").explode()
        vv = vv[vv != ""]
        top = vv[~vv.str.startswith("OTHER")].value_counts()
        print("  variants (patients): " + ", ".join(f"{k} {v}" for k, v in top.items()))
        oth = vv[vv.str.startswith("OTHER")]
        if len(oth):
            print(f"  other variant spellings (patients, review by hand): {oth.nunique():,} distinct, {len(oth):,} mentions")
    nr = sub[~sub["genetic_v3"].isin(RESULT)]
    print(f"  without a result but labeled wild-type clinically: "
          f"{int(nr['clinical_label_wild_type'].fillna(False).astype(bool).sum()):,}")


report(pt, "ALL patients with notes (+ cohort patients without notes)")
if a.cohort:
    report(pt[pt["PAT_MRN_ID"].isin(ids)], f"EVALUATED COHORT {a.cohort.split('/')[-1]}")
if a.report_subset:
    s = pd.read_csv(a.report_subset, dtype=str)
    if a.subset_tier:
        s = s[s["final_classification"] == a.subset_tier]
    report(pt.set_index("PAT_MRN_ID").reindex(sorted(set(s["PAT_MRN_ID"]))).fillna(
        {"genetic_v3": "not_documented", "variants": ""}).reset_index(),
        f"SUBSET {a.report_subset.split('/')[-1]}" + (f" [{a.subset_tier}]" if a.subset_tier else ""))






