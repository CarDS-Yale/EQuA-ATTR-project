

import pandas as pd, re
c = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
FAM = [('V142I (legacy V122I)', r"V\s*142\s*I\b|V\s*122\s*I\b|Val\s*-?\s*1[24]2\s*(?:[Il]le|Ille)|c\.424G>A"),
       ('V142L as written (likely I/l typo of V142I)', r"V142L\b"),
       ('V30M (V50M)', r"V\s*30\s*M\b|V\s*50\s*M\b|Val\s*-?\s*[35]0\s*Met|c\.148G>A"),
       ('T60A (T80A)', r"T\s*60\s*A\b|T\s*80\s*A\b|Thr\s*-?\s*[68]0\s*Ala|c\.238A>G"),
       ('L58H (L78H)', r"L58H\b|L78H\b|Leu\s*-?\s*[57]8\s*His"),
       ('I68L (I88L)', r"I68L\b|I88L\b|Ile\s*-?\s*[68]8\s*Leu"),
       ('F64L (F84L)', r"F64L\b|F84L\b|Phe\s*-?\s*[68]4\s*Leu"),
       ('A97S (A117S)', r"A97S\b|A117S\b|Ala\s*-?\s*(?:97|117)\s*Ser"),
       ('E89Q (E109Q)', r"E89Q\b|E109Q\b|Glu\s*-?\s*(?:89|109)\s*Gln")]
names = c['variant_names'].fillna('').astype(str)
geno = c['genotype'].fillna('')
has = geno.isin(['HEREDITARY_VARIANT', 'CONFLICTING'])
print(f"(not counted: patients with a variant NAME but a non-variant genotype, e.g. a negative test for V142I: {int(((names.str.strip() != '') & ~has).sum())})")
d = c[has].copy(); nm = names[has]
def fams(s):
    out = {lab for lab, rx in FAM if re.search(rx, s, re.I)}
    if not out:
        out = {'unspecified ("TTR mutation", none named)' if s.strip() == '' or not re.search(r"\d", s) else 'other named variant'}
    return out
d['fam'] = [fams(s) for s in nm]
t = d['final_classification']
d['group'] = 'other tiers'
d.loc[t == 'CONFIRMED_POSITIVE', 'group'] = 'ATTR-CM positive (validated by a/b/e)'
d.loc[t == 'TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE', 'group'] = 'carrier, no disease evidence'
print(f"Patients with a variant genotype: {len(d)} (unique MRN {d['PAT_MRN_ID'].nunique()})")
print(d['group'].value_counts().to_string())
ex = d.explode('fam').reset_index(drop=True)
tab = pd.crosstab(ex['fam'], ex['group'])
print("\nUNIQUE PATIENTS per variant family (a patient with two named variants is counted in both rows, so rows can add to more than the group size):")
print(tab.to_string())
print(f"\nPatients with 2+ different variant families named: {int((d['fam'].apply(len) > 1).sum())}")
oth = nm[d['fam'].apply(lambda f: 'other named variant' in f).values]
if len(oth):
    print("\n'other named variant' strings as written (variant names only):")
    print(oth.value_counts().head(20).to_string())