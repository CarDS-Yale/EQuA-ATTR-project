

import pandas as pd, re, sys
f = sys.argv[1] if len(sys.argv) > 1 else 'tmp_confirmed634.csv'
c = pd.read_csv(f, dtype=str)
def P(leg_from, pos, to, aa_from, aa_to):
    pre = pos + 20
    one = rf"\b{leg_from}\s*-?\s*(?:{pos}|{pre})\s*{to}\b"
    three = rf"\b{aa_from}\s*-?\s*(?:{pos}|{pre})\s*{aa_to}"
    return one + "|" + three
VAR = [
 ('V122I / p.V142I', 'cardiac-predominant', P('V', 122, 'I', 'Val', r'(?:[Il]le|Ille|I\b)') + r"|c\.424G>A"),
 ('I68L / p.I88L',   'cardiac-predominant', P('I', 68, 'L', r'I{1,2}[el]e?', 'Leu')),
 ('L111M / p.L131M', 'cardiac-predominant', P('L', 111, 'M', 'Leu', 'Met')),
 ('V30M / p.V50M',   'mixed (late onset)',   P('V', 30, 'M', 'Val', 'Met') + r"|c\.148G>A"),
 ('T60A / p.T80A',   'mixed',                P('T', 60, 'A', 'Thr', 'Ala') + r"|c\.238A>G"),
 ('E89Q / p.E109Q',  'mixed',                P('E', 89, 'Q', 'Glu', 'Gln')),
 ('P24S / p.P44S',   'mixed',                P('P', 24, 'S', 'Pro', 'Ser')),
 ('I107V / p.I127V', 'mixed',                P('I', 107, 'V', r'I{1,2}[el]e?', 'Val')),
 ('W41L / p.W61L',   'mixed',                P('W', 41, 'L', 'Trp', 'Leu')),
 ('T49A / p.T69A',   'mixed',                P('T', 49, 'A', 'Thr', 'Ala')),
 ('H88R / p.H108R',  'mixed',                P('H', 88, 'R', 'His', 'Arg')),
 ('V20I / p.V40I',   'mixed',                P('V', 20, 'I', 'Val', r'(?:[Il]le|I\b)')),
 ('A45T / p.A65T',   'mixed',                P('A', 45, 'T', 'Ala', 'Thr')),
 ('V142L as written (likely V142I typo)', 'cardiac-predominant (if typo)', r"\bV142L\b"),
]
def classify(s):
    hits = [(lab, ph) for lab, ph, rx in VAR if re.search(rx, s, re.I)]
    if hits: return hits
    if not re.search(r"\d", s): return [('unspecified ("TTR mutation", none named)', 'unknown')]
    return [('other named variant: ' + s[:60], 'not in reference list')]
groups = [('ATTR-CM positive (variant WITH disease)', (c['final_classification'] == 'CONFIRMED_POSITIVE') & c['genotype'].eq('HEREDITARY_VARIANT')),
          ('carrier WITHOUT disease evidence', c['final_classification'].eq('TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE'))]
for name, m in groups:
    d = c[m]; nm = d['variant_names'].fillna('').astype(str)
    rows = [(i, lab, ph) for i, s in zip(d.index, nm) for lab, ph in classify(s)]
    r = pd.DataFrame(rows, columns=['i', 'variant', 'phenotype'])
    print(f"\n=== {name}: {len(d)} patients ===")
    print("patients by phenotype group (a patient with 2 variants can appear in 2 groups):")
    print(r.drop_duplicates(['i', 'phenotype']).groupby('phenotype')['i'].nunique().sort_values(ascending=False).to_string())
    print("patients by variant:")
    print(r.drop_duplicates(['i', 'variant']).groupby('variant')['i'].nunique().sort_values(ascending=False).to_string())
    print(f"patients with 2+ different named variants: {int((r.drop_duplicates(['i','variant']).groupby('i').size() > 1).sum())}")
