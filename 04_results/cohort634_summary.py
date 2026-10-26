


import pandas as pd, re
c = pd.read_csv('tmp_confirmed634.csv', dtype=str)
conf = c[c['final_classification'] == 'CONFIRMED_POSITIVE'].copy()
n = len(conf)
pct = lambda k, d: f"{k} ({100*k/d:.1f}%)" if d else f"{k}"
print(f"=== CORROBORATED ATTR-CM POSITIVES: {n} ===")
g = conf['genotype'].fillna('')
tested = g.isin(['HEREDITARY_VARIANT', 'NO_VARIANT_FOUND', 'CONFLICTING'])
print("\n-- TTR genetic testing --")
print(f"documented result: {pct(int(tested.sum()), n)}")
print(f"   variant found (hereditary ATTR-CM): {pct(int((g=='HEREDITARY_VARIANT').sum()), n)}")
print(f"   no variant found (wild-type): {pct(int((g=='NO_VARIANT_FOUND').sum()), n)}")
print(f"   conflicting: {int((g=='CONFLICTING').sum())}")
print(f"ordered, no result documented: {pct(int((g=='ORDERED_NO_RESULT').sum()), n)}")
print(f"no genetic testing documented: {pct(int((g=='NO_RESULT_DOCUMENTED').sum()), n)}")
print(f"CHECK sum = {int(tested.sum()) + int((g=='ORDERED_NO_RESULT').sum()) + int((g=='NO_RESULT_DOCUMENTED').sum())} (must be {n})")
if 'presumed_subtype' in conf.columns:
    print("\nAmong 'no testing documented', subtype presumed from notes:")
    print(conf.loc[g=='NO_RESULT_DOCUMENTED', 'presumed_subtype'].fillna('none stated').value_counts().to_string())
FAM = [('V142I (V122I)', r"V\s*142\s*I\b|V\s*122\s*I\b|Val\s*-?\s*1[24]2\s*(?:[Il]le|Ille)|c\.424G>A|Val142I\b"),
       ('V30M (V50M)', r"V\s*30\s*M\b|V\s*50\s*M\b|Val\s*-?\s*[35]0\s*Met|c\.148G>A"),
       ('T60A (T80A)', r"T\s*60\s*A\b|T\s*80\s*A\b|Thr\s*-?\s*[68]0\s*Ala|c\.238A>G"),
       ('F64L (F84L)', r"F64L\b|F84L\b|Phe\s*-?\s*[68]4\s*Leu"),
       ('I68L (I88L)', r"I68L\b|I88L\b|I{1,2}[el]e?\s*-?\s*[68]8\s*Leu")]
def fam(s):
    out = {lab for lab, rx in FAM if re.search(rx, s, re.I)}
    if not out:
        out = {'unspecified' if not re.search(r"\d", s) else 'other named variant'}
    return out
def variant_table(df, label):
    nm = df['variant_names'].fillna('').astype(str) if 'variant_names' in df.columns else pd.Series('', index=df.index)
    print(f"\n-- Variant type, {label} (unique patients; n={len(df)}) --")
    print(nm.apply(fam).explode().value_counts().to_string())
variant_table(conf[g == 'HEREDITARY_VARIANT'], 'hereditary ATTR-CM (variant WITH disease)')
variant_table(c[c['final_classification'] == 'TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE'], 'carriers WITHOUT disease evidence')
print("\n-- Evidence supporting the corroborated positives --")
src = conf['positive_evidence_source'].fillna('')
for lab, rx in [('drug + amyloid code', 'MEDICATION\\+DX'), ('positive PYP report', 'PYP_IMAGING_SCRIPT'), ('note statement (with other evidence)', 'NOTES_DISEASE_STATEMENT'), ('note-documented biopsy', 'NOTES_BIOPSY_STATEMENT')]:
    print(f"{lab}: {pct(int(src.str.contains(rx).sum()), n)}")
drug = src.str.contains('MEDICATION\\+DX')
print(f"identifiable by drug+code: {pct(int(drug.sum()), n)} | needed PYP or biopsy note (no drug+code): {pct(int((~drug).sum()), n)}")
print(f"\nLLM-only positives set aside (NLP_ONLY_UNCONFIRMED): {int((c['final_classification']=='NLP_ONLY_UNCONFIRMED').sum())}")
