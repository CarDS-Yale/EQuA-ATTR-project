

import pandas as pd
c = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
m = c['PAT_MRN_ID'].fillna('').str.strip()
norm = m.str.lstrip('0')
print(f"rows {len(c)} | unique MRN {m.nunique()} | blank MRN {int((m == '').sum())} | unique after trimming spaces and leading zeros {norm.nunique()}")
pos = c['final_classification'] == 'CONFIRMED_POSITIVE'
print(f"POSITIVE rows {int(pos.sum())} | unique MRN {m[pos].nunique()} | unique normalized {norm[pos].nunique()}")
if 'PAT_ID' in c.columns:
    pid = c['PAT_ID'].fillna('').str.strip()
    d = c[pid != ''].groupby(pid[pid != ''])['PAT_MRN_ID'].nunique()
    print(f"PAT_IDs linked to 2+ different MRNs (same person twice): {int((d > 1).sum())}")
T = lambda col: c[col].fillna('').str.strip().str.lower().isin({'true', '1', 'yes', '1.0'}) if col in c.columns else pd.Series(False, index=c.index)
rule_tier = c['tier_before_adjudication'].fillna('') if 'tier_before_adjudication' in c.columns else pd.Series('', index=c.index)
rule_tier = rule_tier.where(rule_tier != '', c['final_classification'])
held = rule_tier.eq('REVIEW_NOTE_VS_NEG_IMAGING')
recount = T('cat_a') | T('cat_b') | (T('cat_e') & ~held)
adj = c['adjudication'].fillna('') if 'adjudication' in c.columns else pd.Series('', index=c.index)
recount = recount.where(adj == '', adj.eq('ATTRCM_POS'))
print(f"Independent recount from cat_a / cat_b / cat_e + your decisions: {int(recount.sum())} | agrees with the tier for {int((recount == pos).sum())} of {len(c)} patients")
print(f"positives with NONE of a, b, e and no hand decision: {int((pos & ~(T('cat_a') | T('cat_b') | T('cat_e')) & (adj == '')).sum())} (must be 0)")