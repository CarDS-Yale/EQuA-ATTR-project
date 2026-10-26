


import pandas as pd
c = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/tmp_confirmed634.csv', dtype=str)
k = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_candidate_cohort.csv', dtype=str)
t = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_care_gap_timeline.csv', dtype=str, usecols=['PAT_MRN_ID', 'first_signal_date']).drop_duplicates('PAT_MRN_ID')
conf = c[(c['final_classification'] == 'CONFIRMED_POSITIVE') & (c['positive_evidence_source'] != 'MEDICATION+DX')]
cols = ['PAT_MRN_ID', 'flag_medication_cm_specific', 'flag_medication_pn_primary_only', 'med_order_evidence', 'med_admin_evidence']
conf = conf.drop(columns=[x for x in cols[1:] + ['first_signal_date'] if x in conf.columns])
m = conf.merge(k[cols].drop_duplicates('PAT_MRN_ID'), on='PAT_MRN_ID', how='left').merge(t, on='PAT_MRN_ID', how='left')
T = lambda s: s.fillna('').astype(str).str.strip().str.lower().isin({'true', '1', 'yes', '1.0'})
txt = (m['med_order_evidence'].fillna('') + ' ' + m['med_admin_evidence'].fillna('')).str.lower()
strict = T(m['flag_medication_cm_specific'])
cm_ind = strict | txt.str.contains('vutrisiran|amvuttra')
any_attr = cm_ind | T(m['flag_medication_pn_primary_only']) | txt.str.contains('patisiran|onpattro|inotersen|tegsedi')
alive = m['DEATH_DATE'].isna() | m['DEATH_DATE'].fillna('').str.strip().eq('')
after = pd.to_datetime(m['first_signal_date'], errors='coerce') >= pd.Timestamp('2019-05-01')
print(f"Identified without medication criteria: {len(m)}")
def rep(mask, label):
    n = int(mask.sum())
    print(f"\n{label}: n = {n}")
    for nm, on in [('tafamidis/acoramidis (current)', strict), ('+ vutrisiran (all ATTR-CM-indicated)', cm_ind), ('any ATTR-directed drug (+ patisiran/inotersen)', any_attr)]:
        k_ = int((mask & on).sum())
        print(f"   {nm}: treated {k_} | NOT treated {n-k_} ({100*(n-k_)/n:.1f}%)" if n else f"   {nm}: n/a")
rep(after & alive, "A) first signal after May 2019 AND alive")
rep(alive, "B) alive at last data capture (no date restriction)")
print(f"\nvutrisiran mentioned in med records among the {len(m)}: {int(txt.str.contains('vutrisiran|amvuttra').sum())}")
