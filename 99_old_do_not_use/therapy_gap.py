



import pandas as pd
final = pd.read_csv('attr_cm_final_classification.csv', dtype=str)

non_med_positive = final[
    (final['final_classification'] == 'CONFIRMED_POSITIVE') &
    (final['positive_evidence_source'].str.contains('NOTES_GENETICS|PYP_IMAGING_SCRIPT', na=False, regex=True))
]
on_therapy = non_med_positive['flag_medication_cm_specific'].astype(str).str.lower().eq('true').sum()
print(f'{len(non_med_positive)} patients confirmed positive via genetics/imaging (independent of medication)')
print(f'{on_therapy} ({100*on_therapy/len(non_med_positive):.1f}%) of those are documented on CM-specific therapy')
print(f'{len(non_med_positive)-on_therapy} ({100*(len(non_med_positive)-on_therapy)/len(non_med_positive):.1f}%) have NO documented CM-specific therapy — a real treatment gap')
















'''
import pandas as pd
final = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
cohort = pd.read_csv('attr_cm_candidate_cohort.csv', dtype=str)

non_med_positive = final[
    (final['final_classification'] == 'CONFIRMED_POSITIVE') &
    (final['positive_evidence_source'].str.contains('NOTES_GENETICS|PYP_IMAGING_SCRIPT', na=False, regex=True))
]
merged = non_med_positive.merge(cohort[['PAT_MRN_ID','flag_medication_cm_specific']], on='PAT_MRN_ID', how='left')
on_therapy = merged['flag_medication_cm_specific'].astype(str).str.lower().eq('true').sum()
print(f'{len(merged)} patients confirmed positive via genetics/imaging (independent of medication)')
print(f'{on_therapy} ({100*on_therapy/len(merged):.1f}%) of those are documented on CM-specific therapy')
print(f'{len(merged)-on_therapy} ({100*(len(merged)-on_therapy)/len(merged):.1f}%) have NO documented CM-specific therapy — a real treatment gap')

'''








'''

import pandas as pd
final = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
cohort = pd.read_csv('attr_cm_candidate_cohort.csv', dtype=str)
print('Cohort columns:', list(cohort.columns))

non_med_positive = final[
    (final['final_classification'] == 'CONFIRMED_POSITIVE') &
    (final['positive_evidence_source'].str.contains('NOTES_GENETICS|PYP_IMAGING_SCRIPT', na=False, regex=True))
]
merged = non_med_positive.merge(cohort[['PAT_MRN_ID','flag_medication_cm_specific']], on='PAT_MRN_ID', how='left')
on_therapy = merged['flag_medication_cm_specific'].astype(str).str.lower().eq('true').sum()
print(f'{len(merged)} patients confirmed positive via genetics/imaging (independent of medication)')
print(f'{on_therapy} ({100*on_therapy/len(merged):.1f}%) of those are documented on CM-specific therapy')
print(f'{len(merged)-on_therapy} ({100*(len(merged)-on_therapy)/len(merged):.1f}%) have NO documented CM-specific therapy — a real treatment gap')
'''