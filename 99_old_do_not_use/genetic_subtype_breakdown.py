

import pandas as pd
final = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_cm_final_classification.csv', dtype=str)
rollup = pd.read_csv('~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results/attr_genetics_patient_rollup.csv', dtype=str)

gen_confirmed_mrns = set(final.loc[final['subtype_basis']=='GENETICALLY_CONFIRMED', 'PAT_MRN_ID'])
gen_rows = rollup[rollup['PAT_MRN_ID'].isin(gen_confirmed_mrns)]

status = gen_rows['overall_genetic_status'].astype(str)
n_variant = status.str.startswith('variant').sum()
n_wild = status.str.startswith('wild_type').sum()
print(f'Genetically confirmed: {len(gen_rows)}')
print(f'  Variant (hereditary): {n_variant} ({100*n_variant/len(gen_rows):.1f}%)')
print(f'  Wild-type: {n_wild} ({100*n_wild/len(gen_rows):.1f}%)')

print('\nTop specific variants:')
all_variants = gen_rows['variants_found'].dropna().str.split('; ').explode()
print(all_variants.value_counts().head(10).to_string())