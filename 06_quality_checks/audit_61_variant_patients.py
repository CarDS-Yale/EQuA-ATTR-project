


import pandas as pd
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)

variant_mrns = set(final.loc[final['subtype_basis']=='GENETICALLY_CONFIRMED', 'PAT_MRN_ID']) & \
               set(rollup.loc[rollup['overall_genetic_status'].str.startswith('variant', na=False), 'PAT_MRN_ID'])
rows = rollup[rollup['PAT_MRN_ID'].isin(variant_mrns)]
print(f"Total patients in the 'variant (hereditary)' bucket: {len(rows)}")

# Known REAL TTR variant names -- from the reference list you found, plus the
# common ones already seen in your data tonight
KNOWN_TTR_VARIANTS = re.compile(
    r"V142I|Val142Ile|V122I|Val122Ile|V30M|Val30Met|T60A|Thr60Ala|"
    r"I68L|Ile68Leu|L111M|Leu111Met|P24S|Pro24Ser|E89Q|Glu89Gln|"
    r"I107V|Ile107Val|W41L|Trp41Leu|T49A|Thr49Ala|H88R|His88Arg|"
    r"V20I|Val20Ile|A45T|Ala45Thr|G65L|Gly65Leu",
    re.IGNORECASE,
)
NON_TTR_GENES = re.compile(r"\bTTN\b|BRCA1|BRCA2|APC|ATM|DSP|CHEK2|GLA|MYH7|LMNA|VCP", re.IGNORECASE)

matches_known = rows['variants_found'].fillna('').str.contains(KNOWN_TTR_VARIANTS)
has_non_ttr_gene = rows['variants_found'].fillna('').str.contains(NON_TTR_GENES)

print(f"  Matches a known, real TTR variant name: {matches_known.sum()}")
print(f"  Contains a non-TTR gene name (TTN/BRCA/etc - needs review): {has_non_ttr_gene.sum()}")
print(f"  Neither (unrecognized, verify manually): {(~matches_known & ~has_non_ttr_gene).sum()}")

if has_non_ttr_gene.sum() > 0:
    print(f"\n!!! PATIENTS NEEDING IMMEDIATE REVIEW (counted as variant-positive, but evidence mentions a non-TTR gene) !!!")
    for _, r in rows[has_non_ttr_gene].iterrows():
        print(f"  {r['PAT_MRN_ID']}: {r['variants_found']}")