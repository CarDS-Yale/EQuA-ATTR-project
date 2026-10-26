

import pandas as pd
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
rollup = pd.read_csv(f'{BASE}/attr_genetics_patient_rollup.csv', dtype=str)

# Real, verified variant-phenotype associations (from your own reference lookup tonight)
CARDIAC_PREDOMINANT = re.compile(
    r"I68L|Ile68Leu|L111M|Leu111Met|V122I|Val122Ile", re.IGNORECASE)
MIXED_PHENOTYPE = re.compile(
    r"V30M|Val30Met|P24S|Pro24Ser|E89Q|Glu89Gln|I107V|Ile107Val|W41L|Trp41Leu|"
    r"T49A|Thr49Ala|T60A|Thr60Ala|H88R|His88Arg|V20I|Val20Ile|A45T|Ala45Thr|G65L|Gly65Leu",
    re.IGNORECASE)
NON_TTR_GENE = re.compile(
    r"\bTTN\b|BRCA1|BRCA2|APC|ATM|DSP|CHEK2|GLA|MYH7|LMNA|VCP|MUTYH|FBN1|RYR1|SCN5A|PKP2|FLNC",
    re.IGNORECASE)
V142I_NOTE = re.compile(r"V142I|Val142Ile", re.IGNORECASE)  # the common US ATTR-CM variant, same as V122I per your note

variants = rollup['variants_found'].dropna()

def categorize(text):
    if CARDIAC_PREDOMINANT.search(text) or V142I_NOTE.search(text):
        return "CARDIAC_PREDOMINANT_TTR"
    if MIXED_PHENOTYPE.search(text):
        return "MIXED_PHENOTYPE_TTR (cardiac + neuropathic -- needs independent cardiac corroboration)"
    if NON_TTR_GENE.search(text):
        return "NON_TTR_GENE (conflation -- exclude)"
    return "UNRECOGNIZED (manual review needed)"

categorized = variants.apply(categorize)
print("Variant text categorization (patient-note level, not deduplicated):")
print(categorized.value_counts().to_string())

print(f"\n--- Sample UNRECOGNIZED entries (need manual classification) ---")
unrecognized_mask = categorized == "UNRECOGNIZED (manual review needed)"
for v in variants[unrecognized_mask].unique()[:15]:
    print(f"  {v}")