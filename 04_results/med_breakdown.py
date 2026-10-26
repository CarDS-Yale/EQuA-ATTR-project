

import pandas as pd
import re

BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
final = pd.read_csv(f'{BASE}/attr_cm_final_classification.csv', dtype=str)

med_positive = final[
    (final['final_classification'] == 'CONFIRMED_POSITIVE') &
    (final.get('_med_positive', final.get('flag_medication_cm_specific', '')).astype(str).str.lower().isin(['true','1']))
]
print(f"Patients identified via medication+diagnosis code: {len(med_positive)}")

TAFAMIDIS = re.compile(r"tafamidis|vyndaqel|vyndamax", re.IGNORECASE)
ACORAMIDIS = re.compile(r"acoramidis|attruby", re.IGNORECASE)

med_evidence = med_positive['med_order_evidence'].fillna('') + ' ' + med_positive['med_admin_evidence'].fillna('')
has_tafamidis = med_evidence.str.contains(TAFAMIDIS)
has_acoramidis = med_evidence.str.contains(ACORAMIDIS)

print(f"  Tafamidis: {has_tafamidis.sum()}")
print(f"  Acoramidis: {has_acoramidis.sum()}")
print(f"  Both: {(has_tafamidis & has_acoramidis).sum()}")
print(f"  Neither matched (check column names): {(~has_tafamidis & ~has_acoramidis).sum()}")