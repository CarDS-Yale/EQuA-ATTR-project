

#!/bin/bash
cd ~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results || exit 1
cp --update=none table1_summary.txt table1_summary_all_positives_oct6.txt
python3 - << 'PY'
import pandas as pd
c = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
c.loc[(c['final_classification']=='CONFIRMED_POSITIVE') & (c['genotype']=='HEREDITARY_VARIANT'), 'final_classification'] = 'VARIANT_ATTRCM_POSITIVE'
c.to_csv('tmp_table1_groups.csv', index=False)
print((c['final_classification']=='VARIANT_ATTRCM_POSITIVE').sum(), 'variant positives labeled')
PY
echo; echo "===== VARIANT ATTR-CM POSITIVES ====="
python3 ~/ollama_space/ollama_scripts/equa_attr_table1_demographics.py --classification-file tmp_table1_groups.csv --tier VARIANT_ATTRCM_POSITIVE
echo; echo "===== CARRIERS, NO DISEASE EVIDENCE ====="
python3 ~/ollama_space/ollama_scripts/equa_attr_table1_demographics.py --classification-file tmp_table1_groups.csv --tier TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE
