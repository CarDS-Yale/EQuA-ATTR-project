


#!/bin/bash
cd ~/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results || { echo "results folder not found"; exit 1; }
OUT=final_numbers_$(date +%Y%m%d_%H%M).txt
step() { echo; echo "######################## $1 ########################"; }
{
step "1. CLASSIFICATION"
cp attr_cm_final_classification.csv attr_cm_final_classification_prev.csv 2>/dev/null
python3 ~/ollama_space/ollama_scripts/equa_attr_final_cohort_classification.py --cohort-file attr_cm_candidate_cohort.csv --genetics-rollup attr_genetics_patient_rollup.csv --genetics-raw attr_genetics_ALL_combined.csv --pyp-results attr_pyp_results_latest.csv --adjudications manual_adjudications_log.txt || echo ">>> STEP FAILED"
step "2. TABLE 1 (age, sex, race, comorbidities)"
python3 ~/ollama_space/ollama_scripts/equa_attr_table1_demographics.py --classification-file attr_cm_final_classification.csv --tier CONFIRMED_POSITIVE || echo ">>> STEP FAILED"
step "3. MORTALITY"
python3 ~/ollama_space/ollama_scripts/mortality_check_fixed.py || echo ">>> STEP FAILED"
step "4. MEDICATION BREAKDOWN (tafamidis vs acoramidis)"
python3 ~/ollama_space/ollama_scripts/med_breakdown.py || echo ">>> STEP FAILED"
step "5. THERAPY GAP, PI version (after May 2019, alive)"
python3 ~/ollama_space/ollama_scripts/dmt_eligible_check.py || echo ">>> STEP FAILED"
step "6. PRIOR-SCREEN CONCORDANCE"
python3 ~/ollama_space/ollama_scripts/crosscheck_prior_screen.py --classification-file attr_cm_final_classification.csv --screening-file "03132026-Favour-screening-TTR-2.xlsx" || echo ">>> STEP FAILED"
step "7. CARE-GAP BREAKDOWN"
python3 ~/ollama_space/ollama_scripts/care_gap_full_breakdown.py || echo ">>> STEP FAILED"
} 2>&1 | tee "$OUT"
echo; echo "Saved to $OUT"