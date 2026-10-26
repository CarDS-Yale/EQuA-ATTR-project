

import pandas as pd
final = pd.read_csv('attr_cm_final_classification.csv', dtype=str)
gap = pd.read_csv('attr_care_gap_timeline.csv', dtype=str)

care_gap = final[final['final_classification'] == 'CARE_GAP'].copy()
print(f"Total CARE_GAP patients: {len(care_gap)}\n")

seen_unresolved_mrns = set(gap.loc[gap['CARE_GAP_seen_but_unresolved'].astype(str).str.lower() == 'true', 'PAT_MRN_ID'])
seen = care_gap[care_gap['PAT_MRN_ID'].isin(seen_unresolved_mrns)]
not_seen = care_gap[~care_gap['PAT_MRN_ID'].isin(seen_unresolved_mrns)]
print("=== FOLLOW-UP STATUS ===")
print(f"  Seen by cardiology after signal, still unresolved: {len(seen)}")
print(f"  No confirmed cardiology follow-up documented: {len(not_seen)}\n")

signal_cols = [c for c in care_gap.columns if c.startswith("flag_")
               and c not in ("flag_al_any_evidence", "flag_dx_al_amyloid_only",
                              "flag_al_lab_workup", "flag_family_hx_cardiac_red_flag",
                              "flag_diflunisal_only", "flag_medication_pn_primary_only")]
print("=== TRIGGERING SIGNAL(S) — why each patient entered the candidate pool ===")
for col in signal_cols:
    if col in care_gap.columns:
        n = care_gap[col].astype(str).str.lower().isin(["true", "1"]).sum()
        print(f"  {col}: {n} ({100*n/len(care_gap):.1f}%)")

present_signal_cols = [c for c in signal_cols if c in care_gap.columns]
signal_count = care_gap[present_signal_cols].apply(
    lambda col: col.astype(str).str.lower().isin(["true", "1"])
).sum(axis=1)
print(f"\n=== SIGNAL STRENGTH ===")
print(f"  Flagged by exactly ONE signal type (weaker, worth spot-checking): {(signal_count == 1).sum()}")
print(f"  Flagged by 2+ signal types (stronger corroboration): {(signal_count >= 2).sum()}")