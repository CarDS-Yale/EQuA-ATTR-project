

#!/usr/bin/env python3
"""
Final ATTR-CM classification, rule v2 (Oct 2026).

DISEASE -- what makes a patient ATTR-CM positive (any one):
  A  tafamidis/acoramidis (CM-specific) + an amyloid diagnosis code or name   [structured cohort]
  B  a POSITIVE PYP report                                                    [PYP report script]
  E  a note documenting ATTR-CM (clinical/imaging statement, or an explicit ATTR-CM
     disease phrase)                                                          [LLM notes extraction]
     GUARD: if E is the ONLY support and there is negative imaging (negative PYP report,
     ruled-out note, or negative-PYP wording) the patient goes to a REVIEW tier.
     EXCEPTION: a note-documented biopsy / mass-spec statement counts as positive even with negative imaging
     (biopsy outranks scintigraphy); the patient is flagged discordant_imaging.
A diagnosis CODE or NAME alone (structured, or a bare "(HC CODE)" label in a note) is never positive.
GENETICS (your c and d) counts toward the cohort ONLY together with A, B or E: a TTR variant found = c, no variant found = d,
each requires a, b or e; genetics alone never makes a patient positive. It is its OWN classification (genotype column):
  HEREDITARY_VARIANT / NO_VARIANT_FOUND / CONFLICTING / ORDERED_NO_RESULT / NO_RESULT_DOCUMENTED
  (NO_RESULT_DOCUMENTED = nothing in the notes; it does NOT mean the patient was never tested.)
Each source is tracked separately because the three pipelines cover different patients.
HAND DECISIONS OVERRIDE THE RULE: lines "MRN | CODE | reason" in manual_adjudications_log.txt
(ATTRCM_POS, ATTRCM_NEG, TTR_CARRIER, ASK; the last line for a patient wins) replace the rule's tier for that
patient. Every override is reported (section 9) and the rule's own tier is kept in tier_before_adjudication.

Inputs (all produced by earlier steps):
  --cohort-file      structured candidates (attr_cm_candidate_cohort.csv)
  --genetics-rollup  patient-level genetics rollup (used only for the 'ordered, no result' flag)
  --genetics-raw     the COMBINED row-level genetics file (default attr_genetics_ALL_combined.csv)
  --pyp-results      PYP report results
  --care-gap         accepted for compatibility, NOT used (care-gap never changed any tier)
"""
import os
import re
import argparse
import pandas as pd

# ---------------- switches: each is ONE line to flip ----------------
MED_DX_BROAD = True        # drug corroboration = any amyloid dx code OR name flag (False = ATTR-specific codes only)
EXTEND_TTR_GATE = True     # accepts bare "ATTR" and "wild-type ... amyloid" as TTR-specific language (False = the older, narrower gate)

TTR_STRICT = re.compile(
    r"\bTTR\b|transthyretin|\bV142I\b|Val142Ile|\bV122I\b|Val122Ile|\bV30M\b|Val30Met|"
    r"\bT60A\b|Thr60Ala|hATTR|wtATTR|ATTRv\b|ATTRwt\b|p\.Val142|p\.V142", re.I)
TTR_EXT = re.compile(TTR_STRICT.pattern + r"|\bATTR\b|wild[- ]?type\s+(?:cardiac\s+)?amyloid", re.I)
NEG = re.compile(
    r"(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT))[^.;]{0,40}?\b(?:negative|normal|unremarkable)\b|"
    r"\b(?:negative|normal)\b[^.;]{0,25}?(?:PYP|pyrophosphate|scintigraphy|amyloid (?:scan|imaging|SPECT))", re.I)
POS = re.compile(r"\bpositive\b|abnormal|grade\s*[23]|score\s*(?:of|was)?\s*[23]|(?<!not )consistent with TTR|false[- ]negative|falsely[- ]negative|despite (?:a |the )?(?:negative|normal)", re.I)
CARRIER_CUE = re.compile(
    r"\bcarrier\b|no diagnosis of (?:cardiac )?amyloid|no (?:known )?(?:cardiac )?amyloid|not (?:been )?diagnosed|"
    r"no cardiac (?:involvement|uptake)|without (?:evidence of )?(?:cardiac )?amyloid|asymptomatic", re.I)
DISEASE_PHRASE = re.compile(
    r"(?:ATTR|TTR|transthyretin)[\s-]*(?:CM\b|cardiomyopathy|cardiac amyloid|amyloid cardiomyopathy)|"
    r"(?:hereditary|variant|wild[- ]?type)[\s-]+(?:ATTR|TTR|transthyretin)?[\s-]*(?:CM\b|cardiomyopathy|cardiac amyloid)", re.I)


BIOPSY_PHRASE = re.compile(
    r"endomyocardial[\s-]+proven[\s-]+biopsy|biopsy[\s-]*(?:proven|confirmed)|"
    r"(?:proven|confirmed)\s+(?:by|on)\s+(?:an?\s+)?(?:endomyocardial\s+)?biopsy|mass[\s-]*spec(?:trometry)?", re.I)


DX_LABEL = re.compile(r"\(HC CODE\)|\bHC code\b", re.I)   # problem-list / diagnosis-list entries embedded in notes
BASIS_CUE = re.compile(
    r"scan|PYP|pyrophosphate|scintigraph|imaging|SPECT|biops|mass[\s-]*spec|genetic|tafamidis|vyndam|acoramidis|attruby|"
    r"echo|\bMRI\b|based on|confirmed|diagnosed (?:by|with|based)|diagnosis (?:by|based)", re.I)


ADJ_RE = re.compile(r"^\s*(\S+)\s*\|\s*(ATTRCM_POS|ATTRCM_NEG|TTR_CARRIER|ASK)\s*\|(.*)$")


def tf(s):
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def find_col(columns, name):
    for c in columns:
        if c.upper() == name.upper():
            return c
    return None


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    ap = argparse.ArgumentParser(description="Final ATTR-CM classification (rule v2)")
    ap.add_argument("--cohort-file", required=True)
    ap.add_argument("--genetics-rollup", required=True)
    ap.add_argument("--genetics-raw", default="attr_genetics_ALL_combined.csv")
    ap.add_argument("--pyp-results", required=True)
    ap.add_argument("--care-gap", default=None)
    ap.add_argument("--adjudications", default="manual_adjudications_log.txt")
    ap.add_argument("--out", default="attr_cm_final_classification.csv")
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort_file, dtype=str)
    roll = pd.read_csv(args.genetics_rollup, dtype=str)
    pyp = pd.read_csv(args.pyp_results, dtype=str)
    wanted = {"PAT_MRN_ID", "attr_type", "type_basis", "has_genetic_testing", "variant_mutation", "evidence_text"}
    raw = pd.read_csv(args.genetics_raw, dtype=str, usecols=lambda c: c in wanted)
    for c in wanted:
        if c not in raw.columns:
            raw[c] = ""
    print(f"cohort {len(cohort)} | rollup {len(roll)} | raw genetics rows {len(raw)} | pyp rows {len(pyp)}")
    mrn_col = find_col(cohort.columns, "PAT_MRN_ID")
    adj = {}
    if args.adjudications and os.path.exists(args.adjudications):
        for line in open(args.adjudications):
            mm = ADJ_RE.match(line)
            if mm:
                adj[mm.group(1)] = (mm.group(2), mm.group(3).split("|")[0].strip())
        print(f"hand decisions read from {args.adjudications}: {len(adj)} patients")
    else:
        print(f"[note] no adjudication file found at {args.adjudications}; rule-only classification")

    # ---------------- row-level work on the genetics extraction ----------------
    atype = raw["attr_type"].fillna("").str.strip().str.lower()
    basis = raw["type_basis"].fillna("").str.strip().str.lower()
    ev = raw["evidence_text"].fillna("").astype(str)
    txt = raw["variant_mutation"].fillna("").astype(str) + " " + ev
    is_vw = atype.isin(["variant", "wild_type"])
    neg = ev.str.contains(NEG)
    neg_clean = neg & ~ev.where(neg, "").str.contains(POS)
    carrier = txt.where(is_vw, "").str.contains(CARRIER_CUE)
    dphrase = txt.where(is_vw, "").str.contains(DISEASE_PHRASE)
    ttr_s = txt.where(is_vw, "").str.contains(TTR_STRICT)
    ttr_x = txt.where(is_vw, "").str.contains(TTR_EXT)
    bx = txt.where(is_vw, "").str.contains(BIOPSY_PHRASE)
    label_only = txt.where(is_vw, "").str.contains(DX_LABEL) & ~txt.where(is_vw, "").str.contains(BASIS_CUE)
    base_ok = is_vw & ~carrier & ~neg_clean & ~label_only & (basis.eq("clinical_or_imaging") | dphrase)
    rows = pd.DataFrame({
        "PAT_MRN_ID": raw["PAT_MRN_ID"],
        "disease_s": base_ok & ttr_s,
        "disease_x": base_ok & ttr_x,
        "bx_s": base_ok & ttr_s & bx,
        "bx_x": base_ok & ttr_x & bx,
        "gen_variant": (atype == "variant") & basis.eq("genetic") & ttr_s,
        "gen_wild": (atype == "wild_type") & basis.eq("genetic") & ttr_s,
        "neg_img": atype.eq("ruled_out") | neg_clean,
        "pres_wild": base_ok & ttr_s & atype.eq("wild_type") & basis.eq("clinical_or_imaging"),
        "pres_var": base_ok & ttr_s & atype.eq("variant") & basis.eq("clinical_or_imaging"),
    })
    n_carrier_rows = int((is_vw & carrier & (basis.eq("clinical_or_imaging") | dphrase)).sum())
    n_label_rows = int((is_vw & label_only & ttr_x & (basis.eq("clinical_or_imaging") | dphrase)).sum())
    agg = rows.groupby("PAT_MRN_ID").any()
    E_s = set(agg.index[agg["disease_s"]])
    E_x = set(agg.index[agg["disease_x"]])
    E_set = E_x if EXTEND_TTR_GATE else E_s
    BX_set = set(agg.index[agg["bx_x" if EXTEND_TTR_GATE else "bx_s"]])
    GV = set(agg.index[agg["gen_variant"]])
    GW = set(agg.index[agg["gen_wild"]])
    NEGN = set(agg.index[agg["neg_img"]])
    PW = set(agg.index[agg["pres_wild"]])
    PV = set(agg.index[agg["pres_var"]])
    gv_rows = raw[rows["gen_variant"]]
    names = gv_rows.groupby("PAT_MRN_ID")["variant_mutation"].apply(
        lambda s: "; ".join(sorted({x.strip() for x in s.dropna()
                                    if x.strip() and x.strip().lower() not in ("nan", "not specified", "error")})))
    pending = set(roll.loc[tf(roll["genetic_testing_pending_no_result"]), "PAT_MRN_ID"]) \
        if "genetic_testing_pending_no_result" in roll.columns else set()

    # ---------------- PYP report source ----------------
    pr = pyp["pyp_result"].astype(str).str.lower()
    pyp_pos = set(pyp.loc[pr == "positive", "PAT_MRN_ID"])
    pyp_neg = set(pyp.loc[pr.isin(["negative", "equivocal_borderline"]), "PAT_MRN_ID"])
    pyp_any = pyp_pos | pyp_neg

    # ---------------- candidate pool = structured cohort + patients found only in notes/reports ----------------
    structured_mrns = set(cohort[mrn_col].dropna())
    extra_mrns = (E_set | GV | NEGN | pyp_any | set(adj)) - structured_mrns
    print(f"patients found ONLY through notes/reports (no structured signal): {len(extra_mrns)}")
    extra = pd.DataFrame({mrn_col: sorted(extra_mrns)})
    for c in cohort.columns:
        if c not in extra.columns:
            extra[c] = False if c.startswith("flag_") else ""
    cohort["_in_structured"] = True
    extra["_in_structured"] = False
    cohort = pd.concat([cohort, extra], ignore_index=True)
    m = cohort[mrn_col]

    # ---------------- structured signals ----------------
    med_col = find_col(cohort.columns, "flag_medication_cm_specific")
    if med_col:
        med = tf(cohort[med_col])
    else:
        print("[WARNING] flag_medication_cm_specific missing; using unsplit medication flags (less precise)")
        med = pd.Series(False, index=cohort.index)
        for c in ("flag_medication_order", "flag_medication_admin"):
            if find_col(cohort.columns, c):
                med = med | tf(cohort[find_col(cohort.columns, c)])
    dx_cols = ["flag_dx_attr_specific"] + (["flag_dx_amyloid_family", "flag_dx_name_match"] if MED_DX_BROAD else [])
    dx = pd.Series(False, index=cohort.index)
    for c in dx_cols:
        if find_col(cohort.columns, c):
            dx = dx | tf(cohort[find_col(cohort.columns, c)])
    signal_cols = [c for c in cohort.columns if c.startswith("flag_") and c not in (
        "flag_al_any_evidence", "flag_dx_al_amyloid_only", "flag_al_lab_workup",
        "flag_family_hx_cardiac_red_flag", "flag_diflunisal_only")]
    has_signal = cohort[signal_cols].apply(tf).any(axis=1)
    al_col = find_col(cohort.columns, "flag_al_any_evidence")
    al_only = (tf(cohort[al_col]) if al_col else pd.Series(False, index=cohort.index)) & ~has_signal

    # ---------------- per-source columns (each source stands alone) ----------------
    cohort["src_med_dx"] = med & dx
    cohort["_med_positive"] = cohort["src_med_dx"]
    cohort["src_pyp_report"] = m.map(lambda x: "positive+negative" if (x in pyp_pos and x in pyp_neg)
                                     else "positive" if x in pyp_pos else "negative" if x in pyp_neg else "none")
    cohort["src_notes_disease_statement"] = m.isin(E_set)
    cohort["src_notes_negative_imaging"] = m.isin(NEGN)
    cohort["src_notes_biopsy_statement"] = m.isin(BX_set)

    def genotype(x):
        if x in GV and x in GW:
            return "CONFLICTING"
        if x in GV:
            return "HEREDITARY_VARIANT"
        if x in GW:
            return "NO_VARIANT_FOUND"
        if x in pending:
            return "ORDERED_NO_RESULT"
        return "NO_RESULT_DOCUMENTED"
    cohort["genotype"] = m.map(genotype)
    cohort["presumed_subtype"] = m.map(lambda x: "presumed_variant_and_wild_type" if (x in PV and x in PW)
                                       else "presumed_variant" if x in PV else "presumed_wild_type" if x in PW else "")
    cohort["variant_names"] = m.map(names).fillna("")

    # ---------------- tiers ----------------
    A = cohort["src_med_dx"]
    B = cohort["src_pyp_report"].isin(["positive", "positive+negative"])
    E = cohort["src_notes_disease_statement"]
    BX = cohort["src_notes_biopsy_statement"]
    NEGIMG = cohort["src_pyp_report"].isin(["negative", "positive+negative"]) | cohort["src_notes_negative_imaging"]
    tier = pd.Series("INSUFFICIENT_EVIDENCE", index=cohort.index)
    tier[has_signal] = "CARE_GAP"
    tier[NEGIMG] = "CONFIRMED_NEGATIVE"
    tier[cohort["genotype"].eq("HEREDITARY_VARIANT")] = "TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE"
    tier[E & NEGIMG] = "REVIEW_NOTE_VS_NEG_IMAGING"
    tier[E & ~NEGIMG] = "CONFIRMED_POSITIVE"
    tier[BX] = "CONFIRMED_POSITIVE"   # biopsy-proven statement outranks a negative scan (a false negative)
    tier[A | B] = "CONFIRMED_POSITIVE"
    tier[al_only] = "EXCLUDED_AL_ONLY"
    rule_tier = tier.copy()
    nd = pd.Series("INSUFFICIENT_EVIDENCE", index=cohort.index)      # the ladder with no disease evidence
    nd[has_signal] = "CARE_GAP"
    nd[NEGIMG] = "CONFIRMED_NEGATIVE"
    nd[cohort["genotype"].eq("HEREDITARY_VARIANT")] = "TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE"
    adj_code = m.map(lambda x: adj.get(x, ("", ""))[0])
    adj_reason = m.map(lambda x: adj.get(x, ("", ""))[1])
    tier[adj_code.eq("ATTRCM_POS")] = "CONFIRMED_POSITIVE"
    tier[adj_code.eq("ATTRCM_NEG")] = "CONFIRMED_NEGATIVE"
    tier[adj_code.eq("TTR_CARRIER")] = nd[adj_code.eq("TTR_CARRIER")]
    tier[adj_code.eq("ASK")] = "REVIEW_PENDING_ADJUDICATION"
    cohort["adjudication"] = adj_code
    cohort["adjudication_reason"] = adj_reason
    cohort["tier_before_adjudication"] = rule_tier
    cohort["final_classification"] = tier
    pos = tier.eq("CONFIRMED_POSITIVE")
    cohort["discordant_imaging"] = pos & NEGIMG
    src = pd.Series("NOT_APPLICABLE", index=cohort.index)
    note_tok = E.map({True: "NOTES_DISEASE_STATEMENT", False: ""}).where(~BX, "NOTES_BIOPSY_STATEMENT")
    parts = (B.map({True: "PYP_IMAGING_SCRIPT", False: ""}) + "|" + note_tok +
             "|" + A.map({True: "MEDICATION+DX", False: ""}))
    src[pos] = parts[pos].map(lambda s: "+".join(p for p in s.split("|") if p))
    manual_pos = adj_code.eq("ATTRCM_POS")
    src[manual_pos] = src[manual_pos].map(
        lambda x: "MANUAL_ADJUDICATION" if x in ("", "NOT_APPLICABLE") else x + "+MANUAL_ADJUDICATION")
    cohort["positive_evidence_source"] = src
    cohort["subtype_basis"] = cohort["genotype"].where(pos, "NOT_APPLICABLE")
    cohort["resolution_source"] = (cohort["_in_structured"].map({True: "STRUCTURED", False: ""}) + "|" +
                                   (E | NEGIMG | cohort["genotype"].isin(["HEREDITARY_VARIANT", "NO_VARIANT_FOUND", "CONFLICTING"])
                                    ).map({True: "NOTES", False: ""}) + "|" +
                                   cohort["src_pyp_report"].ne("none").map({True: "PYP_IMAGING_SCRIPT", False: ""})
                                   ).map(lambda s: "+".join(p for p in s.split("|") if p) or "NONE")
    cohort["cat_a"] = A
    cohort["cat_b"] = B | BX
    cohort["cat_e"] = E
    cohort["cat_c"] = pos & cohort["genotype"].eq("HEREDITARY_VARIANT")
    cohort["cat_d"] = pos & cohort["genotype"].eq("NO_VARIANT_FOUND")
    cohort["cat_c_d_conflicting"] = pos & cohort["genotype"].eq("CONFLICTING")
    cohort.to_csv(args.out, index=False)

    # =================== REPORT ===================
    section("1. TIERS (rule v2)")
    print(tier.value_counts().to_string())
    print(f"TOTAL evaluated: {len(cohort)}")
    section("2. WHAT SUPPORTS EACH POSITIVE (A = drug+code, B = positive PYP report, E = note statement)")
    print(src[pos].value_counts().to_string())
    n_pos = int(pos.sum())
    print(f"positives: {n_pos} | supported by ONE source only: {int((src[pos].str.count(r'PYP_IMAGING_SCRIPT|NOTES_DISEASE_STATEMENT|NOTES_BIOPSY_STATEMENT|MEDICATION|MANUAL_ADJUDICATION') == 1).sum())}")
    print(f"positives with discordant imaging (a negative scan/statement as well): {int(cohort['discordant_imaging'].sum())}")
    print(f"CHECK positives resting on a diagnosis code/name alone (none of A, B, E, and not hand-adjudicated): "
          f"{int((pos & ~(A | B | E) & ~manual_pos).sum())}  (must be 0)")
    print(f"positives that rest ONLY on your hand adjudication (no A, B or E): {int((pos & ~(A | B | E) & manual_pos).sum())}")
    print(f"positives supported by a note-documented biopsy/mass-spec statement: {int((pos & BX).sum())} "
          f"(of these, also with negative imaging, i.e. scan called a false negative: {int((pos & BX & NEGIMG).sum())})")
    section("2b. YOUR a-e CATEGORIES (a patient can be in several)")
    nn = lambda x: int(x.sum())
    print(f"a  ATTR-CM drug + diagnosis code/name ............................ {nn(cohort['cat_a'])}")
    print(f"b  positive PYP report, or a note-documented biopsy .............. {nn(cohort['cat_b'])}")
    print(f"c  TTR variant found AND validated by a, b or e .................. {nn(cohort['cat_c'])}")
    print(f"d  NO variant found (negative genetic test) AND validated ........ {nn(cohort['cat_d'])}")
    print(f"   conflicting genotype AND validated ............................ {nn(cohort['cat_c_d_conflicting'])}")
    print(f"e  NLP: a note documenting ATTR-CM ............................... {nn(cohort['cat_e'])} "
          f"(of these, held in a review tier and not counted positive: {nn(cohort['cat_e'] & ~pos)})")
    print(f"ATTR-CM positive = a or b or e (plus any hand decision) = {nn(pos)}; c and d add no one, each requires a, b or e")
    print(f"in the evaluated pool with a variant but NOT validated (carriers, not counted): "
          f"{nn(cohort['genotype'].eq('HEREDITARY_VARIANT') & ~pos)}")
    print(f"in the evaluated pool with a negative genetic test but NOT validated (not counted): "
          f"{nn(cohort['genotype'].eq('NO_VARIANT_FOUND') & ~pos)}")
    section("3. GENOTYPE (own classification; counts toward the cohort only together with A, B or E)")
    print("Among ATTR-CM POSITIVE patients:")
    print(cohort.loc[pos, "genotype"].value_counts().to_string())
    nodoc = pos & cohort["genotype"].eq("NO_RESULT_DOCUMENTED")
    print("  of the 'no result documented' positives, presumed subtype from notes:")
    print(cohort.loc[nodoc, "presumed_subtype"].replace("", "none stated").value_counts().to_string())
    print("Top TTR variants named among positives with a hereditary variant:")
    print(cohort.loc[pos & cohort["genotype"].eq("HEREDITARY_VARIANT"), "variant_names"]
          .str.split("; ").explode().replace("", pd.NA).dropna().value_counts().head(10).to_string())
    doc = cohort[cohort["genotype"].isin(["HEREDITARY_VARIANT", "NO_VARIANT_FOUND", "CONFLICTING"])]
    print("\nEVERY patient with a documented genetic result, by tier:")
    print(pd.crosstab(doc["genotype"], doc["final_classification"], margins=True).to_string())
    car = cohort["final_classification"].eq("TTR_VARIANT_CARRIER_NO_DISEASE_EVIDENCE")
    car_noimg = car & cohort["src_pyp_report"].eq("none") & ~cohort["src_notes_negative_imaging"]
    print(f"\nTTR variant carriers with NO disease evidence: {int(car.sum())} "
          f"(negative imaging: {int((car & NEGIMG).sum())}; no imaging of any kind found: {int(car_noimg.sum())})")
    section("4. ASCERTAINMENT (Aim 3) under the new definition")
    nA = int((pos & A).sum())
    print(f"positives: {n_pos}")
    print(f"  identified by structured drug+code alone (A): {nA} ({100*nA/max(n_pos,1):.1f}%)")
    print(f"  NOT identifiable by A, needed the PYP report or note NLP: {n_pos-nA} ({100*(n_pos-nA)/max(n_pos,1):.1f}%)")
    nstruct = int((pos & ~cohort['_in_structured']).sum())
    print(f"  of those, found ONLY through notes/reports (no structured signal at all): {nstruct} ({100*nstruct/max(n_pos,1):.1f}%)")
    if find_col(cohort.columns, "flag_dx_attr_specific"):
        nd = int((pos & tf(cohort[find_col(cohort.columns, 'flag_dx_attr_specific')])).sum())
        print(f"  (looser comparator) positives carrying an ATTR-specific diagnosis code: {nd} ({100*nd/max(n_pos,1):.1f}%)")
    section("5. THERAPY GAP, unrestricted (positives with B or E, i.e. not identified by the drug itself)")
    indep = pos & (B | E)
    onmed = indep & (tf(cohort[med_col]) if med_col else False)
    print(f"positives independent of the drug: {int(indep.sum())} | on CM-specific drug: {int(onmed.sum())} | "
          f"no documented drug: {int(indep.sum()-onmed.sum())}")
    print("(PI version restricted to diagnosed after May 2019 and alive: run dmt_eligible_check.py)")
    section("6. FOLLOW-UP STATUS BY GROUP")
    gdoc = cohort.loc[pos, "genotype"].isin(["HEREDITARY_VARIANT", "NO_VARIANT_FOUND", "CONFLICTING"])
    print(f"positives with a documented genetic result: {int(gdoc.sum())} of {n_pos} ({100*gdoc.sum()/max(n_pos,1):.1f}%)")
    print(f"positives with genetic test ordered, no result: {int(cohort.loc[pos,'genotype'].eq('ORDERED_NO_RESULT').sum())}")
    print(f"positives confirmed by a positive PYP report (B): {int((pos & B).sum())}; "
          f"by drug+code or a note only (no PYP report positive): {int((pos & ~B).sum())}")
    print(f"variant carriers with no disease evidence and no imaging found: {int(car_noimg.sum())}")
    section("7. REVIEW LISTS (written to files in this folder)")
    r1 = cohort[tier.eq("REVIEW_NOTE_VS_NEG_IMAGING")]
    r2 = cohort[cohort["genotype"].isin(["HEREDITARY_VARIANT", "CONFLICTING"]) & ~pos & cohort["adjudication"].eq("")]
    keep = [mrn_col, "final_classification", "genotype", "src_pyp_report", "src_notes_disease_statement", "src_notes_negative_imaging"]
    r1[keep].to_csv("review_note_vs_neg_imaging.csv", index=False)
    r2[keep + ["variant_names"]].to_csv("review_genotype_only.csv", index=False)
    print(f"review_note_vs_neg_imaging.csv: {len(r1)} patients (note says ATTR-CM, imaging negative, nothing else supports it)")
    print(f"review_genotype_only.csv: {len(r2)} patients (variant or conflicting genotype, no disease evidence)")
    section("8. SWITCHES AND TRANSPARENCY")
    print(f"MED_DX_BROAD={MED_DX_BROAD} | EXTEND_TTR_GATE={EXTEND_TTR_GATE}")
    gain = len(E_x - E_s - set(m[A | B]))
    print(f"patients supported ONLY by notes and ONLY because of the widened TTR wording (bare 'ATTR', 'wild-type ... amyloid'): {gain} "
          f"({'INCLUDED in the positives' if EXTEND_TTR_GATE else 'NOT included; EXTEND_TTR_GATE is off'})")
    print(f"note rows ignored as bare diagnosis-list labels ((HC CODE) with no scan/biopsy/genetic/drug wording): {n_label_rows}")
    print(f"note rows ignored as disease statements because of carrier/negation wording: {n_carrier_rows}")
    section("9. HAND ADJUDICATIONS (your logged decisions override the rule)")
    decided = cohort[cohort["adjudication"] != ""]
    if len(decided) == 0:
        print("none applied (no adjudication file, or it contained no decision lines)")
    else:
        print(f"decisions applied: {len(decided)}  {decided['adjudication'].value_counts().to_dict()}")
        in_pool_before = decided[mrn_col].isin(structured_mrns | E_set | GV | NEGN | pyp_any)
        print(f"patients that entered the evaluated pool ONLY through a decision: {int((~in_pool_before).sum())}")
        chg = decided[decided["tier_before_adjudication"] != decided["final_classification"]]
        print(f"decisions that CHANGED the rule's tier: {len(chg)} of {len(decided)}")
        print(pd.crosstab(decided["adjudication"], decided["tier_before_adjudication"], margins=True).to_string())
        ok = decided[decided["adjudication"] != "ASK"]
        rule_pos = ok["tier_before_adjudication"].eq("CONFIRMED_POSITIVE")
        agree = ((ok["adjudication"] == "ATTRCM_POS") & rule_pos) | (ok["adjudication"].isin(["ATTRCM_NEG", "TTR_CARRIER"]) & ~rule_pos)
        print(f"the rule agreed with your hand label for {int(agree.sum())} of {len(ok)} decided patients (ASK excluded); "
              "this is agreement on a hand-picked review set, not accuracy")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()