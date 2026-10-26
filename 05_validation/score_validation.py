
import pandas as pd, numpy as np, math
BASE = "/home/fao9/work_tmp/equa_attr_preprocessing_results/equa_attr_ollama_results"
sh = pd.read_csv(f"{BASE}/validation_sheet.csv", dtype=str)
ky = pd.read_csv(f"{BASE}/validation_key.csv", dtype=str)
pop = dict(l.strip().split('=') for l in open(f"{BASE}/validation_population.txt") if '=' in l)
N1, N0 = int(pop['N_LLM_POS']), int(pop['N_LLM_NEG'])
d = sh.merge(ky, on=['sample_id', 'PAT_MRN_ID'])
d['h'] = d['human_attr_cm (Y/N/U)'].fillna('').str.strip().str.upper().str[:1]
print(f"Sample {len(d)} | labeled {int((d['h'] != '').sum())} | Y {int((d['h']=='Y').sum())} N {int((d['h']=='N').sum())} U {int((d['h']=='U').sum())} | blank {int((d['h']=='').sum())}")
def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'),) * 3
    p = k / n; den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den; h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return p, max(0, c - h), min(1, c + h)
def f(x):
    return f"{100*x:.0f}%" if x == x else "n/a"
def ss(tp, fp, tn, fn):
    p_ = tp / (tp + fp) if tp + fp else float('nan'); q_ = tn / (tn + fn) if tn + fn else float('nan')
    etp, efp, etn, efn = N1 * p_, N1 * (1 - p_), N0 * q_, N0 * (1 - q_)
    return etp / (etp + efn), etn / (etn + efp)
def report(dd, label):
    pos, neg = dd[dd['llm_call'] == 'LLM_POS'], dd[dd['llm_call'] == 'LLM_NEG']
    TP, FP = int((pos['h'] == 'Y').sum()), int((pos['h'] == 'N').sum())
    TN, FN = int((neg['h'] == 'N').sum()), int((neg['h'] == 'Y').sum())
    print(f"\n=== {label} ===")
    print(f"2x2 in the sample: TP {TP} | FP {FP} | FN {FN} | TN {TN} | total {TP+FP+FN+TN}")
    ppv = wilson(TP, TP + FP); npv = wilson(TN, TN + FN)
    print(f"PPV {TP}/{TP+FP} = {f(ppv[0])} (95% CI {f(ppv[1])}-{f(ppv[2])}) | NPV {TN}/{TN+FN} = {f(npv[0])} (95% CI {f(npv[1])}-{f(npv[2])})  [Wilson]")
    se, sp = ss(TP, FP, TN, FN)
    rng = np.random.default_rng(1); bs = []
    pv, nv = (pos['h'] == 'Y').values, (neg['h'] == 'N').values
    for _ in range(4000):
        a_ = rng.choice(pv, len(pv)); b_ = rng.choice(nv, len(nv))
        bs.append(ss(a_.sum(), len(a_) - a_.sum(), b_.sum(), len(b_) - b_.sum()))
    bs = np.array(bs, dtype=float)
    lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
    print(f"Sensitivity {f(se)} (95% CI {f(lo[0])}-{f(hi[0])}) | Specificity {f(sp)} (95% CI {f(lo[1])}-{f(hi[1])})  [re-weighted to the pool: {N1} LLM-positive, {N0} LLM-negative; bootstrap CI]")
    n_ = TP + FP + TN + FN
    if n_:
        po = (TP + TN) / n_; pe = ((TP + FP) * (TP + FN) + (TN + FN) * (TN + FP)) / n_**2
        k = (po - pe) / (1 - pe) if pe < 1 else float('nan')
        print(f"Raw agreement in the sample {f(po)} | Cohen's kappa {k:.2f} (in the enriched 50/50 sample)")
    print("By the pipeline's final tier, LLM-positive stratum:")
    for t, gg in pos.groupby(pos['final_classification'].fillna('?')):
        print(f"   {t}: {int((gg['h']=='Y').sum())}/{int(gg['h'].isin(['Y','N']).sum())} confirmed")
    print("By the pipeline's final tier, LLM-negative stratum (human says Y = LLM missed it):")
    for t, gg in neg.groupby(neg['final_classification'].fillna('?')):
        print(f"   {t}: human Y {int((gg['h']=='Y').sum())} of {int(gg['h'].isin(['Y','N']).sum())}")
lab = d[d['h'].isin(['Y', 'N'])]
report(lab, "PRIMARY: uncertain (U) excluded")
d2 = d[d['h'] != ''].copy(); d2.loc[d2['h'] == 'U', 'h'] = 'N'
report(d2, "SENSITIVITY: uncertain counted as NOT ATTR-CM")
print(f"\nU by LLM call: LLM_POS {int(((d['h']=='U') & (d['llm_call']=='LLM_POS')).sum())} | LLM_NEG {int(((d['h']=='U') & (d['llm_call']=='LLM_NEG')).sum())}")
alc = 'AL_workup (done-negative / done-abnormal / not found)'
a = d[alc].fillna('').str.strip().str.lower(); a = a[a != '']
if len(a):
    print('AL workup among reviewed patients:', a.value_counts().to_dict())
