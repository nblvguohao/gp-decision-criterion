"""Supplementary Table S17: the principal results for the fifth dataset, CUBIC ear weight.

china_summary.csv used to be written outside any script in the repository; this computes
every entry through the same canonical functions as the other datasets (build_summary.py,
partition.py, threshold_model.py). Needs analysis/crosscrop/results/china_panel_wide.csv,
built by panel_china.py from the downloads listed in china_DATA_SOURCE.md.

Writes analysis/crosscrop/results/china_summary.csv
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
from analyse_species import metric_table, reversal_ci, rank_interval_width
from does_it_help import outcome_reliability
import threshold_model as tm

RES = "analysis/crosscrop/results"
PATH, MN = f"{RES}/china_panel_wide.csv", 25
if not os.path.exists(PATH):
    sys.exit(f"{PATH} not present: build it with panel_china.py first")
P = pd.read_csv(PATH)
T = metric_table(P, MN)
pt, lo, hi, M = reversal_ci(T, np.random.default_rng(0))
base = P[~P.method.str.contains("__")]
Tb = metric_table(base, MN)
bpt, blo, bhi, bM = reversal_ci(Tb, np.random.default_rng(0))
w, _ = rank_interval_width(T)
rel = outcome_reliability(P, MN, np.random.default_rng(0))
one = P[P.method == P.method.iloc[0]]
PT = pd.read_csv(f"{RES}/partition_by_dataset.csv").set_index("dataset").loc["Chinese maize (CUBIC)"]
meths, envs, R, G, _ = tm.build(base, 0.10, MN)
row = dict(dataset="CUBIC", methods=M, methods_unaugmented=bM, envs=one.Env.nunique(),
           genos_per_env=int(one.groupby("Env").k.nunique().median()),
           tier_I=int(PT.tier_I), tier_II=int(PT.tier_II), tier_III=int(PT.tier_III),
           max_drift_tierII_affine=float(PT.max_drift_tierII_affine),
           min_drift_tierIII_affine=float(PT.min_drift_tierIII_affine),
           reversal=pt, ci_lo=lo, ci_hi=hi, reversal_unaugmented=bpt, ci_lo_unaug=blo, ci_hi_unaug=bhi,
           width=w, width_frac=w / M, reliability=rel,
           threshold_estimable=len(envs) >= tm.MIN_ENV)
pd.DataFrame([row]).to_csv(f"{RES}/china_summary.csv", index=False)
for k, v in row.items():
    print(f"{k:>24}: {round(v, 4) if isinstance(v, float) else v}")
