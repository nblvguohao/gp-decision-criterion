"""GPverdict applied in the paper (Section 3.7; Supplementary Section S18, Tables S23-S24).

1. Validation: on the spring wheat base panel GPverdict must reproduce the reversal rate of
   Table S4 and the outcome-test recoveries of Table S7 (base panel) exactly.
2. Application: the verdict for the 17 base methods of the Chinese maize (CUBIC) panel, with
   the paper's 25-genotype environment rule and 10 % selection.

GPverdict itself lives in tool/gpverdict (public at github.com/nblvguohao/gpverdict).
Writes analysis/crosscrop/results/gpverdict_validation.csv
       analysis/crosscrop/results/gpverdict_cubic.json
       analysis/crosscrop/results/gpverdict_cubic_methods.csv
       analysis/crosscrop/results/gpverdict_cells.csv (Fig. 6c)
"""
import json, os, sys
sys.path.insert(0, "tool")
import numpy as np, pandas as pd
import gpverdict as gv

RES = "analysis/crosscrop/results"

# ---- 1. validation on spring wheat
P = pd.read_csv(f"{RES}/ursn_panel_wide.csv"); P = P[~P.method.str.contains("__")]
df = gv.load(P); ET = gv.env_table(df, 0.10, 12); S = gv.summarise(df, ET)
rev, pairs = gv.reversal(S, "rmse", "pearson")
o = gv.outcome_test(df, 0.10, 12, B=400, seed=0)
paper_rev = pd.read_csv(f"{RES}/unaugmented_panels.csv").query("dataset == 'spring wheat' and panel == 'base'").rate.iloc[0]
J = json.load(open(f"{RES}/does_it_help.json"))["SPRING WHEAT (FHB) (unaugmented)"]
keys = {"spearman": "mean_spearman_r (Tier I)", "pearson": "mean_pearson_r (Tier II, 2024 official)", "rmse": "mean_RMSE (Tier III, 2022 official)"}
paper_rec = {k: (J[v] - J["random"]) / (J["oracle"] - J["random"]) for k, v in keys.items()}
V = [dict(quantity="reversal rate, mean_RMSE vs mean_pearson_r", gpverdict=rev, paper=paper_rev)]
V += [dict(quantity=f"outcome-test recovery, {k}", gpverdict=o["recovery"][k], paper=paper_rec[k]) for k in keys]
V = pd.DataFrame(V); V["abs_difference"] = (V.gpverdict - V.paper).abs()
V.to_csv(f"{RES}/gpverdict_validation.csv", index=False)
print(V.to_string(index=False))
assert (V.abs_difference < 1e-9).all(), "GPverdict no longer reproduces the paper"

# ---- design points for Fig. 6c: genotype-environment cells per method in each dataset (paper rules)
G = "analysis/g2f_leaderboard/results"
obs = pd.read_csv(f"{G}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y", "Hybrid": "k"})
mz = []
for s in ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953", "EnBiSys_sub243568", "NicheSquad_sub244985"]:
    pr = pd.read_csv(f"{G}/{s}.csv").rename(columns={"Yield_Mg_ha": "p", "Hybrid": "k"})
    d = obs.merge(pr, on=["Env", "k"]).dropna(subset=["y", "p"]); d["method"] = s; mz.append(d)
PTS = [("maize (G2F 2022 test set)", pd.concat(mz), 20)]
for lab, f, mn in (("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12), ("soybean", "nust_panel_wide.csv", 25)):
    Q = pd.read_csv(f"{RES}/{f}"); PTS.append((lab, Q[~Q.method.str.contains("__")], mn))
cells = []
for lab, Q, mn in PTS:
    E = gv.env_table(gv.load(Q), 0.10, mn); c = float(E.groupby("method").n.sum().median())
    cells.append(dict(dataset=lab, cells=c, resolvable_gap=gv.resolvable_gap(c, 0.10)))
pd.DataFrame(cells).to_csv(f"{RES}/gpverdict_cells.csv", index=False)
print(pd.DataFrame(cells).round(4).to_string(index=False))

# ---- 2. CUBIC
path = f"{RES}/china_panel_wide.csv"
if not os.path.exists(path):
    print(f"{path} not present (built by panel_china.py with RUN_CUBIC=1); CUBIC application skipped")
    sys.exit(0)
C = pd.read_csv(path); C = C[~C.method.str.contains("__")]
v = gv.verdict(C, frac=0.10, min_n=25, B=400, seed=0)
S = v["summary"]; order = S.spearman.sort_values(ascending=False)
inv = v["invariance"].set_index("key").tier
out = dict(methods=v["data"]["methods"], environments=v["data"]["environments"], genotypes=v["data"]["genotypes"],
           cells=v["data"]["cells"], tier_I=sorted(inv[inv == "I"].index), tier_II=sorted(inv[inv == "II"].index),
           tier_III=sorted(inv[inv == "III"].index), leader=v["leader"], pearson_leader=v["pearson_pick"], rmse_pick=v["rmse_pick"],
           ranking=[(m, float(order[m])) for m in order.index], gap=v["resolution"]["gap"],
           panel_range=list(v["resolution"]["panel_range"]), tied=v["resolution"]["tied_with_leader"],
           margin=v["resolution"]["leader_margin"], cells_to_resolve_margin=v["resolution"]["cells_to_resolve_margin"],
           reversal_rmse_pearson=v["reversal_rmse_pearson"], method_pairs=v["method_pairs"],
           outcome=dict(design=v["outcome"]["design"], splits=v["outcome"]["splits"], recovery=v["outcome"]["recovery"],
                        reliability=v["outcome"]["reliability"]))
json.dump(out, open(f"{RES}/gpverdict_cubic.json", "w"), indent=1)
T = S[["spearman", "pearson", "rmse", "sel_diff"]].copy(); T["rank_spearman"] = v["ranks"].spearman; T["rank_rmse"] = v["ranks"].rmse
T.sort_values("rank_spearman").to_csv(f"{RES}/gpverdict_cubic_methods.csv")
print(json.dumps({k: out[k] for k in ("leader", "pearson_leader", "rmse_pick", "gap", "tied", "margin", "cells_to_resolve_margin", "reversal_rmse_pearson")}, indent=1))
print("outcome", {k: round(100 * x, 1) for k, x in out["outcome"]["recovery"].items()}, "reliability", round(out["outcome"]["reliability"], 3))
