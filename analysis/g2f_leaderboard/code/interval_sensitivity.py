"""Is the 95% rank-interval width an artefact of redundancy in the metric panel?

Fourteen of the 22 official G2F metrics are variants within the error-magnitude
family, so one can reasonably ask whether the reported interval width is a
property of the leaderboard or of how many near-duplicate metrics the organisers
happened to publish. This recomputes the conformal rank intervals of
`rank_intervals.py` on de-duplicated and family-restricted metric sets.

Writes analysis/g2f_leaderboard/results/rank_interval_sensitivity.csv
(Supplementary Table S8).
"""
import numpy as np, pandas as pd

ALPHA = 0.05
SRC = "analysis/g2f_leaderboard/results/S1_best_per_team.csv"
OUT = "analysis/g2f_leaderboard/results/rank_interval_sensitivity.csv"

df = pd.read_csv(SRC)
df = df[df["Team Name"].notna()].copy()
teams = df["Team Name"].to_numpy(); M = len(teams)

LOWER = {"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
         "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE",
         "mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE = {"lineRegressSlope","mean_lineRegressSlope"}
METRICS = [c for c in df.columns if c not in ("id","Team Name","Submitted At")]
ERRFAM = [m for m in METRICS if m in LOWER or m in ("r2_score","mean_r2_score")]
CORFAM = ["pearson_r","spearman_r","r2_pearson","mean_pearson_r","mean_spearman_r","mean_r2_pearson"]

def ranks(col):
    v = df[col].astype(float).to_numpy()
    s = -np.abs(v-1.0) if col in SLOPE else (-v if col in LOWER else v)
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(M, int); r[order] = np.arange(1, M+1)
    return r

R = pd.DataFrame({m: ranks(m) for m in METRICS}, index=teams)

def conformal(cols, alpha=ALPHA):
    Rc = R[cols].to_numpy()
    rhat = np.median(Rc, axis=1)
    s = np.abs(Rc - rhat[:, None]).ravel()
    n = len(s); k = int(np.ceil((n+1)*(1-alpha)))
    q = np.sort(s)[min(k, n)-1]
    lo = np.clip(np.round(rhat-q), 1, M).astype(int)
    hi = np.clip(np.round(rhat+q), 1, M).astype(int)
    return lo, hi, hi-lo+1

SETS = [
 ("all 22 official metrics", METRICS),
 ("six distinct quantities, within environment",
  ["mean_RMSE","mean_MAE","mean_r2_score","mean_pearson_r","mean_spearman_r","mean_lineRegressSlope"]),
 ("six distinct quantities, pooled",
  ["RMSE","MAE","r2_score","pearson_r","spearman_r","lineRegressSlope"]),
 ("family-balanced (2 error, 2 correlation, 1 slope)",
  ["mean_RMSE","mean_MAE","mean_pearson_r","mean_spearman_r","mean_lineRegressSlope"]),
 ("error-magnitude family only", ERRFAM),
 ("correlation family only", CORFAM),
 ("Tier I and II metrics only", ["mean_spearman_r","mean_pearson_r"]),
]

winner = df.loc[df["mean_RMSE"].astype(float).idxmin(), "Team Name"]
rows = []
for name, cols in SETS:
    lo, hi, w = conformal(cols)
    i = list(teams).index(winner)
    rows.append(dict(metric_set=name, n_metrics=len(cols),
                     median_width=float(np.median(w)),
                     pct_of_board=round(float(np.median(w))/M*100, 1),
                     winner_lo=int(lo[i]), winner_hi=int(hi[i])))
out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"winner under the 2022 official metric: {winner}\n")
print(out.to_string(index=False))
print(f"\nwrote {OUT}")
