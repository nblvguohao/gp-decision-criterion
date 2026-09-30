"""Pairwise reversal-rate analysis of the 2022 G2F leaderboard.

Method transferred from arXiv:2603.03493 (GRN benchmark ranking reversals):
  reversal  <=>  delta_1 * delta_2 < 0   for a team pair under two metrics
  CI        <=>  Wilson binomial interval
  null      <=>  permute team<->score mapping within each metric
Inputs are official numbers only (Supplemental Table S1, Washburn et al. 2024 Genetics).
"""
import itertools, numpy as np, pandas as pd

SRC = "analysis/g2f_leaderboard/results/S1_best_per_team.csv"
df = pd.read_csv(SRC)
df = df[df["Team Name"].notna()].copy()

LOWER_BETTER = {"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
                "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE = {"lineRegressSlope","mean_lineRegressSlope"}
METRICS = [c for c in df.columns if c not in ("id","Team Name","Submitted At")]

def oriented(col):
    v = df[col].astype(float).to_numpy()
    if col in SLOPE:  return -np.abs(v - 1.0)      # calibration: closer to 1 is better
    return -v if col in LOWER_BETTER else v         # higher = better after orientation

S = {m: oriented(m) for m in METRICS}
teams = df["Team Name"].to_numpy()
n = len(teams)
pairs = list(itertools.combinations(range(n), 2))

def wilson(k, N, z=1.96):
    if N == 0: return (np.nan, np.nan)
    p = k / N; d = 1 + z*z/N
    c = (p + z*z/(2*N)) / d
    h = z*np.sqrt(p*(1-p)/N + z*z/(4*N*N)) / d
    return (max(0, c-h), min(1, c+h))

def reversal_rate(a, b, idx=None):
    P = pairs if idx is None else [(i,j) for i,j in pairs if i in idx and j in idx]
    d1 = np.array([a[i]-a[j] for i,j in P]); d2 = np.array([b[i]-b[j] for i,j in P])
    ok = np.isfinite(d1) & np.isfinite(d2)
    d1, d2 = d1[ok], d2[ok]
    k = int(np.sum(d1*d2 < 0))
    return k, len(d1)

M22, M24 = "mean_RMSE", "mean_pearson_r"
print(f"teams n = {n}   pairs = {len(pairs)}   metrics = {len(METRICS)}\n")
print("="*72)
print("HEADLINE — the two metrics the organizers actually used")
print(f"  2022 official ranking metric : {M22}  (per-environment RMSE, averaged)")
print(f"  2024 official ranking metric : {M24}  (mean within-environment Pearson r)")
k, N = reversal_rate(S[M22], S[M24])
lo, hi = wilson(k, N)
print(f"\n  pairwise reversal rate = {k}/{N} = {k/N:.3f}   Wilson 95% CI [{lo:.3f}, {hi:.3f}]")

rng = np.random.default_rng(20260908)
null = []
for _ in range(5000):
    a = rng.permutation(S[M22]); b = rng.permutation(S[M24])
    kk, NN = reversal_rate(a, b)
    null.append(kk/NN)
null = np.array(null)
print(f"  permutation null (5000x, team<->score mapping destroyed): "
      f"mean {null.mean():.3f}  [{np.percentile(null,2.5):.3f}, {np.percentile(null,97.5):.3f}]")

top = set(np.argsort(-S[M22])[:10])
kt, Nt = reversal_rate(S[M22], S[M24], idx=top)
lo_t, hi_t = wilson(kt, Nt)
print(f"\n  restricted to top-10 teams by the 2022 metric: {kt}/{Nt} = {kt/Nt:.3f}  95% CI [{lo_t:.3f}, {hi_t:.3f}]")

print("\n" + "="*72)
print("BREADTH — all metric pairs")
rows = []
for m1, m2 in itertools.combinations(METRICS, 2):
    k, N = reversal_rate(S[m1], S[m2])
    if N: rows.append((m1, m2, k, N, k/N))
R = pd.DataFrame(rows, columns=["metric_a","metric_b","reversals","pairs","rate"])
print(f"  {len(R)} metric pairs;  reversal rate: median {R['rate'].median():.3f}  "
      f"IQR [{R['rate'].quantile(.25):.3f}, {R['rate'].quantile(.75):.3f}]  max {R['rate'].max():.3f}")
print(f"  metric pairs with >10% of team pairs reversed: {(R['rate']>0.10).mean()*100:.1f}%")
print(f"  metric pairs with >20% of team pairs reversed: {(R['rate']>0.20).mean()*100:.1f}%")
R.sort_values("rate", ascending=False).to_csv("analysis/g2f_leaderboard/results/reversal_by_metric_pair.csv", index=False)
print("\n  top 8 most-discordant metric pairs:")
for _, r in R.sort_values("rate", ascending=False).head(8).iterrows():
    print(f"    {r['rate']:.3f}  {r['metric_a']:<26} vs {r['metric_b']}")
