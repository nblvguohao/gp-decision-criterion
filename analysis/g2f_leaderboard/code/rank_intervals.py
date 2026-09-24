"""Conformal rank prediction intervals for the 2022 G2F leaderboard.

Framework transferred from arXiv:2606.08679 (Rank Intervals for Leaderboards):
  models = 30 competition teams (M)
  tasks  = 22 official evaluation metrics (T), each inducing one full ranking
  goal   = for each team, a prediction interval for the rank it would receive
           under a metric the organizers have not used yet.

Split/jackknife conformal on ranks:
  point predictor  rhat_i = median_t r_{i,t}  over calibration metrics
  nonconformity    s_{i,t} = |r_{i,t} - rhat_i|
  interval         [rhat_i - q, rhat_i + q],  q = ceil((n+1)(1-alpha))-th order stat
Coverage is checked two ways: leave-one-metric-out, and a real held-out event
(the organizers actually switched to mean_pearson_r for the 2024 edition).
"""
import numpy as np, pandas as pd

ALPHA = 0.05
df = pd.read_csv("analysis/g2f_leaderboard/results/S1_best_per_team.csv")
df = df[df["Team Name"].notna()].copy()
teams = df["Team Name"].to_numpy(); M = len(teams)

LOWER = {"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
         "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE = {"lineRegressSlope","mean_lineRegressSlope"}
METRICS = [c for c in df.columns if c not in ("id","Team Name","Submitted At")]

def ranks(col):
    v = df[col].astype(float).to_numpy()
    s = -np.abs(v-1.0) if col in SLOPE else (-v if col in LOWER else v)
    order = np.argsort(-s, kind="mergesort")
    r = np.empty(M, int); r[order] = np.arange(1, M+1)
    return r

R = pd.DataFrame({m: ranks(m) for m in METRICS}, index=teams)   # M x T
T = R.shape[1]
print(f"M = {M} teams, T = {T} official metrics\n")

def conformal(cal_metrics, alpha=ALPHA):
    Rc = R[cal_metrics].to_numpy()
    rhat = np.median(Rc, axis=1)
    s = np.abs(Rc - rhat[:, None]).ravel()
    n = len(s)
    k = int(np.ceil((n+1)*(1-alpha)))
    q = np.sort(s)[min(k, n)-1]
    return rhat, q

# ---- 1. full-data intervals -------------------------------------------------
rhat, q = conformal(METRICS)
lo = np.clip(np.round(rhat-q), 1, M).astype(int)
hi = np.clip(np.round(rhat+q), 1, M).astype(int)
width = hi-lo+1
print("="*74)
print(f"95% conformal rank interval, calibrated on all {T} official metrics   (q = {q:.1f})")
print(f"  median interval width = {np.median(width):.0f} of {M} places "
      f"({np.median(width)/M*100:.0f}% of the leaderboard)")
out = pd.DataFrame({"team":teams,"median_rank":rhat.astype(int),"lo":lo,"hi":hi,"width":width}) \
        .sort_values("median_rank")
print("\n  top 10 teams by median rank across all official metrics:")
print(f"    {'team':<22}{'median':>7}{'95% rank interval':>22}")
for _, r in out.head(10).iterrows():
    print(f"    {r['team']:<22}{r['median_rank']:>7}      [{r['lo']:>2}, {r['hi']:>2}]  (width {r['width']})")
out.to_csv("analysis/g2f_leaderboard/results/rank_intervals.csv", index=False)

# ---- 2. leave-one-metric-out coverage --------------------------------------
cov = []
for held in METRICS:
    cal = [m for m in METRICS if m != held]
    rh, qq = conformal(cal)
    truth = R[held].to_numpy()
    cov.append(np.mean((truth >= rh-qq) & (truth <= rh+qq)))
cov = np.array(cov)
print("\n" + "="*74)
print(f"leave-one-metric-out coverage: mean {cov.mean()*100:.1f}%  "
      f"(nominal 95%), min {cov.min()*100:.0f}%, metrics below nominal: {(cov<0.95).sum()}/{T}")

# ---- 3. the real held-out event: the 2024 metric ---------------------------
HELD = "mean_pearson_r"
cal = [m for m in METRICS if m != HELD]
rh, qq = conformal(cal)
truth = R[HELD].to_numpy()
inside = (truth >= rh-qq) & (truth <= rh+qq)
print("\n" + "="*74)
print(f"REAL HELD-OUT EVENT — organizers switched the official metric to '{HELD}' in 2024")
print(f"  intervals built WITHOUT that metric;  coverage = {inside.mean()*100:.1f}%  "
      f"({inside.sum()}/{M} teams landed inside their predicted interval)")
miss = pd.DataFrame({"team":teams,"predicted":np.round(rh).astype(int),
                     "interval":[f"[{max(1,int(round(a-qq)))}, {min(M,int(round(a+qq)))}]" for a in rh],
                     "actual_2024_rank":truth})[~inside]
if len(miss):
    print(f"\n  teams that fell OUTSIDE their interval ({len(miss)}):")
    for _, r in miss.sort_values("actual_2024_rank").iterrows():
        print(f"    {r['team']:<22} predicted {r['interval']:<10} actual {r['actual_2024_rank']}")
