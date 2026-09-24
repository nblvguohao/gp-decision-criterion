"""The rank-interval statistic without the conformal apparatus, all six datasets.

Section 3.5 reports the median width of a 95 % split-conformal rank interval, calibrated
across the 22 official metrics. `rank_intervals.py` and `analyse_species.rank_interval_width`
pool the nonconformity scores over every (method, metric) cell, so the quantile q is a single
global number and the unclipped width 2q+1 is identical for every method on the board: the
only source of per-method variation in the reported widths is clipping at ranks 1 and M.
Conformal validity would additionally require the 22 metrics to be exchangeable, which they
are not -- fourteen are variants inside the error-magnitude family.

This computes the descriptive statistic that answers the same question over the census of
metrics the organisers actually published, needing no exchangeability assumption:

  observed rank range   max_t r_it - min_t r_it + 1      per method, across the 22 metrics
  observed IQR          the interquartile span of the same 22 ranks
  escape rate           the fraction of the M x 22 observed rank cells that fall OUTSIDE
                        their own method's clipped conformal interval

The escape rate is the diagnostic: a conformal interval that is wider than the observed
range and still fails to contain part of it is symmetric about a median that the rank
distribution is not symmetric about.

Intervals are metric-clustered bootstraps (resampling the 22 metrics with replacement),
which is the resampling unit that matches the claim -- the claim is about the choice of
metric, so the metric is the cluster.

Conventions follow Methods 2.3 exactly, via `analyse_species`: min_n = 25 genotypes per
environment, 12 for spring wheat, slopes scored as -|slope-1|, error metrics negated.

Writes analysis/crosscrop/results/rank_census.csv
       analysis/crosscrop/results/rank_census_per_method.csv
       analysis/crosscrop/results/rank_matrix_maize.csv  (30 teams x 22 metrics, for Fig. 4)
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
from analyse_species import metric_table, orient, METRICS

RES = "analysis/crosscrop/results"
G2F = "analysis/g2f_leaderboard/results"
B = 2000
SEED = 20260910

SETS = [("rice",         f"{RES}/rice_panel_wide.csv",  25),
        ("wheat",        f"{RES}/wheat_panel_wide.csv", 25),
        ("common bean",  f"{RES}/bean_panel_wide.csv",  25),
        ("spring wheat", f"{RES}/ursn_panel_wide.csv",  12),
        ("soybean",      f"{RES}/nust_panel_wide.csv",  25)]


def rank_matrix(T):
    """methods x 22 metrics matrix of ranks, 1 = best after orientation."""
    M = len(T)
    out = {}
    for m in METRICS:
        o = np.argsort(-orient(T, m), kind="mergesort")
        r = np.empty(M, int)
        r[o] = np.arange(1, M + 1)
        out[m] = r
    return pd.DataFrame(out, index=T.index)


def conformal(R):
    """The paper's estimator. Returns q, the clipped bounds, and the widths."""
    M = len(R)
    Rc = R.to_numpy()
    rh = np.median(Rc, axis=1)
    s = np.sort(np.abs(Rc - rh[:, None]).ravel())
    n = len(s)
    q = s[min(int(np.ceil((n + 1) * 0.95)), n) - 1]
    lo = np.clip(np.round(rh - q), 1, M)
    hi = np.clip(np.round(rh + q), 1, M)
    return float(q), lo, hi, hi - lo + 1


def census(R):
    Rc = R.to_numpy()
    rng_ = Rc.max(axis=1) - Rc.min(axis=1) + 1
    iqr = np.percentile(Rc, 75, axis=1) - np.percentile(Rc, 25, axis=1)
    return rng_, iqr


def boot_median_range(R, rng):
    """Metric-clustered bootstrap of the median observed rank range."""
    cols = np.arange(R.shape[1])
    out = []
    Rc = R.to_numpy()
    for _ in range(B):
        c = rng.integers(0, len(cols), len(cols))
        S = Rc[:, c]
        out.append(np.median(S.max(axis=1) - S.min(axis=1) + 1))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


rows, per_method = [], []
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------ maize
S1 = pd.read_csv(f"{G2F}/S1_best_per_team.csv")
S1 = S1[S1["Team Name"].notna()].set_index("Team Name")[METRICS]
tables = [("maize", S1)]

for label, path, mn in scope.kept(SETS):
    P = pd.read_csv(path)
    tables.append((label, metric_table(P, mn)[METRICS]))
    print(f"  metric table built: {label}", flush=True)

for label, T in tables:
    M = len(T)
    R = rank_matrix(T)
    if label == "maize":                       # the full rank matrix, for the Fig. 4 blob plot
        R.to_csv(f"{RES}/rank_matrix_maize.csv")
    q, lo, hi, w = conformal(R)
    rng_, iqr = census(R)
    lo_ci, hi_ci = boot_median_range(R, rng)
    Rc = R.to_numpy()
    escape = float(np.mean((Rc < lo[:, None]) | (Rc > hi[:, None])))
    # the leader under the 2024-style official metric, and its observed span. The
    # miscalibration variants are affine maps of their parent and tie with it on this
    # metric, so the leader is taken among the base methods: a variant is never reported
    # as the leader merely by tie order.
    score = np.asarray(orient(T, "mean_pearson_r"), dtype=float)
    cand = [i for i, m in enumerate(T.index) if "__" not in str(m)] or list(range(len(T)))
    lead = T.index[max(cand, key=lambda i: score[i])]
    li = list(T.index).index(lead)
    rows.append(dict(
        dataset=label, methods=M,
        conformal_q=q, conformal_unclipped_width=2 * q + 1,
        conformal_median_width=float(np.median(w)),
        conformal_width_frac=float(np.median(w)) / M,
        n_clipped=int(((lo == 1) | (hi == M)).sum()),
        unclipped_width_sd=float(np.std(2 * q + 1 + np.zeros(M))),
        clipped_width_sd=float(np.std(w)),
        census_median_range=float(np.median(rng_)),
        census_range_frac=float(np.median(rng_)) / M,
        census_range_ci_lo=lo_ci, census_range_ci_hi=hi_ci,
        census_median_iqr=float(np.median(iqr)),
        census_iqr_frac=float(np.median(iqr)) / M,
        escape_rate=escape,
        leader=lead,
        leader_conformal=f"[{int(lo[li])}, {int(hi[li])}]",
        leader_observed=f"[{int(Rc[li].min())}, {int(Rc[li].max())}]",
        distinct_rank_vectors=int(len(set(map(tuple, R.T.to_numpy())))),
    ))
    for i, name in enumerate(T.index):
        per_method.append(dict(dataset=label, method=name,
                               median_rank=float(np.median(Rc[i])),
                               conformal_lo=int(lo[i]), conformal_hi=int(hi[i]),
                               conformal_width=int(w[i]),
                               observed_lo=int(Rc[i].min()), observed_hi=int(Rc[i].max()),
                               observed_range=int(rng_[i]), observed_iqr=float(iqr[i])))

C = pd.DataFrame(rows)
C.to_csv(f"{RES}/rank_census.csv", index=False)
pd.DataFrame(per_method).to_csv(f"{RES}/rank_census_per_method.csv", index=False)

pd.set_option("display.width", 220)
print("\n" + "=" * 112)
print("The conformal width against the census of the same 22 metrics\n")
print(f"  {'dataset':<14}{'M':>4}{'conformal':>22}{'census range':>24}{'census IQR':>14}{'escape':>9}")
print("  " + "-" * 96)
for _, r in C.iterrows():
    print(f"  {r['dataset']:<14}{r['methods']:>4}"
          f"{r['conformal_median_width']:>9.0f} ({r['conformal_width_frac']*100:>3.0f} %)"
          f"{'':>2}q={r['conformal_q']:<5.1f}"
          f"{r['census_median_range']:>9.0f} ({r['census_range_frac']*100:>3.0f} %)"
          f"  [{r['census_range_ci_lo']:.0f}, {r['census_range_ci_hi']:.0f}]"
          f"{r['census_median_iqr']:>8.1f} ({r['census_iqr_frac']*100:>3.0f} %)"
          f"{r['escape_rate']*100:>8.1f} %")

print("\n  Every dataset: the unclipped conformal width is one number for all methods")
print("  (sd = 0 by construction); the reported median is that number after clipping.\n")
print(f"  {'dataset':<14}{'unclipped':>10}{'clipped median':>16}{'clipped sd':>12}"
      f"{'n clipped':>11}{'distinct rankings of 22':>25}")
print("  " + "-" * 88)
for _, r in C.iterrows():
    print(f"  {r['dataset']:<14}{r['conformal_unclipped_width']:>10.0f}"
          f"{r['conformal_median_width']:>16.0f}{r['clipped_width_sd']:>12.2f}"
          f"{r['n_clipped']:>11d}{r['distinct_rank_vectors']:>25d}")

print(f"\n  {'dataset':<14}{'leader by mean_pearson_r':<28}{'conformal':>12}{'observed':>12}")
print("  " + "-" * 66)
for _, r in C.iterrows():
    print(f"  {r['dataset']:<14}{str(r['leader'])[:27]:<28}{r['leader_conformal']:>12}{r['leader_observed']:>12}")

print(f"\n-> written {RES}/rank_census.csv and {RES}/rank_census_per_method.csv")
