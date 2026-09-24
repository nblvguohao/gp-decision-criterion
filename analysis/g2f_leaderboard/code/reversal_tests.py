"""Which pairs of official metrics reverse more than 5 % of team pairs?

For each of the 231 pairs of the 22 official metrics, the reversal rate over the 435
pairs of the 30 teams is tested against H0: rate <= 0.05. Team pairs share teams, so
they are not independent trials; the test resamples teams (4,000 draws, self-pairs
excluded, as in analyse_species.rev_from). Primary: one-sided Wald z with the bootstrap
standard error. Sensitivity: percentile P = share of bootstrap rates <= 0.05, floored at
1 / (B + 1). Benjamini-Hochberg control across the 231 pairs; BH remains valid under the
positive dependence among these tests (Benjamini & Yekutieli 2001), but the 231 are not
231 independent findings.

Writes analysis/g2f_leaderboard/results/reversal_tests.csv
"""
import sys, itertools, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0, "analysis/crosscrop/code")
from analyse_species import orient, rev_from, METRICS

RES = "analysis/g2f_leaderboard/results"
B, H0 = 4000, 0.05
G = pd.read_csv(f"{RES}/S1_best_per_team.csv")
G = G[G["Team Name"].notna()].set_index("Team Name")
n = len(G)
S = {m: orient(G, m) for m in METRICS}
rng = np.random.default_rng(0)
idx = [rng.integers(0, n, n) for _ in range(B)]


def bh(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[o] = np.minimum(q, 1)
    return out


rows = []
for m1, m2 in itertools.combinations(METRICS, 2):
    a, b = S[m1], S[m2]
    pt = rev_from(a, b, list(range(n)))
    bs = np.array([rev_from(a, b, i) for i in idx])
    se = bs.std(ddof=1)
    # a pair whose rate is identical in every resample carries no sampling variability:
    # its z is +inf only if the rate itself exceeds H0, and -inf (P = 1) otherwise. The
    # ten pairs that never reverse (rate 0.000) are the second case, not the first.
    z = (pt - H0) / se if se > 0 else (np.inf if pt > H0 else -np.inf)
    rows.append(dict(metric_a=m1, metric_b=m2, reversal=pt, boot_se=se, z=z,
                     p_wald=norm.sf(z), p_percentile=max((bs <= H0).mean(), 1 / (B + 1))))
R = pd.DataFrame(rows)
R["q_wald"] = bh(R.p_wald); R["q_percentile"] = bh(R.p_percentile)
R.to_csv(f"{RES}/reversal_tests.csv", index=False)
off = R[((R.metric_a == "mean_RMSE") & (R.metric_b == "mean_pearson_r")) |
        ((R.metric_a == "mean_pearson_r") & (R.metric_b == "mean_RMSE"))].iloc[0]
print(f"median reversal over 231 pairs: {R.reversal.median():.3f}")
for lab in ("wald", "percentile"):
    print(f"{lab:>10}: BH 5 % {int((R[f'q_{lab}'] < .05).sum())}, BH 1 % {int((R[f'q_{lab}'] < .01).sum())}; "
          f"official pair reversal {off.reversal:.3f}, adjusted P {off[f'q_{lab}']:.2g}")

# ---------------------------------------------------------------- classes (Section 3.3)
# Reversal within and between the order-sensitive, calibration-invariant metrics (Tiers I
# and II) and the fourteen error-magnitude metrics.
TIER12 = ["mean_spearman_r", "mean_pearson_r"]
ERR = [m for m in METRICS if not any(t in m for t in ("pearson", "spearman", "lineRegressSlope"))
       or m.endswith("r2_score")]
assert len(ERR) == 14, ERR
def _cls(m):
    return "tier12" if m in TIER12 else ("error" if m in ERR else "other")
R["class_a"] = R.metric_a.map(_cls); R["class_b"] = R.metric_b.map(_cls)
rows = []
for lab, sel in (("Tier I-II pair", (R.class_a == "tier12") & (R.class_b == "tier12")),
                 ("within error family", (R.class_a == "error") & (R.class_b == "error")),
                 ("Tier I-II vs error family", ((R.class_a == "tier12") & (R.class_b == "error")) |
                                              ((R.class_a == "error") & (R.class_b == "tier12")))):
    v = R.reversal[sel]
    rows.append(dict(group=lab, pairs=int(sel.sum()), median=v.median(), lo=v.min(), hi=v.max()))
C = pd.DataFrame(rows); C.to_csv(f"{RES}/reversal_by_class.csv", index=False)
print(C.round(3).to_string(index=False))
