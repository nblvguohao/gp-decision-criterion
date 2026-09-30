"""Paired rank-invariance test: does a decision-neutral transform reorder the board?

The invariance test in `invariance.py` and `monotone_test_panels.py`
perturbs ONE submission and measures how far each metric's VALUE moves. Every
claim from Section 3.2 onwards is instead about the ORDER of the methods. The
missing link is the paired version: apply a decision-neutral transform to each
submission INDEPENDENTLY, and count how often the resulting ranking of the teams
differs from the ranking of the untransformed submissions.

The transform is decision-neutral in the strict sense: it is applied within
environment and is strictly increasing there, so every submission's own top-k
pick in every environment is unchanged. The decision each team's predictions
imply is therefore fixed by construction, and any movement in the leaderboard is
movement that no selection decision can justify.

Two transform classes, matching Section 2.4:
  affine    yhat_e -> a_e*yhat_e + b_e,  a_e ~ U(0.5, 2), b_e ~ N(0, sd(y_e))
  monotone  yhat_e -> a random strictly increasing spline through seven sorted
            knots, applied to within-environment ranks

and two tie conventions for the monotone class, because the distinction turns out
to matter:
  ordinal   ranks from argsort, which assigns distinct ranks to tied predictions
            and therefore BREAKS ties -- this is what monotone_test_panels.py does
  tie-safe  average ranks, so exactly tied predictions map to the same value and
            the map is a genuine function of the prediction vector

Only the tie-safe convention is strictly decision-neutral. The ordinal one can
move a tied genotype across the k-th boundary, so it perturbs the decision by the
amount the predictions are tied. Reporting both separates "the metric reorders
the board for free" from "the metric responds to tie-breaking".

Reports, per metric: the probability that the ranking of the five submissions
changes at all, the pair-level reversal rate over the ten team pairs, the
probability that the top-ranked submission changes, and mean Kendall tau against
the untransformed ranking. Intervals are Monte-Carlo over replicates and are
conditional on these five submissions; they carry no uncertainty over the choice
of team, of which there are five.

Writes analysis/g2f_leaderboard/results/paired_invariance.csv
       analysis/g2f_leaderboard/results/paired_invariance_pairs.csv
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress, rankdata, kendalltau
from itertools import combinations

RES = "analysis/g2f_leaderboard/results"
NREP = 400
ALPHA = 0.05

VERIFIED = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944",
            "KernelOfTruth_sub244953", "EnBiSys_sub243568", "NicheSquad_sub244985"]

LOWER = {"MAE", "realative_MAE", "normalized_MAE", "RMSE", "relative_RMSE", "normalized_RMSE"}
SLOPE = {"lineRegressSlope"}

obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})


def load(name):
    pr = pd.read_csv(f"{RES}/{name}.csv").rename(columns={"Yield_Mg_ha": "p"})
    return obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])


def blk(y, p, out, pre=""):
    e = p - y
    out[pre + "RMSE"] = np.sqrt(np.mean(e ** 2))
    out[pre + "MAE"] = np.mean(np.abs(e))
    out[pre + "normalized_RMSE"] = out[pre + "RMSE"] / np.std(y)
    out[pre + "relative_RMSE"] = out[pre + "RMSE"] / np.mean(y)
    out[pre + "normalized_MAE"] = out[pre + "MAE"] / np.std(y)
    out[pre + "realative_MAE"] = out[pre + "MAE"] / np.mean(y)
    out[pre + "r2_score"] = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    r = pearsonr(y, p)[0] if len(np.unique(p)) > 1 else np.nan
    out[pre + "pearson_r"] = r
    out[pre + "r2_pearson"] = r ** 2 if np.isfinite(r) else np.nan
    out[pre + "spearman_r"] = spearmanr(y, p)[0] if len(np.unique(p)) > 1 else np.nan
    out[pre + "lineRegressSlope"] = linregress(p, y).slope if len(np.unique(p)) > 1 else np.nan


def panel22(d):
    """The 22 official metrics: pooled, and within environment then averaged."""
    out = {}
    blk(d["y"].to_numpy(), d["p"].to_numpy(), out)
    per = []
    for _, g in d.groupby("Env", sort=True):
        if len(g) < 3 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        t = {}
        blk(g["y"].to_numpy(), g["p"].to_numpy(), t)
        per.append(t)
    P = pd.DataFrame(per)
    for c in P.columns:
        out["mean_" + c] = P[c].mean()
    return out


def oriented(metric, value):
    base = metric[5:] if metric.startswith("mean_") else metric
    if base in SLOPE:
        return -abs(value - 1.0)
    return -value if base in LOWER else value


def rank_of_teams(values):
    """1 = best. values already oriented so that higher is better."""
    v = np.asarray(values, float)
    order = np.argsort(-v, kind="mergesort")
    r = np.empty(len(v), int)
    r[order] = np.arange(1, len(v) + 1)
    return r


def affine_transform(g, rng):
    p = g["p"].to_numpy()
    a = rng.uniform(0.5, 2.0)
    b = rng.normal(0.0, max(g["y"].to_numpy().std(), 1e-9))
    return a * p + b


def monotone_transform(g, rng, tie_safe):
    p = g["p"].to_numpy()
    n = len(p)
    if tie_safe:
        u = (rankdata(p, method="average") - 0.5) / n
    else:
        u = (np.argsort(np.argsort(p)) + 0.5) / n
    knots = np.sort(rng.uniform(0, 1, 6))
    knots = np.r_[0.0, knots, 1.0]
    vals = np.sort(rng.uniform(0, 1, len(knots)))
    vals[0], vals[-1] = 0.0, 1.0
    return np.interp(u, knots, vals)


def perturb(d, kind, rng, tie_safe=True):
    out = np.empty(len(d))
    for _, g in d.groupby("Env", sort=True):
        if kind == "affine":
            out[g.index.to_numpy()] = affine_transform(g, rng)
        else:
            out[g.index.to_numpy()] = monotone_transform(g, rng, tie_safe)
    d2 = d.copy()
    d2["p"] = out
    return d2


# ------------------------------------------------------------------ tie audit
print("=" * 78)
print("TIE AUDIT — how much room does the tie convention have to matter?\n")
tie_rows = []
data = {}
for name in VERIFIED:
    d = load(name).reset_index(drop=True)
    data[name] = d
    n_tied_cells, n_cells, n_env_with_ties, boundary = 0, 0, 0, 0
    for _, g in d.groupby("Env", sort=True):
        p = g["p"].to_numpy()
        n_cells += len(p)
        vals, counts = np.unique(p, return_counts=True)
        tied = counts[counts > 1].sum()
        n_tied_cells += tied
        if tied:
            n_env_with_ties += 1
            k = max(1, int(round(0.10 * len(p))))
            cut = np.sort(p)[::-1][k - 1]
            if (p == cut).sum() > 1:
                boundary += 1
    tie_rows.append((name, n_cells, n_tied_cells, n_tied_cells / n_cells,
                     n_env_with_ties, d["Env"].nunique(), boundary))
    print(f"  {name:<26} {n_tied_cells:>6}/{n_cells} cells tied "
          f"({n_tied_cells/n_cells*100:5.2f} %), "
          f"{n_env_with_ties}/{d['Env'].nunique()} environments, "
          f"{boundary} with a tie straddling the k=10 % boundary")
TIE = pd.DataFrame(tie_rows, columns=["team", "cells", "tied_cells", "tied_frac",
                                      "envs_with_ties", "envs", "envs_tie_at_boundary"])

# ------------------------------------------------------------------ baseline
base_vals = {name: panel22(data[name]) for name in VERIFIED}
METRICS = [m for m in base_vals[VERIFIED[0]]
           if all(np.isfinite(base_vals[n].get(m, np.nan)) for n in VERIFIED)]
base_rank = {m: rank_of_teams([oriented(m, base_vals[n][m]) for n in VERIFIED])
             for m in METRICS}

print(f"\n{len(METRICS)} of 22 metrics finite for all five submissions; "
      f"{len(VERIFIED)} submissions, {len(list(combinations(range(5), 2)))} team pairs, "
      f"{NREP} replicates per transform class\n")

PAIRS = list(combinations(range(len(VERIFIED)), 2))


def run_class(kind, tie_safe, seed):
    rng = np.random.default_rng(seed)
    changed = {m: 0 for m in METRICS}
    top1 = {m: 0 for m in METRICS}
    revpairs = {m: 0 for m in METRICS}
    taus = {m: [] for m in METRICS}
    relchg = {m: [] for m in METRICS}
    for _ in range(NREP):
        vals = {}
        for name in VERIFIED:
            vals[name] = panel22(perturb(data[name], kind, rng, tie_safe))
        for m in METRICS:
            o = [oriented(m, vals[n][m]) for n in VERIFIED]
            if not all(np.isfinite(o)):
                continue
            r = rank_of_teams(o)
            b = base_rank[m]
            if not np.array_equal(r, b):
                changed[m] += 1
            if np.argmin(r) != np.argmin(b):
                top1[m] += 1
            revpairs[m] += sum(1 for i, j in PAIRS if (r[i] - r[j]) * (b[i] - b[j]) < 0)
            taus[m].append(kendalltau(r, b)[0])
            for n in VERIFIED:
                relchg[m].append(abs(vals[n][m] - base_vals[n][m]) /
                                 max(abs(base_vals[n][m]), 1e-9))
    rows = []
    for m in METRICS:
        k, N = changed[m], NREP
        lo, hi = mc_interval(k, N)
        rows.append(dict(metric=m, transform=kind,
                         tie_convention="tie-safe" if tie_safe else "ordinal",
                         p_ranking_changed=k / N, ci_lo=lo, ci_hi=hi,
                         p_top1_changed=top1[m] / N,
                         pair_reversal_rate=revpairs[m] / (N * len(PAIRS)),
                         mean_kendall_tau=float(np.mean(taus[m])),
                         median_rel_value_change=float(np.median(relchg[m]))))
    return pd.DataFrame(rows)


def mc_interval(k, N, alpha=ALPHA):
    """Clopper-Pearson over independent Monte-Carlo replicates."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, N - k + 1)
    hi = 1.0 if k == N else beta.ppf(1 - alpha / 2, k + 1, N - k)
    return float(lo), float(hi)


frames = [run_class("affine", True, 20260910),
          run_class("monotone", True, 20260911),
          run_class("monotone", False, 20260912)]
R = pd.concat(frames, ignore_index=True)

for kind, tie in [("affine", "tie-safe"), ("monotone", "tie-safe"), ("monotone", "ordinal")]:
    S = R[(R.transform == kind) & (R.tie_convention == tie)].sort_values("p_ranking_changed")
    print("=" * 78)
    print(f"{kind.upper()} transform, {tie} ties — decision unchanged by construction\n")
    print(f"  {'metric':<24}{'P(order changes)':>18}{'95% MC CI':>18}"
          f"{'pair rev':>10}{'P(top1)':>9}{'tau':>7}")
    print("  " + "-" * 84)
    for _, r in S.iterrows():
        print(f"  {r['metric']:<24}{r['p_ranking_changed']:>18.3f}"
              f"   [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              f"{r['pair_reversal_rate']:>10.3f}{r['p_top1_changed']:>9.3f}"
              f"{r['mean_kendall_tau']:>7.2f}")
    stable = S[S.p_ranking_changed == 0]["metric"].tolist()
    print(f"\n  metrics whose ranking of the five never changed in {NREP} replicates: "
          f"{len(stable)} -> {sorted(stable)}")

R.to_csv(f"{RES}/paired_invariance.csv", index=False)
TIE.to_csv(f"{RES}/paired_invariance_pairs.csv", index=False)
print(f"\n-> written {RES}/paired_invariance.csv  ({len(R)} rows)")
print(f"-> written {RES}/paired_invariance_pairs.csv  (tie audit, {len(TIE)} rows)")
