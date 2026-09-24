"""Checks on two load-bearing claims (Supplementary Sections S9-S10). Creates no side effects on any
existing result file; writes only ties_at_boundary.csv, decision_preservation.csv, transform_class.csv and rank_interval_transfer.csv.

CLAIM 1 (Section 2.1 characterisation proposition), attacked at two places:
  1a  TIES.  A strictly increasing map cannot separate tied predictions, so the
      transformation class never touches them.  The premise "the decision is
      unchanged" is therefore about a SET-VALUED top-k when a tie straddles the
      k-th boundary.  This counts, in the five verified G2F submissions and in
      the five constructed panels, how many exact within-environment ties exist
      and how often one straddles the k = 10 % boundary.
  1b  CROSS-ENVIRONMENT AGGREGATION.  The class in the manuscript allows a
      DIFFERENT increasing map per environment.  Under a single COMMON map the
      maximal invariant is the pooled rank pattern, so the admissible class is
      strictly larger and the pooled metrics are readmitted.  This measures how
      many of the 22 official metrics survive each class, so the manuscript's
      exclusion of pooled metrics can be priced.

CLAIM 2 (Section 3.5, the 57 % rank interval):
  2a  The estimator in rank_intervals.py pools nonconformity scores over ALL
      (team, metric) cells, so q is one global number and every team gets the
      same width except where it is clipped at the ends of the board.  Measured
      here.
  2b  Leave-one-metric-out coverage is close to self-coverage: q barely moves
      when one of 22 metrics is dropped.  Measured here.
  2c  The real exchangeability test is leave-one-FAMILY-out: calibrate on the
      error-magnitude family and predict the correlation family, and the
      reverse.  Measured here.
  2d  The plain descriptive alternative: each team's empirical rank range over
      the census of 22 metrics, with a metric-clustered bootstrap.

Writes analysis/g2f_leaderboard/results/ties_at_boundary.csv
       analysis/g2f_leaderboard/results/transform_class.csv
       analysis/g2f_leaderboard/results/rank_interval_transfer.csv
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress

RES = "analysis/g2f_leaderboard/results"
XC = "analysis/crosscrop/results"
FRAC = 0.10
VERIFIED = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
            "EnBiSys_sub243568", "NicheSquad_sub244985"]


def maize_long():
    obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
    out = []
    for n in VERIFIED:
        pr = pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha": "p"})
        d = obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])
        d = d.rename(columns={"Hybrid": "k"})
        d["method"] = n
        out.append(d[["Env", "k", "y", "p", "method"]])
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 1a  TIES ---
def tie_audit(P, label, min_n):
    """Exact within-environment ties in the predictions, and whether a tie group
    straddles the k-th boundary so that the top-k SET is not unique."""
    rows = []
    for (m, e), g in P.groupby(["method", "Env"], sort=False):
        p = g["p"].to_numpy(float)
        n = len(p)
        if n < min_n:
            continue
        u, cnt = np.unique(p, return_counts=True)
        n_tied_cells = int(cnt[cnt > 1].sum())
        n_tie_groups = int((cnt > 1).sum())
        k = max(1, int(np.ceil(FRAC * n)))
        s = np.sort(p)[::-1]
        # a tie straddles the boundary iff the k-th largest value also occurs
        # at a position beyond k
        straddle = bool(np.any(s[k:] == s[k - 1])) if k < n else False
        # how many genotypes are competing for how many remaining slots
        if straddle:
            v = s[k - 1]
            n_at = int(np.sum(p == v))
            n_above = int(np.sum(p > v))
            slots = k - n_above
        else:
            n_at, slots = 0, 0
        rows.append(dict(dataset=label, method=m, Env=e, n=n, k=k,
                         n_tie_groups=n_tie_groups, n_tied_cells=n_tied_cells,
                         frac_tied_cells=n_tied_cells / n, boundary_straddle=straddle,
                         n_tied_at_boundary=n_at, slots_remaining=slots))
    return pd.DataFrame(rows)


# --------------------------------------------- 1b  TRANSFORMATION CLASSES ---
def metrics22(d):
    """The 22-metric panel exactly as analysis/g2f_leaderboard/code/invariance.py
    computes it (pooled block + within-environment block averaged)."""
    out = {}

    def blk(y, p, pre=""):
        e = p - y
        out[pre + "RMSE"] = np.sqrt(np.mean(e ** 2)); out[pre + "MAE"] = np.mean(np.abs(e))
        out[pre + "normalized_RMSE"] = out[pre + "RMSE"] / np.std(y)
        out[pre + "relative_RMSE"] = out[pre + "RMSE"] / np.mean(y)
        out[pre + "normalized_MAE"] = out[pre + "MAE"] / np.std(y)
        out[pre + "realative_MAE"] = out[pre + "MAE"] / np.mean(y)
        out[pre + "r2_score"] = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
        r = pearsonr(y, p)[0] if len(np.unique(p)) > 1 else np.nan
        out[pre + "pearson_r"] = r; out[pre + "r2_pearson"] = r ** 2 if np.isfinite(r) else np.nan
        out[pre + "spearman_r"] = spearmanr(y, p)[0] if len(np.unique(p)) > 1 else np.nan
        out[pre + "lineRegressSlope"] = linregress(p, y).slope if len(np.unique(p)) > 1 else np.nan
    blk(d["y"].to_numpy(), d["p"].to_numpy())
    per = []
    for _, g in d.groupby("Env"):
        if len(g) < 3 or g["p"].nunique() < 2 or g["y"].nunique() < 2:
            continue
        t = {}; y, p = g["y"].to_numpy(), g["p"].to_numpy(); e = p - y
        t["RMSE"] = np.sqrt(np.mean(e ** 2)); t["MAE"] = np.mean(np.abs(e))
        t["normalized_RMSE"] = t["RMSE"] / np.std(y); t["relative_RMSE"] = t["RMSE"] / np.mean(y)
        t["normalized_MAE"] = t["MAE"] / np.std(y); t["realative_MAE"] = t["MAE"] / np.mean(y)
        t["r2_score"] = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
        r = pearsonr(y, p)[0]; t["pearson_r"] = r; t["r2_pearson"] = r ** 2
        t["spearman_r"] = spearmanr(y, p)[0]; t["lineRegressSlope"] = linregress(p, y).slope
        per.append(t)
    Pf = pd.DataFrame(per)
    for c in Pf.columns:
        out["mean_" + c] = Pf[c].mean()
    return out


def repo_monotone(p, rng, tie_safe=True):
    """The monotone map the manuscript's Section 2.4 test uses, reproduced from
    analysis/g2f_leaderboard/code/paired_invariance.py: within-environment ranks
    pushed through a piecewise-linear spline on seven sorted random knots."""
    from scipy.stats import rankdata
    n = len(p)
    u = (rankdata(p, method="average") - 0.5) / n if tie_safe \
        else (np.argsort(np.argsort(p)) + 0.5) / n
    knots = np.sort(rng.uniform(0, 1, 6)); knots = np.r_[0.0, knots, 1.0]
    vals = np.sort(rng.uniform(0, 1, len(knots))); vals[0], vals[-1] = 0.0, 1.0
    return np.interp(u, knots, vals)


def topk_set(p, k):
    """Top-k with an exogenous tie-break (original row order), the convention any
    implementation must adopt; invariant under a genuine strictly increasing map."""
    return frozenset(np.argsort(-p, kind="mergesort")[:k].tolist())


def decision_preservation(MZ, rng, n_rep=200):
    """Is the manuscript's own decision-neutral transform actually decision-
    neutral in floating point? A near-flat segment of the spline collapses
    distinct predictions into a tie, which moves the k-th boundary."""
    rows = []
    for (m, e), g in MZ.groupby(["method", "Env"], sort=False):
        p = g["p"].to_numpy(float); n = len(p)
        if n < 20:
            continue
        k = max(1, int(round(FRAC * n)))
        s0 = topk_set(p, k)
        u0 = len(np.unique(p))
        for tie_safe in (True, False):
            ch = 0; created = 0; broken = 0
            for _ in range(n_rep):
                q = repo_monotone(p, rng, tie_safe)
                if topk_set(q, k) != s0:
                    ch += 1
                u1 = len(np.unique(q))
                created += max(0, u0 - u1)
                broken += max(0, u1 - u0)
            rows.append(dict(method=m, Env=e, n=n, k=k, tie_safe=tie_safe,
                             n_distinct=u0, p_decision_changed=ch / n_rep,
                             mean_values_collapsed=created / n_rep,
                             mean_ties_broken=broken / n_rep))
    return pd.DataFrame(rows)


def monotone_map(rng, lo, hi, n_knots=7):
    """A strictly increasing piecewise-linear map of [lo, hi] onto a random
    increasing knot sequence; strictly increasing on the whole support."""
    xs = np.linspace(lo, hi, n_knots)
    ys = np.sort(rng.uniform(0, 1, n_knots))
    while np.any(np.diff(ys) <= 0):
        ys = np.sort(rng.uniform(0, 1, n_knots))
    return lambda v: np.interp(v, xs, ys)


def transform_class_test(d, rng, n_rep=120):
    """Median relative change of each metric under four transformation classes:
    monotone/affine x (per-environment map | one common map)."""
    base = metrics22(d)
    envs = d["Env"].unique()
    lo, hi = d["p"].min(), d["p"].max()
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    acc = {c: {k: [] for k in base} for c in
           ("monotone_per_env", "monotone_common", "affine_per_env", "affine_common")}
    for _ in range(n_rep):
        # monotone, one map per environment
        d2 = d.copy()
        maps = {e: monotone_map(rng, lo, hi) for e in envs}
        d2["p"] = [maps[e](v) for e, v in zip(d["Env"].to_numpy(), d["p"].to_numpy())]
        acc["monotone_per_env"] = _accum(acc["monotone_per_env"], base, metrics22(d2))
        # monotone, one common map
        g = monotone_map(rng, lo, hi)
        d3 = d.copy(); d3["p"] = g(d["p"].to_numpy())
        acc["monotone_common"] = _accum(acc["monotone_common"], base, metrics22(d3))
        # affine, one pair per environment
        a = pd.Series(rng.uniform(.5, 2.0, len(envs)), index=envs)
        b = pd.Series(rng.normal(0, 2.0, len(envs)), index=envs)
        d4 = d.copy()
        d4["p"] = d["Env"].map(a).to_numpy() * d["p"].to_numpy() + d["Env"].map(b).to_numpy()
        acc["affine_per_env"] = _accum(acc["affine_per_env"], base, metrics22(d4))
        # affine, one common pair
        a0, b0 = rng.uniform(.5, 2.0), rng.normal(0, 2.0)
        d5 = d.copy(); d5["p"] = a0 * d["p"].to_numpy() + b0
        acc["affine_common"] = _accum(acc["affine_common"], base, metrics22(d5))
    return acc


def _accum(store, base, pert):
    for k in base:
        if np.isfinite(base[k]) and np.isfinite(pert[k]):
            store[k].append(abs(pert[k] - base[k]) / max(abs(base[k]), 1e-9))
    return store


# --------------------------------------------------- 2  RANK INTERVALS ------
LOWER = {"MAE", "realative_MAE", "normalized_MAE", "RMSE", "relative_RMSE", "normalized_RMSE",
         "mean_MAE", "mean_realative_MAE", "mean_normalized_MAE", "mean_RMSE",
         "mean_relative_RMSE", "mean_normalized_RMSE"}
SLOPE = {"lineRegressSlope", "mean_lineRegressSlope"}
CORFAM = ["pearson_r", "spearman_r", "r2_pearson", "mean_pearson_r", "mean_spearman_r",
          "mean_r2_pearson"]


def rank_matrix():
    df = pd.read_csv(f"{RES}/S1_best_per_team.csv")
    df = df[df["Team Name"].notna()].copy()
    teams = df["Team Name"].to_numpy(); M = len(teams)
    mets = [c for c in df.columns if c not in ("id", "Team Name", "Submitted At")]

    def ranks(col):
        v = df[col].astype(float).to_numpy()
        s = -np.abs(v - 1.0) if col in SLOPE else (-v if col in LOWER else v)
        order = np.argsort(-s, kind="mergesort")
        r = np.empty(M, int); r[order] = np.arange(1, M + 1)
        return r
    return pd.DataFrame({m: ranks(m) for m in mets}, index=teams), mets, M


def conformal(R, cols, M, alpha=0.05):
    """Exactly the estimator of rank_intervals.py."""
    Rc = R[cols].to_numpy()
    rhat = np.median(Rc, axis=1)
    s = np.abs(Rc - rhat[:, None]).ravel()
    n = len(s); k = int(np.ceil((n + 1) * (1 - alpha)))
    q = np.sort(s)[min(k, n) - 1]
    lo = np.clip(np.round(rhat - q), 1, M).astype(int)
    hi = np.clip(np.round(rhat + q), 1, M).astype(int)
    return rhat, q, lo, hi, hi - lo + 1


if __name__ == "__main__":
    rng = np.random.default_rng(20260910)
    print("=" * 78)
    print("CLAIM 1a — exact within-environment ties, and ties straddling the k=10% cut")
    print("=" * 78)
    MZ = maize_long()
    tabs = [tie_audit(MZ, "maize (5 verified submissions)", 20)]
    for lab, f, mn in scope.kept([("rice", "rice_panel_wide.csv", 25), ("wheat", "wheat_panel_wide.csv", 25),
                       ("common bean", "bean_panel_wide.csv", 25),
                       ("spring wheat", "ursn_panel_wide.csv", 12),
                       ("soybean", "nust_panel_wide.csv", 25)]):
        P = pd.read_csv(f"{XC}/{f}", usecols=lambda c: c in ("Env", "k", "y", "p", "method"))
        tabs.append(tie_audit(P, lab, mn))
    TI = pd.concat(tabs, ignore_index=True)
    summ = (TI.groupby("dataset")
              .agg(cases=("n", "size"),
                   cells=("n", "sum"),
                   tied_cells=("n_tied_cells", "sum"),
                   cases_with_any_tie=("n_tie_groups", lambda x: int((x > 0).sum())),
                   cases_with_boundary_straddle=("boundary_straddle", "sum"))
              .reset_index())
    summ["pct_cells_tied"] = 100 * summ.tied_cells / summ.cells
    summ["pct_cases_any_tie"] = 100 * summ.cases_with_any_tie / summ.cases
    summ["pct_cases_boundary_straddle"] = 100 * summ.cases_with_boundary_straddle / summ.cases
    summ.insert(1, "scope", "dataset")
    # the five verified maize submissions individually: are the ties one team's
    # artefact or a property of the competition data?
    mz = TI[TI.dataset.str.startswith("maize")]
    per = (mz.groupby("method")
             .agg(cases=("n", "size"), cells=("n", "sum"), tied_cells=("n_tied_cells", "sum"),
                  cases_with_any_tie=("n_tie_groups", lambda x: int((x > 0).sum())),
                  cases_with_boundary_straddle=("boundary_straddle", "sum"))
             .reset_index().rename(columns={"method": "dataset"}))
    per.insert(1, "scope", "maize submission")
    per["pct_cells_tied"] = 100 * per.tied_cells / per.cells
    per["pct_cases_any_tie"] = 100 * per.cases_with_any_tie / per.cases
    per["pct_cases_boundary_straddle"] = 100 * per.cases_with_boundary_straddle / per.cases
    summ = pd.concat([summ, per], ignore_index=True)
    summ.to_csv(f"{RES}/ties_at_boundary.csv", index=False)
    print(summ.to_string(index=False))
    # how ambiguous is the top-k set when a tie does straddle it?
    st = TI[TI.boundary_straddle]
    if len(st):
        print(f"\nwhen a tie straddles the k-th place: median {st.n_tied_at_boundary.median():.0f} "
              f"genotypes competing for a median of {st.slots_remaining.median():.0f} remaining "
              f"slots (max {int(st.n_tied_at_boundary.max())} for "
              f"{int(st.loc[st.n_tied_at_boundary.idxmax(), 'slots_remaining'])} slots)")
        print(st.sort_values("n_tied_at_boundary", ascending=False)
                .head(6)[["dataset", "method", "Env", "n", "k", "n_tied_at_boundary",
                          "slots_remaining"]].to_string(index=False))

    print("\n" + "-" * 78)
    print("Is the manuscript's own monotone transform decision-neutral in floating point?")
    DP = decision_preservation(MZ, rng)
    DP.to_csv(f"{RES}/decision_preservation.csv", index=False)
    for ts in (True, False):
        s = DP[DP.tie_safe == ts]
        print(f"  tie convention {'average ranks (tie-safe)' if ts else 'ordinal ranks':<26} "
              f"P(top-10% set changes) = {s.p_decision_changed.mean():.4f}  "
              f"(cases with any change: {int((s.p_decision_changed>0).sum())}/{len(s)}; "
              f"max {s.p_decision_changed.max():.3f}); "
              f"mean distinct values collapsed per replicate "
              f"{s.mean_values_collapsed.mean():.1f}")

    print("\n" + "=" * 78)
    print("CLAIM 1b — per-environment vs one common transformation class")
    print("=" * 78)
    rows = []
    for name in VERIFIED:
        d = MZ[MZ.method == name][["Env", "y", "p"]].copy()
        acc = transform_class_test(d, rng, n_rep=60)
        for cls, store in acc.items():
            for met, vals in store.items():
                if vals:
                    rows.append(dict(submission=name, tclass=cls, metric=met,
                                     median_rel_change=float(np.median(vals))))
    TC = pd.DataFrame(rows)
    TCm = (TC.groupby(["tclass", "metric"])["median_rel_change"].median().reset_index())
    TCm["invariant"] = TCm.median_rel_change < 1e-3
    TCm.to_csv(f"{RES}/transform_class.csv", index=False)
    for cls in ("monotone_per_env", "monotone_common", "affine_per_env", "affine_common"):
        sub = TCm[TCm.tclass == cls]
        inv = sorted(sub[sub.invariant].metric)
        print(f"{cls:<20} admissible {len(inv):>2}/22 : {', '.join(inv) if inv else '(none)'}")

    print("\n" + "=" * 78)
    print("CLAIM 2 — what the conformal rank interval actually is")
    print("=" * 78)
    R, METS, M = rank_matrix()
    rhat, q, lo, hi, w = conformal(R, METS, M)
    print(f"q = {q:.1f}; nominal unclipped width = 2q+1 = {2*q+1:.0f} places on a {M}-team board")
    print(f"teams whose interval is clipped at 1 or {M}: {int(np.sum((rhat-q<1)|(rhat+q>M)))}/{M}")
    print(f"distinct interval widths across the {M} teams: {sorted(set(w.tolist()))}")
    print(f"median width {np.median(w):.0f} ({np.median(w)/M*100:.0f}% of board)")

    # 2b leave-one-metric-out: how much does q move?
    qs, cov = [], []
    for held in METS:
        cal = [m for m in METS if m != held]
        rh, qq, *_ = conformal(R, cal, M)
        qs.append(qq)
        truth = R[held].to_numpy()
        cov.append(np.mean((truth >= rh - qq) & (truth <= rh + qq)))
    qs = np.array(qs); cov = np.array(cov)
    print(f"\nleave-one-metric-out: q ranges {qs.min():.1f}-{qs.max():.1f} "
          f"(full-panel q = {q:.1f}); mean coverage {cov.mean()*100:.1f}%")

    # 2c leave-one-FAMILY-out — the actual exchangeability test
    ERRFAM = [m for m in METS if m in LOWER or m in ("r2_score", "mean_r2_score")]
    SLOPEFAM = [m for m in METS if m in SLOPE]
    fam_rows = []
    for calname, cal, tstname, tst in [
            ("error-magnitude (14)", ERRFAM, "correlation (6)", CORFAM),
            ("correlation (6)", CORFAM, "error-magnitude (14)", ERRFAM),
            ("error-magnitude (14)", ERRFAM, "slope (2)", SLOPEFAM),
            ("all but slope (20)", ERRFAM + CORFAM, "slope (2)", SLOPEFAM)]:
        rh, qq, *_ = conformal(R, cal, M)
        c = np.mean([np.mean((R[t].to_numpy() >= rh - qq) & (R[t].to_numpy() <= rh + qq))
                     for t in tst])
        fam_rows.append(dict(calibrate_on=calname, predict=tstname, q=float(qq),
                             nominal_width=float(2 * qq + 1), coverage=float(c)))
        print(f"calibrate on {calname:<22} -> predict {tstname:<20} "
              f"q={qq:>4.1f}  coverage {c*100:>5.1f}%  (nominal 95%)")

    # 2d plain descriptive alternative, with a metric-clustered bootstrap
    Rm = R.to_numpy()
    obs_range = Rm.max(1) - Rm.min(1) + 1
    iqr = np.percentile(Rm, 75, axis=1) - np.percentile(Rm, 25, axis=1) + 1
    B = 2000
    med_rng, med_iqr, med_conf = [], [], []
    for _ in range(B):
        idx = rng.integers(0, len(METS), len(METS))
        Rb = Rm[:, idx]
        med_rng.append(np.median(Rb.max(1) - Rb.min(1) + 1))
        med_iqr.append(np.median(np.percentile(Rb, 75, axis=1) - np.percentile(Rb, 25, axis=1) + 1))
        cols = [METS[i] for i in idx]
        med_conf.append(np.median(conformal(R, cols, M)[4]))
    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)
    print(f"\nDESCRIPTIVE ALTERNATIVE over the census of 22 official metrics")
    print(f"  median observed rank range (max-min+1) = {np.median(obs_range):.0f} places "
          f"({np.median(obs_range)/M*100:.0f}% of board); metric-clustered 95% CI "
          f"[{ci(med_rng)[0]:.0f}, {ci(med_rng)[1]:.0f}]")
    print(f"  median interquartile rank range        = {np.median(iqr):.1f} places "
          f"({np.median(iqr)/M*100:.0f}% of board); metric-clustered 95% CI "
          f"[{ci(med_iqr)[0]:.1f}, {ci(med_iqr)[1]:.1f}]")
    print(f"  median conformal width                 = {np.median(w):.0f} places; "
          f"metric-clustered 95% CI [{ci(med_conf)[0]:.0f}, {ci(med_conf)[1]:.0f}]")
    inside = (Rm >= lo[:, None]) & (Rm <= hi[:, None])
    print(f"  observed (team, metric) rank cells falling OUTSIDE the team's own clipped "
          f"conformal interval: {int((~inside).sum())}/{inside.size} "
          f"({(~inside).mean()*100:.1f} %); outside the team's observed range: 0 by definition")
    winner = R.index[int(np.argmin(rhat))]
    wi = list(R.index).index(winner)
    print(f"  team with the best median rank ({winner}): conformal [{lo[wi]}, {hi[wi]}], "
          f"observed rank range [{Rm[wi].min()}, {Rm[wi].max()}]")

    out = pd.DataFrame(fam_rows)
    out2 = pd.DataFrame(dict(
        quantity=["conformal_q", "conformal_nominal_width", "conformal_median_width",
                  "observed_range_median", "observed_iqr_median",
                  "loo_metric_coverage_mean", "loo_metric_q_min", "loo_metric_q_max"],
        value=[q, 2 * q + 1, float(np.median(w)), float(np.median(obs_range)),
               float(np.median(iqr)), float(cov.mean()), float(qs.min()), float(qs.max())],
        ci_lo=[np.nan, np.nan, ci(med_conf)[0], ci(med_rng)[0], ci(med_iqr)[0],
               np.nan, np.nan, np.nan],
        ci_hi=[np.nan, np.nan, ci(med_conf)[1], ci(med_rng)[1], ci(med_iqr)[1],
               np.nan, np.nan, np.nan]))
    pd.concat([out.assign(block="leave_one_family_out"),
               out2.assign(block="width_summary")], ignore_index=True) \
      .to_csv(f"{RES}/rank_interval_transfer.csv", index=False)
    print(f"\nwrote {RES}/ties_at_boundary.csv, {RES}/transform_class.csv, "
          f"{RES}/rank_interval_transfer.csv")
