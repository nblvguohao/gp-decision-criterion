"""Does the outcome test favour the correlation family by construction? (Supplementary Section S8)

QUESTION.  The outcome the scoring rules are judged against — the realised
selection differential of a within-environment top-k pick — is itself a
within-environment rank functional.  It therefore belongs to the same family as
the metrics the manuscript declares admissible, so the correlation family was
guaranteed to win and the outcome test would be a tautology.

TEST.  Change the decision to one the manuscript's own scope conditions say is
NOT invariant to within-environment monotone transformation, holding the
scoring rules, the panels and the split-half protocol fixed.  If the winning
rule follows the decision, the criterion tracks the decision rather than
favouring rank metrics unconditionally.  If the correlation family still wins,
the objection stands.

Three decisions, identical machinery:
  D1  within-environment top-10 %  (the manuscript's decision; rank functional)
  D2  absolute-threshold advancement: advance every genotype-environment cell
      whose PREDICTED value clears a fixed agronomic target tau (the 90th
      percentile of observed values in that dataset), and book the realised
      surplus sum(y - tau) over advanced cells per candidate evaluated.  This
      is a cardinal decision: a within-environment monotone transform of the
      predictions changes which cells clear tau, so it lies outside the
      manuscript's scope conditions by construction.
  D3  pooled top-10 % across environments by predicted value (cross-environment
      selection, the first alternative the manuscript names in Section 2.1).

Scoring rules are unchanged from analysis/crosscrop/code/does_it_help.py.
Intervals are environment-cluster bootstraps (D1, D2); D3 is reported as a
point estimate with the across-split range only, because a pooled top-k quota
is not separable over environments and cannot be resampled the same way.

Writes analysis/crosscrop/results/decision_swap.csv
       analysis/crosscrop/results/decision_swap_d3.csv
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr

FRAC = 0.10
RES = "analysis/g2f_leaderboard/results"
XC = "analysis/crosscrop/results"
VERIFIED = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
            "EnBiSys_sub243568", "NicheSquad_sub244985"]
RULES = [("mean_RMSE", "neg_RMSE"), ("mean_MAE", "neg_MAE"), ("mean_r2_score", "r2_score"),
         ("mean_pearson_r", "pearson"), ("mean_spearman_r", "spearman"),
         ("mean_NDCG@10%", "NDCG")]
ERRFAM = {"mean_RMSE", "mean_MAE", "mean_r2_score"}
CORFAM = {"mean_pearson_r", "mean_spearman_r", "mean_NDCG@10%"}


def ndcg(y, p, k):
    o = np.argsort(-p)[:k]; g = y[o] - y.min()
    dcg = np.sum(g / np.log2(np.arange(2, len(o) + 2)))
    b = np.sort(y)[::-1][:k] - y.min()
    idcg = np.sum(b / np.log2(np.arange(2, len(b) + 2)))
    return dcg / idcg if idcg > 0 else np.nan


def maize_long():
    obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
    out = []
    for n in VERIFIED:
        pr = pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha": "p"})
        d = obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"]).rename(
            columns={"Hybrid": "k"})
        d["method"] = n
        out.append(d[["Env", "k", "y", "p", "method"]])
    return pd.concat(out, ignore_index=True)


def build(P, min_n, tau):
    """Per (method, environment): the six scoring statistics and the ingredients
    of the three decision outcomes."""
    meths = sorted(P.method.unique())
    envs = sorted(P.Env.unique())
    mi = {m: i for i, m in enumerate(meths)}; ei = {e: i for i, e in enumerate(envs)}
    keys = ["neg_RMSE", "neg_MAE", "r2_score", "pearson", "spearman", "NDCG"]
    S = {k: np.full((len(meths), len(envs)), np.nan) for k in keys}
    G1 = np.full((len(meths), len(envs)), np.nan)      # D1 within-env gain
    SURP = np.zeros((len(meths), len(envs)))           # D2 sum(y - tau) over advanced
    NC = np.zeros(len(envs))                           # cells per environment
    for (m, e), g in P.groupby(["method", "Env"], sort=False):
        if len(g) < min_n or g.p.nunique() < 2 or g.y.nunique() < 2:
            continue
        i, j = mi[m], ei[e]
        y = g.y.to_numpy(float); p = g.p.to_numpy(float)
        k = max(1, int(round(FRAC * len(g)))); t = np.argsort(-p)[:k]; err = p - y
        S["neg_RMSE"][i, j] = -np.sqrt(np.mean(err ** 2))
        S["neg_MAE"][i, j] = -np.mean(np.abs(err))
        S["r2_score"][i, j] = 1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)
        S["pearson"][i, j] = pearsonr(y, p)[0]
        S["spearman"][i, j] = spearmanr(y, p)[0]
        S["NDCG"][i, j] = ndcg(y, p, k)
        G1[i, j] = (y[t].mean() - y.mean()) / y.std()
        sel = p >= tau
        SURP[i, j] = np.sum(y[sel] - tau) if sel.any() else 0.0
        NC[j] = len(g)
    ok = ~np.isnan(G1).all(0)                       # environments that survived
    return meths, np.array(envs)[ok], {k: v[:, ok] for k, v in S.items()}, \
        G1[:, ok], SURP[:, ok], NC[ok]


def recover(S, OUT, idxA, idxB, denomB=None):
    """Rank methods on half A by each rule; report the held-out outcome of the
    method each rule picks, normalised to (rule - average)/(oracle - average)."""
    if denomB is None:
        outB = np.nanmean(OUT[:, idxB], axis=1)
    else:
        outB = np.nansum(OUT[:, idxB], axis=1) / max(denomB, 1e-12)
    fin = np.isfinite(outB)
    if fin.sum() < 3:
        return None
    orc = np.nanmax(outB[fin]); rnd = np.nanmean(outB[fin])
    if not np.isfinite(orc - rnd) or orc - rnd <= 0:
        return None
    res = {}
    for lab, key in RULES:
        sA = np.nanmean(S[key][:, idxA], axis=1)
        sA = np.where(np.isfinite(sA) & fin, sA, -np.inf)
        res[lab] = (scope.tied_best_mean(sA, outB) - rnd) / (orc - rnd)
    return res


def reliability(OUT, envs_idx, rng, denom=None, NC=None, B=200):
    rs = []
    for _ in range(B):
        e = rng.permutation(envs_idx); A, Bh = e[:len(e) // 2], e[len(e) // 2:]
        if denom is None:
            a = np.nanmean(OUT[:, A], axis=1); b = np.nanmean(OUT[:, Bh], axis=1)
        else:
            a = np.nansum(OUT[:, A], axis=1) / NC[A].sum()
            b = np.nansum(OUT[:, Bh], axis=1) / NC[Bh].sum()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() > 5:
            rs.append(spearmanr(a[ok], b[ok])[0])
    return float(np.mean(rs))


def run_decision(S, OUT, NC, rng, use_denom, n_splits=300, n_boot=200, boot_splits=25):
    nE = OUT.shape[1]; idx = np.arange(nE)
    point = {lab: [] for lab, _ in RULES}
    for _ in range(n_splits):
        e = rng.permutation(idx); A, Bh = e[:nE // 2], e[nE // 2:]
        r = recover(S, OUT, A, Bh, NC[Bh].sum() if use_denom else None)
        if r:
            for lab in point:
                point[lab].append(r[lab])
    boot = {lab: [] for lab, _ in RULES}
    for _ in range(n_boot):
        bidx = rng.integers(0, nE, nE)
        Sb = {k: v[:, bidx] for k, v in S.items()}
        Ob = OUT[:, bidx]; NCb = NC[bidx]
        acc = {lab: [] for lab, _ in RULES}
        uniq = np.unique(bidx)
        for _ in range(boot_splits):
            # split the DISTINCT resampled environments, so no environment's copies
            # land in both halves (an earlier version split positions and leaked)
            u = rng.permutation(uniq); inA = np.isin(bidx, u[:len(u) // 2])
            A, Bh = np.flatnonzero(inA), np.flatnonzero(~inA)
            r = recover(Sb, Ob, A, Bh, NCb[Bh].sum() if use_denom else None)
            if r:
                for lab in acc:
                    acc[lab].append(r[lab])
        for lab in boot:
            if acc[lab]:
                boot[lab].append(np.mean(acc[lab]))
    # paired contrasts: the families' own intervals overlap because both are wide, so the
    # quantity that carries the claim is the difference within a replicate. Both the Tier II
    # rule (mean_pearson_r) and the Tier I rule (mean_spearman_r) are contrasted with the
    # error-magnitude rule, so that the admissible metric is tested and not only its family.
    def contrast(rule):
        pair = [boot[rule][i] - boot["mean_RMSE"][i]
                for i in range(min(len(boot[rule]), len(boot["mean_RMSE"])))]
        # point estimate: difference of the two rules' split-averaged recoveries (the
        # quantities printed in the table); interval: percentiles of the bootstrap differences
        pt_diff = (np.mean(point[rule]) - np.mean(point["mean_RMSE"])
                   if point[rule] and point["mean_RMSE"] else np.nan)
        return (float(pt_diff), float(np.percentile(pair, 2.5)),
                float(np.percentile(pair, 97.5))) if len(pair) > 10 else (np.nan,) * 3
    return ({lab: float(np.mean(v)) if v else np.nan for lab, v in point.items()},
            {lab: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
             if len(v) > 10 else (np.nan, np.nan) for lab, v in boot.items()},
            contrast("mean_pearson_r"), contrast("mean_spearman_r"))


def run_d3(P, min_n, rng, n_splits=200):
    """Pooled top-10 % across environments; not separable, so computed directly."""
    meths = sorted(P.method.unique())
    cache = {}
    for m in meths:
        d = P[P.method == m]
        d = d.groupby("Env").filter(lambda g: len(g) >= min_n)
        cache[m] = (d.Env.to_numpy(), d.y.to_numpy(float), d.p.to_numpy(float))
    envs = np.array(sorted(set(np.concatenate([v[0] for v in cache.values()]))))
    keys = ["neg_RMSE", "neg_MAE", "r2_score", "pearson", "spearman", "NDCG"]
    S = {k: np.full((len(meths), len(envs)), np.nan) for k in keys}
    ei = {e: j for j, e in enumerate(envs)}
    for i, m in enumerate(meths):
        ev, y, p = cache[m]
        for e in np.unique(ev):
            s = ev == e
            if s.sum() < min_n or len(np.unique(p[s])) < 2:
                continue
            j = ei[e]; yy, pp = y[s], p[s]; err = pp - yy
            k = max(1, int(round(FRAC * s.sum())))
            S["neg_RMSE"][i, j] = -np.sqrt(np.mean(err ** 2))
            S["neg_MAE"][i, j] = -np.mean(np.abs(err))
            S["r2_score"][i, j] = 1 - np.sum(err ** 2) / np.sum((yy - yy.mean()) ** 2)
            S["pearson"][i, j] = pearsonr(yy, pp)[0]
            S["spearman"][i, j] = spearmanr(yy, pp)[0]
            S["NDCG"][i, j] = ndcg(yy, pp, k)
    acc = {lab: [] for lab, _ in RULES}
    nE = len(envs)
    for _ in range(n_splits):
        perm = rng.permutation(nE); A, Bh = perm[:nE // 2], perm[nE // 2:]
        setB = set(envs[Bh])
        outB = np.full(len(meths), np.nan)
        for i, m in enumerate(meths):
            ev, y, p = cache[m]
            msk = np.isin(ev, list(setB))
            if msk.sum() < 20:
                continue
            yy, pp = y[msk], p[msk]
            k = max(1, int(round(FRAC * len(yy))))
            top = np.argsort(-pp)[:k]
            outB[i] = (yy[top].mean() - yy.mean()) / yy.std()
        fin = np.isfinite(outB)
        if fin.sum() < 3:
            continue
        orc = outB[fin].max(); rnd = outB[fin].mean()
        if orc - rnd <= 0:
            continue
        for lab, key in RULES:
            sA = np.nanmean(S[key][:, A], axis=1)
            sA = np.where(np.isfinite(sA) & fin, sA, -np.inf)
            acc[lab].append((scope.tied_best_mean(sA, outB) - rnd) / (orc - rnd))
    return {lab: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if v else (np.nan, np.nan, np.nan) for lab, v in acc.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(20260910)
    SETS = [("maize (5 verified)", None, 20), ("rice", "rice_panel_wide.csv", 25),
            ("common bean", "bean_panel_wide.csv", 25),
            ("spring wheat", "ursn_panel_wide.csv", 12),
            ("soybean", "nust_panel_wide.csv", 25)]
    rows, d3rows = [], []
    for lab, f, mn in scope.kept(SETS):
        P = maize_long() if f is None else pd.read_csv(
            f"{XC}/{f}", usecols=lambda c: c in ("Env", "k", "y", "p", "method"))
        tau = float(np.percentile(P.groupby(["Env", "k"]).y.first().to_numpy(), 90))
        meths, envs, S, G1, SURP, NC = build(P, mn, tau)
        nE = len(envs)
        rel1 = reliability(G1, np.arange(nE), rng)
        rel2 = reliability(SURP, np.arange(nE), rng, denom=True, NC=NC)
        print(f"\n### {lab}: {len(meths)} methods, {nE} environments, tau = {tau:.3f}")
        print(f"    outcome reliability  D1 within-env top-10% = {rel1:.3f}   "
              f"D2 absolute-threshold surplus = {rel2:.3f}")
        for dname, OUT, den in [("D1_within_env_topk", G1, False),
                                ("D2_absolute_threshold", SURP, True)]:
            pt, bt, pd_, sp_ = run_decision(S, OUT, NC, rng, den)
            best = max(pt, key=lambda k: pt[k] if np.isfinite(pt[k]) else -np.inf)
            print(f"    {dname}: best rule = {best}; paired contrasts vs mean_RMSE: "
                  f"mean_pearson_r {pd_[0]*100:+.1f} pp [{pd_[1]*100:+.1f}, {pd_[2]*100:+.1f}], "
                  f"mean_spearman_r {sp_[0]*100:+.1f} pp [{sp_[1]*100:+.1f}, {sp_[2]*100:+.1f}]")
            for rule, _ in RULES:
                lo, hi = bt[rule]
                print(f"        {rule:<18}{pt[rule]*100:>7.1f}%   "
                      f"[{lo*100:>6.1f}, {hi*100:>6.1f}]")
                rows.append(dict(dataset=lab, decision=dname, rule=rule,
                                 family="error" if rule in ERRFAM else "correlation",
                                 recovery=pt[rule], ci_lo=lo, ci_hi=hi,
                                 outcome_reliability=rel1 if den is False else rel2,
                                 n_methods=len(meths), n_env=nE, tau=tau,
                                 paired_pearson_minus_rmse=pd_[0],
                                 paired_ci_lo=pd_[1], paired_ci_hi=pd_[2],
                                 paired_spearman_minus_rmse=sp_[0],
                                 paired_spearman_ci_lo=sp_[1], paired_spearman_ci_hi=sp_[2]))
        d3 = run_d3(P, mn, rng)
        best3 = max(d3, key=lambda k: d3[k][0] if np.isfinite(d3[k][0]) else -np.inf)
        print(f"    D3_pooled_topk (point estimate, across-split range): best rule = {best3}")
        for rule, _ in RULES:
            m, lo, hi = d3[rule]
            print(f"        {rule:<18}{m*100:>7.1f}%   [{lo*100:>6.1f}, {hi*100:>6.1f}]")
            d3rows.append(dict(dataset=lab, decision="D3_pooled_topk", rule=rule,
                               family="error" if rule in ERRFAM else "correlation",
                               recovery=m, split_lo=lo, split_hi=hi, n_env=nE))
    pd.DataFrame(rows).to_csv(f"{XC}/decision_swap.csv", index=False)
    pd.DataFrame(d3rows).to_csv(f"{XC}/decision_swap_d3.csv", index=False)
    print(f"\nwrote {XC}/decision_swap.csv and {XC}/decision_swap_d3.csv")
