"""Selection on genotype means across environments (Supplementary Section S15).

Multi-environment programmes often advance genotypes on their mean performance over a
trial network rather than within each trial. This is the decision "D4":

  D4  across-environment selection. Each genotype's prediction and observation are
      centred within every environment (removing environment main effects, as a
      two-way model would) and averaged over the environments of one half of the
      network in which it was tested at least twice; the top 10 % of genotypes by
      predicted mean are advanced, and the outcome is the mean observed (environment-
      standardised) value of the advanced genotypes minus that of all eligible ones.

By Section 2.1 the metrics admissible for D4 depend on the predictions only through the
ranking of those genotype means; none of the 22 official metrics has that form. The
outcome test of decision_swap.py is repeated for D4 with the same split-half and
environment-bootstrap protocol, adding two rules computed on genotype means: their rank
correlation (the D4 analogue of Tier I) and their Pearson correlation.

Its own random stream, so the D1-D3 results in decision_swap.csv are untouched.

Writes analysis/crosscrop/results/decision_genotype_means.csv
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code")
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import rankdata, spearmanr
import scope
import decision_swap as ds

FRAC, MIN_REPS = 0.10, 2
XC = "analysis/crosscrop/results"
RULES = ds.RULES + [("genotype-mean spearman", "gm_spearman"), ("genotype-mean pearson", "gm_pearson")]
FAMILY = {**{r: "error" for r in ds.ERRFAM}, **{r: "correlation" for r in ds.CORFAM},
          "genotype-mean spearman": "genotype mean", "genotype-mean pearson": "genotype mean"}


def matrices(P, meths, envs):
    """Genotype x environment arrays: centred predictions per method, standardised observations."""
    P = P[P.Env.isin(envs)].copy()
    gen = np.array(sorted(P.k.unique())); gi = {g: i for i, g in enumerate(gen)}
    ei = {e: j for j, e in enumerate(envs)}; mi = {m: i for i, m in enumerate(meths)}
    P["gi"] = P.k.map(gi); P["ej"] = P.Env.map(ei); P["mi"] = P.method.map(mi)
    P["pc"] = P.p - P.groupby(["method", "Env"]).p.transform("mean")
    ref = P[P.method == meths[0]]
    yz = (ref.y - ref.groupby("Env").y.transform("mean")) / ref.groupby("Env").y.transform(lambda v: v.std(ddof=0))
    G, E = len(gen), len(envs)
    Y = np.zeros((G, E)); M = np.zeros((G, E))
    Y[ref.gi, ref.ej] = yz.to_numpy(); M[ref.gi, ref.ej] = 1.0
    PC = np.zeros((len(meths), G, E), dtype=np.float64)
    PC[P.mi.to_numpy(), P.gi.to_numpy(), P.ej.to_numpy()] = P.pc.to_numpy()
    return PC, Y, M


def gmeans(PC, Y, M, w):
    cnt = M @ w
    elig = cnt >= MIN_REPS
    ybar = np.where(elig, (Y @ w) / np.maximum(cnt, 1e-12), np.nan)
    pbar = (PC @ w) / np.maximum(cnt, 1e-12)            # methods x genotypes
    return elig, ybar, pbar


def outcome(PC, Y, M, w):
    elig, ybar, pbar = gmeans(PC, Y, M, w)
    n = int(elig.sum())
    if n < 20:
        return None
    k = max(1, int(round(FRAC * n)))
    pb = np.where(elig[None, :], pbar, -np.inf)
    top = np.argpartition(-pb, k - 1, axis=1)[:, :k]
    return np.nanmean(ybar[top], axis=1) - np.nanmean(ybar[elig])


def gm_scores(PC, Y, M, w):
    elig, ybar, pbar = gmeans(PC, Y, M, w)
    if elig.sum() < 20:
        return np.full(PC.shape[0], np.nan), np.full(PC.shape[0], np.nan)
    yb = ybar[elig]; pb = pbar[:, elig]
    def corr(a, B):
        a = a - a.mean(); B = B - B.mean(axis=1, keepdims=True)
        return (B @ a) / np.sqrt((B ** 2).sum(1) * (a ** 2).sum())
    return corr(rankdata(yb), rankdata(pb, axis=1)), corr(yb, pb)


def evaluate(S, PC, Y, M, envidx, A, B):
    """Recovery of each rule: rank methods on half A, outcome on half B."""
    E = Y.shape[1]
    wA = np.bincount(envidx[A], minlength=E).astype(float)
    wB = np.bincount(envidx[B], minlength=E).astype(float)
    outB = outcome(PC, Y, M, wB)
    if outB is None:
        return None
    fin = np.isfinite(outB)
    orc, rnd = outB[fin].max(), outB[fin].mean()
    if orc - rnd <= 0:
        return None
    gs, gp = gm_scores(PC, Y, M, wA)
    res = {}
    for lab, key in RULES:
        if key == "gm_spearman":
            sA = gs
        elif key == "gm_pearson":
            sA = gp
        else:
            sA = np.nanmean(S[key][:, envidx[A]], axis=1)
        sA = np.where(np.isfinite(sA) & fin, sA, -np.inf)
        res[lab] = (scope.tied_best_mean(sA, outB) - rnd) / (orc - rnd)
    return res


def reliability(PC, Y, M, rng, B=200):
    E = Y.shape[1]; rs = []
    for _ in range(B):
        e = rng.permutation(E); a, b = e[:E // 2], e[E // 2:]
        oa = outcome(PC, Y, M, np.bincount(a, minlength=E).astype(float))
        ob = outcome(PC, Y, M, np.bincount(b, minlength=E).astype(float))
        if oa is not None and ob is not None:
            rs.append(spearmanr(oa, ob)[0])
    return float(np.mean(rs))


def run(S, PC, Y, M, rng, n_splits=300, n_boot=200, boot_splits=25):
    E = Y.shape[1]; idx = np.arange(E)
    point = {lab: [] for lab, _ in RULES}
    for _ in range(n_splits):
        e = rng.permutation(idx)
        r = evaluate(S, PC, Y, M, idx, e[:E // 2], e[E // 2:])
        if r:
            for lab in point:
                if np.isfinite(r[lab]): point[lab].append(r[lab])
    boot = {lab: [] for lab, _ in RULES}
    for _ in range(n_boot):
        bidx = rng.integers(0, E, E); uniq = np.unique(bidx); acc = {lab: [] for lab, _ in RULES}
        for _ in range(boot_splits):
            u = rng.permutation(uniq); inA = np.isin(bidx, u[:len(u) // 2])
            r = evaluate(S, PC, Y, M, bidx, np.flatnonzero(inA), np.flatnonzero(~inA))
            if r:
                # a rule that cannot score on a tiny resampled half (no genotype seen twice)
                # returns NaN for that split; it is skipped rather than allowed to poison the mean
                for lab in acc:
                    if np.isfinite(r[lab]): acc[lab].append(r[lab])
        for lab in boot:
            boot[lab].append(np.mean(acc[lab]) if acc[lab] else np.nan)
    pt = {lab: float(np.mean(v)) if v else np.nan for lab, v in point.items()}
    fin = {lab: [x for x in v if np.isfinite(x)] for lab, v in boot.items()}
    ci = {lab: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10
          else (np.nan, np.nan) for lab, v in fin.items()}
    def contrast(a, b):
        d = [x - y for x, y in zip(boot[a], boot[b]) if np.isfinite(x) and np.isfinite(y)]
        n = len(d)
        return (pt[a] - pt[b], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))) \
            if n > 10 else (np.nan,) * 3
    con = {"gm_spearman_minus_rmse": contrast("genotype-mean spearman", "mean_RMSE"),
           "spearman_minus_rmse": contrast("mean_spearman_r", "mean_RMSE"),
           "pearson_minus_rmse": contrast("mean_pearson_r", "mean_RMSE"),
           "spearman_minus_gm_spearman": contrast("mean_spearman_r", "genotype-mean spearman")}
    return pt, ci, con


if __name__ == "__main__":
    rng = np.random.default_rng(20260924)
    SETS = [("maize (5 verified)", None, 20), ("rice", "rice_panel_wide.csv", 25),
            ("common bean", "bean_panel_wide.csv", 25),
            ("spring wheat", "ursn_panel_wide.csv", 12),
            ("soybean", "nust_panel_wide.csv", 25)]
    rows = []
    for lab, f, mn in scope.kept(SETS):
        P = ds.maize_long() if f is None else pd.read_csv(
            f"{XC}/{f}", usecols=lambda c: c in ("Env", "k", "y", "p", "method"))
        meths, envs, S, G1, SURP, NC = ds.build(P, mn, np.inf)
        PC, Y, M = matrices(P, meths, list(envs))
        rel = reliability(PC, Y, M, rng)
        if len(meths) < 10:   # five maize submissions: too few methods for a reliability, as in the main text
            rel = np.nan
        pt, ci, con = run(S, PC, Y, M, rng)
        n_g = int((M.sum(1) >= 2 * MIN_REPS).sum())
        print(f"\n### {lab}: {len(meths)} methods, {len(envs)} environments, "
              f"{M.shape[0]} genotypes; D4 outcome reliability = {rel:.3f}")
        for rule, _ in RULES:
            print(f"    {rule:<24}{pt[rule]*100:>7.1f}%   [{ci[rule][0]*100:>6.1f}, {ci[rule][1]*100:>6.1f}]")
        for k, (m, lo, hi) in con.items():
            print(f"    {k:<28}{m*100:+7.1f} pp [{lo*100:+.1f}, {hi*100:+.1f}]")
        for rule, _ in RULES:
            rows.append(dict(dataset=lab, decision="D4_genotype_mean_topk", rule=rule, family=FAMILY[rule],
                             recovery=pt[rule], ci_lo=ci[rule][0], ci_hi=ci[rule][1],
                             outcome_reliability=rel, n_methods=len(meths), n_env=len(envs),
                             n_genotypes=M.shape[0],
                             **{f"{k}{s}": v for k, t in con.items() for s, v in zip(("", "_lo", "_hi"), t)}))
    pd.DataFrame(rows).to_csv(f"{XC}/decision_genotype_means.csv", index=False)
    print(f"\nwrote {XC}/decision_genotype_means.csv")
