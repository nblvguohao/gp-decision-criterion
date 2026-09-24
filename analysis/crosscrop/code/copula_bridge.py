"""Does non-normality explain the breeder's-equation shortfall?

`monotone_test_panels.py` CHECK A already ruled out the finite-population selection
intensity. The remaining textbook explanation is that dz = i * r * sd assumes the
prediction and the phenotype are jointly normal within an environment, which
yield trials are not obliged to be. This separates the two ways that assumption
can fail.

For each environment, three benchmarks for the standardised selection
differential of a top-k pick, all holding n, k and the observed dependence fixed:

  N   normal margins, finite population, dependence matched on Pearson r
      i_fin(n,k) * r                                    (the paper's benchmark)
  C0  Gaussian copula, NORMAL margin for y, dependence matched on Spearman rho
      isolates the effect of matching on rank rather than on Pearson r
  C1  Gaussian copula, EMPIRICAL margin for y, dependence matched on Spearman rho
      the observed within-environment yield distribution, whatever its shape

C1 - C0 is what the shape of the yield distribution costs; observed - C1 is what
neither the margin nor a Gaussian dependence can account for, and can only be a
departure of the dependence itself from the Gaussian copula, i.e. predictions
that carry less information about the top of the distribution than their overall
correlation implies.

Writes analysis/crosscrop/results/copula_bridge.csv
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, norm

RES = "analysis/g2f_leaderboard/results"
OUT = "analysis/crosscrop/results/copula_bridge.csv"
V = ["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
     "EnBiSys_sub243568","NicheSquad_sub244985"]
B = 4000                      # copula replicates per environment
rng = np.random.default_rng(0)

obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
def load(n):
    pr = pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha":"p"})
    return obs.merge(pr, on=["Env","Hybrid"]).dropna(subset=["y","p"])

def i_fin(n, k, reps=40000):
    """E[mean of the top k of n standard normals]"""
    Z = rng.standard_normal((reps, n)); Z.sort(axis=1)
    return Z[:, -k:].mean(axis=1).mean()

def copula(y_sorted, n, k, rho_s, empirical):
    """Mean standardised differential of a top-k pick under a Gaussian copula.

    y_sorted : the environment's observed yields, sorted ascending
    rho_s    : Spearman correlation to reproduce; the Gaussian copula parameter
               is rho_z = 2 sin(pi * rho_s / 6)
    empirical: True  -> y takes the observed values, assigned by rank
               False -> y takes normal scores, assigned by rank
    """
    rho_z = 2*np.sin(np.pi*rho_s/6)
    vals = y_sorted if empirical else norm.ppf((np.arange(1, n+1)-0.375)/(n+0.25))
    mu, sd = vals.mean(), vals.std()
    z1 = rng.standard_normal((B, n))
    z2 = rho_z*z1 + np.sqrt(max(1-rho_z**2, 0.0))*rng.standard_normal((B, n))
    # rank of z2 within each replicate -> the value that rank carries
    order = np.argsort(z2, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(n)[None, :].repeat(B, 0), axis=1)
    yv = vals[ranks]
    top = np.argsort(-z1, axis=1)[:, :k]
    sel = np.take_along_axis(yv, top, axis=1).mean(axis=1)
    return ((sel - mu)/sd).mean()

rows = []
for name in V:
    d = load(name)
    for f in (0.05, 0.10, 0.20):
        rec = {"obs": [], "N": [], "C0": [], "C1": []}
        for _, g in d.groupby("Env"):
            if len(g) < 20 or g.p.nunique() < 2: continue
            y = g.y.to_numpy(); p = g.p.to_numpy()
            n = len(g); k = max(1, int(round(f*n)))
            t = np.argsort(-p)[:k]
            r  = pearsonr(y, p)[0]
            rs = spearmanr(y, p)[0]
            ys = np.sort(y)
            rec["obs"].append((y[t].mean()-y.mean())/y.std())
            rec["N"].append(i_fin(n, k)*r)
            rec["C0"].append(copula(ys, n, k, rs, empirical=False))
            rec["C1"].append(copula(ys, n, k, rs, empirical=True))
        a = {kk: np.array(v) for kk, v in rec.items()}
        ne = len(a["obs"])
        rows.append(dict(
            team=name, frac=f, n_env=ne,
            observed=a["obs"].mean(),
            bench_N=a["N"].mean(), bench_C0=a["C0"].mean(), bench_C1=a["C1"].mean(),
            gap_N=(a["obs"]-a["N"]).mean(),  se_N=(a["obs"]-a["N"]).std(ddof=1)/np.sqrt(ne),
            gap_C1=(a["obs"]-a["C1"]).mean(), se_C1=(a["obs"]-a["C1"]).std(ddof=1)/np.sqrt(ne),
            margin_effect=(a["C1"]-a["C0"]).mean()))

D = pd.DataFrame(rows)
D.to_csv(OUT, index=False)

print(f"{'submission':<26}{'f':>6}{'observed':>10}{'bench N':>9}{'bench C1':>10}"
      f"{'gap vs N':>11}{'gap vs C1':>12}{'SE':>7}")
print("-"*91)
for _, r in D.iterrows():
    print(f"{r.team:<26}{r.frac:>6.2f}{r.observed:>10.3f}{r.bench_N:>9.3f}{r.bench_C1:>10.3f}"
          f"{r.gap_N:>+11.3f}{r.gap_C1:>+12.3f}{r.se_C1:>7.3f}")
print()
for f, s in D.groupby("frac"):
    expl = (1-abs(s.gap_C1.mean())/abs(s.gap_N.mean()))*100
    print(f"  f={f:.2f}:  gap vs normal benchmark {s.gap_N.mean():+.3f} SD"
          f"   vs empirical-margin copula {s.gap_C1.mean():+.3f} SD"
          f"   ({expl:.0f} % of the gap explained by the margin and the rank matching)")
    print(f"           mean margin effect (C1 - C0) {s.margin_effect.mean():+.3f} SD;"
          f"  mean |t| of the residual gap "
          f"{np.mean(np.abs(s.gap_C1/s.se_C1)):.1f}")
print(f"\nwrote {OUT}")
