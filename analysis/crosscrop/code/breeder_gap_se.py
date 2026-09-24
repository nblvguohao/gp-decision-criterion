"""Section 3.6, first paragraph: the breeder's-equation gap with its standard errors.

The gaps against the textbook benchmark i_k * r are computed exactly as in
`monotone_test_panels.py` CHECK A (same environments, same k, same standardisation);
this script adds what CHECK A never computed: the standard error of each
submission's mean gap across environments, the t statistic and the two-sided P
value (t distribution, df = environments - 1). The environment is the
replication unit, so each submission contributes one gap per environment.

The same gap on the soybean panel (64 methods, environments with at least 25
genotypes, f = 0.10), averaged over methods within each environment and then
over environments, with a 2,000-draw bootstrap over environments.

Writes analysis/crosscrop/results/breeder_gap_se.csv
       analysis/crosscrop/results/breeder_gap_soybean.csv
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, norm, t as tdist

RES = "analysis/g2f_leaderboard/results"
OUT = "analysis/crosscrop/results/breeder_gap_se.csv"
V = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
     "EnBiSys_sub243568", "NicheSquad_sub244985"]

obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
def load(name):
    pr = pd.read_csv(f"{RES}/{name}.csv").rename(columns={"Yield_Mg_ha": "p"})
    return obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])

def i_inf(f):
    x = norm.ppf(1 - f)
    return norm.pdf(x) / f

rows = []
for name in V:
    d = load(name)
    for f in (0.05, 0.10, 0.20):
        gaps, observed = [], []
        for _, g in d.groupby("Env"):
            if len(g) < 20 or g.p.nunique() < 2:
                continue
            y, p = g.y.to_numpy(), g.p.to_numpy()
            n = len(g); k = max(1, int(round(f * n)))
            top = np.argsort(-p)[:k]
            r = pearsonr(y, p)[0]
            o = (y[top].mean() - y.mean()) / y.std()
            observed.append(o)
            gaps.append(o - i_inf(k / n) * r)
        gaps = np.array(gaps); m = len(gaps)
        se = gaps.std(ddof=1) / np.sqrt(m)
        tt = gaps.mean() / se
        rows.append(dict(team=name, frac=f, n_env=m, mean_observed=np.mean(observed),
                         gap=gaps.mean(), se=se, t=tt,
                         p_two_sided=2 * tdist.sf(abs(tt), m - 1)))

R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
main = R[R.frac.isin([0.10, 0.20])]
pd.set_option("display.width", 160)
print(R.round(4).to_string(index=False))
print(f"\nten comparisons (f = 0.10, 0.20): mean gap {main.gap.mean():+.4f}; "
      f"|t| {main.t.abs().min():.2f}-{main.t.abs().max():.2f}; "
      f"se {main.se.min():.4f}-{main.se.max():.4f}; "
      f"P<0.05 in {(main.p_two_sided < 0.05).sum()}/10, P<=0.001 in {(main.p_two_sided <= 0.001).sum()}/10")
print(f"negative mean observed differential: "
      f"{R[R.mean_observed < 0][['team', 'frac', 'mean_observed']].to_dict('records')}")

# ---------------------------------------------------------------- method panels
# The same gap on each constructed panel, averaged over methods within each environment
# and then over environments, with a 2,000-draw bootstrap over environments; for all
# methods and for parent methods only, at f = 0.10 and 0.20.
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope

# Finite-population benchmark. i_inf(k/n) * r is the large-n expectation; with a median of
# 16 lines per spring-wheat environment and k = 2 it overstates what a Gaussian predictor
# delivers. The benchmark below is the expectation of the very statistic computed here,
# (mean y of the top k by p - mean y) / sd(y), for (y, p) bivariate normal with correlation r
# at the environment's own n and k, by simulation on a grid of r and linear interpolation.
R_GRID = np.linspace(-1, 1, 41)
_BENCH = {}
def bench(n, k, reps=20000):
    if (n, k) not in _BENCH:
        g = np.random.default_rng(10_000 * n + k)
        z1 = g.standard_normal((reps, n)); z2 = g.standard_normal((reps, n))
        top = np.argsort(-z1, axis=1)[:, :k]            # selection depends on p = z1 only
        ev = []
        for r in R_GRID:
            y = r * z1 + np.sqrt(max(0.0, 1 - r * r)) * z2
            o = (np.take_along_axis(y, top, axis=1).mean(1) - y.mean(1)) / y.std(1)
            ev.append(o.mean())
        _BENCH[(n, k)] = np.array(ev)
    return _BENCH[(n, k)]

def panel_gap(P, min_n, f):
    rows = []
    for (e, m), g in P.groupby(["Env", "method"]):
        if len(g) < min_n or g.p.nunique() < 2 or g.y.nunique() < 2:
            continue
        y, p = g.y.to_numpy(), g.p.to_numpy()
        n = len(g); k = max(1, int(round(f * n)))
        top = np.argsort(-p)[:k]
        r = pearsonr(y, p)[0]
        o = (y[top].mean() - y.mean()) / y.std()
        rows.append((e, m, o - i_inf(k / n) * r, o - np.interp(r, R_GRID, bench(n, k))))
    return pd.DataFrame(rows, columns=["Env", "method", "gap", "gap_fin"])

PANELS = [("common bean", "analysis/crosscrop/results/bean_panel_wide.csv", 25),
          ("spring wheat", "analysis/crosscrop/results/ursn_panel_wide.csv", 12),
          ("soybean", "analysis/crosscrop/results/nust_panel_wide.csv", 25)]
out = []
for lab, path, mn in scope.kept(PANELS):
    P = pd.read_csv(path)
    for subset, PP in (("all methods", P), ("parent methods", P[~P.method.str.contains("__")])):
        for f in (0.10, 0.20):
            G = panel_gap(PP, mn, f)
            ge = G.groupby("Env").gap.mean().to_numpy()
            rng = np.random.default_rng(0)
            boot = rng.choice(ge, size=(2000, len(ge)), replace=True).mean(axis=1)
            gf = G.groupby("Env").gap_fin.mean().to_numpy()
            bootf = np.random.default_rng(1).choice(gf, size=(2000, len(gf)), replace=True).mean(axis=1)
            out.append(dict(dataset=lab, methods=subset, frac=f, n_env=len(ge), n_methods=G.method.nunique(),
                            gap=ge.mean(), ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5),
                            gap_fin=gf.mean(), ci_lo_fin=np.percentile(bootf, 2.5),
                            ci_hi_fin=np.percentile(bootf, 97.5)))
PG = pd.DataFrame(out)
PG.to_csv("analysis/crosscrop/results/breeder_gap_panels.csv", index=False)
PG[(PG.dataset == "soybean") & (PG.methods == "all methods") & (PG.frac == 0.10)].drop(columns="methods").to_csv(
    "analysis/crosscrop/results/breeder_gap_soybean.csv", index=False)
print("\nmethod panels:\n" + PG.round(4).to_string(index=False))
