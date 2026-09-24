"""Can exactly tied predictions explain the shortfall below i*r?

Where several genotypes share the k-th largest predicted value, the top-k set is not
unique (Supplementary Section S10) and the realised selection differential depends on
which of the tied genotypes the sort happens to take. If that arbitrary choice drove the
shortfall of Section 3.4, resolving every tie in the selected set's favour would remove
it. This computes the realised differential of each verified G2F submission under three
tie-breaks — the one the analysis uses (numpy's sort), the best case (tied genotypes
ordered by observed value, highest first) and the worst case — and the resulting gap
against i*r, with the same environments and k as breeder_gap_se.py.

Writes analysis/crosscrop/results/tie_effect_on_gap.csv
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import norm, pearsonr

RES = "analysis/g2f_leaderboard/results"
OUT = "analysis/crosscrop/results/tie_effect_on_gap.csv"
V = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
     "EnBiSys_sub243568", "NicheSquad_sub244985"]
obs = pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})


def i_inf(f):
    x = norm.ppf(1 - f)
    return norm.pdf(x) / f


rows = []
for name in V:
    pr = pd.read_csv(f"{RES}/{name}.csv").rename(columns={"Yield_Mg_ha": "p"})
    d = obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])
    for f in (0.05, 0.10, 0.20):
        g_as, g_best, g_worst, straddle = [], [], [], 0
        for _, g in d.groupby("Env"):
            if len(g) < 20 or g.p.nunique() < 2:
                continue
            y, p = g.y.to_numpy(), g.p.to_numpy()
            n = len(g); k = max(1, int(round(f * n)))
            r = pearsonr(y, p)[0]; proj = i_inf(k / n) * r
            def diff(order):
                t = order[:k]
                return (y[t].mean() - y.mean()) / y.std()
            g_as.append(diff(np.argsort(-p)) - proj)                              # as analysed
            g_best.append(diff(np.lexsort((-y, -p))) - proj)                      # ties: best first
            g_worst.append(diff(np.lexsort((y, -p))) - proj)                      # ties: worst first
            s = np.sort(p)[::-1]
            straddle += int(k < n and s[k] == s[k - 1])
        rows.append(dict(team=name, frac=f, n_env=len(g_as), env_with_boundary_tie=straddle,
                         gap_as_analysed=np.mean(g_as), gap_best_case=np.mean(g_best),
                         gap_worst_case=np.mean(g_worst)))
R = pd.DataFrame(rows)
R.to_csv(OUT, index=False)
pd.set_option("display.width", 200)
print(R.round(4).to_string(index=False))
m = R.groupby("frac")[["gap_as_analysed", "gap_best_case", "gap_worst_case"]].mean()
print("\nmean over the five submissions:\n" + m.round(4).to_string())
print(f"\nlargest move from the tie-break, best minus as-analysed: "
      f"{(R.gap_best_case - R.gap_as_analysed).max():.4f} SD; the shortfall itself is "
      f"{-R.gap_as_analysed.min():.4f} to {-R.gap_as_analysed.max():.4f} SD")
print(f"wrote {OUT}")
