"""The resolution threshold against the tests that compare two correlations (Meng, Rosenthal and Rubin
1992, used in this journal by Fernandes et al. 2024).

Question: when a test declares two methods' accuracies different, does the more accurate method also
select better material? The threshold dr* answers the second question directly; a correlation test
answers the first and assumes independent cells.

For every pair of base methods in each constructed panel (the pairs of the threshold analysis:
f = 0.10, at least 8 shared environments, ties on accuracy or outcome dropped; threshold_model.py):
  dr, dg   mean differences in within-environment Pearson r and in realised selection differential
  agree    the more accurate method also has the larger realised differential
  three tests of dr, two-sided at 0.05:
    meng_pooled    Meng's z on all shared cells after centring y and both predictions within each
                   environment (cells treated as independent, N = cells), as a pooled analysis would
    meng_stouffer  Meng's z within each environment, combined by Stouffer's method (cells independent
                   within environments, environments independent)
    env_t          t test of the per-environment differences in r over environments (environments as
                   the units; no assumption about cells)
Reported per panel and test: pairs declared different, the share of them in which the more accurate
method selects better, and the share with dr at or above the panel's threshold (threshold_estimates.csv).
Also the smallest dr a Meng test declares at N cells under the Gaussian model, against 3/sqrt(N).

Descriptive; no seed needed (no resampling). Output (new files): analysis/supplementary/results/meng_vs_threshold.csv,
meng_vs_threshold_pairs.csv, meng_design.csv.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import norm, t as tdist
GPDC = os.environ.get("GPDC", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(os.path.dirname(HERE), "results")
OUT, OUTP, OUTD = (os.path.join(RES, f) for f in ("meng_vs_threshold.csv", "meng_vs_threshold_pairs.csv", "meng_design.csv"))
os.chdir(GPDC); sys.path.insert(0, "analysis/crosscrop/code")
import scope
X = "analysis/crosscrop/results"
SETS = [("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12), ("soybean", "nust_panel_wide.csv.gz", 25)]
FRAC, MIN_ENV = 0.10, 8


def meng_z(r1, r2, r12, n):
    """Meng, Rosenthal & Rubin (1992): z for H0 rho1 = rho2, two correlations sharing y."""
    rbar2 = (r1 ** 2 + r2 ** 2) / 2
    f = min(1.0, (1 - r12) / (2 * (1 - rbar2)))
    h = (1 - f * rbar2) / (1 - rbar2)
    return (np.arctanh(r1) - np.arctanh(r2)) * np.sqrt((n - 3) / (2 * (1 - r12) * h))


def per_env(P, mn):
    out = {}
    for (m, e), g in P.groupby(["method", "Env"]):
        if len(g) < mn or g.p.nunique() < 2 or g.y.nunique() < 2:
            continue
        g = g.sort_values("k"); y, p = g.y.to_numpy(float), g.p.to_numpy(float)
        k = max(1, int(round(FRAC * len(g)))); top = np.argsort(-p, kind="mergesort")[:k]
        out[(m, e)] = dict(k=g.k.to_numpy(), y=y, p=p, r=np.corrcoef(y, p)[0, 1], g=(y[top].mean() - y.mean()) / y.std())
    return out


def main():
    if os.path.exists(OUT) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{os.path.relpath(OUT, GPDC)} exists; new results go to new files")
    TE = pd.read_csv(f"{X}/threshold_estimates.csv").query("methods == 'parent methods' and frac == 0.10").set_index("dataset")
    rows, prs = [], []
    for lab, f, mn in SETS:
        P = pd.read_csv(f"{X}/{f}"); P = P[~P.method.str.contains("__")]
        E = per_env(P, mn); meths = sorted(P.method.unique())
        for a in range(len(meths)):
            for b in range(a + 1, len(meths)):
                envs = sorted({e for (m, e) in E if m == meths[a]} & {e for (m, e) in E if m == meths[b]})
                if len(envs) < MIN_ENV:
                    continue
                d_r = np.array([E[(meths[b], e)]["r"] - E[(meths[a], e)]["r"] for e in envs])
                d_g = np.array([E[(meths[b], e)]["g"] - E[(meths[a], e)]["g"] for e in envs])
                dr, dg = d_r.mean(), d_g.mean()
                if abs(dr) < scope.ACC_TIE or abs(dg) < scope.OUT_TIE:
                    continue
                s = 1 if dr > 0 else -1; hi, lo = (meths[b], meths[a]) if s > 0 else (meths[a], meths[b])
                zs, Y, P1, P2 = [], [], [], []
                for e in envs:
                    A, B = E[(hi, e)], E[(lo, e)]
                    assert (A["k"] == B["k"]).all()
                    r12 = np.corrcoef(A["p"], B["p"])[0, 1]; n = len(A["y"])
                    zs.append(meng_z(A["r"], B["r"], min(r12, 0.9999), n))
                    Y.append(A["y"] - A["y"].mean()); P1.append(A["p"] - A["p"].mean()); P2.append(B["p"] - B["p"].mean())
                Y, P1, P2 = map(np.concatenate, (Y, P1, P2))
                r1, r2, r12 = np.corrcoef(Y, P1)[0, 1], np.corrcoef(Y, P2)[0, 1], np.corrcoef(P1, P2)[0, 1]
                z_pool = meng_z(r1, r2, min(r12, 0.9999), len(Y))
                z_st = np.sum(zs) / np.sqrt(len(zs))
                t_env = (s * d_r).mean() / ((s * d_r).std(ddof=1) / np.sqrt(len(envs)))
                prs.append(dict(dataset=lab, hi=hi, lo=lo, n_env=len(envs), cells=len(Y), dr=abs(dr), dg=s * dg, agree=s * dg > 0,
                                p_meng_pooled=2 * norm.sf(abs(z_pool)), p_meng_stouffer=2 * norm.sf(abs(z_st)),
                                p_env_t=2 * tdist.sf(abs(t_env), len(envs) - 1)))
        D = pd.DataFrame([p for p in prs if p["dataset"] == lab]); thr = TE.loc[lab, "threshold"]
        for test in ("meng_pooled", "meng_stouffer", "env_t"):
            sig = D[D[f"p_{test}"] < 0.05]
            rows.append(dict(dataset=lab, test=test, pairs=len(D), declared=len(sig), share_declared=len(sig) / len(D),
                             agree_if_declared=sig.agree.mean() if len(sig) else np.nan,
                             beyond_threshold_if_declared=(sig.dr >= thr).mean() if len(sig) else np.nan,
                             agree_if_not_declared=D[D[f"p_{test}"] >= 0.05].agree.mean(), threshold=thr))
        above = D[D.dr >= thr]
        rows.append(dict(dataset=lab, test="dr >= threshold", pairs=len(D), declared=len(above), share_declared=len(above) / len(D),
                         agree_if_declared=above.agree.mean() if len(above) else np.nan, beyond_threshold_if_declared=1.0 if len(above) else np.nan,
                         agree_if_not_declared=D[D.dr < thr].agree.mean(), threshold=thr))
    T = pd.DataFrame(rows); T.to_csv(OUT, index=False); pd.DataFrame(prs).to_csv(OUTP, index=False)
    # design: smallest dr declared by a pooled Meng test at N cells, mean accuracy 0.3, prediction correlation r12
    des = []
    for N in (500, 1000, 2000, 4000, 8000, 16000):
        for r12 in (0.5, 0.8, 0.95):
            lo_, hi_ = 0.0, 0.5
            for _ in range(60):
                mid = (lo_ + hi_) / 2
                if abs(meng_z(0.3 + mid / 2, 0.3 - mid / 2, r12, N)) >= norm.isf(0.025): hi_ = mid
                else: lo_ = mid
            des.append(dict(cells=N, r12=r12, meng_min_dr=hi_, three_over_sqrtN=3 / np.sqrt(N)))
    pd.DataFrame(des).to_csv(OUTD, index=False)
    pd.set_option("display.width", 200)
    print(T.round(3).to_string(index=False)); print(pd.DataFrame(des).round(4).to_string(index=False))
    print(f"-> {os.path.relpath(OUT, GPDC)}\n-> {os.path.relpath(OUTP, GPDC)}\n-> {os.path.relpath(OUTD, GPDC)}")


if __name__ == "__main__":
    main()
