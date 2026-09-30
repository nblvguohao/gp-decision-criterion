"""Exact decomposition of pooled Pearson r and pooled MSE into an environment-mean part and a
within-environment part, applied to this paper's predictions.

Proposition (stated and proved in Supplementary Section S20 of the paper): write every prediction as yhat_ij = m_j + t d_ij, m_j the prediction mean
of environment j and d_ij its within-environment deviation (no assumption: this is notation). With
cell weights w_j = n_j / N,
    a   = sum_j w_j (m_j - mbar)(ybar_j - ybar)      v_b = sum_j w_j (m_j - mbar)^2
    u   = sqrt(sum_j w_j var_j(yhat))                k   = sum_j w_j cov_j(yhat, y) / u
    S   = var(y)
then  pooled r = (a + k u) / sqrt((v_b + u^2) S),  maximised over the scale u at u* = k v_b / a,
      r_max^2 = (a^2 / v_b + k^2) / S,  and  phi = (a^2 / v_b) / (a^2 / v_b + k^2)
is the share of r_max^2 carried by the environment means; pooled MSE = sum_j w_j (m_j - ybar_j)^2
+ u^2 - 2 k u + sum_j w_j var_j(y), whose first term does not depend on the within-environment order.

For every method: the identity error, pooled r, mean within-environment Pearson r, a, v_b, u, k, u*,
phi, and the share of pooled MSE carried by the environment-mean term. Also runs a
simulation check of the identities (500 random designs, seed 20260930). Output (new files):
analysis/supplementary/results/pooled_decomposition.csv, pooled_decomposition_simcheck.txt.
"""
import os, sys
import numpy as np, pandas as pd

GPDC = os.environ.get("GPDC", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "pooled_decomposition.csv")
SIM = os.path.join(RES, "pooled_decomposition_simcheck.txt")


def decompose(env, y, p):
    envs, e = np.unique(env, return_inverse=True)
    n = np.bincount(e).astype(float); w = n / n.sum()
    ybar_j = np.bincount(e, y) / n; m = np.bincount(e, p) / n
    ybar = (w * ybar_j).sum(); mbar = (w * m).sum()
    a = (w * (m - mbar) * (ybar_j - ybar)).sum(); v_b = (w * (m - mbar) ** 2).sum()
    dev = p - m[e]
    u2 = (w * np.bincount(e, dev * dev) / n).sum(); u = np.sqrt(u2)
    k = (w * np.bincount(e, dev * (y - ybar_j[e])) / n).sum() / u
    S = y.var()
    r_form = (a + k * u) / np.sqrt((v_b + u2) * S)
    r_num = np.corrcoef(p, y)[0, 1]
    mse = np.mean((p - y) ** 2); env_term = (w * (m - ybar_j) ** 2).sum()
    rw = []
    for j in range(len(envs)):
        s = e == j
        if s.sum() > 2 and p[s].std() > 0 and y[s].std() > 0:
            rw.append(np.corrcoef(p[s], y[s])[0, 1])
    return dict(n_env=len(envs), cells=len(y), pooled_r=r_num, identity_error=abs(r_num - r_form),
                within_r=float(np.mean(rw)), a=a, v_b=v_b, u=u, k=k, S=S,
                u_star=k * v_b / a if a > 0 else np.nan,
                phi=(a * a / v_b) / (a * a / v_b + k * k) if a > 0 and v_b > 0 else np.nan,
                mse_env_share=env_term / mse)


def data():
    G = f"{GPDC}/analysis/g2f_leaderboard/results"
    obs = pd.read_csv(f"{G}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
    for s in ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
              "EnBiSys_sub243568", "NicheSquad_sub244985"]:
        pr = pd.read_csv(f"{G}/{s}.csv").rename(columns={"Yield_Mg_ha": "p"})
        d = obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])
        yield "maize (5 verified submissions)", s, d
    X = f"{GPDC}/analysis/crosscrop/results"
    for lab, f, mn in (("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12),
                       ("soybean", "nust_panel_wide.csv.gz", 25)):
        P = pd.read_csv(f"{X}/{f}")
        nk = P.groupby(["method", "Env"]).y.transform("size"); P = P[nk >= mn]
        for s in sorted(m for m in P.method.unique() if "__" not in m):
            yield lab, s, P[P.method == s]


def simcheck():
    """Simulation check of the identities over random designs."""
    rng = np.random.default_rng(20260930); worst = dict(r=0.0, ustar=0.0, rmax=0.0, umse=0.0); cases = 0
    for _ in range(500):
        J = int(rng.integers(3, 40)); n = rng.integers(3, 300, J); env = np.repeat(np.arange(J), n)
        mu = rng.normal(0, rng.uniform(0.5, 5), J); sig = rng.uniform(0.3, 3, J); g = rng.normal(size=len(env))
        y = mu[env] + sig[env] * g
        m = 0.6 * mu + rng.normal(0, rng.uniform(0.1, 3), J)
        d = rng.uniform(-0.2, 1.0) * g + rng.normal(size=len(env)) * rng.uniform(0.2, 2)
        d = d * rng.uniform(0.2, 3, J)[env]; d = d - (np.bincount(env, d) / n)[env]
        w = n / n.sum(); ybar_j = np.bincount(env, y) / n; mbar = (w * m).sum()
        a = (w * (m - mbar) * (ybar_j - (w * ybar_j).sum())).sum(); v_b = (w * (m - mbar) ** 2).sum()
        q = (w * np.bincount(env, d * d) / n).sum(); k = (w * np.bincount(env, d * (y - ybar_j[env])) / n).sum() / np.sqrt(q)
        S = y.var()
        if a <= 0 or k <= 0: continue
        cases += 1
        ts = np.linspace(0, 6 * k * v_b / a / np.sqrt(q), 4001)[1:]
        r_num = np.array([np.corrcoef(m[env] + t * d, y)[0, 1] for t in ts]); u = ts * np.sqrt(q)
        r_form = (a + k * u) / np.sqrt((v_b + u ** 2) * S)
        worst["r"] = max(worst["r"], np.abs(r_num - r_form).max())
        us = k * v_b / a; worst["ustar"] = max(worst["ustar"], abs(u[np.argmax(r_num)] - us) / us - (u[1] - u[0]) / us)
        worst["rmax"] = max(worst["rmax"], abs(r_num.max() - np.sqrt((a * a / v_b + k * k) / S)))
        mse = np.array([((m[env] + t * d - y) ** 2).mean() for t in ts])
        worst["umse"] = max(worst["umse"], abs(u[np.argmin(mse)] - k) - (u[1] - u[0]))
    return (f"cases with a > 0 and k > 0: {cases} of 500\n"
            f"max |r_numeric - r_formula|: {worst['r']:.2e}\n"
            f"max relative error of u* beyond one grid step: {max(worst['ustar'], 0):.2e}\n"
            f"max |max r_numeric - r_max|: {worst['rmax']:.2e}\n"
            f"max |argmin MSE - k| beyond one grid step: {max(worst['umse'], 0):.2e}\n")


def main():
    if os.path.exists(OUT) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{os.path.relpath(OUT, GPDC)} exists; new results go to new files")
    rows = [dict(dataset=lab, method=s, **decompose(d.Env.to_numpy(), d.y.to_numpy(float), d.p.to_numpy(float)))
            for lab, s, d in data()]
    T = pd.DataFrame(rows); T.to_csv(OUT, index=False)
    pd.set_option("display.width", 220)
    print(T.groupby("dataset")[["pooled_r", "within_r", "phi", "mse_env_share", "identity_error"]]
          .agg(["min", "median", "max"]).round(4).to_string())
    txt = simcheck(); open(SIM, "w").write(txt); print(txt)
    print(f"-> {os.path.relpath(OUT, GPDC)}\n-> {os.path.relpath(SIM, GPDC)}")


if __name__ == "__main__":
    main()
