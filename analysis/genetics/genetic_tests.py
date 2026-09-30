"""Tests of the analysis plan analysis_plan.md, s.6.

Inputs: analysis/genetics/results/genetic_params.csv (this directory) and analysis/atlas/atlas.csv.
Output: analysis/genetics/results/genetic_tests.csv (refuses to overwrite).

Primary: for each predictor P in (h2_g, s_GE, P3_between_env_r) against O1 = reversal, the within-
dataset rank correlation rho_w: ranks of outcome and predictor within each dataset, centred on the
dataset mean, then the Pearson correlation of the centred ranks over all combinations. Two-sided p
from 10,000 permutations of the predictor within datasets (seed 20260929); Holm over the three.
Secondary (no correction): O2 = number tied with the leader, O3 = recovery_corr - recovery_rmse
(reliability >= 0.5 only); the within-dataset rank regression O1 ~ P + C1 (coefficient of P, same
permutation scheme on P); pooled Spearman ignoring datasets (descriptive only).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "results", "genetic_tests.csv")
B, SEED = 10_000, 20260929
PRED = ["h2_g", "s_GE", "P3_between_env_r"]


def centred_ranks(v, groups):
    out = np.empty(len(v))
    for g in np.unique(groups):
        m = groups == g
        r = rankdata(v[m]); out[m] = r - r.mean()
    return out


def rho_w(x, y, groups):
    a, b = centred_ranks(x, groups), centred_ranks(y, groups)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def beta_w(x, c, y, groups):
    X = np.column_stack([centred_ranks(x, groups), centred_ranks(c, groups)])
    return float(np.linalg.lstsq(X, centred_ranks(y, groups), rcond=None)[0][0])


def perm_p(stat, x, groups, rng, *args):
    obs = stat(x, *args)
    null = np.empty(B)
    for b in range(B):
        xp = x.copy()
        for g in np.unique(groups):
            m = np.flatnonzero(groups == g); xp[m] = xp[rng.permutation(m)]
        null[b] = stat(xp, *args)
    return obs, float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (B + 1))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, j in enumerate(o):
        run = max(run, min(1, (len(p) - i) * p[j])); adj[j] = run
    return adj


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; new results go to new files")
    G = pd.read_csv(os.path.join(HERE, "results", "genetic_params.csv"))
    A = pd.read_csv(os.path.join(REPO, "analysis/atlas/atlas.csv"))
    A["O2_tied"] = A.tied.fillna("").map(lambda s: len([t for t in s.split(",") if t.strip()]))
    A["O3_corr_minus_rmse"] = np.where(A.reliability >= 0.5, A.recovery_corr - A.recovery_rmse, np.nan)
    D = G.merge(A[["id", "reversal", "O2_tied", "O3_corr_minus_rmse", "reliability"]], on="id", how="inner")
    assert len(D) == len(G) == 52, (len(D), len(G))
    rng = np.random.default_rng(SEED)
    rows = []
    for outcome, role in (("reversal", "primary"), ("O2_tied", "secondary"), ("O3_corr_minus_rmse", "secondary")):
        for p in PRED:
            d = D.dropna(subset=[outcome, p])
            grp = d.dataset.to_numpy()
            obs, pv = perm_p(lambda x, y, g: rho_w(x, y, g), d[p].to_numpy(float), grp, rng, d[outcome].to_numpy(float), grp)
            rows.append(dict(outcome=outcome, predictor=p, role=role, model="rho_w", n=len(d),
                             n_datasets=d.dataset.nunique(), estimate=obs, p_perm=pv,
                             pooled_spearman=spearmanr(d[p], d[outcome])[0]))
    T = pd.DataFrame(rows)
    prim = T.role == "primary"
    T.loc[prim, "p_holm"] = holm(T.loc[prim, "p_perm"].to_numpy())
    extra = []
    for p in PRED:                                   # O1 ~ P + C1, within-dataset ranks
        d = D.dropna(subset=["reversal", p, "C1_calib_sd"]); grp = d.dataset.to_numpy()
        obs, pv = perm_p(lambda x, c, y, g: beta_w(x, c, y, g), d[p].to_numpy(float), grp, rng,
                         d.C1_calib_sd.to_numpy(float), d.reversal.to_numpy(float), grp)
        extra.append(dict(outcome="reversal", predictor=p, role="secondary", model="rank_beta_adj_C1", n=len(d),
                          n_datasets=d.dataset.nunique(), estimate=obs, p_perm=pv))
    d = D.dropna(subset=["reversal", "C1_calib_sd"]); grp = d.dataset.to_numpy()
    obs, pv = perm_p(lambda x, y, g: rho_w(x, y, g), d.C1_calib_sd.to_numpy(float), grp, rng, d.reversal.to_numpy(float), grp)
    extra.append(dict(outcome="reversal", predictor="C1_calib_sd", role="secondary", model="rho_w", n=len(d),
                      n_datasets=d.dataset.nunique(), estimate=obs, p_perm=pv, pooled_spearman=spearmanr(d.C1_calib_sd, d.reversal)[0]))
    T = pd.concat([T, pd.DataFrame(extra)], ignore_index=True)
    T.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nper dataset (descriptive):")
    print(D.groupby("dataset")[["h2_g", "s_GE", "P3_between_env_r", "C1_calib_sd", "reversal"]].median().round(3).to_string())
    print(f"-> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
