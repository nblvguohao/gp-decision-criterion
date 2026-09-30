"""Where do the evaluation metrics of recent multi-environment studies fall in the tiers of the admissibility criterion?

Metrics, as defined in the papers (formulas transcribed from the published full texts):
  MSED     Eckhoff et al. 2026 (Theor. Appl. Genet. 139:206), eq. 3: per environment
           sum_{i != i'} [(y_i - y_i') - (z_i - z_i')]^2 / (n (n - 1)) = 2 sum (f_i - fbar)^2 / (n - 1),
           f = y - z; reported rooted and averaged over environments ("rooted MSED").
  MSEPD    Tadese et al. 2024 (Theor. Appl. Genet. 137:181), eq. 5: the same pairwise differences summed over
           locations with the global denominator M I (I - 1); with unequal n we use sum_j n_j (n_j - 1).
  CI       coincidence index, Hamblin & Zimmermann 1986, as used by Fernandes et al. 2024 (eq. 7):
           (B - R) / (M - R), B = genotypes in both the predicted and the observed top fraction, R = the
           number expected by chance (M^2 / n), M = the number selected; top 20 % (Fernandes) and 10 %.
           Fernandes et al. describe the denominator term as the total number of hybrids; the variant
           (B - R) / (n - R) is also tested (CI_totalT). Computed within environment, then averaged.
  NDCG@k   Chen et al. 2024 (Theor. Appl. Genet. 137:270), eqs. 15-17: linear gain v, discount 1 / log2(i + 1), k = 20 %
           and 10 % of each environment; computed within environment, then averaged.
  and, for reference, within-environment Spearman and Pearson r and per-environment RMSE.

Tests (same definitions as the paper's Section 2.4; tolerance 1e-3 on the median relative change):
  affine    per environment p -> a_e p + b_e, a_e ~ U(0.5, 2), b_e ~ N(0, SD(y_e)^2)
  monotone  per environment a random strictly increasing piecewise-linear map of the average ranks
  reversal  per environment p -> -p
Tier I: unmoved by affine and monotone, moved by reversal. Tier II: unmoved by affine only, moved by
reversal. Tier III: otherwise.

Data: the five verified G2F submissions and the first six parent methods of the common bean, spring
wheat and soybean panels, read from this repository. Output (new file):
analysis/supplementary/results/tier_extra_metrics.csv. Seed 20260929, 40 replicates per method and transform.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata

GPDC = os.environ.get("GPDC", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "results", "tier_extra_metrics.csv")
SEED, REPS, TOL = 20260929, 40, 1e-3


def per_env_metrics(y, p):
    n = len(y); out = {}
    f = y - p
    msed = 2 * np.sum((f - f.mean()) ** 2) / (n - 1)
    out["MSED_rooted"] = np.sqrt(msed)
    out["_msepd_num"] = msed * n * (n - 1); out["_msepd_den"] = n * (n - 1)
    for fr in (0.20, 0.10):
        k = max(1, int(round(fr * n)))
        top_p = set(np.argsort(-p, kind="mergesort")[:k]); top_y = set(np.argsort(-y, kind="mergesort")[:k])
        B = len(top_p & top_y); R = k * k / n
        out[f"CI@{int(fr*100)}"] = (B - R) / (k - R)
        out[f"CI_totalT@{int(fr*100)}"] = (B - R) / (n - R)
        order = np.argsort(-p, kind="mergesort")[:k]; ideal = np.sort(y)[::-1][:k]
        disc = 1 / np.log2(np.arange(2, k + 2))
        den = np.sum(ideal * disc)
        out[f"NDCG@{int(fr*100)}"] = np.sum(y[order] * disc) / den if den != 0 else np.nan
    out["spearman"] = spearmanr(y, p)[0]; out["pearson"] = pearsonr(y, p)[0]
    out["RMSE"] = np.sqrt(np.mean(f ** 2))
    return out


def metrics(d, min_n):
    rows = []
    for _, g in d.groupby("Env"):
        if len(g) < min_n or g.p.nunique() < 2 or g.y.nunique() < 2:
            continue
        rows.append(per_env_metrics(g.y.to_numpy(float), g.p.to_numpy(float)))
    D = pd.DataFrame(rows)
    m = {c: D[c].mean() for c in D.columns if not c.startswith("_")}
    m["MSEPD"] = D["_msepd_num"].sum() / D["_msepd_den"].sum()
    return m


def mono(p, rng):
    u = (rankdata(p, method="average") - 0.5) / len(p)
    kn = np.r_[0, np.sort(rng.uniform(0, 1, 6)), 1]
    vl = np.sort(rng.uniform(0, 1, len(kn))); vl[0], vl[-1] = 0, 1
    return np.interp(u, kn, vl)


def datasets():
    G = f"{GPDC}/analysis/g2f_leaderboard/results"
    obs = pd.read_csv(f"{G}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
    mz = []
    for s in ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
              "EnBiSys_sub243568", "NicheSquad_sub244985"]:
        pr = pd.read_csv(f"{G}/{s}.csv").rename(columns={"Yield_Mg_ha": "p"})
        d = obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"]); d["method"] = s; mz.append(d)
    yield "maize (5 verified submissions)", pd.concat(mz, ignore_index=True), 20, None
    X = f"{GPDC}/analysis/crosscrop/results"
    for lab, f, mn in (("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12),
                       ("soybean", "nust_panel_wide.csv.gz", 25)):
        P = pd.read_csv(f"{X}/{f}")
        ms = sorted(m for m in P.method.unique() if "__" not in m)[:6]
        yield lab, P[P.method.isin(ms)], mn, ms


def main():
    if os.path.exists(OUT) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{os.path.relpath(OUT, GPDC)} exists; new results go to new files")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rng = np.random.default_rng(SEED); rows = []
    for lab, P, mn, _ in datasets():
        for s in P.method.unique():
            d = P[P.method == s][["Env", "y", "p"]].reset_index(drop=True)
            base = metrics(d, mn)
            acc = {}
            for kind in ("affine", "monotone", "reversal"):
                for _ in range(REPS if kind != "reversal" else 1):
                    d2 = d.copy()
                    if kind == "affine":
                        e = d.Env.unique()
                        a = pd.Series(rng.uniform(.5, 2., len(e)), index=e)
                        sd = d.groupby("Env").y.std(ddof=0).reindex(e).to_numpy()
                        b = pd.Series(rng.normal(0, 1.0, len(e)) * sd, index=e)
                        d2["p"] = d.Env.map(a).to_numpy() * d.p.to_numpy() + d.Env.map(b).to_numpy()
                    elif kind == "monotone":
                        d2["p"] = d.groupby("Env").p.transform(lambda t: mono(t.to_numpy(), rng))
                    else:
                        d2["p"] = -d.p
                    m = metrics(d2, mn)
                    for k in base:
                        if np.isfinite(base[k]) and np.isfinite(m[k]):
                            acc.setdefault((k, kind), []).append(abs(m[k] - base[k]) / max(abs(base[k]), 1e-9))
            for (k, kind), v in acc.items():
                rows.append(dict(dataset=lab, method=s, metric=k, test=kind, rel_change=float(np.median(v))))
    R = pd.DataFrame(rows)
    T = R.groupby(["dataset", "metric", "test"]).rel_change.median().unstack("test").reset_index()
    def tier(r):
        if r.reversal < TOL: return "III"
        if r.affine < TOL and r.monotone < TOL: return "I"
        if r.affine < TOL: return "II"
        return "III"
    T["tier"] = T.apply(tier, axis=1)
    T.to_csv(OUT, index=False)
    pd.set_option("display.width", 200)
    print(T.to_string(index=False, float_format=lambda v: f"{v:.2e}"))
    print("\ntier by metric across datasets:")
    print(T.groupby("metric").tier.agg(lambda s: "/".join(sorted(set(s)))).to_string())
    print(f"-> {os.path.relpath(OUT, GPDC)}")


if __name__ == "__main__":
    main()
