"""The admissibility partition of the 22 official metrics, in every dataset.

Three properties per metric, each measured on the dataset's own predictions:
  monotone  invariant to an independent random strictly increasing map within each
            environment (average ranks, so exactly tied predictions stay tied; a
            piecewise-linear map through six sorted random interior knots)
  affine    invariant to yhat_e -> a_e * yhat_e + b_e, a_e ~ U(0.5, 2), b_e ~ N(0, SD(y_e)^2)
  order     order-sensitive: the metric changes when every environment's predictions are
            reversed (yhat_e -> -yhat_e). A metric that scores a ranking and its reverse
            alike (the squared correlations) cannot tell a good selection from the worst
            one, whatever its invariance.
A property holds when the median relative change is below 1e-3 (invariance) or at least
1e-3 (order sensitivity).
  Tier I    monotone-invariant and order-sensitive
  Tier II   affine-invariant and order-sensitive, not Tier I
  Tier III  everything else (with order-insensitive metrics flagged separately)

Maize uses the five verified G2F submissions; each constructed panel uses its parent
methods (the miscalibration variants are affine copies and add nothing here).

Also records, per dataset, where the method ranked first by mean_r2_pearson stands on
mean_pearson_r (an order-insensitive metric can crown the most reversed ranking).

Writes analysis/crosscrop/results/partition_metrics.csv
       analysis/crosscrop/results/partition_by_dataset.csv
"""
import sys, warnings, os, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
import scope
from analyse_species import panel22, METRICS
from scipy.stats import rankdata

TOL = 1e-3
REPS = 20
RES = "analysis/crosscrop/results"
G2F = "analysis/g2f_leaderboard/results"
VERIFIED = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
            "EnBiSys_sub243568", "NicheSquad_sub244985"]


def mono(p, rng):
    u = (rankdata(p, method="average") - 0.5) / len(p)
    knots = np.r_[0, np.sort(rng.uniform(0, 1, 6)), 1]
    vals = np.sort(rng.uniform(0, 1, len(knots))); vals[0], vals[-1] = 0, 1
    return np.interp(u, knots, vals)


def perturb(d, kind, rng):
    d2 = d.copy()
    if kind == "monotone":
        d2["p"] = d.groupby("Env").p.transform(lambda s: mono(s.to_numpy(), rng))
    elif kind == "affine":
        envs = d.Env.unique()
        sd = d.groupby("Env").y.std(ddof=0).reindex(envs).to_numpy()
        a = pd.Series(rng.uniform(0.5, 2.0, len(envs)), index=envs)
        b = pd.Series(rng.normal(0, 1.0, len(envs)) * sd, index=envs)
        d2["p"] = d.Env.map(a).to_numpy() * d.p.to_numpy() + d.Env.map(b).to_numpy()
    elif kind == "reverse":
        d2["p"] = -d.p
    return d2


def rel(a, b):
    return abs(b - a) / max(abs(a), 1e-9)


R2_CHAMP = {}


def analyse(label, methods, min_n, rng):
    acc = {k: {"monotone": [], "affine": [], "reverse": []} for k in METRICS}
    bases = []
    for d in methods:
        base = panel22(d, min_n); bases.append(base)
        for kind, reps in (("monotone", REPS), ("affine", REPS), ("reverse", 1)):
            for _ in range(reps):
                m = panel22(perturb(d, kind, rng), min_n)
                for k in METRICS:
                    if np.isfinite(base.get(k, np.nan)) and np.isfinite(m.get(k, np.nan)):
                        acc[k][kind].append(rel(base[k], m[k]))
    # what an order-insensitive metric rewards: the method it ranks first, and where that
    # method stands on the within-environment Pearson correlation
    B = pd.DataFrame(bases)
    ch = int(B.mean_r2_pearson.idxmax())
    R2_CHAMP[label] = dict(r2_champion_pearson=B.mean_pearson_r[ch],
                           r2_champion_pearson_rank=int((B.mean_pearson_r > B.mean_pearson_r[ch]).sum()) + 1,
                           n_methods=len(B))
    rows = []
    for k in METRICS:
        mo, af, rv = (np.median(acc[k][x]) if acc[k][x] else np.nan for x in ("monotone", "affine", "reverse"))
        inv_m, inv_a, order = mo < TOL, af < TOL, rv >= TOL
        tier = "I" if (inv_m and order) else ("II" if (inv_a and order) else "III")
        rows.append(dict(dataset=label, metric=k, monotone_drift=mo, affine_drift=af, reverse_change=rv,
                         monotone_invariant=inv_m, affine_invariant=inv_a, order_sensitive=order, tier=tier))
    return rows


def maize_methods():
    obs = pd.read_csv(f"{G2F}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
    out = []
    for n in VERIFIED:
        pr = pd.read_csv(f"{G2F}/{n}.csv").rename(columns={"Yield_Mg_ha": "p"})
        out.append(obs.merge(pr, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])[["Env", "y", "p"]])
    return out


def panel_methods(path):
    P = pd.read_csv(path)
    P = P[~P.method.str.contains("__")]
    return [g[["Env", "y", "p"]] for _, g in P.groupby("method")]


SETS = [("maize", None, 20),
        ("common bean", f"{RES}/bean_panel_wide.csv", 25),
        ("spring wheat", f"{RES}/ursn_panel_wide.csv", 12),
        ("soybean", f"{RES}/nust_panel_wide.csv", 25),
        ("Chinese maize (CUBIC)", f"{RES}/china_panel_wide.csv", 25)]

rng = np.random.default_rng(20260923)
rows = []
for label, path, mn in scope.kept(SETS):
    if path is not None and not os.path.exists(path):
        print(f"{label}: {path} not present, skipped"); continue
    meths = maize_methods() if path is None else panel_methods(path)
    rows += analyse(label, meths, mn, rng)
    print(f"{label}: {len(meths)} methods done")
R = pd.DataFrame(rows)
R.to_csv(f"{RES}/partition_metrics.csv", index=False)
S = (R.groupby("dataset", sort=False)
       .apply(lambda g: pd.Series(dict(tier_I=int((g.tier == "I").sum()), tier_II=int((g.tier == "II").sum()),
                                        tier_III=int((g.tier == "III").sum()),
                                        order_insensitive=", ".join(g.metric[~g.order_sensitive]),
                                        max_drift_tierI_monotone=g.monotone_drift[g.tier == "I"].max(),
                                        max_drift_tierII_affine=g.affine_drift[g.tier.isin(["I", "II"])].max(),
                                        min_drift_tierIII_affine=g.affine_drift[(g.tier == "III") & g.order_sensitive].min())))
       .reset_index())
S = S.merge(pd.DataFrame(R2_CHAMP).T.rename_axis("dataset").reset_index(), on="dataset")
S.to_csv(f"{RES}/partition_by_dataset.csv", index=False)
pd.set_option("display.width", 200)
print(S.to_string(index=False))
