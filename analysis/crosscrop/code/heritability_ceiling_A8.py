"""The correlation ceiling as a function of replication.

A ceiling sqrt(H2) estimated only from replicated cells (0.796) is estimated at the
replication level of the *training* data, but the quantity it bounds is the
correlation with the *2022 test-set* observed value.  Those are the same number
only if the test value rests on the same number of plots.

What a ceiling actually is.  Write the scored value for a cell (one hybrid in
one environment) as the mean of n plots,

    ybar_n = g + ebar_n ,   Var(g) = s2g ,   Var(ebar_n) = s2e / n ,

where s2e is *everything* non-genetic that varies between plots of the same
hybrid in the same environment (block, position, harvest error).  Blocks are
not removed: plots of one hybrid sit in different replicates by design, so
between-block variation is part of what separates a plot from the genotypic
mean.  For any predictor yhat,

    corr(yhat, ybar_n) = corr(yhat, g) * sqrt(H2(n)) <= sqrt(H2(n)),
    H2(n) = s2g / (s2g + s2e / n).

Three consequences the manuscript currently collapses into one number:
  * n = 1        -- ceiling for selecting on a single unreplicated plot;
  * n = n_test   -- ceiling for a competition scored against an n-plot mean;
  * n -> infinity-- H2 -> 1, ceiling -> 1.  Correlation with the true
                    within-environment genotypic value is NOT bounded below 1
                    by phenotyping error.  The ceiling is a property of the
                    measurement the metric is computed against, not of the
                    trait.

When n varies across cells, the relevant summary is the harmonic mean:
Var(ybar) = s2g + s2e * E[1/n], so n_eff = 1 / E[1/n].

Estimator.  Unbalanced one-way random-effects ANOVA (Searle's method-of-
moments, the estimator behind Piepho & Moehring 2007 for this design):

    s2e_hat = MSW,  s2g_hat = (MSB - MSW) / n0,
    n0 = (N - sum_i n_i^2 / N) / (a - 1).

This departs from analysis/crosscrop/code/heritability.py in one respect: that
script keeps only entries with n_i >= 2 and estimates s2g as
var(entry means) - s2e/nbar over that subset.  Entries with two surviving plots
are not a random subset of entries -- they are enriched for checks and for
cells with no plot loss -- and the subset's nbar (~2.02) is not the replication
of the environment (harmonic mean ~1.29 over all cells).  Both estimators are
computed and reported side by side.

Outputs
  analysis/crosscrop/results/heritability_ceiling_A8.csv         per environment / trial
  analysis/crosscrop/results/heritability_ceiling_A8_bounds.csv  summary + headline recomputation

Run from the repository root with the project interpreter.
"""
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

RES_CC = "analysis/crosscrop/results"
RES_LB = "analysis/g2f_leaderboard/results"
DATA = os.environ.get("GP_DATA", "data/raw")
VEF_DIR = os.environ.get("VEF_DIR", f"{DATA}/vef")
RNG = np.random.default_rng(20260910)
B_BOOT = 2000
NS = [1, 2, 3, 4]


# ----------------------------------------------------------------- estimators
def anova_components(y, gid):
    """Unbalanced one-way random-effects MoM components.

    Returns (s2g, s2e, a, N, n0, n_harm, n_bar, frac_single) or None if the
    design carries too little information (fewer than 20 entries or fewer than
    20 error degrees of freedom).
    """
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(gid)}).dropna()
    if df.empty:
        return None
    grp = df.groupby("g")["y"]
    n_i = grp.size().to_numpy(float)
    m_i = grp.mean().to_numpy(float)
    a, N = len(n_i), n_i.sum()
    if a < 20 or (N - a) < 20:
        return None
    ssw = float(((df["y"] - df["g"].map(grp.mean()).to_numpy()) ** 2).sum())
    gm = df["y"].mean()
    ssb = float((n_i * (m_i - gm) ** 2).sum())
    msw = ssw / (N - a)
    msb = ssb / (a - 1)
    n0 = (N - (n_i ** 2).sum() / N) / (a - 1)
    s2g = (msb - msw) / n0
    return dict(s2g=s2g, s2e=msw, entries=int(a), plots=int(N), n0=n0,
                n_harm=float(1.0 / np.mean(1.0 / n_i)), n_bar=float(n_i.mean()),
                frac_single=float((n_i == 1).mean()))


def legacy_components(y, gid):
    """The estimator in heritability.py: replicated entries only."""
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(gid)}).dropna()
    n = df.groupby("g").size()
    if (n >= 2).sum() < 20:
        return None
    sub = df[df["g"].isin(n[n >= 2].index)]
    m = sub.groupby("g")["y"].agg(["mean", "var", "size"])
    s2e = float(np.nansum(m["var"] * (m["size"] - 1)) / np.nansum(m["size"] - 1))
    nbar = float(m["size"].mean())
    s2g = float(m["mean"].var(ddof=1) - s2e / nbar)
    return dict(s2g=s2g, s2e=s2e, nbar=nbar)


def h2_at(s2g, s2e, n):
    return s2g / (s2g + s2e / n)


def ceiling_at(s2g, s2e, n):
    return float(np.sqrt(h2_at(s2g, s2e, n)))


def boot_median_ceiling(tab, n, b=B_BOOT, rng=RNG):
    """Percentile CI for the median ceiling, resampling environments (the cluster)."""
    vals = np.sqrt(h2_at(tab["s2g"].to_numpy(), tab["s2e"].to_numpy(), n))
    k = len(vals)
    idx = rng.integers(0, k, size=(b, k))
    meds = np.median(vals[idx], axis=1)
    return float(np.median(vals)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


# ------------------------------------------------------------------ maize G2F
print("=" * 78)
print("MAIZE -- G2F 2014-2021 training data, plot level\n")
src = f"{DATA}/g2f/1_Training_Trait_Data_2014_2021.csv"
d = pd.read_csv(src, low_memory=False,
                usecols=["Env", "Field_Location", "Hybrid", "Replicate", "Yield_Mg_ha"]
                ).dropna(subset=["Yield_Mg_ha"])
cell_n = d.groupby(["Env", "Hybrid"]).size()
print(f"{len(d)} plots, {d.Env.nunique()} environments, {d.Hybrid.nunique()} hybrids, "
      f"{len(cell_n)} environment x hybrid cells")
print(f"plots per cell: median {cell_n.median():.0f}, mean {cell_n.mean():.3f}, "
      f"harmonic mean {1 / np.mean(1 / cell_n):.3f}, {100 * (cell_n == 1).mean():.1f} % singletons")

rows = []
for e, g in d.groupby("Env"):
    est = anova_components(g["Yield_Mg_ha"], g["Hybrid"])
    if est is None or not np.isfinite(est["s2g"]) or est["s2g"] <= 0:
        continue
    leg = legacy_components(g["Yield_Mg_ha"], g["Hybrid"])
    r = dict(dataset="maize_G2F", unit=e, **est)
    for n in NS:
        r[f"ceiling_n{n}"] = ceiling_at(est["s2g"], est["s2e"], n)
    r["ceiling_n_harm"] = ceiling_at(est["s2g"], est["s2e"], est["n_harm"])
    r["ceiling_inf"] = 1.0
    r["H2_n1"] = h2_at(est["s2g"], est["s2e"], 1)
    r["H2_n2"] = h2_at(est["s2g"], est["s2e"], 2)
    if leg is not None and np.isfinite(leg["s2g"]) and leg["s2g"] > 0:
        r["legacy_H2"] = h2_at(leg["s2g"], leg["s2e"], leg["nbar"])
        r["legacy_ceiling"] = float(np.sqrt(r["legacy_H2"]))
        r["legacy_nbar"] = leg["nbar"]
    rows.append(r)
M = pd.DataFrame(rows)
print(f"\n{len(M)} environments retained (>=20 entries and >=20 error df, positive s2g)")
print(f"median plots per cell within those environments: harmonic {M.n_harm.median():.3f}, "
      f"arithmetic {M.n_bar.median():.3f}")

print("\nCeiling sqrt(H2(n)), median over environments, 95 % CI by bootstrap over environments:")
maize_boot = {}
for n in NS:
    med, lo, hi = boot_median_ceiling(M, n)
    maize_boot[n] = (med, lo, hi)
    print(f"  n = {n}      H2 = {np.median(h2_at(M.s2g, M.s2e, n)):.3f}   "
          f"ceiling = {med:.3f}  [{lo:.3f}, {hi:.3f}]")
print(f"  n -> inf   H2 = 1.000   ceiling = 1.000  (exact, by construction)")

# exact reproduction of heritability.py, on its own environment-retention rule
leg_rows = []
for e, g in d.groupby("Env"):
    leg = legacy_components(g["Yield_Mg_ha"], g["Hybrid"])
    if leg is None or not np.isfinite(leg["s2g"]) or leg["s2g"] <= 0:
        continue
    leg_rows.append(dict(unit=e, H2=h2_at(leg["s2g"], leg["s2e"], leg["nbar"]), nbar=leg["nbar"]))
L = pd.DataFrame(leg_rows)
leg_pub_H2 = float(L.H2.median())
leg_pub = float(np.sqrt(L.H2).median())
print(f"\nlegacy estimator on its own retention rule ({len(L)} env): median H2 {leg_pub_H2:.3f}, "
      f"ceiling {leg_pub:.3f}  (published: H2 0.634, ceiling 0.796)")
leg_med = float(M["legacy_ceiling"].median())
print(f"legacy estimator on the 192 ANOVA environments  : median H2 {M.legacy_H2.median():.3f}, "
      f"ceiling {leg_med:.3f} at its own nbar = {M.legacy_nbar.median():.2f}")
print(f"same-n comparison, ANOVA estimator at n = 2     : ceiling {maize_boot[2][0]:.3f} "
      f"(difference from legacy {maize_boot[2][0] - leg_med:+.3f}: the estimator change alone)")

# replicate-block sensitivity: remove replicate main effects before the ANOVA
rows_rcbd = []
for e, g in d.groupby("Env"):
    if g["Replicate"].nunique() < 2:
        continue
    adj = g.assign(y=g["Yield_Mg_ha"] - g.groupby("Replicate")["Yield_Mg_ha"].transform("mean")
                   + g["Yield_Mg_ha"].mean())
    est = anova_components(adj["y"], adj["Hybrid"])
    if est is None or not np.isfinite(est["s2g"]) or est["s2g"] <= 0:
        continue
    rows_rcbd.append(dict(unit=e, s2g=est["s2g"], s2e=est["s2e"]))
R = pd.DataFrame(rows_rcbd)
rcbd1 = float(np.median(np.sqrt(h2_at(R.s2g, R.s2e, 1))))
rcbd2 = float(np.median(np.sqrt(h2_at(R.s2g, R.s2e, 2))))
print(f"sensitivity, replicate main effects removed     : ceiling {rcbd1:.3f} (n=1), "
      f"{rcbd2:.3f} (n=2), over {len(R)} environments")


# -------------------------------- how many plots stand behind a 2022 test value
print("\n" + "=" * 78)
print("2022 TEST SET -- replication is not recorded; a weak one-sided diagnostic\n")
obs = pd.read_csv(f"{RES_LB}/Final_Observed_Yield.csv")
tpl = pd.read_csv(f"{RES_LB}/1_Submission_Template_2022.csv")
obs["loc"] = obs["Env"].str.replace(r"_2022$", "", regex=True)
print(f"answer key: {len(obs)} cells, {obs.Env.nunique()} environments, {obs.Hybrid.nunique()} "
      f"hybrids, {obs.duplicated(['Env','Hybrid']).sum()} duplicate cells")
print(f"submission template: {len(tpl)} cells over {tpl.Env.nunique()} environments; "
      f"{len(tpl[tpl.Env.isin(obs.Env)])} of them in the 23 scored environments")
print("  -> one value per environment x hybrid cell, no replicate column, no plot count.")
print("  -> Lima et al. 2023 BMC Res Notes 16:148 (doi:10.1186/s13104-023-06421-z) and Washburn")
print("     et al. 2025 Genetics 229:iyae195 (doi:10.1093/genetics/iyae195, body text read from")
print("     the bioRxiv version doi:10.1101/2024.09.13.612969) describe the key as one yield per")
print("     hybrid per location and do not state its replication; nor does the 2024-edition note")
print("     (doi:10.1186/s13104-026-07629-5).  CyVerse doi:10.25739/tq5e-ak26 was not reachable")
print("     from this sandbox (the host redirects automated clients to an access-check page).")

# per location-year components from the training data
ly = []
for (fl, e), g in d.groupby(["Field_Location", "Env"]):
    est = anova_components(g["Yield_Mg_ha"], g["Hybrid"])
    if est is None or not np.isfinite(est["s2g"]) or est["s2g"] <= 0:
        continue
    ly.append(dict(loc=fl, Env=e, s2g=est["s2g"], s2e=est["s2e"]))
LY = pd.DataFrame(ly)
locmed = LY.groupby("loc")[["s2g", "s2e"]].median()
locmin = LY.groupby("loc")["s2e"].min().rename("s2e_min_year")

V = obs.groupby(["Env", "loc"]).Yield_Mg_ha.agg(V="var", cells="size").reset_index()
V = V.merge(locmed, left_on="loc", right_index=True, how="left").merge(
    locmin, left_on="loc", right_index=True, how="left")
V["pred_n1"] = V.s2g + V.s2e
V["pred_n2"] = V.s2g + V.s2e / 2
V["n_implied"] = V.s2e / (V.V - V.s2g)
V.loc[(V.V - V.s2g) <= 0, "n_implied"] = np.inf
matched = V.dropna(subset=["s2e"])
print(f"\n{len(matched)} of {len(V)} scored environments have their location in the training data")
print(f"observed within-environment variance V vs the location's training-year median:")
print(f"  median V / (s2g + s2e)   = {(matched.V / matched.pred_n1).median():.2f}   "
      f"(would be 1.0 if the key were single plots)")
print(f"  median V / (s2g + s2e/2) = {(matched.V / matched.pred_n2).median():.2f}   "
      f"(would be 1.0 if the key were two-plot means)")
print(f"  V below the location's median plot-error variance s2e   : "
      f"{(matched.V < matched.s2e).sum()} of {len(matched)} environments")
print(f"  V below the location's SMALLEST single-year s2e         : "
      f"{(matched.V < matched.s2e_min_year).sum()} of {len(matched)} environments")
ni = matched.n_implied.replace(np.inf, np.nan).dropna()
idx = RNG.integers(0, len(ni), size=(B_BOOT, len(ni)))
nb_med = np.median(ni.to_numpy()[idx], axis=1)
print(f"  implied n = s2e / (V - s2g), median {np.median(ni):.2f} "
      f"[{np.percentile(nb_med, 2.5):.2f}, {np.percentile(nb_med, 97.5):.2f}] over "
      f"{len(ni)} environments ({(matched.n_implied == np.inf).sum()} gave V <= s2g, no solution)")
print("  This diagnostic is WEAK and biased upward: it assumes the 2022 genetic and error")
print("  variances at a location equal the 2014-2021 medians there, while the 548 test")
print("  hybrids are largely new germplasm, and the competition file was 'lightly filtered")
print("  for quality'.  Both shrink V and inflate the implied n.  It argues against the key")
print("  being single plots; it does not establish a value, so the report brackets n.")


# ------------------------------------------------------- common bean, two ways
print("\n" + "=" * 78)
print("COMMON BEAN -- VEF, published BLUE standard errors vs raw plots\n")
blue = pd.read_csv(f"{VEF_DIR}/VEF_BLUE_data.csv")
raw = pd.read_csv(f"{VEF_DIR}/VEF_raw_phenotypic_data.csv")
raw.columns = [c.strip() for c in raw.columns]
raw = raw.dropna(subset=["Yd"])

brows = []
for t, g in blue.groupby("Trial"):
    v = g["Yd_BLUE"].dropna()
    se = g.loc[v.index, "Yd_SE"]
    rel = 1 - float(np.mean(se ** 2)) / float(v.var(ddof=1))
    rr = raw[raw.Trial == t]
    est = anova_components(rr["Yd"], rr["Line"]) if len(rr) else None
    row = dict(dataset="bean_VEF", unit=t, blue_reliability=rel,
               blue_ceiling=float(np.sqrt(rel)) if rel > 0 else np.nan, blue_lines=len(v))
    if est is not None and np.isfinite(est["s2g"]) and est["s2g"] > 0:
        row.update(est)
        for n in NS:
            row[f"ceiling_n{n}"] = ceiling_at(est["s2g"], est["s2e"], n)
        row["ceiling_n_harm"] = ceiling_at(est["s2g"], est["s2e"], est["n_harm"])
        row["ceiling_inf"] = 1.0
        row["H2_n1"] = h2_at(est["s2g"], est["s2e"], 1)
        row["H2_n2"] = h2_at(est["s2g"], est["s2e"], 2)
    brows.append(row)
Bn = pd.DataFrame(brows)
have_raw = Bn.dropna(subset=["s2g"])
print(f"{len(Bn)} trials with published BLUEs; {len(have_raw)} of them also have raw plot data")
print(f"replication behind the BLUEs (raw plots per line): harmonic mean "
      f"{have_raw.n_harm.median():.2f} (median over trials), {100 * have_raw.frac_single.median():.0f} % singletons")
print(f"\npublished-SE reliability route (manuscript's 0.773): median ceiling "
      f"{Bn.blue_ceiling.median():.3f}")
for n in NS:
    print(f"raw-plot ANOVA, n = {n}                              : median ceiling "
          f"{have_raw[f'ceiling_n{n}'].median():.3f}")
bean_at_harm = float(have_raw["ceiling_n_harm"].median())
print(f"raw-plot ANOVA at each trial's own harmonic n         : median ceiling {bean_at_harm:.3f}")
print(f"  -> the SE route and the plot route agree to "
      f"{abs(Bn.blue_ceiling.median() - bean_at_harm):.3f}; both describe a ~"
      f"{have_raw.n_harm.median():.1f}-plot mean, not a single plot")
bean_boot = {}
for n in NS:
    bean_boot[n] = boot_median_ceiling(have_raw, n)


# ------------------------------------------------- what the headline numbers do
print("\n" + "=" * 78)
print("HEADLINE NUMBERS RECOMPUTED AGAINST EACH CEILING\n")
lb = pd.read_csv(f"{RES_LB}/S1_best_per_team.csv")
lb = lb[lb["Team Name"].notna()]
r = lb["mean_pearson_r"].dropna().sort_values(ascending=False)
r1, r2, r3 = r.iloc[0], r.iloc[1], r.iloc[2]
gap12, gap13 = r1 - r2, r1 - r3
print(f"leaderboard mean_pearson_r: 1st {r1:.4f}, 2nd {r2:.4f}, 3rd {r3:.4f}; "
      f"gap 1-2 {gap12:.4f}, gap 1-3 {gap13:.4f}")

xs = pd.read_csv(f"{RES_CC}/cross_dataset_summary.csv")
thr = xs.dropna(subset=["threshold"]).set_index("dataset")["threshold"].to_dict()
thr_lo, thr_hi = min(thr.values()), max(thr.values())
print(f"surrogate thresholds in the repository: "
      f"{', '.join(f'{k} {v:.3f}' for k, v in sorted(thr.items(), key=lambda kv: kv[1]))}")

n_train_eff = float(1 / np.mean(1 / cell_n))
n_implied = float(np.median(ni))
scen = [("n=1  single plot", maize_boot[1]),
        (f"n={n_train_eff:.2f} training effective (harmonic)", boot_median_ceiling(M, n_train_eff)),
        (f"n={n_implied:.2f} implied by the 2022 variance diagnostic",
         boot_median_ceiling(M, n_implied)),
        ("n=2  two-plot mean", maize_boot[2]),
        ("n=3  three-plot mean", maize_boot[3]),
        ("n=4  four-plot mean", maize_boot[4]),
        ("legacy estimator as published (manuscript)", (leg_pub, np.nan, np.nan))]
out = []
for name, (c, lo, hi) in scen:
    out.append(dict(scenario=name, ceiling=c, ceiling_lo=lo, ceiling_hi=hi,
                    winner_frac_of_ceiling=r1 / c,
                    thr_min_frac_of_ceiling=thr_lo / c,
                    thr_max_frac_of_ceiling=thr_hi / c,
                    gap13_frac_of_ceiling=gap13 / c,
                    gain_to_ceiling_pct=(c - r1) / r1 * 100,
                    winner_exceeds_ceiling=bool(r1 > c)))
O = pd.DataFrame(out)
print()
print(O.assign(**{c: O[c].map(lambda v: f"{v:.3f}" if np.isfinite(v) else "")
                  for c in ["ceiling", "ceiling_lo", "ceiling_hi"]})
      .assign(**{c: (100 * O[c]).map("{:.1f}%".format)
                 for c in ["winner_frac_of_ceiling", "thr_min_frac_of_ceiling",
                           "thr_max_frac_of_ceiling", "gap13_frac_of_ceiling"]},
              gain_to_ceiling_pct=O.gain_to_ceiling_pct.map("{:.0f}%".format))
      .to_string(index=False))

print("\nThe bracket for the 2022 test set (replication not established, see report):")
c_lo, c_hi = maize_boot[1][0], maize_boot[2][0]
print(f"  ceiling in [{c_lo:.3f}, {c_hi:.3f}]  -> winner reached "
      f"[{100 * r1 / c_hi:.0f} %, {100 * r1 / c_lo:.0f} %] of attainable")
print(f"                                       thresholds are "
      f"[{100 * thr_lo / c_hi:.0f} %, {100 * thr_hi / c_lo:.0f} %] of the ceiling")
print(f"                                       gap 1-3 is "
      f"[{100 * gap13 / c_hi:.1f} %, {100 * gap13 / c_lo:.1f} %] of the ceiling")
print(f"  ratios that do NOT move: threshold / gap13 = "
      f"{thr_lo / gap13:.1f} to {thr_hi / gap13:.1f} (scale-free)")

# ------------------------------------------------------------------- write out
per_unit = pd.concat([M, Bn], ignore_index=True, sort=False)
per_unit.to_csv(f"{RES_CC}/heritability_ceiling_A8.csv", index=False)

bounds = []
for n in NS:
    m_, l_, h_ = maize_boot[n]
    bounds.append(dict(dataset="maize_G2F", quantity=f"ceiling_n{n}", value=m_, lo=l_, hi=h_,
                       n_units=len(M)))
    b_, bl_, bh_ = bean_boot[n]
    bounds.append(dict(dataset="bean_VEF_rawplots", quantity=f"ceiling_n{n}", value=b_, lo=bl_,
                       hi=bh_, n_units=len(have_raw)))
bounds += [
    dict(dataset="maize_G2F", quantity="ceiling_inf", value=1.0, lo=1.0, hi=1.0, n_units=len(M)),
    # the exact reproduction of the published numbers, on heritability.py's own retention
    # rule (>=20 entries with n>=2).  These two rows are the ones the manuscript quotes.
    dict(dataset="maize_G2F", quantity="legacy_H2_own_retention_rule_PUBLISHED_0.634",
         value=leg_pub_H2, lo=np.nan, hi=np.nan, n_units=len(L)),
    dict(dataset="maize_G2F", quantity="legacy_ceiling_own_retention_rule_PUBLISHED_0.796",
         value=leg_pub, lo=np.nan, hi=np.nan, n_units=len(L)),
    # the same estimator restricted to the 192 environments the ANOVA retains -- a DIFFERENT
    # quantity, reported only so the two estimators are compared on one environment set.
    dict(dataset="maize_G2F", quantity="ceiling_legacy_estimator_on_192_ANOVA_envs", value=leg_med,
         lo=np.nan, hi=np.nan, n_units=len(M)),
    dict(dataset="maize_G2F", quantity="ceiling_n1_repblock_removed", value=rcbd1,
         lo=np.nan, hi=np.nan, n_units=len(R)),
    dict(dataset="maize_G2F", quantity="ceiling_n2_repblock_removed", value=rcbd2,
         lo=np.nan, hi=np.nan, n_units=len(R)),
    dict(dataset="maize_G2F", quantity="train_plots_per_cell_harmonic",
         value=float(1 / np.mean(1 / cell_n)), lo=np.nan, hi=np.nan, n_units=len(cell_n)),
    dict(dataset="maize_G2F", quantity="train_plots_per_cell_frac_single",
         value=float((cell_n == 1).mean()), lo=np.nan, hi=np.nan, n_units=len(cell_n)),
    dict(dataset="bean_VEF_BLUEse", quantity="ceiling_published_SE_route",
         value=float(Bn.blue_ceiling.median()), lo=np.nan, hi=np.nan, n_units=len(Bn)),
    dict(dataset="bean_VEF_rawplots", quantity="ceiling_at_own_harmonic_n", value=bean_at_harm,
         lo=np.nan, hi=np.nan, n_units=len(have_raw)),
    dict(dataset="bean_VEF_rawplots", quantity="plots_per_cell_harmonic",
         value=float(have_raw.n_harm.median()), lo=np.nan, hi=np.nan, n_units=len(have_raw)),
]
Bo = pd.DataFrame(bounds)
Bo = pd.concat([Bo, O.assign(dataset="headline").rename(columns={"scenario": "quantity"})],
               ignore_index=True, sort=False)
Bo.to_csv(f"{RES_CC}/heritability_ceiling_A8_bounds.csv", index=False)
print(f"\nwrote {RES_CC}/heritability_ceiling_A8.csv ({len(per_unit)} rows) and "
      f"{RES_CC}/heritability_ceiling_A8_bounds.csv ({len(Bo)} rows)")
