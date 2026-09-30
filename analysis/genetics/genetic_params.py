"""Genetic parameters of every Atlas combination (analysis plan analysis_plan.md, s.4-5).

For each of the 52 Atlas combinations with a built method panel (G2F excluded), with the Atlas's own
data preparation (same loader, environment floor and marker principal components):

  P1  genomic heritability within an environment   h2_g = (d0 + d1) / (d0 + d1 + 1)
  P2  G x E share of genomic variance               s_GE = d1 / (d0 + d1)      (r_g = 1 - s_GE)
  P3  median between-environment phenotypic correlation of genotype values over environment pairs
      sharing >= 25 genotypes (model-free; mixes heritability and r_g)
  C1  SD across the panel's methods of their mean within-environment calibration slope (observed on
      predicted)

d0 = s2_u / s2_e and d1 = s2_v / s2_e come from one REML fit of the panels' marker x environment GBLUP
(analysis/crosscrop/code/mixed_models.fit with group = environment) on all the data, the features scaled
by scale_features so that the mean diagonal of WW' is one. The kernel is built from the Atlas's
leading principal components.

Run from anywhere; data are read from the repository root (ATLAS_ROOT), with the primary data under GP_DATA
(default data/raw) and the Atlas caches and panels (local only; rebuilt by analysis/atlas/atlas.py and prep_processed.py). Writes
analysis/genetics/results/genetic_params.csv in THIS checkout; existing files are never overwritten
(a run refuses to start if the output exists).
"""
import os, sys, time, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "genetic_params.csv")
ROOT = os.environ.get("ATLAS_ROOT", os.path.dirname(os.path.dirname(HERE)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis/atlas"))
sys.path.insert(0, os.path.join(ROOT, "analysis/crosscrop/code"))
import types
# The loaders need neither the panel builder (scikit-learn) nor the tool; stub them so that the
# analysis runs in an environment with numpy, scipy and pandas only.
for _m in ("panel_design", "gpverdict"):
    sys.modules.setdefault(_m, types.ModuleType(_m))
import atlas                                   # loaders and catalogue
import mixed_models as mm

MIN_SHARED = 25


def between_env_corr(ph):
    W = ph.pivot_table(index="k", columns="Env", values="y", aggfunc="mean")
    r = []
    cols = list(W.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = W[cols[i]], W[cols[j]]
            m = a.notna() & b.notna()
            if m.sum() >= MIN_SHARED and a[m].std() > 0 and b[m].std() > 0:
                r.append(np.corrcoef(a[m], b[m])[0, 1])
    return (float(np.median(r)) if r else np.nan), len(r)


def calib_spread(entry):
    pan = f"analysis/atlas/panels/{entry['id']}.csv.gz"
    P = pd.read_csv(pan)
    n = P.groupby(["method", "Env"]).k.transform("size")
    P = P[n >= entry["min_n"]]
    slopes = {}
    for (m, e), g in P.groupby(["method", "Env"]):
        v = g.p.var()
        if v > 0:
            slopes.setdefault(m, []).append(np.cov(g.p, g.y)[0, 1] / v)
    ms = np.array([np.mean(s) for s in slopes.values()])
    return float(np.std(ms, ddof=1)), len(ms)


def main():
    if os.path.exists(OUT):
        sys.exit(f"{OUT} exists; new results go to new files")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for entry in atlas.catalogue():
        if entry["dataset"] == "G2F":
            continue
        t0 = time.time()
        ph, PC = atlas.load(entry)
        W = mm.scale_features(PC[ph.gi.to_numpy()])
        fit = mm.fit(ph.y.to_numpy(), ph.Env.to_numpy(), W, group=ph.Env.to_numpy())
        d0, d1 = float(fit["d0"]), float(fit["d1"])
        p3, npairs = between_env_corr(ph)
        c1, nm = calib_spread(entry)
        rows.append(dict(id=entry["id"], dataset=entry["dataset"], trait=entry["trait"],
                         n_env=ph.Env.nunique(), n_genotypes=ph.k.nunique(), cells=len(ph), n_pc=PC.shape[1],
                         d0=d0, d1=d1, h2_g=(d0 + d1) / (d0 + d1 + 1), s_GE=d1 / (d0 + d1) if d0 + d1 > 0 else np.nan,
                         r_g_cs=d0 / (d0 + d1) if d0 + d1 > 0 else np.nan,
                         P3_between_env_r=p3, env_pairs=npairs, C1_calib_sd=c1, n_methods=nm,
                         seconds=round(time.time() - t0, 1)))
        r = rows[-1]
        print(f"{r['id']:<28} env {r['n_env']:>4} geno {r['n_genotypes']:>5}  h2_g {r['h2_g']:.3f}  s_GE {r['s_GE']:.3f}"
              f"  P3 {r['P3_between_env_r']:.3f}  C1 {r['C1_calib_sd']:.3f}  ({r['seconds']} s)", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"-> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
