"""Surrogate thresholds in every constructed panel (Section 3.7, Table 2, Tables S5, S10).

Primary analysis: parent methods only (the miscalibration variants are affine copies of
their parents and duplicate pairs), f = 0.10, logistic-through-one-half estimator of
threshold_model.py, 95 % interval from a two-way (method x environment) cluster bootstrap.
Also reported: the isotonic crossing, the out-of-sample threshold, all methods (variants
included, ties dropped), and f = 0.05 and 0.20. Maize has five verified submissions (ten
pairs), too few to estimate; its threshold is transferred from the design floor in
threshold_theory.py.

Writes analysis/crosscrop/results/threshold_estimates.csv
       analysis/crosscrop/results/threshold_pairs.csv (the primary pairs, for Fig. 5A)
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
import scope
import threshold_model as tm

RES = "analysis/crosscrop/results"
SETS = [("common bean", f"{RES}/bean_panel_wide.csv", 25),
        ("spring wheat", f"{RES}/ursn_panel_wide.csv", 12),
        ("soybean", f"{RES}/nust_panel_wide.csv", 25)]
rows = []; PAIRS = []
for lab, path, mn in scope.kept(SETS):
    P0 = pd.read_csv(path)
    for subset in ("parent methods", "all methods"):
        P = P0[~P0.method.str.contains("__")] if subset == "parent methods" else P0
        for f in (0.05, 0.10, 0.20):
            meths, envs, R, G, TOP = tm.build(P, f, mn)
            dr, dg = tm.pairs(R, G)
            if subset == "parent methods" and f == 0.10:
                PAIRS.append(pd.DataFrame(dict(dataset=lab, dr=dr, dg=dg)))
            row = dict(dataset=lab, methods=subset, frac=f, n_methods=len(meths), n_env=len(envs),
                       n_pairs=len(dr), beta=tm.fit_beta(dr, dg), threshold=tm.threshold(dr, dg),
                       isotonic=tm.isotonic_crossing(dr, dg),
                       p_agree_below_001=float((dg[dr < 0.01] > 0).mean()) if (dr < 0.01).any() else np.nan,
                       n_below_001=int((dr < 0.01).sum()))
            if f == 0.10:
                rng = np.random.default_rng(20260923)
                bs = tm.boot_threshold(R, G, meths, rng)
                ok = ~np.isnan(bs)                       # +inf = bounded from below only
                q = np.sort(bs[ok]); lo_i, hi_i = int(np.floor(0.025 * (len(q) - 1))), int(np.ceil(0.975 * (len(q) - 1)))
                row.update(ci_lo=q[lo_i], ci_hi=q[hi_i],             # order statistics; +inf = not bounded above
                           boot_unbounded=float(np.mean(np.isinf(bs[ok]))), boot_failed=float(np.mean(~ok)),
                           oos=tm.oos_threshold(R, G, np.random.default_rng(1)))
                ov, nov = tm.overlap_at_equal_accuracy(R, TOP, meths)
                row.update(overlap_below_001=ov, n_overlap_pairs=nov)
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
T = pd.DataFrame(rows)
T.to_csv(f"{RES}/threshold_estimates.csv", index=False)
pd.concat(PAIRS).to_csv(f"{RES}/threshold_pairs.csv", index=False)
pd.set_option("display.width", 220)
print(T.round(3).to_string(index=False))
