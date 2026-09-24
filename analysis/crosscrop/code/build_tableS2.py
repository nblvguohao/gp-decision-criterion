"""Emit Supplementary Table S2 as markdown, from the computed result files.

The table used to be maintained by hand, so it drifted out of step with the
analysis. Run from the repository root:
    python analysis/crosscrop/code/build_tableS2.py

Maize drifts are those plotted in Fig. 1a (invariance.py, monotone_test_panels.py); the
reversal change, the tier and the soybean drift come from partition.py.
"""
import numpy as np, pandas as pd

G = "analysis/g2f_leaderboard/results"; R = "analysis/crosscrop/results"
mz_a = pd.read_csv(f"{G}/invariance_test.csv").set_index("metric")["median"]
mz_m = pd.read_csv(f"{G}/monotone_test.csv").set_index("metric")["median"]
mz_c = pd.read_csv(f"{G}/coverage_by_metric_class.csv").set_index("metric")
PM = pd.read_csv(f"{R}/partition_metrics.csv")
pm = PM[PM.dataset == "maize"].set_index("metric")
sy = PM[PM.dataset == "soybean"].set_index("metric")
sy_c = pd.read_csv(f"{R}/nust_coverage_wide.csv").set_index("metric")["coverage"]
assert set(PM.groupby("metric").tier.nunique()) == {1}, "tier differs between datasets"


def fmt(v):
    if not np.isfinite(v): return "—"
    if v == 0: return "0.0"
    if v >= 1: return f"{v:.2f}"
    e = int(np.floor(np.log10(abs(v)))); m = v / 10**e
    return f"{m:.1f} × 10{str(e).translate(str.maketrans('-0123456789','⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"


rows = []
order = pm.assign(t=pm.tier.map({"I": 0, "II": 1, "III": 2})).sort_values(["t", "affine_drift"]).index
for m in order:
    agg = "within-env" if m.startswith("mean_") else "pooled"
    typ = mz_c.loc[m, "type"].replace("error-magnitude", "error magnitude") if m in mz_c.index else "—"
    tier = pm.loc[m, "tier"]; tier = f"**{tier}**" if tier != "III" else tier
    rows.append(f"| `{m}` | {agg} | {typ} | {fmt(mz_a[m])} | {fmt(mz_m[m])} | "
                f"{'yes' if pm.loc[m, 'order_sensitive'] else '**no**'} | {tier} | "
                f"{fmt(sy.loc[m, 'affine_drift'])} | {mz_c.loc[m,'coverage']:.2f} | {sy_c.get(m, np.nan):.2f} |")

print("| Metric | Aggregation | Quantity | Affine drift, maize | Monotone drift, maize | Order-sensitive | Tier | "
      "Affine drift, soybean | Coverage, maize | Coverage, soybean |")
print("|---|---|---|---:|---:|:---:|:---:|---:|---:|---:|")
print("\n".join(rows))
t12 = pm.tier.isin(["I", "II"])
print(f"\n<!-- Tier I {int((pm.tier=='I').sum())}, Tier II {int((pm.tier=='II').sum())}, Tier III {int((pm.tier=='III').sum())}; "
      f"max affine drift Tier I/II {mz_a[pm.index[t12]].max():.3g}; smallest affine drift among order-sensitive Tier III "
      f"{mz_a[pm.index[(~t12) & pm.order_sensitive]].min():.3g}; smallest monotone drift outside Tier I "
      f"{mz_m[pm.index[pm.tier != 'I']].min():.3g} -->")
