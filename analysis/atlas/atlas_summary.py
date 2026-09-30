"""Summary of the Atlas for the paper (Supplementary Section S21).

Reads analysis/atlas/atlas.csv and the per-combination JSON (for the tie-ensemble comparison) and
writes analysis/atlas/atlas_summary.json with every number the text quotes.

Tie ensemble: the equal-weight ensemble of the methods tied with the leader is compared with the
single leader out of sample (tie_ensemble.py) in every combination with at least eight
environments (environment split). Combinations share datasets, so the pooled test treats the
dataset as the unit: the mean difference per dataset, then a bootstrap over datasets. The ensemble
is reported as an improvement only if that interval excludes zero.
"""
import json, glob, os
import numpy as np, pandas as pd

A = "analysis/atlas"
T = pd.read_csv(f"{A}/atlas.csv")
out = dict(n=len(T), datasets=int(T.dataset.nunique()), crops=int(T.crop.nunique()),
           per_dataset=T.groupby("dataset").size().to_dict())
out["reversal_median"] = float(T.reversal.median())
out["reversal_min"], out["reversal_max"] = float(T.reversal.min()), float(T.reversal.max())
out["tied_median"] = float(T.tied.fillna("").str.split(", ").map(len).median())
out["tied_ge3"] = int((T.tied.fillna("").str.split(", ").map(len) >= 3).sum())
out["rmse_pick_not_leader"] = int((T.rmse_pick != T.leader).sum())
out["rmse_pick_outside_tied"] = int(sum(r.rmse_pick not in str(r.tied).split(", ") for r in T.itertuples()))
rel = T[T.reliability >= 0.5].copy()
rel["corr"] = rel[["recovery_spearman", "recovery_pearson"]].mean(axis=1)
out["n_reliable"] = len(rel)
out["corr_beats_rmse"] = int((rel["corr"] > rel.recovery_rmse + 1e-9).sum())
out["rmse_beats_corr"] = int((rel.recovery_rmse > rel["corr"] + 1e-9).sum())
out["median_corr_minus_rmse"] = float((rel["corr"] - rel.recovery_rmse).median())
out["median_recovery_corr"] = float(rel["corr"].median()); out["median_recovery_rmse"] = float(rel.recovery_rmse.median())

rows = []
for f in sorted(glob.glob(f"{A}/results/*.json")):
    r = json.load(open(f))
    t = r.get("tie") or {}
    if t and "tie_minus_leader" in t:
        rows.append(dict(id=r["id"], dataset=r["dataset"], d_leader=t["tie_minus_leader"], lo=t["tie_minus_leader_lo"],
                         hi=t["tie_minus_leader_hi"], d_rmse=t["tie_minus_rmse"], n_tied=t["mean_tied"]))
TE = pd.DataFrame(rows)
if len(TE):
    per = TE.groupby("dataset").d_leader.mean()
    rng = np.random.default_rng(20260927)
    boot = [per.sample(len(per), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(4000)]
    out["tie"] = dict(n=len(TE), n_datasets=int(per.size), mean_diff=float(TE.d_leader.mean()),
                      dataset_mean=float(per.mean()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                      positive=int((TE.d_leader > 0).sum()), ci_above_zero=int((TE.lo > 0).sum()), ci_below_zero=int((TE.hi < 0).sum()),
                      per_dataset={k: float(v) for k, v in per.items()})
    TE.to_csv(f"{A}/tie_ensemble_atlas.csv", index=False)
json.dump(out, open(f"{A}/atlas_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
