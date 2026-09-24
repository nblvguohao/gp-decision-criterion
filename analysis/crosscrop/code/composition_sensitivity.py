"""Does the outcome test depend on which methods are in the panel? (Supplementary Table S22)

The outcome test of does_it_help.py is repeated on sub-panels:
  - the full panel, as a check that this script reproduces does_it_help.json;
  - the panel without the two GBLUP models (and their variants), which were added to the
    panels after the other methods;
  - each base panel with one base method (and its variants) left out in turn.
Recovery is (rule - average method) / (best method on the held-out half - average method),
as in Supplementary Table S7. Splits: 400 for the full and GBLUP-free panels (seed 0, as
does_it_help.py), 200 for each leave-one-method-out panel.

Writes analysis/crosscrop/results/composition_sensitivity.csv
"""
import sys; sys.path.insert(0, "analysis/crosscrop/code")
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import scope
from does_it_help import per_env, RULES

XC = "analysis/crosscrop/results"
SETS = [("common bean", f"{XC}/bean_panel_wide.csv", 25),
        ("spring wheat", f"{XC}/ursn_panel_wide.csv", 12),
        ("soybean", f"{XC}/nust_panel_wide.csv", 25)]
GBLUP = {"gblup", "gblup_gxe"}
base = lambda m: m.split("__")[0]


def recoveries(cache, meths, B, seed):
    """does_it_help.run on a method subset, returning recovery per rule."""
    rng = np.random.default_rng(seed)
    meths = sorted(meths)
    envs = sorted(set().union(*[set(cache[m]) for m in meths]))
    raw = {key: [] for _, key in RULES}; orc, rnd = [], []
    for _ in range(B):
        e = rng.permutation(envs); A, Bh = set(e[:len(e) // 2]), set(e[len(e) // 2:])
        gainB = {m: np.mean([cache[m][x]["GAIN"] for x in Bh if x in cache[m]]) for m in meths}
        for _, key in RULES:
            sA = {m: np.mean([cache[m][x][key] for x in A if x in cache[m]]) for m in meths}
            raw[key].append(scope.tied_best_mean([sA[m] for m in meths], [gainB[m] for m in meths]))
        orc.append(max(gainB.values())); rnd.append(np.mean(list(gainB.values())))
    o, r = np.mean(orc), np.mean(rnd)
    return {key: (np.mean(v) - r) / (o - r) for key, v in raw.items()}


rows = []
for lab, path, mn in SETS:
    P = pd.read_csv(path)
    allm = sorted(P.method.unique())
    cache = {m: per_env(P[P.method == m], mn) for m in allm}
    for panel, pool in (("augmented", allm), ("base", [m for m in allm if "__" not in m])):
        cases = [("none", pool, 400), ("GBLUP models", [m for m in pool if base(m) not in GBLUP], 400)]
        if panel == "base":
            cases += [(b, [m for m in pool if base(m) != b], 200) for b in sorted({base(m) for m in pool})]
        for removed, meths, B in cases:
            rec = recoveries(cache, meths, B, 0)
            rows.append(dict(dataset=lab, panel=panel, removed=removed, n_methods=len(meths), splits=B,
                             **{key: rec[key] for _, key in RULES}))
        print(f"{lab:<13} {panel:<9} done ({len(cases)} cases)", flush=True)

D = pd.DataFrame(rows)
D["corr_lo"] = D[["pearson", "spearman"]].min(1); D["corr_hi"] = D[["pearson", "spearman"]].max(1)
D["err_lo"] = D[["neg_RMSE", "neg_MAE", "r2_score"]].min(1); D["err_hi"] = D[["neg_RMSE", "neg_MAE", "r2_score"]].max(1)
D.to_csv(f"{XC}/composition_sensitivity.csv", index=False)
print(f"wrote {XC}/composition_sensitivity.csv")
