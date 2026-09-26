"""Spring-wheat counterexample for Fig. 1D: mean_r2_pearson cannot tell a ranking from its
reverse. Every base method is scored as fitted and with its predictions reversed within each
environment; squaring the correlation gives the two the same score although one selects
above the environment mean and the other below it.
Writes analysis/crosscrop/results/counterexample_ursn.csv
"""
import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
import numpy as np, pandas as pd
from analyse_species import panel22

RES = "analysis/crosscrop/results"
P = pd.read_csv(f"{RES}/ursn_panel_wide.csv")
P = P[~P.method.str.contains("__")]
rows = []
for m, d0 in P.groupby("method"):
    for rev in (False, True):
        d = d0.assign(p=-d0.p) if rev else d0
        b = panel22(d, 12)
        sg = []
        for e, g in d.groupby("Env"):
            if len(g) < 12 or g.p.nunique() < 2 or g.y.nunique() < 2:
                continue
            y, p = g.y.to_numpy(), g.p.to_numpy(); k = max(1, int(round(.10 * len(g))))
            top = np.argsort(-p, kind="mergesort")[:k]
            sg.append((y[top].mean() - y.mean()) / y.std())
        rows.append(dict(method=m, reversed=rev, mean_r2_pearson=b["mean_r2_pearson"],
                         mean_pearson_r=b["mean_pearson_r"], mean_spearman_r=b["mean_spearman_r"],
                         SG_f10=float(np.mean(sg)), n_env=len(sg)))
T = pd.DataFrame(rows).sort_values(["mean_r2_pearson", "reversed"], ascending=[False, True])
T.to_csv(f"{RES}/counterexample_ursn.csv", index=False)
print(T.round(3).to_string(index=False))
