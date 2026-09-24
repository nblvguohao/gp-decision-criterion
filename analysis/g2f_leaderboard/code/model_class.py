"""Does model class explain the reordering between the two official metrics?

Rank movement of each team between the 2022 official metric (mean_RMSE) and the
2024 official metric (mean_pearson_r), grouped by the strategy labels of
Supplemental Table S3 of the competition paper. A team may carry several labels.
For each label, the mean movement of its members is compared with a null that
permutes the label across teams (20,000 permutations, two-sided about the
overall mean). Reported twice: permuting across all 30 teams (teams without a
label in S3 are members of no class) and across the 27 labelled teams only.

Writes analysis/g2f_leaderboard/results/model_class_permutation.csv
"""
import numpy as np, pandas as pd

RES = "analysis/g2f_leaderboard/results"
S3 = pd.read_csv(f"{RES}/Supplemental_Table_S3_GENETICS-2024-307594.tsv", sep="\t", skiprows=2)
S3 = S3[S3["Team Name"].notna()]
G = pd.read_csv(f"{RES}/S1_best_per_team.csv"); G = G[G["Team Name"].notna()].copy()
G["move"] = G.mean_RMSE.rank(method="min") - G.mean_pearson_r.rank(ascending=False, method="min")
key = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
S3["key"] = key(S3["Team Name"]); G["key"] = key(G["Team Name"])
M = G.merge(S3.drop(columns="Team Name"), on="key", how="left")
M["labelled"] = M["Rank"].notna()
CLASSES = ["Classical Machine Learning", "Linear/Mixed/BLUP", "Deep Learning/Neural Net", "Ensemble Model"]
rows = []
for scope, X in (("all 30 teams", M), ("27 labelled teams", M[M.labelled])):
    rng = np.random.default_rng(0)
    mv = X.move.to_numpy()
    for c in CLASSES:
        lab = (pd.to_numeric(X[c], errors="coerce").fillna(0) > 0).to_numpy()
        obs = mv[lab].mean()
        null = np.array([mv[rng.permutation(lab)].mean() for _ in range(20000)])
        p = float((np.abs(null - mv.mean()) >= abs(obs - mv.mean())).mean())
        rows.append(dict(scope=scope, model_class=c, n=int(lab.sum()), mean_move=obs, perm_p=p))
R = pd.DataFrame(rows)
R.to_csv(f"{RES}/model_class_permutation.csv", index=False)
print(R.round(3).to_string(index=False))
