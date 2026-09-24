"""Does the leaderboard analysis depend on which submission stands for a team?

The 2022 competition ranked teams on the mean within-environment RMSE, so the paper
takes each team's best submission under that metric (build_inputs.best_per_team), which
reproduces the published leaderboard. That rule is itself one of the two metrics being
compared, so this recomputes the leaderboard quantities under four other rules:
the best submission under the 2024 official metric, under the admissible metric, the
team's most recent submission, and every scored submission with no per-team selection.

For each rule: the reversal rate between the two official metrics, the top three by
mean_pearson_r with their scores, the first-to-second and first-to-third gaps, and how
many entries lie within the Gaussian design value of the leader (threshold_theory.py).

Writes analysis/g2f_leaderboard/results/selection_rule_sensitivity.csv
"""
import sys, itertools, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/g2f_leaderboard/code")
from build_inputs import METRICS, orient

G = "analysis/g2f_leaderboard/results"
X = "analysis/crosscrop/results"
D = pd.read_csv(f"{G}/S1_parsed.csv").dropna(subset=METRICS)
FLOOR = float(pd.read_csv(f"{X}/threshold_theory.csv").set_index("dataset").loc["maize (G2F design)", "theory_lam05"])


def rate(T):
    a, b = orient(T, "mean_RMSE"), orient(T, "mean_pearson_r")
    pr = list(itertools.combinations(range(len(T)), 2))
    d1 = np.array([a[i] - a[j] for i, j in pr]); d2 = np.array([b[i] - b[j] for i, j in pr])
    ok = (np.abs(d1) >= 1e-9) & (np.abs(d2) >= 1e-9)
    return float((d1[ok] * d2[ok] < 0).mean()), int(ok.sum())


RULES = [("best mean_RMSE (2022 official; used in the paper)", lambda d: d.sort_values("mean_RMSE").groupby("Team Name", as_index=False).first()),
         ("best mean_pearson_r (2024 official)", lambda d: d.sort_values("mean_pearson_r", ascending=False).groupby("Team Name", as_index=False).first()),
         ("best mean_spearman_r (admissible)", lambda d: d.sort_values("mean_spearman_r", ascending=False).groupby("Team Name", as_index=False).first()),
         ("most recent submission", lambda d: d.sort_values("Submitted At", ascending=False).groupby("Team Name", as_index=False).first()),
         ("every scored submission, no selection", lambda d: d.copy())]
rows = []
for lab, f in RULES:
    T = f(D).reset_index(drop=True)
    r, npairs = rate(T)
    o = T.sort_values("mean_pearson_r", ascending=False).reset_index(drop=True)
    v = o.mean_pearson_r.to_numpy()
    rows.append(dict(rule=lab, entries=len(T), pairs=npairs, reversal=r,
                     first=o["Team Name"][0], second=o["Team Name"][1], third=o["Team Name"][2],
                     r1=v[0], r2=v[1], r3=v[2], gap12=v[0] - v[1], gap13=v[0] - v[2],
                     within_floor=int((v > v[0] - FLOOR).sum())))
T = pd.DataFrame(rows)
T.to_csv(f"{G}/selection_rule_sensitivity.csv", index=False)
pd.set_option("display.width", 250)
print(f"{len(D)} of {len(pd.read_csv(f'{G}/S1_parsed.csv'))} submissions carry all 22 metrics; "
      f"Gaussian design value {FLOOR:.3f}\n")
print(T.round(4).to_string(index=False))
