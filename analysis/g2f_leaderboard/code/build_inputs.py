"""Rebuild the derived leaderboard inputs used by every downstream script and figure.

Source : Supplemental Table S1 of Washburn et al. (2025) Genetics 229:iyae195
         (129 successful submissions x 30 teams x 22 officially computed metrics).
Outputs: S1_parsed.csv            all 129 submissions, one row each
         S1_best_per_team.csv     30 rows, each team's best submission under the
                                  2022 official metric (lowest mean_RMSE)
         coverage_by_metric_class.csv   leave-one-metric-out conformal coverage
         tierI_ranking.csv        2022 / 2024 / Tier-I rankings side by side

Run from the repository root:  python analysis/g2f_leaderboard/code/build_inputs.py
"""
import itertools, numpy as np, pandas as pd

RES = "analysis/g2f_leaderboard/results"
SRC = f"{RES}/Supplemental_Table_S1_GENETICS-2024-307594.tsv"

METRICS = ["pearson_r","spearman_r","lineRegressSlope","r2_score","r2_pearson",
           "MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
           "mean_pearson_r","mean_spearman_r","mean_lineRegressSlope","mean_r2_score",
           "mean_r2_pearson","mean_MAE","mean_realative_MAE","mean_normalized_MAE",
           "mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"]
# metrics where the ideal value is 1 (a calibration slope), and where lower is better
SLOPE = {"lineRegressSlope","mean_lineRegressSlope"}
LOWER = {"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
         "mean_MAE","mean_realative_MAE","mean_normalized_MAE",
         "mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}


def parse_S1():
    """Header sits on the third line; one blank spacer column separates the
    globally-computed block from the per-environment averaged block."""
    raw = [l.rstrip("\n").split("\t") for l in open(SRC, encoding="utf-8")]
    hdr = raw[2]
    keep = [i for i, h in enumerate(hdr) if h.strip()]
    cols = [hdr[i].strip() for i in keep]
    rows = [[r[i] if i < len(r) else "" for i in keep] for r in raw[3:] if any(x.strip() for x in r)]
    d = pd.DataFrame(rows, columns=cols)
    for c in METRICS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["id"] = pd.to_numeric(d["id"], errors="coerce")
    return d[["id", "Team Name", "Submitted At"] + METRICS]


def best_per_team(d):
    """The 2022 competition ranked on the mean within-environment RMSE, so a
    team's best submission is the one minimising mean_RMSE. Submissions for which
    the organisers could not compute every one of the 22 metrics are excluded,
    because a team must carry a value on all of them to enter the comparison."""
    b = (d.dropna(subset=METRICS)
           .sort_values("mean_RMSE")
           .groupby("Team Name", as_index=False)
           .first())
    return b.sort_values("Team Name")[["id", "Team Name", "Submitted At"] + METRICS]


def orient(d, m):
    v = d[m].to_numpy(float)
    return -np.abs(v - 1.) if m in SLOPE else (-v if m in LOWER else v)


def coverage(d):
    """Split-conformal rank intervals, leaving each metric out of the calibration
    set in turn; coverage is the fraction of teams whose rank under the held-out
    metric falls inside the interval built from the other 21."""
    M = len(d)
    S = {m: orient(d, m) for m in METRICS}

    def rk(m):
        o = np.argsort(-S[m], kind="mergesort")
        r = np.empty(M, int); r[o] = np.arange(1, M + 1); return r

    R = pd.DataFrame({m: rk(m) for m in METRICS})

    def conf(cal, a=.05):
        Rc = R[cal].to_numpy(); rh = np.median(Rc, axis=1)
        s = np.sort(np.abs(Rc - rh[:, None]).ravel()); n = len(s)
        return rh, s[min(int(np.ceil((n + 1) * (1 - a))), n) - 1]

    out = []
    for h in METRICS:
        rh, q = conf([m for m in METRICS if m != h])
        r = R[h].to_numpy()
        out.append((h, float(np.mean((r >= rh - q) & (r <= rh + q)))))
    c = pd.DataFrame(out, columns=["metric", "coverage"])
    c["aggregation"] = ["within-env" if m.startswith("mean_") else "pooled" for m in c.metric]
    c["type"] = ["discrimination" if (m in SLOPE or ("pearson" in m or "spearman" in m)
                                      and not m.endswith("r2_score")) else "error-magnitude"
                 for m in c.metric]
    c["cell"] = c.aggregation + " x " + c.type
    return c[["metric", "aggregation", "type", "coverage", "cell"]]


def tierI(d):
    """2022 official = mean_RMSE (lower better); 2024 official = mean_pearson_r;
    Tier I = mean_spearman_r, the only fully admissible official metric."""
    t = d.set_index("Team Name")
    r = pd.DataFrame({
        "official_2022": t.mean_RMSE.rank(method="min"),
        "official_2024": (-t.mean_pearson_r).rank(method="min"),
        "tier_I": (-t.mean_spearman_r).rank(method="min")}).reset_index()
    r = r.rename(columns={"Team Name": "team"})
    r["move_2022_to_TierI"] = r.official_2022 - r.tier_I
    return r.sort_values("team")


if __name__ == "__main__":
    d = parse_S1()
    d.to_csv(f"{RES}/S1_parsed.csv", index=False)
    b = best_per_team(d)
    b.to_csv(f"{RES}/S1_best_per_team.csv", index=False)
    coverage(b).to_csv(f"{RES}/coverage_by_metric_class.csv", index=False)
    tierI(b).to_csv(f"{RES}/tierI_ranking.csv", index=False)
    print(f"S1_parsed.csv            {len(d)} submissions")
    print(f"S1_best_per_team.csv     {len(b)} teams")
    print(f"coverage_by_metric_class.csv  {len(METRICS)} metrics")
    print(f"tierI_ranking.csv        {len(b)} teams")
