"""Build the cross-dataset summary behind Table 2 and Fig. 3.

Every value in that table used to be typed into the figure script by hand, so it
could not be checked and did not change when the analysis did. This script
computes all of them, through the canonical functions in analyse_species.py and
does_it_help.py, and writes:

    analysis/crosscrop/results/cross_dataset_summary.csv

Run from the repository root:  python analysis/crosscrop/code/build_summary.py
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
from analyse_species import metric_table, reversal_ci, rank_interval_width, METRICS, orient
from does_it_help import outcome_reliability

RES  = "analysis/crosscrop/results"
GRES = "analysis/g2f_leaderboard/results"

# min_n follows Methods 2.3: environments with fewer than 25 genotypes are
# skipped, relaxed to 12 for spring wheat, whose nursery design gives a median
# of 16 lines per environment.
SETS = [("rice",         "IRRI elite", f"{RES}/rice_panel_wide.csv",  25),
        ("wheat",        "CAIGE",      f"{RES}/wheat_panel_wide.csv", 25),
        ("common bean",  "VEF",        f"{RES}/bean_panel_wide.csv",  25),
        ("spring wheat", "URSN",       f"{RES}/ursn_panel_wide.csv",  12),
        ("soybean",      "NUST",       f"{RES}/nust_panel_wide.csv",  25)]

if __name__ == "__main__":
    rows = []

    # ---- maize: published organiser values only, no panel of ours ----
    G = pd.read_csv(f"{GRES}/S1_best_per_team.csv")
    G = G[G["Team Name"].notna()]
    T = G.set_index("Team Name")[METRICS]
    pt, lo, hi, M = reversal_ci(T, np.random.default_rng(0))
    w, _ = rank_interval_width(T)
    rows.append(dict(dataset="maize", source="G2F 2022", methods=M, envs=23,
                     genos_per_env=445, reversal=pt, ci_lo=lo, ci_hi=hi,
                     width=w, width_frac=w / M, threshold=np.nan, reliability=np.nan))
    print(f"{'maize':14s} methods={M:3d} envs={23:4d} reversal={pt:.3f} "
          f"[{lo:.3f},{hi:.3f}] width={w:.0f}/{M}")

    # the top-ten subset, with a clustered interval of its own so that Fig. 2B
    # plots two intervals of the same kind rather than mixing Wilson with bootstrap
    top10 = T.assign(_o=orient(T, "mean_RMSE")).sort_values("_o", ascending=False).head(10)[METRICS]
    tp, tlo, thi, tn = reversal_ci(top10, np.random.default_rng(0))
    pd.DataFrame([dict(subset="top10", rate=tp, ci_lo=tlo, ci_hi=thi, n=tn)]).to_csv(
        f"{GRES}/top10_reversal.csv", index=False)
    print(f"{'  top 10':14s} methods={tn:3d}          reversal={tp:.3f} [{tlo:.3f},{thi:.3f}]")

    # ---- the five method panels ----
    for key, src, path, mn in scope.kept(SETS):
        P = pd.read_csv(path)
        T = metric_table(P, mn)
        pt, lo, hi, M = reversal_ci(T, np.random.default_rng(0))
        w, _ = rank_interval_width(T)
        # surrogate threshold: logistic estimator, parent methods, f = 0.10 (threshold_estimate.py)
        TE = pd.read_csv(f"{RES}/threshold_estimates.csv")
        te = TE[(TE.dataset == key) & (TE.methods == "parent methods") & (TE.frac == 0.10)]
        thr = float(te.threshold.iloc[0]) if len(te) else np.nan
        rel = outcome_reliability(P, mn, np.random.default_rng(0))
        one = P[P.method == P.method.iloc[0]]
        ge = int(one.groupby("Env").k.nunique().median())
        rows.append(dict(dataset=key, source=src, methods=M, envs=one.Env.nunique(),
                         genos_per_env=ge, reversal=pt, ci_lo=lo, ci_hi=hi,
                         width=w, width_frac=w / M, threshold=thr, reliability=rel))
        print(f"{key:14s} methods={M:3d} envs={one.Env.nunique():4d} reversal={pt:.3f} "
              f"[{lo:.3f},{hi:.3f}] width={w:.0f}/{M} "
              f"thr={thr:.3f} rel={rel:.3f}" if np.isfinite(thr) else
              f"{key:14s} methods={M:3d} envs={one.Env.nunique():4d} reversal={pt:.3f} "
              f"[{lo:.3f},{hi:.3f}] width={w:.0f}/{M} thr=n.r.  rel={rel:.3f}")

    pd.DataFrame(rows).to_csv(f"{RES}/cross_dataset_summary.csv", index=False)
    print(f"\n-> written {RES}/cross_dataset_summary.csv")
