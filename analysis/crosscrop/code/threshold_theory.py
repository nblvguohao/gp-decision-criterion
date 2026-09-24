"""A normal-theory design curve for the surrogate threshold.

The empirical thresholds are single numbers tied to one crop and one method panel,
so a breeding programme cannot transfer them.
This asks what the threshold *should* be under a plain Gaussian model, as a
function of the three things a programme controls or knows: the number of
genotypes per environment n, the number of environments E, and the selection
fraction f.

Model. Within an environment, y ~ N(0,1) over n genotypes and each method
predicts yhat_m = r_m*y + sqrt(1-r_m^2)*e_m. Method errors share a fraction
lambda of their variance (lambda = 0, methods err independently; lambda -> 1,
methods err alike), which is the one feature of a real panel a pure accuracy
parameter cannot express. The simulated panel is then put through exactly the
estimator used on the data — mean within-environment Pearson r and mean realised
selection differential per pair, threshold from the logistic-through-one-half
fit of threshold_model.py (the dr at which the accuracy winner is also the field
winner with probability 0.95).

Writes analysis/crosscrop/results/threshold_theory.csv       (matched configurations)
       analysis/crosscrop/results/threshold_design.csv       (the design grid)
       analysis/crosscrop/results/threshold_theory_curve.csv (Fig. 5a)
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, sys, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
import threshold_model as tm

RES = "analysis/crosscrop/results"
rng = np.random.default_rng(7)


def simulate_panel(r_true, n, E, f, lam, rng):
    """Per-method mean within-environment r and mean realised differential."""
    M = len(r_true); k = max(1, int(round(f*n)))
    a = r_true[:, None]; b = np.sqrt(1 - r_true**2)[:, None]
    Rsum = np.zeros(M); Gsum = np.zeros(M)
    for _ in range(E):
        y = rng.standard_normal(n)
        u = rng.standard_normal(n)
        v = rng.standard_normal((M, n))
        e = np.sqrt(lam)*u + np.sqrt(1-lam)*v
        P = a*y + b*e
        yc = y - y.mean()
        Pc = P - P.mean(axis=1, keepdims=True)
        Rsum += (Pc @ yc) / (np.linalg.norm(Pc, axis=1)*np.linalg.norm(yc) + 1e-12)
        top = np.argpartition(-P, k-1, axis=1)[:, :k]
        Gsum += (y[top].mean(axis=1) - y.mean()) / y.std()
    return Rsum/E, Gsum/E


def sim_pairs(r_true, n, E, f, lam, rng):
    rbar, gbar = simulate_panel(r_true, n, E, f, lam, rng)
    i, j = np.triu_indices(len(r_true), 1)
    dr = rbar[j] - rbar[i]; dg = gbar[j] - gbar[i]
    s = np.where(dr < 0, -1.0, 1.0)
    return dr * s, dg * s


def theory_threshold(r_true, n, E, f, lam, reps, rng):
    return np.array([tm.threshold(*sim_pairs(r_true, n, E, f, lam, rng)) for _ in range(reps)], float)


# ---------------------------------------------------------------- part 1
# Each panel matched on its own design (parent methods, genotypes per environment,
# environments, accuracy range), and the G2F competition design, whose five verified
# submissions are too few to estimate a threshold directly.
SETS = [("common bean",  f"{RES}/bean_panel_wide.csv", 25),
        ("spring wheat", f"{RES}/ursn_panel_wide.csv", 12),
        ("soybean",      f"{RES}/nust_panel_wide.csv", 25)]
FRAC = 0.10
REPS = 40
EMP = pd.read_csv(f"{RES}/threshold_estimates.csv")
EMP = EMP[(EMP.methods == "parent methods") & (EMP.frac == FRAC)].set_index("dataset")


def design(P, min_n):
    g = P.groupby(["method", "Env"]).size().rename("n").reset_index(); g = g[g.n >= min_n]
    rr = []
    for m, d in P.groupby("method"):
        v = [np.corrcoef(gg.y, gg.p)[0, 1] for e, gg in d.groupby("Env")
             if len(gg) >= min_n and gg.p.nunique() > 1 and gg.y.nunique() > 1]
        if v: rr.append(np.mean(v))
    lo, hi = np.percentile(rr, [2.5, 97.5])
    return P.method.nunique(), int(round(g.n.median())), int(round(g.groupby("method").size().median())), lo, hi


configs = []
for lab, path, min_n in scope.kept(SETS):
    P = pd.read_csv(path); P = P[~P.method.str.contains("__")]
    configs.append((lab,) + design(P, min_n) + (float(EMP.loc[lab, "threshold"]),))
Gt = pd.read_csv("analysis/g2f_leaderboard/results/S1_best_per_team.csv").dropna(subset=["mean_pearson_r"])
glo, ghi = np.percentile(Gt.mean_pearson_r, [2.5, 97.5])
configs.append(("maize (G2F design)", len(Gt), 445, 23, glo, ghi, np.nan))

print("=" * 94)
print("Matched configurations: the empirical threshold against its normal-theory counterpart\n")
print(f"{'dataset':<22}{'M':>4}{'n/env':>7}{'env':>5}{'r range':>14}{'empirical':>11}"
      f"{'theory l=0':>12}{'l=0.5':>9}{'l=0.8':>9}")
rows = []
for lab, M, n_med, E_med, lo, hi, emp in configs:
    r_true = np.clip(np.linspace(lo, hi, M), 0.01, 0.95)
    th = {lam: theory_threshold(r_true, n_med, E_med, FRAC, lam, REPS, rng) for lam in (0.0, 0.5, 0.8)}
    med = {lam: float(np.nanmedian(v)) for lam, v in th.items()}
    print(f"{lab:<22}{M:>4}{n_med:>7}{E_med:>5}{f'{lo:.2f}–{hi:.2f}':>14}"
          f"{('—' if np.isnan(emp) else f'{emp:.3f}'):>11}{med[0.0]:>12.3f}{med[0.5]:>9.3f}{med[0.8]:>9.3f}")
    rows.append(dict(dataset=lab, M=M, n_per_env=n_med, n_env=E_med, r_lo=lo, r_hi=hi, empirical=emp,
                     theory_lam0=med[0.0], theory_lam05=med[0.5], theory_lam08=med[0.8],
                     ratio_lam05=emp / med[0.5] if np.isfinite(emp) else np.nan))
TT = pd.DataFrame(rows)
ratios = TT.ratio_lam05.dropna()
mz = TT[TT.dataset == "maize (G2F design)"].iloc[0]
TT.loc[TT.dataset == "maize (G2F design)", "transfer_lo"] = mz.theory_lam05 * ratios.min()
TT.loc[TT.dataset == "maize (G2F design)", "transfer_hi"] = mz.theory_lam05 * ratios.max()
TT.to_csv(f"{RES}/threshold_theory.csv", index=False)
print(f"\nobserved / theory (lambda = 0.5): {ratios.min():.2f}–{ratios.max():.2f}; "
      f"maize transfer {mz.theory_lam05 * ratios.min():.3f}–{mz.theory_lam05 * ratios.max():.3f}")

# ---------------------------------------------------------------- part 2
# The design grid a breeding programme can read its own number off.
print("\n" + "="*94)
print("Design grid: threshold dr* under the Gaussian model (lambda = 0.5, 40 methods,")
print("accuracies spread over 0.05–0.45)\n")
GRID_R = np.linspace(0.05, 0.45, 40)
grid = []
for f in (0.05, 0.10, 0.20):
    for n in (25, 50, 100, 400):
        for E in (10, 20, 50, 100, 200):
            v = theory_threshold(GRID_R, n, E, f, 0.5, 25, rng)
            grid.append(dict(f=f, n_per_env=n, n_env=E,
                             dr_star=np.nanmedian(v), nan_frac=np.mean(np.isnan(v))))
G = pd.DataFrame(grid)
G.to_csv(f"{RES}/threshold_design.csv", index=False)
for f, s in G.groupby("f"):
    print(f"  selection fraction f = {f:.2f}")
    piv = s.pivot(index="n_per_env", columns="n_env", values="dr_star")
    print(piv.round(3).to_string(), "\n")
print(f"wrote {RES}/threshold_theory.csv and {RES}/threshold_design.csv")

# ---------------------------------------------------------------- part 3
# Fig. 5a: fitted agreement curves on a common grid of dr, observed (parent methods,
# logistic through one-half) and the matched Gaussian model (lambda = 0.5, whose slope is
# logit(0.95) / threshold), for every configuration; maize has the Gaussian curve only.
grid_dr = np.linspace(0, 0.30, 61); C = []
for _, t in TT.iterrows():
    b_emp = float(EMP.loc[t.dataset, "beta"]) if t.dataset in EMP.index else np.nan
    b_th = tm.LOGIT95 / t.theory_lam05
    C.append(pd.DataFrame(dict(dataset=t.dataset, dr=grid_dr,
                               p_observed=1 / (1 + np.exp(-b_emp * grid_dr)),
                               p_gaussian=1 / (1 + np.exp(-b_th * grid_dr)))))
pd.concat(C).to_csv(f"{RES}/threshold_theory_curve.csv", index=False)
print(f"wrote {RES}/threshold_theory_curve.csv")
