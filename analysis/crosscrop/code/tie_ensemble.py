"""Does averaging the methods a trial cannot separate select better material than picking one?

Rules compared out of sample (environments split at random into halves; the rule is fitted on
half A and its realised selection differential at 10 % selection measured on half B, as in the
outcome test of Section 2.4):

  leader       the single method with the highest mean within-environment Pearson r on half A
  rmse         the method with the lowest mean RMSE on half A
  tie_ens      equal-weight ensemble of the methods tied with the leader on half A: those within the
               resolvable gap k/sqrt(N) (N = genotype-environment cells per method in half A) of the
               best on either within-environment correlation, the rule GPverdict reports
  top3_ens     equal-weight ensemble of the three best by mean Pearson r (fixed-size comparison)
  all_ens      equal-weight ensemble of every base method (naive baseline)

Ensembles average predictions standardised within each environment, so a method's calibration
cannot weight it; the ensemble's selections are then scored like any method's. Base methods only
(the miscalibration variants of the panels are affine copies and are left out).

Reported per dataset: mean realised selection differential on half B (phenotypic SD) of each
rule, recovery (rule - average method)/(best single method on half B - average method), and the
paired difference tie_ens - leader with a 95 % interval from environment-cluster bootstrap.

Writes analysis/crosscrop/results/tie_ensemble.csv
"""
import sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code"); sys.path.insert(0, "tool")
from gpverdict.core import resolvable_gap

FRAC = 0.10
XC = "analysis/crosscrop/results"
RULES = ["leader", "rmse", "tie_ens", "top3_ens", "all_ens"]


def prepare(P, min_n):
    """Per environment: standardised predictions (genotypes x methods), standardised y, and per-method stats."""
    P = P[~P.method.str.contains("__")]
    meths = sorted(P.method.unique()); envs = []
    for e, g in P.groupby("Env"):
        W = g.pivot_table(index="k", columns="method", values="p").reindex(columns=meths)
        y = g.groupby("k").y.first().reindex(W.index)
        W = W.dropna(axis=0, how="any")
        y = y.loc[W.index].to_numpy(float)
        if len(y) < min_n or np.std(y) == 0:
            continue
        X = W.to_numpy(float)
        sd = X.std(0); sd[sd == 0] = np.nan
        Z = (X - X.mean(0)) / sd
        yz = (y - y.mean()) / y.std()
        k = max(1, int(round(FRAC * len(y))))
        with np.errstate(invalid="ignore"):
            pear = np.array([np.corrcoef(y, X[:, j])[0, 1] if np.isfinite(sd[j]) else np.nan for j in range(X.shape[1])])
            rk = lambda a: pd.Series(a).rank().to_numpy()
            spear = np.array([np.corrcoef(rk(y), rk(X[:, j]))[0, 1] if np.isfinite(sd[j]) else np.nan for j in range(X.shape[1])])
        rmse = np.sqrt(((X - y[:, None]) ** 2).mean(0))
        envs.append(dict(Z=Z, yz=yz, k=k, n=len(y), pear=pear, spear=spear, rmse=rmse))
    return meths, envs


def gain(env, score):
    s = np.where(np.isfinite(score), score, -np.inf)
    top = np.argsort(-s, kind="mergesort")[:env["k"]]
    return env["yz"][top].mean()


def ens_gain(env, idx):
    return gain(env, np.nanmean(env["Z"][:, idx], axis=1))


def fit_rules(envs, A):
    pear = np.nanmean([envs[i]["pear"] for i in A], axis=0)
    spear = np.nanmean([envs[i]["spear"] for i in A], axis=0)
    rmse = np.nanmean([envs[i]["rmse"] for i in A], axis=0)
    cells = sum(envs[i]["n"] for i in A)
    gap = resolvable_gap(cells, FRAC)
    lead = int(np.nanargmax(pear))
    tied = np.where((np.nanmax(pear) - pear <= gap) | (np.nanmax(spear) - spear <= gap))[0]
    return dict(leader=[lead], rmse=[int(np.nanargmin(rmse))], tie_ens=list(tied),
                top3_ens=list(np.argsort(-np.nan_to_num(pear, nan=-9))[:3]), all_ens=list(range(len(pear)))), gap


def evaluate(envs, A, B):
    rules, gap = fit_rules(envs, A)
    single = np.array([[gain(envs[i], envs[i]["Z"][:, j]) for j in range(envs[i]["Z"].shape[1])] for i in B]).mean(0)
    out = {r: float(np.mean([ens_gain(envs[i], idx) if len(idx) > 1 else gain(envs[i], envs[i]["Z"][:, idx[0]]) for i in B]))
           for r, idx in rules.items()}
    out.update(oracle=float(np.nanmax(single)), average=float(np.nanmean(single)), n_tied=len(rules["tie_ens"]), gap=gap)
    return out


def run(P, min_n, label, n_splits=300, n_boot=200, boot_splits=20, seed=20260927):
    meths, envs = prepare(P, min_n)
    E = len(envs); rng = np.random.default_rng(seed)
    pts = []
    for _ in range(n_splits):
        e = rng.permutation(E); pts.append(evaluate(envs, e[:E // 2], e[E // 2:]))
    T = pd.DataFrame(pts)
    diffs = []
    for _ in range(n_boot):
        b = rng.integers(0, E, E); u = np.unique(b); acc = []
        for _ in range(boot_splits):
            p = rng.permutation(u); A = [i for i in b if i in set(p[:len(p) // 2])]; Bh = [i for i in b if i in set(p[len(p) // 2:])]
            if len(set(A)) < 2 or len(set(Bh)) < 2:
                continue
            r = evaluate(envs, A, Bh); acc.append((r["tie_ens"] - r["leader"], r["tie_ens"] - r["rmse"]))
        if acc:
            diffs.append(np.mean(acc, axis=0))
    D = np.array(diffs)
    row = dict(dataset=label, n_methods=len(meths), n_env=E, mean_tied=T.n_tied.mean(), gap=T.gap.mean(),
               oracle=T.oracle.mean(), average=T.average.mean())
    for r in RULES:
        row[f"gain_{r}"] = T[r].mean()
        row[f"recovery_{r}"] = float(((T[r] - T.average) / (T.oracle - T.average)).mean())
    row.update(tie_minus_leader=float((T.tie_ens - T.leader).mean()),
               tie_minus_leader_lo=float(np.percentile(D[:, 0], 2.5)), tie_minus_leader_hi=float(np.percentile(D[:, 0], 97.5)),
               tie_minus_rmse=float((T.tie_ens - T.rmse).mean()),
               tie_minus_rmse_lo=float(np.percentile(D[:, 1], 2.5)), tie_minus_rmse_hi=float(np.percentile(D[:, 1], 97.5)))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return row


if __name__ == "__main__":
    SETS = [("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12),
            ("soybean", "nust_panel_wide.csv", 25), ("CUBIC ear weight", "china_panel_wide.csv", 25)]
    rows = [run(pd.read_csv(f"{XC}/{f}", usecols=["Env", "k", "y", "p", "method"]), mn, lab) for lab, f, mn in SETS]
    pd.DataFrame(rows).to_csv(f"{XC}/tie_ensemble.csv", index=False)
    print(f"wrote {XC}/tie_ensemble.csv")
