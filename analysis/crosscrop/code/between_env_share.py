"""Share of the observed variance that lies between environments, per dataset.

Supplementary Section S3 uses this to test one candidate explanation for why the
metric-predictability separation holds in some datasets and not in others: if the
separation followed the between-environment share, soybean (where it fails) would have
to be lower than maize (where it holds), and it is not.

The share is SS_between / SS_total of the observed values over the environments that
enter the analysis (for maize, the 2022 answer key; for a panel, the environments the
metric table keeps, i.e. at least min_n genotypes).

Writes analysis/crosscrop/results/between_env_share.csv
"""
import sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
import scope

RES = "analysis/crosscrop/results"
G = "analysis/g2f_leaderboard/results"
SETS = [("common bean", f"{RES}/bean_panel_wide.csv", 25),
        ("spring wheat", f"{RES}/ursn_panel_wide.csv", 12),
        ("soybean", f"{RES}/nust_panel_wide.csv", 25)]


def share(y, env):
    y = np.asarray(y, float); env = np.asarray(env)
    ok = np.isfinite(y); y, env = y[ok], env[ok]
    m = y.mean()
    sst = ((y - m) ** 2).sum()
    d = pd.DataFrame(dict(y=y, e=env)).groupby("e").y
    ssb = (d.count() * (d.mean() - m) ** 2).sum()
    return 100 * ssb / sst, len(np.unique(env)), len(y)


rows = []
obs = pd.read_csv(f"{G}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
s, ne, n = share(obs.y, obs.Env)
rows.append(dict(dataset="maize", scope="2022 answer key", between_env_pct=s, n_env=ne, n_obs=n))
for lab, path, mn in scope.kept(SETS):
    P = pd.read_csv(path)
    one = P[P.method == P.method.iloc[0]]                    # one method: the observed values
    keep = one.groupby("Env").k.size()
    sub = one[one.Env.isin(keep[keep >= mn].index)]
    s, ne, n = share(sub.y, sub.Env)
    rows.append(dict(dataset=lab, scope=f"environments with >= {mn} genotypes", between_env_pct=s, n_env=ne, n_obs=n))
    s2, ne2, n2 = share(one.y, one.Env)
    rows.append(dict(dataset=lab, scope="all environments in the panel", between_env_pct=s2, n_env=ne2, n_obs=n2))
T = pd.DataFrame(rows)
T.to_csv(f"{RES}/between_env_share.csv", index=False)
print(T.round(2).to_string(index=False))
