"""Restore the calibration dimension the least-squares panel removed.

Every method in the base panel is fitted by least squares on the same data, so
all are well calibrated. Real competition entries are not: the G2F leaderboard
spans mean_RMSE 2.33-10.47 (CV 0.49) while the base NUST panel spans 12.5-14.1
(CV 0.02). Calibration is exactly the axis along which error-magnitude and
discrimination metrics diverge (Murphy decomposition), so a panel that is
homogeneous in calibration cannot exhibit the contrast under study.

Variants below apply location, scale and rank-noise distortions -- the same
failure modes real entries display -- and the panel is then checked against the
G2F spread before any conclusion is drawn.
"""
import numpy as np, pandas as pd
RES="analysis/crosscrop/results"
P=pd.read_csv(f"{RES}/nust_panel_predictions.csv")
base=P[~P.method.isin(["global_mean","env_index"])].copy()
rng=np.random.default_rng(3)
src=["ridge_pc50_lo","rf_large","gbm_deep","knn10","markeronly_pc100","reaction_norm","mlp","ens_all"]
new=[]
def add(tag, d, p):
    q=d.copy(); q["method"]=tag; q["p"]=p; new.append(q)

for s in src:
    d=base[base.method==s]
    y,p=d.y.to_numpy(),d.p.to_numpy()
    sd=p.std()
    add(f"{s}__shrunk",   d, p.mean()+0.25*(p-p.mean()))          # dispersion collapse
    add(f"{s}__inflated", d, p.mean()+3.0*(p-p.mean()))           # dispersion blow-up
    add(f"{s}__biased",   d, p+0.8*y.std())                        # systematic offset
for s in src[:6]:
    d=base[base.method==s]; p=d.p.to_numpy()
    for lvl in (0.5,1.5):
        add(f"{s}__noise{lvl}", d, p+rng.normal(0,lvl*p.std(),len(p)))
for s in src[:4]:
    d=base[base.method==s]; p=d.p.to_numpy()
    q=p.copy(); m=rng.random(len(q))<0.35; q[m]=rng.permutation(q[m])
    add(f"{s}__shuffled", d, q)
W=pd.concat([base]+new,ignore_index=True)
W.to_csv(f"{RES}/nust_panel_wide.csv",index=False)
print(f"base {base.method.nunique()} -> wide {W.method.nunique()} methods")
