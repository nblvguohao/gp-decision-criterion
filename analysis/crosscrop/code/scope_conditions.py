"""Are the failures explained by a property measurable IN ADVANCE?

Diagnostic 1 - outcome reliability. No scoring rule can beat the split-half
reliability of the outcome itself. If realised selection gain measured on half A
does not predict gain on half B, there is no stable method ordering to find and
the failure is not the metric's.

Diagnostic 2 - family coupling. If, across the methods present, within-
environment discrimination is strongly correlated with environment-level fit,
the two metric families agree and no taxonomy separation can appear.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr
import sys; sys.path.insert(0,"analysis/crosscrop/code")
from does_it_help import per_env, outcome_reliability

SETS=[("rice","analysis/crosscrop/results/rice_panel_wide.csv",25,True,False),
      ("common bean","analysis/crosscrop/results/bean_panel_wide.csv",25,True,True),
      ("spring wheat","analysis/crosscrop/results/ursn_panel_wide.csv",12,False,True),
      ("soybean","analysis/crosscrop/results/nust_panel_wide.csv",25,False,True),
      ("wheat","analysis/crosscrop/results/wheat_panel_wide.csv",25,True,None)]
rng=np.random.default_rng(0)
print("="*92)
print("DIAGNOSTIC 1 — split-half reliability of the OUTCOME (realised selection gain)\n")
print(f"{'dataset':<15}{'sep?':>6}{'outcome reliability':>22}{'best rule (S2c)':>18}{'ceiling reached':>18}")
print("-"*92)
S2C={"rice":22,"common bean":94,"spring wheat":83,"soybean":83,"wheat":None}
rel={}
for lab,path,mn,sep,_ in SETS:
    P=pd.read_csv(path); meths=sorted(P.method.unique())
    cache={m:per_env(P[P.method==m],mn) for m in meths}
    envs=sorted(set().union(*[set(v) for v in cache.values()]))
    rel[lab]=outcome_reliability(P,mn,np.random.default_rng(0))
    rs=[rel[lab]]
    b=S2C.get(lab)
    print(f"{lab:<15}{('yes' if sep else 'no'):>6}{np.mean(rs):>22.3f}"
          f"{(f'{b}%' if b is not None else 'n/a'):>18}"
          f"{(f'{b/ (np.mean(rs)*100) *100:.0f}%' if b is not None and np.mean(rs)>0 else 'n/a'):>18}")

print("\n"+"="*92)
print("DIAGNOSTIC 2 — coupling between within-environment discrimination and")
print("               environment-level fit, across the methods in each panel\n")
print(f"{'dataset':<15}{'separation?':>13}{'corr(within-env r, env-level fit)':>36}{'n methods':>11}")
print("-"*92)
for lab,path,mn,sep,_ in SETS:
    P=pd.read_csv(path); meths=sorted(P.method.unique())
    wi=[];ef=[]
    for m in meths:
        d=P[P.method==m]
        rs=[];
        for e,g in d.groupby("Env"):
            if len(g)<mn or g.p.nunique()<2 or g.y.nunique()<2: continue
            rs.append(pearsonr(g.y.to_numpy(),g.p.to_numpy())[0])
        em_p=d.groupby("Env").p.mean(); em_y=d.groupby("Env").y.mean()
        if len(rs)<5 or em_p.nunique()<3: continue
        wi.append(np.mean(rs)); ef.append(pearsonr(em_y.to_numpy(),em_p.to_numpy())[0])
    wi,ef=np.array(wi),np.array(ef); ok=np.isfinite(wi)&np.isfinite(ef)
    c=pearsonr(wi[ok],ef[ok])[0] if ok.sum()>5 else np.nan
    print(f"{lab:<15}{('YES' if sep else 'no' if sep is False else '?'):>13}{c:>36.3f}{int(ok.sum()):>11}")
print("""
Reading. Diagnostic 1 bounds what any scoring rule can achieve: a rule cannot
rank methods more reliably than the outcome ranks itself across environment
samples. Diagnostic 2 asks whether the two metric families are even separable in
a given panel.""")
