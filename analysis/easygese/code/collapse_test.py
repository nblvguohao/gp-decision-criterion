"""Controlled test of the design-level claim (exploratory, 9 September 2026; reported in
Supplementary Section S16, not carried into the paper). The withdrawn rice and CAIGE panels
are skipped (analysis/crosscrop/code/scope.py).

Claim: metric disagreement requires the evaluation to retain within-environment
structure. Test it WITHIN each dataset by collapsing it -- averaging each
genotype across environments, exactly what a pooled cross-validation design does
-- and recomputing the r-vs-RMSE reversal rate on the same methods.
If the claim holds, reversal should fall towards the level EasyGeSe reports for
its pooled design (9.1%).
"""
import sys; sys.path.insert(0, "analysis/crosscrop/code")
import warnings, itertools, numpy as np, pandas as pd
import scope
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr

def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)
def rev(a,b):
    pr=list(itertools.combinations(range(len(a)),2))
    d1=np.array([a[i]-a[j] for i,j in pr]); d2=np.array([b[i]-b[j] for i,j in pr])
    ok=np.isfinite(d1)&np.isfinite(d2)
    return int((d1[ok]*d2[ok]<0).sum()), int(ok.sum())

SETS=[("rice","analysis/crosscrop/results/rice_panel_wide.csv",25),
      ("wheat","analysis/crosscrop/results/wheat_panel_wide.csv",25),
      ("common bean","analysis/crosscrop/results/bean_panel_wide.csv",25),
      ("spring wheat","analysis/crosscrop/results/ursn_panel_wide.csv",12),
      ("soybean","analysis/crosscrop/results/nust_panel_wide.csv",25)]
print("Same methods, same data, two evaluation designs\n")
print(f"{'dataset':<15}{'within-env design':>26}{'collapsed (pooled) design':>30}")
print(f"{'':<15}{'r vs RMSE reversal':>26}{'r vs RMSE reversal':>30}")
print("-"*74)
out=[]
for lab,path,mn in scope.kept(SETS):
    P=pd.read_csv(path)
    # --- design A: within-environment metrics, averaged over environments
    A_r=[];A_e=[]
    for m,d in P.groupby("method"):
        rs=[];es=[]
        for e,g in d.groupby("Env"):
            if len(g)<mn or g.p.nunique()<2 or g.y.nunique()<2: continue
            y=g.y.to_numpy(); p=g.p.to_numpy()
            rs.append(pearsonr(y,p)[0]); es.append(np.sqrt(np.mean((p-y)**2)))
        A_r.append(np.mean(rs)); A_e.append(-np.mean(es))
    kA,nA=rev(np.array(A_r),np.array(A_e)); lA,hA=wilson(kA,nA)
    # --- design B: collapse each genotype across environments, then pooled metrics
    B_r=[];B_e=[]
    for m,d in P.groupby("method"):
        c=d.groupby("k").agg(y=("y","mean"),p=("p","mean"))
        y=c.y.to_numpy(); p=c.p.to_numpy()
        B_r.append(pearsonr(y,p)[0] if len(np.unique(p))>1 else np.nan)
        B_e.append(-np.sqrt(np.mean((p-y)**2)))
    kB,nB=rev(np.array(B_r),np.array(B_e)); lB,hB=wilson(kB,nB)
    print(f"{lab:<15}{f'{kA/nA:.3f}  [{lA:.3f}, {hA:.3f}]':>26}{f'{kB/nB:.3f}  [{lB:.3f}, {hB:.3f}]':>30}")
    out.append((lab,kA/nA,kB/nB))
O=pd.DataFrame(out,columns=["ds","within","collapsed"])
print(f"\n  mean reversal, within-environment design : {O.within.mean():.3f}")
print(f"  mean reversal, collapsed design          : {O.collapsed.mean():.3f}")
print(f"  EasyGeSe (10 species, pooled by design)  : 0.091")
print(f"\n  collapsing reduces reversal in {int((O.collapsed<O.within).sum())} of {len(O)} datasets")
O.to_csv("analysis/easygese/results/collapse_test.csv",index=False)
