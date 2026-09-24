"""Decision-invariance test.

Fingerprint claim (Metrics Reloaded style): the decision is "select the top-k
genotypes WITHIN each environment". That decision is unchanged by any
strictly increasing within-environment transform of the predictions.
=> A metric is admissible for this decision only if it is invariant to
   per-environment affine rescaling  yhat_e -> a_e * yhat_e + b_e,  a_e > 0.
This script applies random per-environment affine transforms to real submissions
and measures how much each official metric moves.
"""
import warnings, numpy as np, pandas as pd, glob, os
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress

RES="analysis/g2f_leaderboard/results"
obs=pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
VERIFIED=["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
          "EnBiSys_sub243568","NicheSquad_sub244985"]

def metrics(d):
    """22-metric panel, pooled and within-environment averaged."""
    out={}
    def blk(y,p,pre=""):
        e=p-y
        out[pre+"RMSE"]=np.sqrt(np.mean(e**2)); out[pre+"MAE"]=np.mean(np.abs(e))
        out[pre+"normalized_RMSE"]=out[pre+"RMSE"]/np.std(y); out[pre+"relative_RMSE"]=out[pre+"RMSE"]/np.mean(y)
        out[pre+"normalized_MAE"]=out[pre+"MAE"]/np.std(y);   out[pre+"realative_MAE"]=out[pre+"MAE"]/np.mean(y)
        out[pre+"r2_score"]=1-np.sum(e**2)/np.sum((y-y.mean())**2)
        r=pearsonr(y,p)[0] if len(np.unique(p))>1 else np.nan
        out[pre+"pearson_r"]=r; out[pre+"r2_pearson"]=r**2 if np.isfinite(r) else np.nan
        out[pre+"spearman_r"]=spearmanr(y,p)[0] if len(np.unique(p))>1 else np.nan
        out[pre+"lineRegressSlope"]=linregress(p,y).slope if len(np.unique(p))>1 else np.nan
    blk(d["y"].to_numpy(), d["p"].to_numpy())
    per=[]
    for _,g in d.groupby("Env"):
        if len(g)<3 or g["p"].nunique()<2 or g["y"].nunique()<2: continue
        t={}; y,p=g["y"].to_numpy(),g["p"].to_numpy(); e=p-y
        t["RMSE"]=np.sqrt(np.mean(e**2)); t["MAE"]=np.mean(np.abs(e))
        t["normalized_RMSE"]=t["RMSE"]/np.std(y); t["relative_RMSE"]=t["RMSE"]/np.mean(y)
        t["normalized_MAE"]=t["MAE"]/np.std(y);   t["realative_MAE"]=t["MAE"]/np.mean(y)
        t["r2_score"]=1-np.sum(e**2)/np.sum((y-y.mean())**2)
        r=pearsonr(y,p)[0]; t["pearson_r"]=r; t["r2_pearson"]=r**2
        t["spearman_r"]=spearmanr(y,p)[0]; t["lineRegressSlope"]=linregress(p,y).slope
        per.append(t)
    P=pd.DataFrame(per)
    for c in P.columns: out["mean_"+c]=P[c].mean()
    return out

rng=np.random.default_rng(11)
rows=[]
for name in VERIFIED:
    f=f"{RES}/{name}.csv"
    if not os.path.exists(f): continue
    pr=pd.read_csv(f).rename(columns={"Yield_Mg_ha":"p"})
    d=obs.merge(pr,on=["Env","Hybrid"]).dropna(subset=["y","p"])
    base=metrics(d)
    for rep in range(200):
        d2=d.copy()
        a=pd.Series(rng.uniform(.5,2.0,d["Env"].nunique()), index=d["Env"].unique())
        sd=d.groupby("Env").y.std(ddof=0).reindex(d["Env"].unique()).to_numpy()   # b_e ~ N(0, SD(y_e)^2)
        b=pd.Series(rng.normal(0,1.0,d["Env"].nunique())*sd, index=d["Env"].unique())
        d2["p"]=d["Env"].map(a).to_numpy()*d["p"].to_numpy()+d["Env"].map(b).to_numpy()
        m=metrics(d2)
        for k in base:
            if np.isfinite(base[k]) and np.isfinite(m[k]):
                rows.append((name,k,base[k],m[k]))
R=pd.DataFrame(rows,columns=["team","metric","base","perturbed"])
R["rel_change"]=np.abs(R.perturbed-R.base)/np.maximum(np.abs(R.base),1e-9)
S=R.groupby("metric")["rel_change"].agg(median="median",p95=lambda x:np.percentile(x,95)).reset_index()
S["invariant"]=S["p95"]<1e-8
S=S.sort_values("median")
print(f"per-environment affine perturbation  yhat_e -> a_e*yhat_e + b_e,  a_e~U(0.5,2), b_e~N(0,SD(y_e)^2)")
print(f"{len(VERIFIED)} verified submissions x 200 random perturbations\n")
print(f"{'metric':<24}{'median |rel change|':>21}{'  invariant?':>13}")
print("-"*60)
for _,r in S.iterrows():
    print(f"{r['metric']:<24}{r['median']:>21.2e}   {'YES' if r['invariant'] else 'no'}")
inv=set(S[S.invariant]['metric'])
print(f"\nINVARIANT metrics ({len(inv)}): {sorted(inv)}")
S.to_csv(f"{RES}/invariance_test.csv",index=False)
