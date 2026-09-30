"""Is the admissibility criterion constructive as well as destructive?

(a) Do decision-oriented metrics proposed elsewhere -- NDCG@k (Blondel et al.
    2015), top-k selection gain, top-k hit rate, Kendall tau -- pass the
    invariance test that RMSE fails?
(b) If the leaderboard is ranked by admissible metrics only, is the ranking
    stable, or does the instability persist?
"""
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, kendalltau

def ndcg_at(y,p,k):
    o=np.argsort(-p)[:k]; g=y[o]-y.min()
    dcg=np.sum(g/np.log2(np.arange(2,len(o)+2)))
    b=np.sort(y)[::-1][:k]-y.min()
    idcg=np.sum(b/np.log2(np.arange(2,len(b)+2)))
    return dcg/idcg if idcg>0 else np.nan

def decision_metrics(d,frac=.10,min_n=25):
    per=[]
    for _,g in d.groupby("Env"):
        if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
        y=g.y.to_numpy(); p=g.p.to_numpy(); k=max(1,int(round(frac*len(g))))
        top=np.argsort(-p,kind="mergesort")[:k]; best=np.argsort(-y,kind="mergesort")[:k]
        per.append(dict(
            SG=(y[top].mean()-y.mean())/y.std(),
            hit=len(set(top)&set(best))/k,
            NDCG=ndcg_at(y,p,k),
            kendall=kendalltau(y,p)[0],
            pearson=pearsonr(y,p)[0],
            RMSE=np.sqrt(np.mean((p-y)**2)),
        ))
    D=pd.DataFrame(per)
    return {("mean_"+c):D[c].mean() for c in D.columns}

PAN={"maize (5 verified submissions)":None,
     "common bean":"analysis/crosscrop/results/bean_panel_wide.csv",
     "soybean":"analysis/crosscrop/results/nust_panel_wide.csv"}
obs=pd.read_csv("analysis/g2f_leaderboard/results/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
V=["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
   "EnBiSys_sub243568","NicheSquad_sub244985"]
mz=[]
for n in V:
    pr=pd.read_csv(f"analysis/g2f_leaderboard/results/{n}.csv").rename(columns={"Yield_Mg_ha":"p"})
    d=obs.merge(pr,on=["Env","Hybrid"]).dropna(subset=["y","p"]); d["method"]=n; mz.append(d)
MZ=pd.concat(mz,ignore_index=True)

def mono_transform(p,rng):
    """Random strictly increasing function of p: average ranks (ties stay tied), then a
    piecewise-linear map through six sorted random interior knots."""
    from scipy.stats import rankdata
    u=(rankdata(p,method="average")-0.5)/len(p)
    kn=np.sort(rng.uniform(0,1,6)); kn=np.r_[0,kn,1]
    vl=np.sort(rng.uniform(0,1,len(kn))); vl[0]=0; vl[-1]=1
    return np.interp(u,kn,vl)

print("(a) INVARIANCE of decision-oriented metrics, under a per-environment affine")
print("    transform and under any strictly increasing within-environment transform\n")
print(f"{'dataset':<28}{'metric':<12}{'affine':>13}{'monotone':>13}{'admissible?':>13}")
print("-"*80)
rows=[]
for kind in ("affine","monotone"):
    rng=np.random.default_rng(4)
    for lab,path in PAN.items():
        P = MZ if path is None else pd.read_csv(path)
        # all five verified submissions; for a panel, its first six parent methods
        ms=list(P.method.unique()) if path is None else sorted(m for m in P.method.unique() if "__" not in m)[:6]
        acc={}
        for s in ms:
            d=P[P.method==s][["Env","y","p"]]
            base=decision_metrics(d)
            for _ in range(40):
                d2=d.copy()
                if kind=="affine":
                    e=d.Env.unique()
                    a=pd.Series(rng.uniform(.5,2.,len(e)),index=e)
                    sd=d.groupby("Env").y.std(ddof=0).reindex(e).to_numpy()   # b_e ~ N(0, SD(y_e)^2)
                    b=pd.Series(rng.normal(0,1.0,len(e))*sd,index=e)
                    d2["p"]=d.Env.map(a).to_numpy()*d.p.to_numpy()+d.Env.map(b).to_numpy()
                else:
                    d2["p"]=d.groupby("Env").p.transform(lambda t: mono_transform(t.to_numpy(),rng))
                m=decision_metrics(d2)
                for k in base:
                    if np.isfinite(base[k]) and np.isfinite(m[k]):
                        acc.setdefault(k,[]).append(abs(m[k]-base[k])/max(abs(base[k]),1e-9))
        for k in ("mean_SG","mean_hit","mean_NDCG","mean_kendall","mean_pearson","mean_RMSE"):
            if k in acc:
                rows.append(dict(dataset=lab,metric=k.replace("mean_",""),kind=kind,
                                 median=float(np.median(acc[k]))))
C=pd.DataFrame(rows).pivot_table(index=["dataset","metric"],columns="kind",values="median").reset_index()
ORD={"SG":0,"hit":1,"NDCG":2,"kendall":3,"pearson":4,"RMSE":5}
C=C.sort_values(["dataset","metric"],key=lambda c:c.map(ORD) if c.name=="metric" else c)
for _,r in C.iterrows():
    ok="  YES" if max(r.affine,r.monotone)<1e-3 else "  no"
    print(f"{r.dataset:<28}{r.metric:<12}{r.affine:>13.2e}{r.monotone:>13.2e}{ok:>13}")
C.to_csv("analysis/g2f_leaderboard/results/decision_metrics_invariance.csv",index=False)
print("\n-> written analysis/g2f_leaderboard/results/decision_metrics_invariance.csv")
print(f"   RMSE under the affine transform ranges {C[C.metric=='RMSE'].affine.min():.2f}"
      f"-{C[C.metric=='RMSE'].affine.max():.2f} across the three datasets\n")
