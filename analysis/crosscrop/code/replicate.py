"""Replicate the G2F metric-structure analysis on NUST soybean."""
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress
RES="analysis/crosscrop/results"
P=pd.read_csv(f"{RES}/nust_panel_wide.csv")

def panel22(d):
    o={}
    def blk(y,p,pre=""):
        e=p-y
        o[pre+"RMSE"]=np.sqrt(np.mean(e**2)); o[pre+"MAE"]=np.mean(np.abs(e))
        o[pre+"normalized_RMSE"]=o[pre+"RMSE"]/np.std(y); o[pre+"relative_RMSE"]=o[pre+"RMSE"]/np.mean(y)
        o[pre+"normalized_MAE"]=o[pre+"MAE"]/np.std(y);   o[pre+"realative_MAE"]=o[pre+"MAE"]/np.mean(y)
        o[pre+"r2_score"]=1-np.sum(e**2)/np.sum((y-y.mean())**2)
        u=len(np.unique(p))>1
        r=pearsonr(y,p)[0] if u else np.nan
        o[pre+"pearson_r"]=r; o[pre+"r2_pearson"]=r**2 if np.isfinite(r) else np.nan
        o[pre+"spearman_r"]=spearmanr(y,p)[0] if u else np.nan
        o[pre+"lineRegressSlope"]=linregress(p,y).slope if u else np.nan
    blk(d.y.to_numpy(), d.p.to_numpy())
    per=[]
    for _,g in d.groupby("Env"):
        if len(g)<25 or g.p.nunique()<2 or g.y.nunique()<2: continue   # the paper's rule (Section 2.3)
        t={}; blk_y,blk_p=g.y.to_numpy(),g.p.to_numpy(); e=blk_p-blk_y
        t["RMSE"]=np.sqrt(np.mean(e**2)); t["MAE"]=np.mean(np.abs(e))
        t["normalized_RMSE"]=t["RMSE"]/np.std(blk_y); t["relative_RMSE"]=t["RMSE"]/np.mean(blk_y)
        t["normalized_MAE"]=t["MAE"]/np.std(blk_y);   t["realative_MAE"]=t["MAE"]/np.mean(blk_y)
        t["r2_score"]=1-np.sum(e**2)/np.sum((blk_y-blk_y.mean())**2)
        r=pearsonr(blk_y,blk_p)[0]; t["pearson_r"]=r; t["r2_pearson"]=r**2
        t["spearman_r"]=spearmanr(blk_y,blk_p)[0]; t["lineRegressSlope"]=linregress(blk_p,blk_y).slope
        per.append(t)
    D=pd.DataFrame(per)
    for c in D.columns: o["mean_"+c]=D[c].mean()
    return o

# ---------- metrics table: methods x 22 metrics (pooled over the 5 forward years)
rows=[]
for m,d in P.groupby("method"):
    r=panel22(d); r["method"]=m; rows.append(r)
T=pd.DataFrame(rows).set_index("method")
METRICS=[c for c in T.columns]
LOWER={"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
       "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE={"lineRegressSlope","mean_lineRegressSlope"}
def orient(c):
    v=T[c].to_numpy(float)
    return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)
S={m:orient(m) for m in METRICS}
M=len(T); names=T.index.to_numpy()
pairs=list(itertools.combinations(range(M),2))
def rev(a,b,idx=None):
    Pr=pairs if idx is None else [(i,j) for i,j in pairs if i in idx and j in idx]
    d1=np.array([a[i]-a[j] for i,j in Pr]); d2=np.array([b[i]-b[j] for i,j in Pr])
    ok=np.isfinite(d1)&np.isfinite(d2); d1,d2=d1[ok],d2[ok]
    return int((d1*d2<0).sum()), len(d1)
def wilson(k,N,z=1.96):
    p=k/N; d=1+z*z/N; c=(p+z*z/(2*N))/d; h=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/d
    return max(0,c-h),min(1,c+h)
print("="*74); print(f"NUST soybean — {M} methods, {P.Env.nunique()} environments, 5 forward years\n")
k,N=rev(S["mean_RMSE"],S["mean_pearson_r"]); lo,hi=wilson(k,N)
print(f"[1] pairwise reversal, mean_RMSE vs mean_pearson_r : {k}/{N} = {k/N:.3f}  95% CI [{lo:.3f},{hi:.3f}]")
top=set(np.argsort(-S["mean_RMSE"])[:10]); kt,Nt=rev(S["mean_RMSE"],S["mean_pearson_r"],idx=top)
lt,ht=wilson(kt,Nt); print(f"    restricted to top-10 methods                   : {kt}/{Nt} = {kt/Nt:.3f}  95% CI [{lt:.3f},{ht:.3f}]")

def rk(c):
    s=S[c]; o=np.argsort(-s,kind="mergesort"); r=np.empty(M,int); r[o]=np.arange(1,M+1); return r
R=pd.DataFrame({m:rk(m) for m in METRICS})
def conf(cal,a=.05):
    Rc=R[cal].to_numpy(); rh=np.median(Rc,axis=1); s=np.abs(Rc-rh[:,None]).ravel()
    n=len(s); kk=int(np.ceil((n+1)*(1-a))); return rh,np.sort(s)[min(kk,n)-1]
rh,q=conf(METRICS); w=np.clip(np.round(rh+q),1,M)-np.clip(np.round(rh-q),1,M)+1
print(f"\n[2] median 95% conformal rank-interval width      : {np.median(w):.0f} of {M} "
      f"({np.median(w)/M*100:.0f}% of the board)")
cov=[]
for h in METRICS:
    rr,qq=conf([m for m in METRICS if m!=h]); t=R[h].to_numpy()
    cov.append((h,np.mean((t>=rr-qq)&(t<=rr+qq))))
C=pd.DataFrame(cov,columns=["metric","coverage"])
def agg(m): return "within-env" if m.startswith("mean_") else "pooled"
def typ(m):
    if m in SLOPE: return "discrimination"
    if ("pearson" in m or "spearman" in m) and not m.endswith("r2_score"): return "discrimination"
    return "error-magnitude"
C["cell"]=C.metric.map(agg)+" x "+C.metric.map(typ)
KEY="within-env x discrimination"
we=C[C.cell==KEY].coverage; ot=C[C.cell!=KEY].coverage
print(f"\n[3] leave-one-metric-out coverage, mean           : {C.coverage.mean():.3f}")
print(f"    within-env x discrimination (n={len(we)})           : [{we.min():.2f}, {we.max():.2f}]")
print(f"    all other metrics           (n={len(ot)})          : [{ot.min():.2f}, {ot.max():.2f}]")
sep = we.max()<ot.min()
print(f"    SEPARATION: {'PERFECT — no overlap, gap %.3f'%(ot.min()-we.max()) if sep else 'OVERLAPPING'}")
C.sort_values("coverage").to_csv(f"{RES}/nust_coverage_wide.csv",index=False)
T.to_csv(f"{RES}/nust_metrics_wide.csv")
print("\n    lowest-coverage metrics:")
for _,r in C.sort_values("coverage").head(6).iterrows():
    print(f"      {r['metric']:<24}{r['coverage']:.2f}   {r['cell']}")
