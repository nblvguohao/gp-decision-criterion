"""Run the full metric-validity battery on any (Env, k, y, p, method) panel."""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, linregress, norm

LOWER={"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
       "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE={"lineRegressSlope","mean_lineRegressSlope"}
def blk(y,p,o,pre=""):
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
def panel22(d,min_n=15):
    o={}; blk(d.y.to_numpy(),d.p.to_numpy(),o)
    per=[]
    for _,g in d.groupby("Env"):
        if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
        t={}; blk(g.y.to_numpy(),g.p.to_numpy(),t); per.append(t)
    D=pd.DataFrame(per)
    for c in D.columns: o["mean_"+c]=D[c].mean()
    return o
BASE=["RMSE","MAE","normalized_RMSE","relative_RMSE","normalized_MAE","realative_MAE",
      "r2_score","pearson_r","r2_pearson","spearman_r","lineRegressSlope"]
METRICS=BASE+["mean_"+c for c in BASE]

def orient(T,c):
    """Higher is better. Slopes are scored as -|slope-1|."""
    v=T[c].to_numpy(float)
    return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)

def rev_from(a,b,idx):
    """Reversal rate between two oriented score vectors over the methods in idx.
    Self-pairs are excluded so a bootstrap resample cannot manufacture agreement."""
    pr=[(i,j) for i,j in itertools.combinations(range(len(idx)),2) if idx[i]!=idx[j]]
    d1=np.array([a[idx[i]]-a[idx[j]] for i,j in pr])
    d2=np.array([b[idx[i]]-b[idx[j]] for i,j in pr])
    ok=scope.untied(d1,d2)          # a pair tied on either metric cannot reverse
    return (d1[ok]*d2[ok]<0).sum()/max(ok.sum(),1)

def families(T):
    """Method families: a parent method and its miscalibration variants ("parent__variant")
    are one cluster. Competition teams are each their own family."""
    fam=pd.Series([str(m).split("__")[0] for m in T.index])
    return [np.flatnonzero((fam==f).to_numpy()) for f in pd.unique(fam)]

def reversal_ci(T,rng,B=2000,m1="mean_RMSE",m2="mean_pearson_r"):
    """THE canonical reversal rate and its 95% interval, resampling method families.
    Pairs tied on either metric are excluded (scope.untied). Every table and figure in
    the paper uses this function; do not re-implement it."""
    a,b=orient(T,m1),orient(T,m2); n=len(T)
    pt=rev_from(a,b,list(range(n)))
    F=families(T)
    bs=[rev_from(a,b,np.concatenate([F[k] for k in rng.integers(0,len(F),len(F))])) for _ in range(B)]
    lo,hi=np.percentile(bs,[2.5,97.5])
    return float(pt),float(lo),float(hi),n

def rank_interval_width(T):
    """Median width of the 95% rank interval across the 22 metrics: a common half-width,
    the 95th percentile of |rank - median rank| pooled over method-metric cells (fitted
    and evaluated on the same cells, so conformal-style, not split-conformal)."""
    M=len(T)
    def rk(m):
        o=np.argsort(-orient(T,m),kind="mergesort"); r=np.empty(M,int); r[o]=np.arange(1,M+1); return r
    R=pd.DataFrame({m:rk(m) for m in METRICS})
    Rc=R.to_numpy(); rh=np.median(Rc,axis=1)
    s=np.sort(np.abs(Rc-rh[:,None]).ravel()); n=len(s)
    q=s[min(int(np.ceil((n+1)*.95)),n)-1]
    w=np.clip(np.round(rh+q),1,M)-np.clip(np.round(rh-q),1,M)+1
    return float(np.median(w)),M

def metric_table(P,min_n):
    """(Env,k,y,p,method) panel -> methods x 22 metrics."""
    return pd.DataFrame([{**panel22(d,min_n),"method":m} for m,d in P.groupby("method")]).set_index("method")

def wilson(k,N,z=1.96):
    p=k/N; d=1+z*z/N; c=(p+z*z/(2*N))/d; h=z*np.sqrt(p*(1-p)/N+z*z/(4*N*N))/d
    return max(0,c-h),min(1,c+h)
def i_k(f): x=norm.ppf(1-f); return norm.pdf(x)/f

def run(path,label,min_n=15,frac=.10):
    P=pd.read_csv(path)
    print("="*76); print(f"{label}   {P.method.nunique()} methods · {P.Env.nunique()} environments · {len(P)} rows")
    T=pd.DataFrame([{**panel22(d,min_n),"method":m} for m,d in P.groupby("method")]).set_index("method")
    T=T.dropna(axis=1,how="all"); METRICS=list(T.columns); M=len(T)
    def orient(c):
        v=T[c].to_numpy(float)
        return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)
    S={m:orient(m) for m in METRICS}
    pairs=list(itertools.combinations(range(M),2))
    def rev(a,b,idx=None):
        Pr=pairs if idx is None else [(i,j) for i,j in pairs if i in idx and j in idx]
        d1=np.array([a[i]-a[j] for i,j in Pr]); d2=np.array([b[i]-b[j] for i,j in Pr])
        ok=np.isfinite(d1)&np.isfinite(d2); return int((d1[ok]*d2[ok]<0).sum()), int(ok.sum())
    k,N=rev(S["mean_RMSE"],S["mean_pearson_r"]); lo,hi=wilson(k,N)
    print(f"  [1] reversal mean_RMSE vs mean_pearson_r : {k/N:.3f}  [{lo:.3f},{hi:.3f}]  ({k}/{N})")
    def rk(c):
        s=S[c]; o=np.argsort(-s,kind="mergesort"); r=np.empty(M,int); r[o]=np.arange(1,M+1); return r
    R=pd.DataFrame({m:rk(m) for m in METRICS})
    def conf(cal,a=.05):
        Rc=R[cal].to_numpy(); rh=np.median(Rc,axis=1); s=np.abs(Rc-rh[:,None]).ravel()
        n=len(s); kk=int(np.ceil((n+1)*(1-a))); return rh,np.sort(s)[min(kk,n)-1]
    rh,q=conf(METRICS); w=np.clip(np.round(rh+q),1,M)-np.clip(np.round(rh-q),1,M)+1
    print(f"  [2] median 95% rank-interval width       : {np.median(w):.0f} of {M} ({np.median(w)/M*100:.0f}%)")
    cov=pd.DataFrame([(h,np.mean((R[h].to_numpy()>=conf([m for m in METRICS if m!=h])[0]-conf([m for m in METRICS if m!=h])[1])
                      &(R[h].to_numpy()<=conf([m for m in METRICS if m!=h])[0]+conf([m for m in METRICS if m!=h])[1])))
                      for h in METRICS],columns=["metric","coverage"])
    agg=lambda m:"within-env" if m.startswith("mean_") else "pooled"
    typ=lambda m:"discrimination" if (m in SLOPE or (("pearson" in m or "spearman" in m) and not m.endswith("r2_score"))) else "error-magnitude"
    cov["cell"]=cov.metric.map(agg)+" x "+cov.metric.map(typ)
    KEY="within-env x discrimination"
    we=cov[cov.cell==KEY].coverage; ot=cov[cov.cell!=KEY].coverage
    sep=we.max()<ot.min()
    print(f"  [3] coverage: within-env×discrim [{we.min():.2f},{we.max():.2f}]  others [{ot.min():.2f},{ot.max():.2f}]"
          f"  -> {'SEPARATED (gap %.3f)'%(ot.min()-we.max()) if sep else 'overlapping'}")
    # invariance
    rng=np.random.default_rng(9); rr=[]
    for s in list(P.method.unique())[:6]:
        d=P[P.method==s][["Env","y","p"]]; base=panel22(d,min_n)
        for _ in range(60):
            e=d.Env.unique()
            a=pd.Series(rng.uniform(.5,2.,len(e)),index=e); b=pd.Series(rng.normal(0,d.y.std(),len(e)),index=e)
            d2=d.copy(); d2["p"]=d.Env.map(a).to_numpy()*d.p.to_numpy()+d.Env.map(b).to_numpy()
            m=panel22(d2,min_n)
            for kk in base:
                if np.isfinite(base[kk]) and np.isfinite(m[kk]):
                    rr.append((kk,abs(m[kk]-base[kk])/max(abs(base[kk]),1e-9)))
    IV=pd.DataFrame(rr,columns=["metric","rel"]).groupby("metric").rel.median()
    # tolerance 1e-3: Pearson-family drift is ~1e-16, Spearman ~1e-5 (floating-point
    # reordering of tied predictions), every inadmissible metric moves >0.07.
    inv=sorted(IV[IV<1e-3].index)
    lo=IV[IV<1e-3].max() if len(inv) else np.nan; hi=IV[IV>=1e-3].min() if len(inv)<len(IV) else np.nan
    print(f"  [4] invariant metrics                    : {len(inv)} of {len(IV)}  {inv}")
    print(f"      max drift among them {lo:.1e}  vs  min change among the rest {hi:.1e}  "
          f"({np.log10(hi/lo):.1f} orders of magnitude apart)")
    # threshold
    cache={}
    for m,d in P.groupby("method"):
        Rr={};Gg={};Ss={}
        for e,g in d.groupby("Env"):
            if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
            y=g.y.to_numpy(); p=g.p.to_numpy(); kk=max(1,int(round(frac*len(g))))
            t=np.argsort(-p)[:kk]
            Rr[e]=pearsonr(y,p)[0]; Gg[e]=(y[t].mean()-y.mean())/y.std(); Ss[e]=frozenset(g.k.to_numpy()[t])
        cache[m]=(Rr,Gg,Ss)
    rows=[]
    for a,b in itertools.combinations(sorted(cache),2):
        Ra,Ga,Sa=cache[a]; Rb,Gb,Sb=cache[b]; env=sorted(set(Ra)&set(Rb))
        if len(env)<max(8,min_n//2): continue
        rows.append((np.mean([Rb[e]-Ra[e] for e in env]),np.mean([Gb[e]-Ga[e] for e in env]),
                     np.mean([len(Sa[e]&Sb[e])/max(len(Sa[e]),1) for e in env])))
    D=scope.drop_ties(pd.DataFrame(rows,columns=["dr","dg","overlap"]))
    f=D.dr<0; D.loc[f,["dr","dg"]]*=-1; D["agree"]=D.dg>0
    eq=D[D.dr<0.01]
    print(f"  [5] at equal accuracy (Δr<0.01, {len(eq)} pairs): P(field=accuracy winner) "
          f"{eq.agree.mean()*100:.0f}%, shortlist overlap {eq.overlap.mean()*100:.0f}%")
    thr=None
    for lo_ in np.arange(0,.30,.01):
        s=D[(D.dr>=lo_)&(D.dr<lo_+.04)]
        if len(s)>=15 and s.agree.mean()>=.95: thr=lo_; break
    print(f"      threshold Δr for 95% agreement        : {thr if thr is not None else 'not reached'}")
    return D

if __name__=="__main__":
    run("analysis/crosscrop/results/bean_panel_wide.csv","COMMON BEAN · VEF Andean panel (10 trials)",min_n=25)
