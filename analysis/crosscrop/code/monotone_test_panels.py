"""Two gating checks raised in round-1 review."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr, norm, linregress
import sys; sys.path.insert(0,"analysis/crosscrop/code")
from analyse_species import panel22

RES="analysis/g2f_leaderboard/results"
obs=pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
V=["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
   "EnBiSys_sub243568","NicheSquad_sub244985"]
def load(n):
    pr=pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha":"p"})
    return obs.merge(pr,on=["Env","Hybrid"]).dropna(subset=["y","p"])

# ---------------------------------------------------------------- CHECK A
print("="*78)
print("CHECK A — is the breeder's-equation shortfall a finite-population artefact?\n")
print("  i_inf = phi(Phi^-1(1-f))/f  assumes an infinite population.")
print("  i_fin(n,k) = E[mean of top-k of n standard normals], by simulation.\n")
rng=np.random.default_rng(0)
def i_fin(n,k,B=40000):
    Z=rng.standard_normal((B,n))
    Z.sort(axis=1)
    return Z[:,-k:].mean(axis=1).mean()
def i_inf(f): x=norm.ppf(1-f); return norm.pdf(x)/f

print(f"{'submission':<26}{'f':>6}{'observed':>10}{'i_inf*r':>10}{'i_fin*r':>10}"
      f"{'gap vs inf':>12}{'gap vs fin':>12}")
print("-"*86)
tot={}
for name in V:
    d=load(name)
    for f in (0.05,0.10,0.20):
        o=[];ai=[];af=[]
        for _,g in d.groupby("Env"):
            if len(g)<20 or g.p.nunique()<2: continue
            y=g.y.to_numpy(); p=g.p.to_numpy(); n=len(g); k=max(1,int(round(f*n)))
            t=np.argsort(-p)[:k]
            r=pearsonr(y,p)[0]
            o.append((y[t].mean()-y.mean())/y.std())
            ai.append(i_inf(k/n)*r); af.append(i_fin(n,k)*r)
        o,ai,af=map(np.array,(o,ai,af))
        print(f"{name:<26}{f:>6.2f}{o.mean():>10.3f}{ai.mean():>10.3f}{af.mean():>10.3f}"
              f"{(o-ai).mean():>+12.3f}{(o-af).mean():>+12.3f}")
        tot.setdefault(f,[]).append(((o-ai).mean(),(o-af).mean()))
print()
for f,v in tot.items():
    v=np.array(v)
    print(f"  f={f:.2f}:  mean gap vs infinite-population {v[:,0].mean():+.3f} SD"
          f"   vs finite-population {v[:,1].mean():+.3f} SD"
          f"   ({(1-abs(v[:,1].mean())/abs(v[:,0].mean()))*100:.0f}% of the gap explained)")

# ---------------------------------------------------------------- CHECK B
print("\n"+"="*78)
print("CHECK B — does the criterion still admit 3 metrics under a GENERAL")
print("          monotone transform, or only the rank-based one?\n")
def panel(d,min_n=20):
    """All 22 official metrics, using the same definitions as every other script."""
    return panel22(d,min_n)

def mono(p,rng):
    """Random strictly increasing function of p. Average ranks keep exactly tied
    predictions tied (ordinal ranks would break ties and so would not be a function of
    p), then a piecewise-linear map through six sorted random interior knots."""
    from scipy.stats import rankdata
    u=(rankdata(p,method="average")-0.5)/len(p)
    knots=np.sort(rng.uniform(0,1,6)); knots=np.r_[0,knots,1]
    vals=np.sort(rng.uniform(0,1,len(knots))); vals[0]=0; vals[-1]=1
    return np.interp(u,knots,vals)
rng2=np.random.default_rng(7)
acc={}
for name in V:
    d=load(name)[["Env","y","p"]]
    base=panel(d)
    for _ in range(120):
        d2=d.copy()
        d2["p"]=d.groupby("Env").p.transform(lambda s: mono(s.to_numpy(),rng2))
        m=panel(d2)
        for k in base:
            if np.isfinite(base[k]) and np.isfinite(m[k]):
                acc.setdefault(k,[]).append(abs(m[k]-base[k])/max(abs(base[k]),1e-9))
print(f"{'metric':<24}{'median |rel change|':>22}{'invariant?':>13}")
print("-"*60)
for k,v in sorted(acc.items(),key=lambda x:np.median(x[1])):
    med=np.median(v)
    print(f"{k:<24}{med:>22.2e}{'  YES' if med<1e-3 else '  no':>13}")
inv=[k for k,v in acc.items() if np.median(v)<1e-3]
print(f"\n  invariant under a GENERAL monotone transform: {len(inv)}  -> {sorted(inv)}")
pd.DataFrame([(k,float(np.median(v)),int(np.median(v)<1e-3)) for k,v in acc.items()],
             columns=["metric","median","invariant"]).sort_values("median").to_csv(
    f"{RES}/monotone_test.csv",index=False)
print(f"  -> written {RES}/monotone_test.csv  ({len(acc)} metrics)")
