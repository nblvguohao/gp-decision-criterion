"""Surrogate threshold for genomic selection.

Decision question: if method B reports a higher within-environment correlation
than method A, how often does B actually select better material in the field?
The threshold is the accuracy gap  dr = r_B - r_A  at which that probability
reaches a stated level. Below it, a reported accuracy gain buys nothing.

Realised value of a method in one environment = standardised selection
differential of its own top-k pick:  (mean y of chosen) - (mean y) / sd(y).
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr

def per_env(d, frac):
    """per-environment within-env r and realised selection differential"""
    R={}; G={}; SEL={}
    for e,g in d.groupby("Env"):
        if len(g)<25 or g.p.nunique()<2 or g.y.nunique()<2: continue
        y=g.y.to_numpy(); p=g.p.to_numpy(); k=max(1,int(round(frac*len(g))))
        t=np.argsort(-p)[:k]
        R[e]=pearsonr(y,p)[0]; G[e]=(y[t].mean()-y.mean())/y.std()
        SEL[e]=frozenset(g.k.to_numpy()[t]) if "k" in g else frozenset(t)
    return R,G,SEL

def analyse(P, label, frac=0.10, min_env=20):
    meth=sorted(P.method.unique())
    cache={m:per_env(P[P.method==m], frac) for m in meth}
    rows=[]
    for a,b in itertools.combinations(meth,2):
        Ra,Ga,Sa=cache[a]; Rb,Gb,Sb=cache[b]
        env=sorted(set(Ra)&set(Rb))
        if len(env)<min_env: continue
        dr=np.mean([Rb[e]-Ra[e] for e in env])
        dg=np.mean([Gb[e]-Ga[e] for e in env])
        ov=np.mean([len(Sa[e]&Sb[e])/max(len(Sa[e]),1) for e in env])
        rows.append((a,b,dr,dg,ov,len(env)))
    D=scope.drop_ties(pd.DataFrame(rows,columns=["A","B","dr","dg","overlap","n_env"]))
    # orient every pair so the accuracy winner is B
    flip=D.dr<0
    D.loc[flip,["dr","dg"]]*=-1
    D["agree"]=D.dg>0                      # accuracy winner also wins in the field
    print(f"\n{'='*72}\n{label}   {len(meth)} methods, {len(D)} pairs, top {frac*100:.0f}% selected")
    print(f"\n  shortlist overlap when the two methods are equally accurate (|dr|<0.01):")
    eq=D[D.dr<0.01]
    if len(eq): print(f"    {len(eq)} pairs — mean overlap {eq.overlap.mean()*100:.0f}% "
                      f"→ {100-eq.overlap.mean()*100:.0f}% of the shortlist differs at equal accuracy")
    bins=[(0,.01),(.01,.02),(.02,.03),(.03,.05),(.05,.08),(.08,.12),(.12,.20),(.20,1)]
    print(f"\n  {'accuracy gap dr':<18}{'pairs':>7}{'P(field winner = accuracy winner)':>36}{'overlap':>10}")
    thr=None
    for lo,hi in bins:
        s=D[(D.dr>=lo)&(D.dr<hi)]
        if len(s)<5: continue
        p=s.agree.mean()
        se=np.sqrt(p*(1-p)/len(s))
        print(f"  {lo:.2f} – {hi:<11.2f}{len(s):>7}{p*100:>28.0f}% ±{se*100:.0f}{s.overlap.mean()*100:>10.0f}%")
        if thr is None and p>=0.95 and len(s)>=10: thr=lo
    print(f"\n  >>> threshold: P reaches 95% only once dr >= {thr if thr is not None else 'NOT REACHED in range'}")
    return D

P=pd.read_csv("analysis/crosscrop/results/nust_panel_wide.csv")
Ds=analyse(P,"SOYBEAN · NUST forward-year panel")

RES="analysis/g2f_leaderboard/results"
obs=pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
V=["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
   "EnBiSys_sub243568","NicheSquad_sub244985"]
mz=[]
for n in V:
    pr=pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha":"p"})
    d=obs.merge(pr,on=["Env","Hybrid"]).dropna(subset=["y","p"])
    d["method"]=n; d["k"]=d.Hybrid; mz.append(d)
Dm=analyse(pd.concat(mz,ignore_index=True),"MAIZE · G2F verified submissions",min_env=15)
Ds.to_csv("analysis/crosscrop/results/threshold_soy.csv",index=False)
Dm.to_csv("analysis/crosscrop/results/threshold_maize.csv",index=False)
