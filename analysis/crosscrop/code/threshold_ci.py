"""Bootstrap CI on the surrogate threshold, and its sensitivity to selection intensity."""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr

def pairs_table(P,frac,min_n):
    cache={}
    for m,d in P.groupby("method"):
        R={};Gv={}
        for e,g in d.groupby("Env"):
            if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
            y=g.y.to_numpy(); p=g.p.to_numpy(); k=max(1,int(round(frac*len(g))))
            t=np.argsort(-p)[:k]
            R[e]=pearsonr(y,p)[0]; Gv[e]=(y[t].mean()-y.mean())/y.std()
        cache[m]=(R,Gv)
    rows=[]
    for a,b in itertools.combinations(sorted(cache),2):
        Ra,Ga=cache[a]; Rb,Gb=cache[b]; env=sorted(set(Ra)&set(Rb))
        if len(env)<max(8,min_n//2): continue
        rows.append((np.mean([Rb[e]-Ra[e] for e in env]),np.mean([Gb[e]-Ga[e] for e in env])))
    D=scope.drop_ties(pd.DataFrame(rows,columns=["dr","dg"]))
    f=D.dr<0; D.loc[f,["dr","dg"]]*=-1; D["agree"]=D.dg>0
    return D

def threshold(D,lvl=.95,win=.04,minp=15):
    for lo in np.arange(0,.40,.005):
        s=D[(D.dr>=lo)&(D.dr<lo+win)]
        if len(s)>=minp and s.agree.mean()>=lvl: return lo
    return np.nan

if __name__=="__main__":
    SETS=[("rice","analysis/crosscrop/results/rice_panel_wide.csv",25),
          ("soybean","analysis/crosscrop/results/nust_panel_wide.csv",25),
          ("spring wheat (FHB)","analysis/crosscrop/results/ursn_panel_wide.csv",12),
          ("wheat","analysis/crosscrop/results/wheat_panel_wide.csv",25),
          ("common bean","analysis/crosscrop/results/bean_panel_wide.csv",25)]
    rng=np.random.default_rng(0)
    print("Surrogate threshold: point estimate, bootstrap CI over method pairs, and")
    print("sensitivity to selection intensity f\n")
    print(f"{'dataset':<20}{'f=0.05':>10}{'f=0.10':>10}{'f=0.20':>10}   {'95% CI at f=0.10':>22}")
    print("-"*76)
    for lab,path,mn in scope.kept(SETS):
        P=pd.read_csv(path); out={}
        for frac in (.05,.10,.20):
            D=pairs_table(P,frac,mn); out[frac]=threshold(D)
            if frac==.10: D10=D
        bs=[]
        for _ in range(400):
            s=D10.sample(len(D10),replace=True)
            t=threshold(s)
            if np.isfinite(t): bs.append(t)
        ci=f"[{np.percentile(bs,2.5):.2f}, {np.percentile(bs,97.5):.2f}]" if len(bs)>50 else "n.r."
        fmt=lambda v: f"{v:.2f}" if np.isfinite(v) else "n.r."
        print(f"{lab:<20}{fmt(out[.05]):>10}{fmt(out[.10]):>10}{fmt(out[.20]):>10}   {ci:>22}")

