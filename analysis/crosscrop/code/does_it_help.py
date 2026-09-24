"""Does scoring on an admissible metric actually deliver better material?

Out-of-sample decision test. Split environments into halves A and B.
  - on half A, rank all methods by a candidate scoring rule
  - take the rule's top-ranked method
  - on half B, measure that method's REALISED selection differential
The rule that yields the highest out-of-sample realised gain is the one a
competition or a breeding programme should score on.
Replication unit is the environment split, not the method.
"""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import pearsonr, spearmanr

FRAC=0.10
def ndcg(y,p,k):
    o=np.argsort(-p)[:k]; g=y[o]-y.min()
    dcg=np.sum(g/np.log2(np.arange(2,len(o)+2)))
    b=np.sort(y)[::-1][:k]-y.min()
    idcg=np.sum(b/np.log2(np.arange(2,len(b)+2)))
    return dcg/idcg if idcg>0 else np.nan

def per_env(d,min_n):
    """for one method: per-environment metric values and realised gain"""
    out={}
    for e,g in d.groupby("Env"):
        if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
        y=g.y.to_numpy(); p=g.p.to_numpy(); k=max(1,int(round(FRAC*len(g))))
        t=np.argsort(-p)[:k]; err=p-y
        out[e]=dict(
            neg_RMSE=-np.sqrt(np.mean(err**2)),
            neg_MAE=-np.mean(np.abs(err)),
            r2_score=1-np.sum(err**2)/np.sum((y-y.mean())**2),
            pearson=pearsonr(y,p)[0],
            spearman=spearmanr(y,p)[0],
            NDCG=ndcg(y,p,k),
            GAIN=(y[t].mean()-y.mean())/y.std(),
        )
    return out

RULES=[("mean_RMSE (Tier III, 2022 official)","neg_RMSE"),
       ("mean_MAE (Tier III)","neg_MAE"),
       ("mean_r2_score (Tier III)","r2_score"),
       ("mean_pearson_r (Tier II, 2024 official)","pearson"),
       ("mean_spearman_r (Tier I)","spearman"),
       ("mean_NDCG@10% (Tier I)","NDCG")]

def run(P,label,min_n,B=400,seed=0):
    rng=np.random.default_rng(seed)
    meths=sorted(P.method.unique())
    cache={m:per_env(P[P.method==m],min_n) for m in meths}
    envs=sorted(set().union(*[set(v) for v in cache.values()]))
    if len(envs)<8: print(f"{label}: too few environments"); return None
    res={lab:[] for lab,_ in RULES}; res["oracle"]=[]; res["random"]=[]
    for _ in range(B):
        e=rng.permutation(envs); A,Bh=set(e[:len(e)//2]),set(e[len(e)//2:])
        gainB={m:np.mean([cache[m][x]["GAIN"] for x in Bh if x in cache[m]]) for m in meths}
        for lab,key in RULES:
            sA={m:np.mean([cache[m][x][key] for x in A if x in cache[m]]) for m in meths}
            res[lab].append(scope.tied_best_mean([sA[m] for m in meths],[gainB[m] for m in meths]))
        res["oracle"].append(max(gainB.values()))
        res["random"].append(np.mean(list(gainB.values())))
    return res

def outcome_reliability(P,min_n,rng,B=200):
    """THE canonical split-half reliability of the OUTCOME (realised selection gain).
    No scoring rule can order methods more reliably than the outcome orders itself,
    so this is the ceiling against which every rule in Table 2 is judged. It is
    computed from the outcome alone and never from any metric."""
    meths=sorted(P.method.unique())
    cache={m:per_env(P[P.method==m],min_n) for m in meths}
    envs=sorted(set().union(*[set(v) for v in cache.values()]))
    rs=[]
    for _ in range(B):
        e=rng.permutation(envs); A,Bh=set(e[:len(e)//2]),set(e[len(e)//2:])
        # iterate the sorted env list, not the sets: Python set iteration order
        # depends on per-process string hashing, which would change the summation
        # order and so the last digits of this statistic between runs
        gA=[np.mean([cache[m][x]["GAIN"] for x in envs if x in A and x in cache[m]]) for m in meths]
        gB=[np.mean([cache[m][x]["GAIN"] for x in envs if x in Bh and x in cache[m]]) for m in meths]
        ok=np.isfinite(gA)&np.isfinite(gB)
        if ok.sum()>5: rs.append(spearmanr(np.array(gA)[ok],np.array(gB)[ok])[0])
    return float(np.mean(rs))


if __name__=="__main__":
    SETS=[("MAIZE (5 verified G2F submissions)",None,20),
          ("RICE","analysis/crosscrop/results/rice_panel_wide.csv",25),
          ("COMMON BEAN","analysis/crosscrop/results/bean_panel_wide.csv",25),
          ("SPRING WHEAT (FHB)","analysis/crosscrop/results/ursn_panel_wide.csv",12),
          ("SOYBEAN","analysis/crosscrop/results/nust_panel_wide.csv",25)]
    RES="analysis/g2f_leaderboard/results"
    obs=pd.read_csv(f"{RES}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha":"y"})
    V=["KernelOfTruth_sub244906","KernelOfTruth_sub244944","KernelOfTruth_sub244953",
       "EnBiSys_sub243568","NicheSquad_sub244985"]
    mz=[]
    for n in V:
        pr=pd.read_csv(f"{RES}/{n}.csv").rename(columns={"Yield_Mg_ha":"p"})
        d=obs.merge(pr,on=["Env","Hybrid"]).dropna(subset=["y","p"]); d["method"]=n; mz.append(d)
    MZ=pd.concat(mz,ignore_index=True)

    print("Out-of-sample realised selection differential (phenotypic SD) of the method")
    print("each scoring rule picks on held-out environments. Higher is better.\n")
    allres={}; rels={}
    jobs=[]
    for lab,path,mn in scope.kept(SETS):
        P=MZ if path is None else pd.read_csv(path)
        jobs.append((lab,P,mn))
        if path is not None:          # the same test without the miscalibration variants
            jobs.append((lab+" (unaugmented)",P[~P.method.str.contains("__")],mn))
    for lab,P,mn in jobs:
        if P.method.nunique()>5:
            rels[lab]=outcome_reliability(P,mn,np.random.default_rng(0))
        r=run(P,lab,mn)
        if r is None: continue
        allres[lab]=r
        print(f"### {lab}   ({P.method.nunique()} methods)")
        orc=np.mean(r["oracle"]); rnd=np.mean(r["random"])
        rows=sorted(((k,np.mean(v)) for k,v in r.items() if k not in("oracle","random")),
                    key=lambda x:-x[1])
        for k,v in rows:
            frac=(v-rnd)/(orc-rnd) if orc>rnd else np.nan
            print(f"    {k:<42}{v:+.4f}   {frac*100:>5.0f}% of the way from average to oracle")
        print(f"    {'(oracle: best method on held-out half)':<42}{orc:+.4f}")
        print(f"    {'(random: average method)':<42}{rnd:+.4f}\n")
    import json
    json.dump({k:{kk:float(np.mean(vv)) for kk,vv in v.items()} for k,v in allres.items()},
              open("analysis/crosscrop/results/does_it_help.json","w"),indent=1)
    json.dump({k:float(v) for k,v in rels.items()},
              open("analysis/crosscrop/results/does_it_help_reliability.json","w"),indent=1)
    print("outcome reliability:", {k:round(v,3) for k,v in rels.items()})

