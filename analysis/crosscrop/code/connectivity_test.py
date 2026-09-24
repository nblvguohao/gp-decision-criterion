"""Decisive test of the connectivity hypothesis.

If low genotype connectivity between environments is why the taxonomy separation
fails in soybean and spring wheat, then restricting those datasets to a
well-connected subset of environments should make the separation appear.
This is a prediction, not a description: the environment subset is chosen by
connectivity alone, using no metric and no outcome.
"""
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"analysis/crosscrop/code")
from analyse_species import panel22, SLOPE, LOWER

def connected_subset(sets, k):
    """greedily grow a set of k environments with high mutual genotype overlap"""
    envs=list(sets)
    best=None
    for seed in envs:
        cur=[seed]
        while len(cur)<k:
            cand=max((e for e in envs if e not in cur),
                     key=lambda e: np.mean([len(sets[e]&sets[c])/max(len(sets[e]|sets[c]),1) for c in cur]))
            cur.append(cand)
        J=np.mean([len(sets[a]&sets[b])/max(len(sets[a]|sets[b]),1)
                   for a,b in itertools.combinations(cur,2)])
        if best is None or J>best[0]: best=(J,cur)
    return best

def coverage_sep(P, min_n):
    T=pd.DataFrame([{**panel22(d,min_n),"method":m} for m,d in P.groupby("method")]).set_index("method")
    T=T.dropna(axis=1,how="all"); MET=list(T.columns); M=len(T)
    def orient(c):
        v=T[c].to_numpy(float)
        return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)
    R=pd.DataFrame({m:pd.Series(-orient(m)).rank(method="first").to_numpy() for m in MET})
    def conf(cal,a=.05):
        Rc=R[cal].to_numpy(); rh=np.median(Rc,axis=1); s=np.abs(Rc-rh[:,None]).ravel()
        n=len(s); kk=int(np.ceil((n+1)*(1-a))); return rh,np.sort(s)[min(kk,n)-1]
    cv=[]
    for h in MET:
        rh,q=conf([m for m in MET if m!=h]); t=R[h].to_numpy()
        cv.append((h,np.mean((t>=rh-q)&(t<=rh+q))))
    C=pd.DataFrame(cv,columns=["metric","coverage"])
    KEY=lambda m: m.startswith("mean_") and (m in SLOPE or (("pearson" in m or "spearman" in m) and not m.endswith("r2_score")))
    we=C[C.metric.map(KEY)].coverage; ot=C[~C.metric.map(KEY)].coverage
    return we.max()<ot.min(), we.min(), we.max(), ot.min(), M

rows=[]
for lab,path,mn,K in (("soybean (NUST)","analysis/crosscrop/results/nust_panel_wide.csv",25,20),
                      ("spring wheat (URSN)","analysis/crosscrop/results/ursn_panel_wide.csv",12,20)):
    P=pd.read_csv(path)
    one=P[P.method==P.method.iloc[0]]
    sets={e:set(g.k) for e,g in one.groupby("Env")}
    sets={e:s for e,s in sets.items() if len(s)>=mn}
    J,sub=connected_subset(sets,K)
    Jall=np.mean([len(sets[a]&sets[b])/max(len(sets[a]|sets[b]),1)
                  for a,b in itertools.combinations(list(sets)[:60],2)])
    print(f"### {lab}")
    print(f"    full dataset : {len(sets)} environments, mean Jaccard {Jall:.3f}")
    print(f"    connected subset chosen by overlap alone : {K} environments, mean Jaccard {J:.3f}")
    for tag,PP in (("full",P),("connected subset",P[P.Env.isin(sub)])):
        try:
            sep,a,b,c,M=coverage_sep(PP,mn)
            rows.append(dict(dataset=lab,subset=tag,n_env=len(sets) if tag=="full" else K,
                             mean_jaccard=Jall if tag=="full" else J,separated=bool(sep),
                             within_min=a,within_max=b,others_min=c,methods=M))
            print(f"      {tag:<18} within-env×discrim [{a:.2f}, {b:.2f}]  others min {c:.2f}  "
                  f"-> {'SEPARATED' if sep else 'overlapping'}   ({M} methods)")
        except Exception as e:
            print(f"      {tag:<18} failed: {type(e).__name__}")
    print()
pd.DataFrame(rows).to_csv("analysis/crosscrop/results/connectivity.csv",index=False)
print("wrote analysis/crosscrop/results/connectivity.csv")
