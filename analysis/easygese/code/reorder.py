"""Independent test of metric-driven reordering on EasyGeSe (exploratory, 9 September 2026;
reported in Supplementary Section S16, not carried into the paper).

EasyGeSe (Quesada-Traver et al., BMC Genomics 2025; Zenodo 15348871) benchmarks
10 genomic prediction models on 93 traits across 10 species spanning plants,
livestock, aquaculture and forest trees, with 5x5-fold cross-validation. Both
Pearson r and RMSE are reported for every model x trait x split.
Nothing here is our construction: data, methods, folds and metrics are theirs.
"""
import itertools, numpy as np, pandas as pd
import os
RAW = os.environ.get("GP_DATA", "data/raw")   # root for third-party primary data; see DATA_SOURCES.md
SP=f"{RAW}/easygese"
d=pd.read_csv(f"{SP}/datasets/results_raw.csv")
print(f"{len(d)} rows | {d.species.nunique()} species | {d.trait.nunique()} traits | "
      f"{d.model.nunique()} models | {d.CV_split.nunique()} splits\n")
print("species:", ", ".join(sorted(d.species.unique())))

m=d.groupby(["species","trait","model"]).agg(r=("correlation","mean"),
                                             rmse=("RMSE","mean"),
                                             n=("correlation","size")).reset_index()
def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; dd=1+z*z/n; c=(p+z*z/(2*n))/dd; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/dd
    return max(0,c-h),min(1,c+h)

rows=[]
for (sp,tr),g in m.groupby(["species","trait"]):
    if len(g)<5: continue
    a=g.r.to_numpy(); b=-g.rmse.to_numpy()          # both oriented higher-better
    pr=list(itertools.combinations(range(len(g)),2))
    d1=np.array([a[i]-a[j] for i,j in pr]); d2=np.array([b[i]-b[j] for i,j in pr])
    ok=np.isfinite(d1)&np.isfinite(d2)
    rows.append((sp,tr,int((d1[ok]*d2[ok]<0).sum()),int(ok.sum()),len(g)))
R=pd.DataFrame(rows,columns=["species","trait","rev","pairs","models"])
R["rate"]=R.rev/R.pairs
K,N=R.rev.sum(),R.pairs.sum(); lo,hi=wilson(K,N)
print(f"\n{'='*72}\nPAIRWISE MODEL REORDERING BETWEEN PEARSON r AND RMSE\n")
print(f"  {len(R)} traits, {R.models.mean():.0f} models each, {N} model pairs in total")
print(f"  overall reversal rate : {K}/{N} = {K/N:.3f}   Wilson 95% CI [{lo:.3f}, {hi:.3f}]")
print(f"  traits with >=1 reversal : {(R.rate>0).sum()} of {len(R)} ({(R.rate>0).mean()*100:.0f}%)")
print(f"  traits with >20% of pairs reversed : {(R.rate>0.2).sum()} ({(R.rate>0.2).mean()*100:.0f}%)")
print(f"\n{'species':<12}{'traits':>8}{'reversal rate':>16}{'95% CI':>20}{'best model differs':>20}")
print("-"*78)
for sp,g in R.groupby("species"):
    k,n=g.rev.sum(),g.pairs.sum(); l,h=wilson(k,n)
    sub=m[m.species==sp]
    diff=0; tot=0
    for tr,gg in sub.groupby("trait"):
        if len(gg)<5: continue
        tot+=1
        if gg.loc[gg.r.idxmax(),"model"]!=gg.loc[gg.rmse.idxmin(),"model"]: diff+=1
    print(f"{sp:<12}{len(g):>8}{k/n:>16.3f}{f'[{l:.3f}, {h:.3f}]':>20}{f'{diff}/{tot}':>20}")
alltr=0; alld=0
for (sp,tr),gg in m.groupby(["species","trait"]):
    if len(gg)<5: continue
    alltr+=1
    if gg.loc[gg.r.idxmax(),"model"]!=gg.loc[gg.rmse.idxmin(),"model"]: alld+=1
print(f"\n  the best model under r differs from the best under RMSE in "
      f"{alld} of {alltr} traits ({alld/alltr*100:.0f}%)")
R.to_csv("analysis/easygese/results/reordering_by_trait.csv",index=False)
