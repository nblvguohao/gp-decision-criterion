"""Is the surrogate threshold near the phenotyping noise floor?

Within-environment correlation between a prediction and an observed phenotype is
bounded above by sqrt(H2), where H2 is the within-environment repeatability of
the phenotype itself. A threshold of Dr = 0.1 means something very different if
the achievable range is [0, 0.95] than if it is [0, 0.35].

Two independent estimates:
  maize  -- replicated plots in the G2F competition training data (2014-2021),
            H2 = sigma2_g / (sigma2_g + sigma2_e / nrep) per environment
  bean   -- published BLUEs with standard errors, reliability =
            1 - mean(SE^2) / var(BLUE)
"""
import os, warnings, glob, numpy as np, pandas as pd
warnings.filterwarnings("ignore")

print("="*78); print("MAIZE — G2F competition training data, replicated plots\n")
cand=sorted(glob.glob(os.environ.get("GP_DATA","data/raw")+"/**/1_Training_Trait_Data_2014_2021.csv",recursive=True))
src=cand[0]; print(f"source: .../{'/'.join(src.split('/')[-4:])}")
d=pd.read_csv(src,low_memory=False,usecols=["Env","Hybrid","Replicate","Yield_Mg_ha"]).dropna(subset=["Yield_Mg_ha"])
print(f"{len(d)} plots, {d.Env.nunique()} environments, {d.Hybrid.nunique()} hybrids")
rows=[]
for e,g in d.groupby("Env"):
    n=g.groupby("Hybrid").size()
    if (n>=2).sum()<20: continue
    gg=g[g.Hybrid.isin(n[n>=2].index)]
    m=gg.groupby("Hybrid").Yield_Mg_ha.agg(["mean","var","size"])
    ve=np.nansum(m["var"]*(m["size"]-1))/np.nansum(m["size"]-1)      # pooled within-entry
    nrep=m["size"].mean()
    vg=m["mean"].var(ddof=1)-ve/nrep                                  # entry-mean var minus error
    if not np.isfinite(vg) or vg<=0: continue
    H2=vg/(vg+ve/nrep)
    rows.append((e,H2,np.sqrt(H2),len(m),nrep))
H=pd.DataFrame(rows,columns=["Env","H2","r_max","entries","nrep"])
print(f"\n{len(H)} environments with usable replication (median {H.nrep.median():.1f} reps)")
print(f"within-environment repeatability H2 : median {H.H2.median():.3f}  "
      f"IQR [{H.H2.quantile(.25):.3f}, {H.H2.quantile(.75):.3f}]")
print(f"ceiling on within-env correlation r  : median {H.r_max.median():.3f}  "
      f"IQR [{H.r_max.quantile(.25):.3f}, {H.r_max.quantile(.75):.3f}]")

print("\n"+"="*78); print("COMMON BEAN — published BLUEs with standard errors\n")
VEF_DIR=os.environ.get("VEF_DIR", os.environ.get("GP_DATA","data/raw")+"/vef")
b=pd.read_csv(f"{VEF_DIR}/VEF_BLUE_data.csv")
rows=[]
for t,g in b.groupby("Trial"):
    v=g["Yd_BLUE"].dropna(); se=g.loc[v.index,"Yd_SE"]
    pev=np.mean(se**2); vp=v.var(ddof=1)
    rel=1-pev/vp
    if np.isfinite(rel) and rel>0: rows.append((t,rel,np.sqrt(rel),len(v)))
B=pd.DataFrame(rows,columns=["Trial","reliability","r_max","n"])
print(B.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print(f"\nmedian reliability {B.reliability.median():.3f}, ceiling r {B.r_max.median():.3f}")

print("\n"+"="*78); print("THE THRESHOLD AGAINST THE CEILING\n")
obs=pd.read_csv("analysis/g2f_leaderboard/results/S1_best_per_team.csv")
obs=obs[obs["Team Name"].notna()]
r=obs["mean_pearson_r"].dropna()
cm=H.r_max.median(); cb=B.r_max.median()
print(f"  maize achievable ceiling (median sqrt(H2))            : {cm:.3f}")
print(f"  best team's reported within-environment r             : {r.max():.3f}"
      f"   = {r.max()/cm*100:.0f}% of the ceiling")
print(f"  spread of the whole 30-team leaderboard               : {r.max()-r.min():.3f}")
print(f"  gap between 1st and 3rd place                         : 0.018"
      f"   = {0.018/cm*100:.1f}% of the ceiling")
for thr,src_ in ((0.07,"spring wheat"),(0.09,"soybean"),(0.14,"rice"),(0.33,"wheat")):
    print(f"  threshold {thr:.2f} ({src_:<13})                       "
          f"= {thr/cm*100:.0f}% of the maize ceiling")
print(f"\n  common bean ceiling for comparison                    : {cb:.3f}")
H.to_csv("analysis/crosscrop/results/heritability_maize.csv",index=False)
B.to_csv("analysis/crosscrop/results/heritability_bean.csv",index=False)
