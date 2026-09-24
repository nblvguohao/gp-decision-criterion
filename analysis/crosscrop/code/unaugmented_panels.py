"""Round-1 review fixes: team-clustered intervals, and unaugmented panels.

Writes analysis/crosscrop/results/unaugmented_panels.csv (one row per dataset and
panel version: rate, pair-level Wilson interval, clustered interval, width ratio)."""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"analysis/crosscrop/code")
from analyse_species import panel22, SLOPE, LOWER, orient, rev_from, reversal_ci

def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)

rng=np.random.default_rng(0)
print("="*80)
print("[3] TEAM-CLUSTERED INTERVALS — pairs share teams, so pair-level binomial")
print("    intervals are too narrow. Resampling TEAMS (2000 replicates).\n")
print(f"{'dataset':<16}{'rate':>7}{'pair-level Wilson 95%':>26}{'team-clustered 95%':>24}{'width x':>9}")
print("-"*84)

G=pd.read_csv("analysis/g2f_leaderboard/results/S1_best_per_team.csv")
G=G[G["Team Name"].notna()].copy()
Sm={c:orient(G.set_index("Team Name"),c) for c in ("mean_RMSE","mean_pearson_r")}
n=len(G); full=list(range(n))
k=sum(1 for i,j in itertools.combinations(full,2)
      if (Sm["mean_RMSE"][i]-Sm["mean_RMSE"][j])*(Sm["mean_pearson_r"][i]-Sm["mean_pearson_r"][j])<0)
N=n*(n-1)//2; rate=k/N; lo,hi=wilson(k,N)
bs=[rev_from(Sm["mean_RMSE"],Sm["mean_pearson_r"],rng.integers(0,n,n)) for _ in range(2000)]
cl,ch=np.percentile(bs,[2.5,97.5])
print(f"{'maize (30 teams)':<16}{rate:>7.3f}{f'[{lo:.3f}, {hi:.3f}]':>26}{f'[{cl:.3f}, {ch:.3f}]':>24}"
      f"{(ch-cl)/(hi-lo):>9.2f}")
OUT=[dict(dataset="maize",panel="30 teams",methods=n,rate=rate,wilson_lo=lo,wilson_hi=hi,ci_lo=cl,ci_hi=ch,width_ratio=(ch-cl)/(hi-lo))]

SETS=[("rice","analysis/crosscrop/results/rice_panel_wide.csv",25),
      ("wheat","analysis/crosscrop/results/wheat_panel_wide.csv",25),
      ("common bean","analysis/crosscrop/results/bean_panel_wide.csv",25),
      ("spring wheat","analysis/crosscrop/results/ursn_panel_wide.csv",12),
      ("soybean","analysis/crosscrop/results/nust_panel_wide.csv",25)]
cache={}
for lab,path,mn in scope.kept(SETS):
    P=pd.read_csv(path); cache[lab]=(P,mn)
    T=pd.DataFrame([{**panel22(d,mn),"method":m} for m,d in P.groupby("method")]).set_index("method")
    a,b=orient(T,"mean_RMSE"),orient(T,"mean_pearson_r"); m=len(T)
    d1=np.array([a[i]-a[j] for i,j in itertools.combinations(range(m),2)]); d2=np.array([b[i]-b[j] for i,j in itertools.combinations(range(m),2)])
    ok=scope.untied(d1,d2); kk=int((d1[ok]*d2[ok]<0).sum()); NN=int(ok.sum()); r0=kk/NN; l0,h0=wilson(kk,NN)
    _,c1,c2,_=reversal_ci(T,rng)            # family-clustered, as in Table 2
    print(f"{lab:<16}{r0:>7.3f}{f'[{l0:.3f}, {h0:.3f}]':>26}{f'[{c1:.3f}, {c2:.3f}]':>24}"
          f"{(c2-c1)/(h0-l0):>9.2f}")
    OUT.append(dict(dataset=lab,panel="augmented",methods=m,rate=r0,wilson_lo=l0,wilson_hi=h0,ci_lo=c1,ci_hi=c2,width_ratio=(c2-c1)/(h0-l0)))

print("\n"+"="*80)
print("[4] UNAUGMENTED PANELS — do the results survive without the")
print("    miscalibration variants added in Methods 10?\n")
print(f"{'dataset':<16}{'methods (aug/base)':>20}{'reversal aug':>14}{'reversal base':>15}{'base 95% CI':>20}")
print("-"*86)
for lab,(P,mn) in cache.items():
    base=P[~P.method.str.contains("__")]
    out=[]
    for tag,PP in (("aug",P),("base",base)):
        T=pd.DataFrame([{**panel22(d,mn),"method":m} for m,d in PP.groupby("method")]).set_index("method")
        a,b=orient(T,"mean_RMSE"),orient(T,"mean_pearson_r"); m=len(T)
        pt,lo,hi,_=reversal_ci(T,rng)          # tie-excluded, family-clustered, as in Table 2
        out.append((m,pt,(lo,hi)))
    (ma,ra,_),(mb,rb,cb)=out
    print(f"{lab:<16}{f'{ma}/{mb}':>20}{ra:>14.3f}{rb:>15.3f}{f'[{cb[0]:.3f}, {cb[1]:.3f}]':>20}")
    OUT.append(dict(dataset=lab,panel="base",methods=mb,rate=rb,wilson_lo=np.nan,wilson_hi=np.nan,ci_lo=cb[0],ci_hi=cb[1],width_ratio=np.nan))
pd.DataFrame(OUT).to_csv("analysis/crosscrop/results/unaugmented_panels.csv",index=False)
