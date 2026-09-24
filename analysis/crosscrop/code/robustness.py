"""Robustness pass: the noise null (Table S3) and environment-bootstrap rank intervals
(Table S6). Writes analysis/crosscrop/results/robustness.csv."""
import sys as _sys; _sys.path.insert(0, "analysis/crosscrop/code"); import scope  # datasets used in the paper
import warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import sys; sys.path.insert(0,"analysis/crosscrop/code")
from analyse_species import panel22, SLOPE, LOWER
from scipy.stats import binomtest

def metrics_on(P, envs, min_n):
    sub=P[P.Env.isin(envs)]
    return pd.DataFrame([{**panel22(d,min_n),"method":m} for m,d in sub.groupby("method")]).set_index("method")


def env_r_matrix(P, min_n):
    """method x environment matrix of within-environment Pearson r, NaN where panel22
    skips the environment (fewer than min_n genotypes or no variation)."""
    from scipy.stats import pearsonr
    rows={}
    for m,d in P.groupby("method"):
        r={}
        for e,g in d.groupby("Env"):
            if len(g)<min_n or g.p.nunique()<2 or g.y.nunique()<2: continue
            r[e]=pearsonr(g.y.to_numpy(),g.p.to_numpy())[0]
        rows[m]=r
    return pd.DataFrame(rows).T

def orient(T,c):
    v=T[c].to_numpy(float)
    return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)

def rev_rate(a,b):
    n=len(a); pr=list(itertools.combinations(range(n),2))
    d1=np.array([a[i]-a[j] for i,j in pr]); d2=np.array([b[i]-b[j] for i,j in pr])
    ok=scope.untied(d1,d2)          # pairs tied on either metric cannot reverse
    return int((d1[ok]*d2[ok]<0).sum()), int(ok.sum())

SETS=[("rice","analysis/crosscrop/results/rice_panel_wide.csv",25),
      ("wheat","analysis/crosscrop/results/wheat_panel_wide.csv",25),
      ("common bean","analysis/crosscrop/results/bean_panel_wide.csv",25),
      ("spring wheat","analysis/crosscrop/results/ursn_panel_wide.csv",12),
      ("soybean","analysis/crosscrop/results/nust_panel_wide.csv",25)]
M22,M24="mean_RMSE","mean_pearson_r"
rng=np.random.default_rng(0)
OUT={}

print("="*80)
print("[1] NOISE NULL — is the metric disagreement larger than sampling noise?\n")
print("    signal : reversal between the two OFFICIAL metrics on all environments")
print("    noise  : reversal between the SAME metric computed on two disjoint")
print("             random halves of the environments (20 splits)\n")
print(f"{'dataset':<16}{'signal':>9}{'noise (mean_RMSE)':>20}{'noise (mean_r)':>16}{'ratio':>8}")
print("-"*80)
for lab,path,mn in scope.kept(SETS):
    P=pd.read_csv(path); envs=P.Env.unique()
    T=metrics_on(P,envs,mn)
    k,n=rev_rate(orient(T,M22),orient(T,M24)); sig=k/n
    nz={M22:[],M24:[]}
    for _ in range(20):
        e=rng.permutation(envs); h1,h2=e[:len(e)//2],e[len(e)//2:]
        T1=metrics_on(P,h1,mn); T2=metrics_on(P,h2,mn)
        common=T1.index.intersection(T2.index)
        for mt in (M22,M24):
            a=orient(T1.loc[common],mt); b=orient(T2.loc[common],mt)
            kk,nn=rev_rate(a,b); nz[mt].append(kk/nn)
    n1,n2=np.mean(nz[M22]),np.mean(nz[M24])
    print(f"{lab:<16}{sig:>9.3f}{n1:>20.3f}{n2:>16.3f}{sig/max((n1+n2)/2,1e-9):>8.1f}x")
    OUT[lab]=dict(dataset=lab,signal=sig,noise_rmse=n1,noise_pearson=n2,signal_over_noise=sig/max((n1+n2)/2,1e-9))

print("\n"+"="*80)
print("[2] BOOTSTRAP RANK INTERVALS — does the conformal width survive an")
print("    assumption-free alternative? (resample environments, 400 replicates)\n")
print(f"{'dataset':<16}{'conformal width':>17}{'bootstrap width':>18}{'methods':>9}")
print("-"*80)
for lab,path,mn in scope.kept(SETS):
    P=pd.read_csv(path); envs=P.Env.unique()
    T=metrics_on(P,envs,mn); T=T.dropna(axis=1,how="all"); MET=list(T.columns); M=len(T)
    R=pd.DataFrame({m:pd.Series(-orient(T,m)).rank(method="first").to_numpy() for m in MET})
    Rc=R.to_numpy(); rh=np.median(Rc,axis=1); s=np.abs(Rc-rh[:,None]).ravel()
    q=np.sort(s)[min(int(np.ceil((len(s)+1)*.95)),len(s))-1]
    conf=np.median(np.clip(rh+q,1,M)-np.clip(rh-q,1,M)+1)
    # Environment bootstrap of the mean within-environment Pearson r. Resampled
    # environments keep their multiplicity (an earlier version de-duplicated them, which
    # made this a ~63 % subsample without replacement and the interval too narrow).
    Rm=env_r_matrix(P,mn).reindex(T.index)
    assert np.allclose(np.nanmean(Rm.to_numpy(),axis=1),T[M24].to_numpy(),atol=1e-10)
    cols=np.arange(Rm.shape[1]); boot=[]
    for _ in range(400):
        e=rng.choice(cols,len(cols),replace=True)
        mb=np.nanmean(Rm.to_numpy()[:,e],axis=1)
        boot.append(pd.Series(-mb).rank(method="first").to_numpy())
    B=np.vstack(boot)
    w=np.percentile(B,97.5,axis=0)-np.percentile(B,2.5,axis=0)+1
    print(f"{lab:<16}{conf:>10.0f} of {M:<4}{np.median(w):>11.0f} of {M:<4}{M:>9}")
    OUT[lab].update(methods=M,envs=len(envs),metric_width=conf,boot_width=float(np.median(w)))
pd.DataFrame(list(OUT.values())).to_csv("analysis/crosscrop/results/robustness.csv",index=False)
