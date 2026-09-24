"""NUST (soybean Uniform Regional Tests) preparation.
GE means 1993-2020 + 5,158 KNN-imputed SNPs. Genotype names matched after
stripping non-alphanumerics (raw matching recovers only 615 of 2,666).
"""
import gzip, re, numpy as np, pandas as pd
import os
RAW = os.environ.get("GP_DATA", "data/raw")   # root for third-party primary data; see DATA_SOURCES.md
SRC=f"{RAW}/nust"
OUT="analysis/crosscrop/results"
norm=lambda s: re.sub(r'[^A-Z0-9]','',str(s).upper())

ge=pd.read_csv(f"{SRC}/nust_ge_means.csv"); ge["k"]=ge.GermplasmId.map(norm)
cnt=ge.groupby("Env").k.nunique(); ge=ge[ge.Env.isin(cnt[cnt>=20].index)]
print(f"GE after >=20 genotypes/env filter: {len(ge)} rows, {ge.Env.nunique()} envs, {ge.k.nunique()} genos")

code={"0/0":0.,"0/1":1.,"1/0":1.,"1/1":2.,"./.":np.nan}
rows=[]; ids=None
with gzip.open(f"{SRC}/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz","rt") as fh:
    for line in fh:
        if line.startswith("##"): continue
        f=line.rstrip("\n").split("\t")
        if line.startswith("#CHROM"): ids=[norm(s) for s in f[9:]]; continue
        rows.append([code.get(g.split(":")[0],np.nan) for g in f[9:]])
X=np.asarray(rows,dtype=np.float64).T                      # samples x markers
print(f"VCF: {X.shape[0]} samples x {X.shape[1]} markers, missing {np.isnan(X).mean()*100:.2f}%")

keep=ids is not None
col_mean=np.nanmean(X,axis=0); X=np.where(np.isnan(X),col_mean,X)
poly=np.nanstd(X,axis=0)>0; X=X[:,poly]
print(f"after dropping {int((~poly).sum())} monomorphic markers: {X.shape}")
Z=(X-X.mean(0))/X.std(0)
U,S,Vt=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
NPC=100; PC=U[:,:NPC]*S[:NPC]
ev=(S**2)/np.sum(S**2)
print(f"PC1-{NPC} explain {ev[:NPC].sum()*100:.1f}% of marker variance")

gid=pd.Index(ids)
keepg=set(gid)&set(ge.k)
ge=ge[ge.k.isin(keepg)].copy()
idx={g:i for i,g in enumerate(gid)}
np.save(f"{OUT}/nust_PC.npy", PC)
pd.Series(list(gid)).to_csv(f"{OUT}/nust_geno_ids.csv",index=False,header=["k"])
ge.to_csv(f"{OUT}/nust_ge.csv",index=False)
print(f"\nfinal: {len(ge)} rows | {ge.Env.nunique()} envs | {ge.k.nunique()} genos | {ge.Year.min()}-{ge.Year.max()}")
print(f"per-env genotypes: median {ge.groupby('Env').k.nunique().median():.0f}, "
      f"IQR {ge.groupby('Env').k.nunique().quantile(.25):.0f}-{ge.groupby('Env').k.nunique().quantile(.75):.0f}")
