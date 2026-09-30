import gzip, re, numpy as np, pandas as pd
ph=pd.read_csv('Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz')
print("pheno file rows", len(ph), "cols", list(ph.columns))
print("Phenotype types:", ph.Phenotype.value_counts().head(15).to_dict())
y=ph[ph.Phenotype=='YieldBuA'].copy()
y['Value']=pd.to_numeric(y.Value, errors='coerce')
y=y.dropna(subset=['Value'])
y['Year']=y.Experiment.str.extract(r'_(\d{4})$')[0].astype(int)
y['Env']=y.Location.astype(str)+'_'+y.Year.astype(str)
print("\n[YIELD] rows %d | experiments %d | locations %d | years %d-%d"%(len(y), y.Experiment.nunique(), y.Location.nunique(), y.Year.min(), y.Year.max()))
print("unique GermplasmId:", y.GermplasmId.nunique())
# collapse replicate plots to genotype-environment means
ge=y.groupby(['Env','Year','GermplasmId'], as_index=False).Value.mean()
print("GE cells (loc-year x geno):", len(ge), "| envs:", ge.Env.nunique())
n=ge.groupby('Env').GermplasmId.nunique()
print("genotypes/env: median %.0f mean %.1f min %d max %d"%(n.median(),n.mean(),n.min(),n.max()))
for th in [50,100,150,200,300]:
    print(f"  envs with >= {th}: {(n>=th).sum()}")
# also experiment-level (test x location x year)
ge2=y.groupby(['Experiment','Location','GermplasmId'], as_index=False).Value.mean()
n2=ge2.groupby(['Experiment','Location']).GermplasmId.nunique()
print("\ntest-loc-year units:", len(n2), "median geno %.0f"%n2.median(), "min", n2.min())

# --- VCF samples ---
with gzip.open('NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz','rt') as f:
    for line in f:
        if line.startswith('#CHROM'):
            samples=line.rstrip('\n').split('\t')[9:]; break
print("\nVCF samples:", len(samples), "sample ids:", samples[:6])
def norm(s):
    s=str(s).upper().strip()
    s=re.sub(r'[^A-Z0-9]','',s)
    return s
gs=set(map(norm,samples))
pids=sorted(y.GermplasmId.astype(str).unique())
pn=[norm(p) for p in pids]
hit=sum(1 for p in pn if p in gs)
print("RAW  match GermplasmId->VCF: %d/%d = %.4f"%(sum(1 for p in pids if p in set(samples)), len(pids), sum(1 for p in pids if p in set(samples))/len(pids)))
print("NORM match GermplasmId->VCF: %d/%d = %.4f"%(hit, len(pids), hit/len(pids)))
# cell-weighted match
gemask=ge.GermplasmId.astype(str).map(lambda p: norm(p) in gs)
print("GE cells covered by genotype: %d/%d = %.4f"%(gemask.sum(), len(ge), gemask.mean()))
sub=ge[gemask]
n3=sub.groupby('Env').GermplasmId.nunique()
print("\nAFTER restricting to genotyped lines:")
print("  envs:", len(n3), "| GE cells:", len(sub))
print("  genotypes/env: median %.0f min %d max %d"%(n3.median(), n3.min(), n3.max()))
for th in [50,100,150,200]:
    print(f"  envs with >= {th} genotyped lines: {(n3>=th).sum()}")
print("  years:", sub.Year.min(),"-",sub.Year.max())
sub.to_csv('nust_ge_means.csv', index=False)
print("\nsaved nust_ge_means.csv")
print("\nexamples pheno ids:", pids[:8])
