"""Common bean (VEF Andean elite panel) method panel.
Data: Harvard Dataverse doi:10.7910/DVN/XCD67U, from Keller et al. 2020,
Front. Plant Sci. 11:1001. 467 elite Andean lines, 5,820 SNPs, 10 trials at
Palmira and Darien (Colombia) under drought / irrigated / phosphorus regimes.
Trait: seed yield (Yd_BLUE). Five-fold cross-validation on genotypes (CV1).
"""
import os, warnings, gzip, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
D=os.environ.get("VEF_DIR", os.environ.get("GP_DATA","data/raw")+"/vef")
OUT="analysis/crosscrop/results"

ph=pd.read_csv(f"{D}/VEF_BLUE_data.csv")
ph=ph[["Trial","Line","Yd_BLUE"]].rename(columns={"Trial":"Env","Line":"k","Yd_BLUE":"y"}).dropna()
print(f"phenotypes: {len(ph)} rows, {ph.Env.nunique()} trials, {ph.k.nunique()} lines")
print(ph.groupby("Env").k.nunique().to_string())

code={"0/0":0.,"0|0":0.,"0/1":1.,"1/0":1.,"0|1":1.,"1|0":1.,"1/1":2.,"1|1":2.}
rows=[]; ids=None
with gzip.open(f"{D}/VEF_Pvulgaris_genotypic_data.vcf.gz","rt") as fh:
    for line in fh:
        if line.startswith("##"): continue
        f=line.rstrip("\n").split("\t")
        if line.startswith("#CHROM"): ids=f[9:]; continue
        rows.append([code.get(g.split(":")[0],np.nan) for g in f[9:]])
X=np.asarray(rows,dtype=np.float64).T
print(f"VCF: {X.shape[0]} samples x {X.shape[1]} markers, missing {np.isnan(X).mean()*100:.2f}%")
X=np.where(np.isnan(X),np.nanmean(X,axis=0),X)
X=X[:,np.nanstd(X,axis=0)>0]
Z=(X-X.mean(0))/X.std(0)
U,S,_=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
NPC=min(80,U.shape[1]); PC=U[:,:NPC]*S[:NPC]
print(f"after filtering {X.shape}, PC1-{NPC} explain {(S**2/np.sum(S**2))[:NPC].sum()*100:.1f}%")
pos={g:i for i,g in enumerate(ids)}
ph=ph[ph.k.isin(pos)].copy(); ph["gi"]=ph.k.map(pos)
cnt=ph.groupby("Env").k.nunique(); ph=ph[ph.Env.isin(cnt[cnt>=25].index)]
print(f"usable: {len(ph)} cells, {ph.Env.nunique()} envs, {ph.k.nunique()} genotypes, "
      f"per-env median {ph.groupby('Env').k.nunique().median():.0f}")

def ridge(Xtr,ytr,Xte,lam):
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xte=np.c_[np.ones(len(Xte)),Xte]
    A=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); A[0,0]-=lam
    return Xte@np.linalg.solve(A,Xtr.T@ytr)
rng=np.random.default_rng(0)
gs=ph.k.unique(); ph["fold"]=ph.k.map(pd.Series(rng.integers(0,5,len(gs)),index=gs))
preds={}
# Environment index: each fold's test genotypes get the mean of that fold's TRAINING genotypes
# in the environment (leave-genotypes-out, CV1). An earlier version averaged the five
# training-fold means, which put every test genotype's own phenotype into its index (four
# folds in five) -- a leak into the environment level of the predictions. Fold-specific
# indices differ slightly within an environment; that is what CV1 predictions look like.
for f in range(5):
    tr=ph[ph.fold!=f]; te=ph[ph.fold==f]
    em=tr.groupby("Env").y.mean(); mu=tr.y.mean(); y_tr=tr.y.to_numpy()
    F=lambda d,n,e=em: np.c_[d.Env.map(e).fillna(mu).to_numpy(), PC[d.gi.to_numpy()][:,:n]]
    for n in (5,20,40,80):
        for lam,t in ((1e0,"lo"),(1e2,"hi")):
            preds.setdefault(f"ridge_pc{n}_{t}",[]).append((te.index,ridge(F(tr,n),y_tr,F(te,n),lam)))
    preds.setdefault("markeronly_pc80",[]).append(
        (te.index,ridge(PC[tr.gi.to_numpy()][:,:80],y_tr,PC[te.gi.to_numpy()][:,:80],1e2)))
    A_,B_=F(tr,80),F(te,80)
    for n,md in (("rf",RandomForestRegressor(n_estimators=300,min_samples_leaf=3,n_jobs=-1,random_state=0)),
                 ("gbm",HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=.06,random_state=1)),
                 ("knn10",KNeighborsRegressor(n_neighbors=10)),
                 ("knn30",KNeighborsRegressor(n_neighbors=30)),
                 ("mlp",MLPRegressor(hidden_layer_sizes=(48,24),max_iter=500,random_state=2,early_stopping=True))):
        md.fit(A_,y_tr); preds.setdefault(n,[]).append((te.index,md.predict(B_)))
# Genomic mixed models (mixed_models.py): GBLUP and the marker x environment GBLUP, on every
# principal component of the markers. Each fold's test genotypes get the environment effects
# estimated in that fold's own fit, as for the index above (no averaging across folds).
import sys; sys.path.insert(0,"analysis/crosscrop/code"); import mixed_models as mm
WF=mm.scale_features(PC)   # same NPC=80 features as the ridge baselines, not the untruncated basis
gen={"gblup":[],"gblup_gxe":[]}
for f in range(5):
    tr=ph[ph.fold!=f]; te=ph[ph.fold==f]
    for name,gxe in (("gblup",False),("gblup_gxe",True)):
        fit=mm.fit(tr.y.to_numpy(),tr.Env.to_numpy(),WF[tr.gi.to_numpy()],tr.Env.to_numpy() if gxe else None)
        be=pd.Series(mm.env_effects(fit))
        lvl=te.Env.map(be).fillna(be.mean()).to_numpy()
        gen[name].append((te.index,lvl+mm.genetic(fit,WF[te.gi.to_numpy()],te.Env.to_numpy() if gxe else None)))
        print(f"  fold {f} {name}: s2u/s2e {fit['d0']:.3f}, s2v/s2e {fit['d1']:.3f}")
for name in gen:
    preds[name]=gen[name]
rows=[]
for m,parts in preds.items():
    s=pd.Series(np.concatenate([p[1] for p in parts]),index=np.concatenate([p[0] for p in parts])).sort_index()
    d=ph.loc[s.index,["Env","k","y"]].copy(); d["p"]=s.to_numpy(); d["method"]=m; rows.append(d)
base=pd.concat(rows,ignore_index=True)
g=base[base.method.isin(["ridge_pc40_lo","rf","gbm"])].groupby(["Env","k","y"],as_index=False).p.mean()
g["method"]="ens"; base=pd.concat([base,g],ignore_index=True)
r2=np.random.default_rng(7); extra=[]
for s in ["ridge_pc40_lo","rf","gbm","mlp","knn10","markeronly_pc80","ens"]:
    d=base[base.method==s]; p=d.p.to_numpy(); y=d.y.to_numpy()
    for tag,q in (("shrunk",p.mean()+.25*(p-p.mean())),("inflated",p.mean()+3*(p-p.mean())),
                  ("biased",p+.8*y.std()),("noise",p+r2.normal(0,p.std(),len(p))),("shuffled",None)):
        if q is None:
            q=p.copy(); m=r2.random(len(q))<.35; q[m]=r2.permutation(q[m])
        e=d.copy(); e["p"]=q; e["method"]=f"{s}__{tag}"; extra.append(e)
W=pd.concat([base]+extra,ignore_index=True)
W.to_csv(f"{OUT}/bean_panel_wide.csv",index=False)
print(f"\nbean panel: {W.method.nunique()} methods x {W.Env.nunique()} envs, {len(W)} rows")
