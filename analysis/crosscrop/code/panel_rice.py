"""Rice (IRRI elite lines) method panel.

Data: Rice (2023) 16:6, doi 10.1186/s12284-023-00623-6 supplementary files.
107 elite breeding lines, 882 SNPs, 15 environments across Asia and Africa
(2018-2020), 24 environmental covariates. Grain yield BLUPs.
Design: 5-fold cross-validation on GENOTYPES (predict untested lines), which is
the scenario a breeder faces and the one the G2F competition posed.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import os
RAW = os.environ.get("GP_DATA", "data/raw")   # root for third-party primary data; see DATA_SOURCES.md
SP=f"{RAW}/rice"
OUT="analysis/crosscrop/results"

ph=pd.read_csv(f"{SP}/12284_2023_623_MOESM4_ESM.csv")
ph=ph[ph.Trait=="YLD"].copy()
ph["Env"]=(ph.Country.str[:2].str.upper()+"_"+ph.Location.str.replace(r'\s+','',regex=True).str[:4]
           +"_"+ph.Year.astype(str)+ph.Season.str[0])
ph["k"]=ph.GID.astype(str); ph=ph.rename(columns={"BLUPs":"y"})[["Env","k","y"]].dropna()

hm=pd.read_csv(f"{SP}/12284_2023_623_MOESM3_ESM.txt",sep="\t")
gids=[str(c) for c in hm.columns[11:]]
call=hm.iloc[:,11:].to_numpy()
uniq=pd.unique(call.ravel()); print("genotype call codes:",uniq[:12])
maj=pd.DataFrame(call).mode(axis=1)[0].to_numpy()          # per-marker most common call
X=np.where(call==maj[:,None],0.,np.where(pd.isna(call),np.nan,2.))
het=np.isin(call,[c for c in uniq if isinstance(c,str) and c in "RYSWKM"])
X=np.where(het,1.,X).T                                     # genotypes x markers
X=np.where(np.isnan(X),np.nanmean(X,axis=0),X)
X=X[:,np.nanstd(X,axis=0)>0]
Z=(X-X.mean(0))/X.std(0)
U,S,_=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
NPC=min(60,U.shape[1]); PC=U[:,:NPC]*S[:NPC]
print(f"markers {X.shape}, PC1-{NPC} explain {(S**2/np.sum(S**2))[:NPC].sum()*100:.1f}%")
pos={g:i for i,g in enumerate(gids)}
ph=ph[ph.k.isin(pos)].copy(); ph["gi"]=ph.k.map(pos)

ec=pd.read_csv(f"{SP}/12284_2023_623_MOESM5_ESM.csv")
print(f"phenotype: {len(ph)} rows, {ph.Env.nunique()} envs, {ph.k.nunique()} genos")

def ridge(Xtr,ytr,Xte,lam):
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xte=np.c_[np.ones(len(Xte)),Xte]
    A=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); A[0,0]-=lam
    return Xte@np.linalg.solve(A,Xtr.T@ytr)

rng=np.random.default_rng(0)
genos=ph.k.unique(); fold=pd.Series(rng.integers(0,5,len(genos)),index=genos)
ph["fold"]=ph.k.map(fold)
preds={}
for f in range(5):
    tr=ph[ph.fold!=f]; te=ph[ph.fold==f]
    em=tr.groupby("Env").y.mean(); mu=tr.y.mean()
    def F(d,npc):
        return np.c_[d.Env.map(em).fillna(mu).to_numpy(), PC[d.gi.to_numpy()][:,:npc]]
    y_tr=tr.y.to_numpy()
    for npc in (5,15,30,60):
        for lam,tag in ((1e0,"lo"),(1e2,"hi")):
            preds.setdefault(f"ridge_pc{npc}_{tag}",[]).append(
                (te.index, ridge(F(tr,npc),y_tr,F(te,npc),lam)))
    preds.setdefault("env_mean_only",[]).append((te.index, te.Env.map(em).fillna(mu).to_numpy()))
    # reaction norm on environmental covariates
    E=ec.set_index("ENV"); common=[c for c in E.columns]
    envz=(E-E.mean())/E.std()
    def FE(d,npc):
        ev=np.array([envz.loc[e,common].to_numpy() if e in envz.index else np.zeros(len(common))
                     for e in d.Env])
        P=PC[d.gi.to_numpy()][:,:npc]
        return np.c_[d.Env.map(em).fillna(mu).to_numpy(), P, ev[:,:6], P[:,:5]*ev[:,[0]]]
    preds.setdefault("reaction_norm_EC",[]).append((te.index, ridge(FE(tr,30),y_tr,FE(te,30),1e1)))
    A=F(tr,60); B=F(te,60)
    for n,md in (("rf",RandomForestRegressor(n_estimators=300,min_samples_leaf=3,n_jobs=-1,random_state=0)),
                 ("gbm",HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=.06,random_state=1)),
                 ("knn10",KNeighborsRegressor(n_neighbors=10)),
                 ("mlp",MLPRegressor(hidden_layer_sizes=(48,24),max_iter=500,random_state=2,early_stopping=True))):
        md.fit(A,y_tr); preds.setdefault(n,[]).append((te.index, md.predict(B)))
rows=[]
for m,parts in preds.items():
    idx=np.concatenate([p[0] for p in parts]); val=np.concatenate([p[1] for p in parts])
    s=pd.Series(val,index=idx).sort_index()
    d=ph.loc[s.index,["Env","k","y"]].copy(); d["p"]=s.to_numpy(); d["method"]=m
    rows.append(d)
base=pd.concat(rows,ignore_index=True)
for a,b in (("ens_lin",["ridge_pc30_lo","ridge_pc60_hi","reaction_norm_EC"]),
            ("ens_ml",["rf","gbm","mlp"])):
    g=base[base.method.isin(b)].groupby(["Env","k","y"],as_index=False).p.mean(); g["method"]=a
    base=pd.concat([base,g],ignore_index=True)
# restore the calibration axis (see widen_panel.py)
r2=np.random.default_rng(7); extra=[]
for s in ["ridge_pc30_lo","rf","gbm","reaction_norm_EC","mlp","knn10","ens_ml"]:
    d=base[base.method==s]; p=d.p.to_numpy(); y=d.y.to_numpy()
    for tag,q in (("shrunk",p.mean()+.25*(p-p.mean())),("inflated",p.mean()+3*(p-p.mean())),
                  ("biased",p+.8*y.std()),("noise",p+r2.normal(0,p.std(),len(p))),
                  ("shuffled",None)):
        if q is None:
            q=p.copy(); m=r2.random(len(q))<.35; q[m]=r2.permutation(q[m])
        e=d.copy(); e["p"]=q; e["method"]=f"{s}__{tag}"; extra.append(e)
W=pd.concat([base]+extra,ignore_index=True)
W.to_csv(f"{OUT}/rice_panel_wide.csv",index=False)
print(f"\nrice panel: {W.method.nunique()} methods x {W.Env.nunique()} environments, {len(W)} rows")
