"""CAIGE bread wheat (Australia) method panel.
15 site-year environments (2017-2018), CIMMYT/ICARDA lines, SNP data from the
CAIGE genotypic workbook. CV1 on genotypes, as for rice.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import os
RAW = os.environ.get("GP_DATA", "data/raw")   # root for third-party primary data; see DATA_SOURCES.md
SP=f"{RAW}/caige"
OUT="analysis/crosscrop/results"
ge=pd.read_csv(f"{OUT}/caige_ge.csv",dtype={"GID":str}).dropna(subset=["GID","y"])
x=pd.ExcelFile(f"{SP}/CAIGE_Bread-Wheat_ICARDA-Line_2015-20_GenotypicData.xlsx")
blocks=[]
for sh in x.sheet_names[1:]:
    d=x.parse(sh)
    cols=[str(c).replace(".0","").strip() for c in d.columns]; d.columns=cols
    mk=next((c for c in cols[:7] if "marker" in c.lower()), None)
    if mk is None: continue
    g=[c for c in cols[7:] if c in set(ge.GID)]
    if not g: continue
    b=d[[mk]+g].dropna(subset=[mk]).drop_duplicates(subset=[mk]).set_index(mk)
    blocks.append(b); print(f"  sheet {sh}: {b.shape[0]} markers x {len(g)} genotypes")
G=pd.concat(blocks,axis=1,join="inner")
G=G.loc[:,~G.columns.duplicated()]
print(f"combined: {G.shape[0]} markers x {G.shape[1]} genotypes")
codes=pd.unique(G.to_numpy().ravel()); print("call codes:",codes[:12])
A=G.to_numpy()
maj=pd.DataFrame(A).mode(axis=1)[0].to_numpy()
miss=pd.isna(A)|(A=="NN")|(A=="--")
het=np.array([[isinstance(v,str) and len(v)==2 and v[0]!=v[1] for v in row] for row in A])
X=np.where(miss,np.nan,np.where(het,1.,np.where(A==maj[:,None],0.,2.)))
X=X.T                                                    # genotypes x markers
X=np.where(np.isnan(X),np.nanmean(np.where(np.isnan(X),np.nan,X),axis=0),X)
X=X[:,np.nanstd(X,axis=0)>0]
Z=(X-X.mean(0))/X.std(0)
U,S,_=np.linalg.svd(Z-Z.mean(0),full_matrices=False)
NPC=min(50,U.shape[1]); PC=U[:,:NPC]*S[:NPC]
print(f"markers {X.shape}, PC1-{NPC} explain {(S**2/np.sum(S**2))[:NPC].sum()*100:.1f}%")
pos={g:i for i,g in enumerate(G.columns)}
ph=ge[ge.GID.isin(pos)].copy(); ph["gi"]=ph.GID.map(pos); ph["k"]=ph.GID
cnt=ph.groupby("Env").GID.nunique(); ph=ph[ph.Env.isin(cnt[cnt>=25].index)]
print(f"phenotype: {len(ph)} cells, {ph.Env.nunique()} envs, {ph.GID.nunique()} genotypes, "
      f"per-env median {ph.groupby('Env').GID.nunique().median():.0f}")
def ridge(Xtr,ytr,Xte,lam):
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xte=np.c_[np.ones(len(Xte)),Xte]
    M=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); M[0,0]-=lam
    return Xte@np.linalg.solve(M,Xtr.T@ytr)
rng=np.random.default_rng(0)
gs=ph.GID.unique(); ph["fold"]=ph.GID.map(pd.Series(rng.integers(0,5,len(gs)),index=gs))
preds={}
for f in range(5):
    tr=ph[ph.fold!=f]; te=ph[ph.fold==f]
    em=tr.groupby("Env").y.mean(); mu=tr.y.mean(); y_tr=tr.y.to_numpy()
    F=lambda d,n: np.c_[d.Env.map(em).fillna(mu).to_numpy(), PC[d.gi.to_numpy()][:,:n]]
    for n in (5,15,30,50):
        for lam,t in ((1e0,"lo"),(1e2,"hi")):
            preds.setdefault(f"ridge_pc{n}_{t}",[]).append((te.index,ridge(F(tr,n),y_tr,F(te,n),lam)))
    preds.setdefault("env_mean_only",[]).append((te.index,te.Env.map(em).fillna(mu).to_numpy()))
    A_,B_=F(tr,50),F(te,50)
    for n,md in (("rf",RandomForestRegressor(n_estimators=300,min_samples_leaf=3,n_jobs=-1,random_state=0)),
                 ("gbm",HistGradientBoostingRegressor(max_depth=4,max_iter=300,learning_rate=.06,random_state=1)),
                 ("knn10",KNeighborsRegressor(n_neighbors=10)),
                 ("mlp",MLPRegressor(hidden_layer_sizes=(48,24),max_iter=500,random_state=2,early_stopping=True))):
        md.fit(A_,y_tr); preds.setdefault(n,[]).append((te.index,md.predict(B_)))
rows=[]
for m,parts in preds.items():
    s=pd.Series(np.concatenate([p[1] for p in parts]),index=np.concatenate([p[0] for p in parts])).sort_index()
    d=ph.loc[s.index,["Env","k","y"]].copy(); d["p"]=s.to_numpy(); d["method"]=m; rows.append(d)
base=pd.concat(rows,ignore_index=True)
g=base[base.method.isin(["ridge_pc30_lo","rf","gbm"])].groupby(["Env","k","y"],as_index=False).p.mean()
g["method"]="ens"; base=pd.concat([base,g],ignore_index=True)
r2=np.random.default_rng(7); extra=[]
for s in ["ridge_pc30_lo","rf","gbm","mlp","knn10","ens"]:
    d=base[base.method==s]; p=d.p.to_numpy(); y=d.y.to_numpy()
    for tag,q in (("shrunk",p.mean()+.25*(p-p.mean())),("inflated",p.mean()+3*(p-p.mean())),
                  ("biased",p+.8*y.std()),("noise",p+r2.normal(0,p.std(),len(p))),("shuffled",None)):
        if q is None:
            q=p.copy(); m=r2.random(len(q))<.35; q[m]=r2.permutation(q[m])
        e=d.copy(); e["p"]=q; e["method"]=f"{s}__{tag}"; extra.append(e)
W=pd.concat([base]+extra,ignore_index=True)
W.to_csv(f"{OUT}/wheat_panel_wide.csv",index=False)
print(f"\nwheat panel: {W.method.nunique()} methods x {W.Env.nunique()} envs, {len(W)} rows")
