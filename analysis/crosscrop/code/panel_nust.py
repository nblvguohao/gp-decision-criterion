"""A panel of forward-year prediction methods for NUST soybean.

Mirrors the G2F competition design: train on all earlier years, predict a
held-out year's environments (mostly new genotypes, some new locations).
The panel spans the same strategy space the 30 real G2F teams occupied
(environment index, BLUP, penalised regression, trees, kernels, neural nets,
reaction norms, ensembles) so that the METRIC structure can be studied.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

RES="analysis/crosscrop/results"
ge=pd.read_csv(f"{RES}/nust_ge.csv")
ge["loc"]=ge.Env.str.replace(r'_\d{4}$','',regex=True)
PC=np.load(f"{RES}/nust_PC.npy")
ids=pd.read_csv(f"{RES}/nust_geno_ids.csv")["k"].tolist()
import sys; sys.path.insert(0,"analysis/crosscrop/code"); import mixed_models as mm
WF=mm.scale_features(PC)   # same NPC=100 features as the ridge baselines, not the untruncated basis
pos={g:i for i,g in enumerate(ids)}
ge=ge[ge.k.isin(pos)].copy()
ge["gi"]=ge.k.map(pos)

def ridge(Xtr,ytr,Xte,lam):
    Xtr=np.c_[np.ones(len(Xtr)),Xtr]; Xte=np.c_[np.ones(len(Xte)),Xte]
    A=Xtr.T@Xtr+lam*np.eye(Xtr.shape[1]); A[0,0]-=lam
    return Xte@np.linalg.solve(A,Xtr.T@ytr)

def build(tr,te,npc):
    lm=tr.groupby("loc").Value.mean(); gm=tr.groupby("k").Value.mean(); mu=tr.Value.mean()
    def feats(d):
        li=d["loc"].map(lm).fillna(mu).to_numpy()
        gg=d["k"].map(gm).fillna(mu).to_numpy()
        P=PC[d.gi.to_numpy()][:,:npc]
        return np.c_[li,gg,P], li, gg
    return feats(tr), feats(te), mu

out=[]
for Y in (2016,2017,2018,2019,2020):
    tr=ge[ge.Year<Y]; te=ge[ge.Year==Y]
    if len(te)<300: continue
    y_tr=tr.Value.to_numpy(); y_te=te.Value.to_numpy()
    preds={}
    (Xtr100,li_tr,gm_tr),(Xte100,li_te,gm_te),mu=build(tr,te,100)
    preds["global_mean"]=np.full(len(te),mu)
    preds["env_index"]=li_te
    preds["geno_mean"]=gm_te
    preds["env_plus_geno"]=li_te+gm_te-mu
    for npc in (10,25,50,100):
        Xtr=Xtr100[:,:2+npc]; Xte=Xte100[:,:2+npc]
        for lam,tag in ((1e1,"lo"),(1e3,"hi")):
            preds[f"ridge_pc{npc}_{tag}"]=ridge(Xtr,y_tr,Xte,lam)
    # marker-only ridge (no phenotype-derived features)
    for npc in (25,100):
        preds[f"markeronly_pc{npc}"]=ridge(PC[tr.gi.to_numpy()][:,:npc],y_tr,PC[te.gi.to_numpy()][:,:npc],1e2)
    # genomic Finlay-Wilkinson reaction norm: geno effect scaled by environment index
    z_tr=(li_tr-li_tr.mean())/(li_tr.std()+1e-9); z_te=(li_te-li_te.mean())/(li_te.std()+1e-9)
    Xfw_tr=np.c_[Xtr100[:,:2+25], PC[tr.gi.to_numpy()][:,:25]*z_tr[:,None]]
    Xfw_te=np.c_[Xte100[:,:2+25], PC[te.gi.to_numpy()][:,:25]*z_te[:,None]]
    preds["reaction_norm"]=ridge(Xfw_tr,y_tr,Xfw_te,1e2)
    preds["gxe_ridge"]=ridge(np.c_[Xfw_tr,z_tr[:,None]],y_tr,np.c_[Xfw_te,z_te[:,None]],1e3)
    for n,md in (("rf_small",RandomForestRegressor(n_estimators=120,max_depth=8,min_samples_leaf=8,n_jobs=-1,random_state=0)),
                 ("rf_large",RandomForestRegressor(n_estimators=300,max_depth=None,min_samples_leaf=2,n_jobs=-1,random_state=1)),
                 ("gbm_shallow",HistGradientBoostingRegressor(max_depth=3,max_iter=250,learning_rate=.06,random_state=2)),
                 ("gbm_deep",HistGradientBoostingRegressor(max_depth=8,max_iter=500,learning_rate=.05,random_state=3)),
                 ("knn10",KNeighborsRegressor(n_neighbors=10)),
                 ("knn50",KNeighborsRegressor(n_neighbors=50)),
                 ("mlp",MLPRegressor(hidden_layer_sizes=(64,32),max_iter=400,random_state=4,early_stopping=True))):
        md.fit(Xtr100,y_tr); preds[n]=md.predict(Xte100)
    # genomic mixed models (mixed_models.py); the G x E term is genotype x location, since the
    # test year's environments are new but its locations mostly are not
    for name,gxe in (("gblup",False),("gblup_gxe",True)):
        fit=mm.fit(y_tr,tr.Env.to_numpy(),WF[tr.gi.to_numpy()],tr["loc"].to_numpy() if gxe else None)
        be=pd.Series(mm.env_effects(fit)); lb=be.groupby(be.index.str.replace(r'_\d{4}$','',regex=True)).mean()
        lvl=te["loc"].map(lb).fillna(be.mean()).to_numpy()
        preds[name]=lvl+mm.genetic(fit,WF[te.gi.to_numpy()],te["loc"].to_numpy() if gxe else None)
        print(f"  {Y} {name}: s2u/s2e {fit['d0']:.3f}, s2v/s2e {fit['d1']:.3f}")
    preds["ens_linear"]=np.mean([preds[k] for k in ("ridge_pc50_lo","ridge_pc100_hi","env_plus_geno")],axis=0)
    preds["ens_tree"]=np.mean([preds[k] for k in ("rf_large","gbm_deep","gbm_shallow")],axis=0)
    preds["ens_all"]=np.mean([preds[k] for k in ("ridge_pc50_lo","rf_large","gbm_deep","knn10","reaction_norm")],axis=0)
    for m,p in preds.items():
        out.append(pd.DataFrame({"year":Y,"method":m,"Env":te.Env.to_numpy(),
                                 "k":te.k.to_numpy(),"y":y_te,"p":p}))
    print(f"{Y}: {len(te)} rows, {te.Env.nunique()} envs, {len(preds)} methods")
P=pd.concat(out,ignore_index=True)
P.to_csv(f"{RES}/nust_panel_predictions.csv",index=False)
print(f"\npanel: {P.method.nunique()} methods x {P.year.nunique()} years -> {len(P)} prediction rows")
