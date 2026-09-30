"""The final method-panel design, shared by the rebuilt rice and CAIGE-wheat panels.

The same methods, cross-validation and variants as panel_ursn.py (the spring-wheat panel
reported in the paper): five-fold leave-genotypes-out CV; ridge on a fold-specific
environment index plus marker principal components at four dimensionalities and two
penalties; a marker-only ridge; random forest, gradient boosting, two k-nearest-neighbour
learners and a multilayer perceptron; GBLUP and the marker x environment GBLUP
(mixed_models.py) on the same principal components; an unweighted ensemble; and the five
miscalibration variants of seven members. Each fold's test genotypes take the environment
index and environment effects estimated from that fold's training genotypes only.

build(ph, PC, out_csv) takes ph with columns Env, k, y, gi (row of PC) and writes the wide
panel file in the format every downstream script reads.
"""
import sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
sys.path.insert(0, "analysis/crosscrop/code")
import mixed_models as mm


def ridge(Xtr, ytr, Xte, lam):
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xte = np.c_[np.ones(len(Xte)), Xte]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); A[0, 0] -= lam
    return Xte @ np.linalg.solve(A, Xtr.T @ ytr)


def build(ph, PC, out_csv, variants=True, n_jobs=-1, verbose=True):
    ph = ph[["Env", "k", "y", "gi"]].reset_index(drop=True).copy()
    NPC = PC.shape[1]
    dims = sorted({5, 20, 40, NPC} & set(range(1, NPC + 1)) | {NPC})
    rng = np.random.default_rng(0)
    gs = ph.k.unique(); ph["fold"] = ph.k.map(pd.Series(rng.integers(0, 5, len(gs)), index=gs))
    preds = {}
    for f in range(5):
        tr = ph[ph.fold != f]; te = ph[ph.fold == f]
        em = tr.groupby("Env").y.mean(); mu = tr.y.mean(); y_tr = tr.y.to_numpy()
        F = lambda d, n, e=em: np.c_[d.Env.map(e).fillna(mu).to_numpy(), PC[d.gi.to_numpy()][:, :n]]
        for n in dims:
            for lam, t in ((1e0, "lo"), (1e2, "hi")):
                preds.setdefault(f"ridge_pc{n}_{t}", []).append((te.index, ridge(F(tr, n), y_tr, F(te, n), lam)))
        preds.setdefault(f"markeronly_pc{NPC}", []).append(
            (te.index, ridge(PC[tr.gi.to_numpy()], y_tr, PC[te.gi.to_numpy()], 1e2)))
        A_, B_ = F(tr, NPC), F(te, NPC)
        for n, md in (("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=3, n_jobs=n_jobs, random_state=0)),
                      ("gbm", HistGradientBoostingRegressor(max_depth=4, max_iter=300, learning_rate=.06, random_state=1)),
                      ("knn10", KNeighborsRegressor(n_neighbors=10)),
                      ("knn30", KNeighborsRegressor(n_neighbors=30)),
                      ("mlp", MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=500, random_state=2, early_stopping=True))):
            md.fit(A_, y_tr); preds.setdefault(n, []).append((te.index, md.predict(B_)))
    WF = mm.scale_features(PC)
    for f in range(5):
        tr = ph[ph.fold != f]; te = ph[ph.fold == f]
        for name, gxe in (("gblup", False), ("gblup_gxe", True)):
            fit = mm.fit(tr.y.to_numpy(), tr.Env.to_numpy(), WF[tr.gi.to_numpy()], tr.Env.to_numpy() if gxe else None)
            be = pd.Series(mm.env_effects(fit))
            lvl = te.Env.map(be).fillna(be.mean()).to_numpy()
            preds.setdefault(name, []).append(
                (te.index, lvl + mm.genetic(fit, WF[te.gi.to_numpy()], te.Env.to_numpy() if gxe else None)))
    rows = []
    for m, parts in preds.items():
        s = pd.Series(np.concatenate([p[1] for p in parts]), index=np.concatenate([p[0] for p in parts])).sort_index()
        d = ph.loc[s.index, ["Env", "k", "y"]].copy(); d["p"] = s.to_numpy(); d["method"] = m; rows.append(d)
    base = pd.concat(rows, ignore_index=True)
    g = base[base.method.isin(["ridge_pc40_lo" if 40 in dims else f"ridge_pc{NPC}_lo", "rf", "gbm"])] \
        .groupby(["Env", "k", "y"], as_index=False).p.mean()
    g["method"] = "ens"; base = pd.concat([base, g], ignore_index=True)
    if not variants:                                   # the atlas uses base methods only
        if out_csv: base.to_csv(out_csv, index=False)
        return base
    r2 = np.random.default_rng(7); extra = []
    parents = ["ridge_pc40_lo" if 40 in dims else f"ridge_pc{NPC}_lo", "rf", "gbm", "mlp", "knn10", f"markeronly_pc{NPC}", "ens"]
    for s in parents:
        d = base[base.method == s]; p = d.p.to_numpy(); y = d.y.to_numpy()
        for tag, q in (("shrunk", p.mean() + .25 * (p - p.mean())), ("inflated", p.mean() + 3 * (p - p.mean())),
                       ("biased", p + .8 * y.std()), ("noise", p + r2.normal(0, p.std(), len(p))), ("shuffled", None)):
            if q is None:
                q = p.copy(); m = r2.random(len(q)) < .35; q[m] = r2.permutation(q[m])
            e = d.copy(); e["p"] = q; e["method"] = f"{s}__{tag}"; extra.append(e)
    W = pd.concat([base] + extra, ignore_index=True)
    W.to_csv(out_csv, index=False)
    if verbose: print(f"{out_csv}: {W.method.nunique()} methods x {W.Env.nunique()} envs, {len(W)} rows")
    return W
