"""China maize (CUBIC) method panel -- the fifth dataset (Supplementary Section S11).

Data
----
CUBIC: 1,404 recombinant inbred lines derived by a complete-diallel plus
unbalanced breeding-like inter-cross from 24 Chinese elite maize founders
(Liu et al., Genome Biol. 2020, 21:20, doi 10.1186/s13059-020-1930-x).

Phenotype: per-site raw measurements of 23 agronomic traits at five Chinese
sites -- JL Gongzhuling (Jilin), LN Shenyang (Liaoning), BJ Beijing,
HeB Shijiazhuang (Hebei), HN Xinxiang (Henan) -- published as Supplemental
Table S12 of Jin et al., Plant Commun. 2023, 4:100473,
doi 10.1016/j.xplc.2022.100473. Site-level weather (Supplemental Table S13)
supplies the environmental covariates for the reaction-norm method.
Default trait: ear weight (EW), the yield trait of this panel.

Genotype: PLINK binary set for population GSTP004 curated by CropGS-Hub
(Chen et al., Nucleic Acids Res. 2024, 52:D1519, doi 10.1093/nar/gkad1062).
The primary deposit (NCBI PRJNA597703 / CNCB-NGDC CRA000171) is raw reads only.

See analysis/crosscrop/results/china_DATA_SOURCE.md for provenance, licences,
accessions and download dates.

Design: 5-fold cross-validation on GENOTYPES (predict untested lines), as for the common-bean
and spring-wheat panels, with the same learners, reduced-rank GBLUP models and miscalibration
variants; the differences are listed in Supplementary Table S18.

Toolchain: built under the same interpreter as every other script (versions in
china_DATA_SOURCE.md); the statistics reported for this dataset come from the same
canonical functions as for the other panels.

Marker thinning: genomic principal components are computed from every
GENO_STEP-th marker of the CropGS-Hub set (default: an evenly spaced subset of
about 100,000 markers) rather than all of them, and the decoded subset is cached
as cubic_geno_thinned.npz. Reported as a deviation; PCs at this density are the
population-structure axes the ridge and machine-learning methods use as
features, not a marker-effect model.
"""
import os, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

RAW = os.environ.get("GP_DATA", "data/raw")   # root for third-party primary data
SP = f"{RAW}/cubic"
OUT = "analysis/crosscrop/results"
TRAIT = os.environ.get("CUBIC_TRAIT", "EW")   # ear weight = the yield trait
SITES = ["JL", "LN", "BJ", "HeB", "HN"]
TARGET_SNP = int(os.environ.get("GENO_TARGET_SNP", 100000))
NPC = 60
CACHE = f"{SP}/cubic_geno_thinned.npz"
# The 4.15 GB PLINK .bed of the CropGS-Hub archive is not kept in the repository
# (see china_DATA_SOURCE.md for the URL, sha256 and unpacking step). Point
# CUBIC_BED_PREFIX at it for the one run that builds CACHE; afterwards the cache
# is all that is needed.
BED = os.environ.get("CUBIC_BED_PREFIX", f"{SP}/cubic_1404_hmp2plink_maf0.02")

# ---------------------------------------------------------------- phenotypes
ph = pd.read_csv(f"{SP}/TableS12_phenotypes.csv")
ph = ph.rename(columns={ph.columns[0]: "k"})
cols = [f"{TRAIT}_{s}" for s in SITES]
missing = [c for c in cols if c not in ph.columns]
if missing:
    sys.exit(f"trait columns absent from Table S12: {missing}")
ph = ph[["k"] + cols].melt(id_vars="k", var_name="Env", value_name="y").dropna()
ph["Env"] = ph.Env.str.split("_").str[-1]
ph["k"] = ph.k.astype(str)
print(f"phenotype ({TRAIT}): {len(ph)} rows, {ph.Env.nunique()} envs, {ph.k.nunique()} genos")
print(ph.groupby("Env").size().to_string())

# ---------------------------------------------------------------- genotypes
def read_bed_thinned(prefix, target):
    """Evenly spaced subset of a PLINK .bed, decoded to allele-1 dosage 0/1/2."""
    fam = pd.read_csv(f"{prefix}.fam", sep=r"\s+", header=None, dtype=str)
    ids = fam[1].tolist()
    n_all = sum(1 for _ in open(f"{prefix}.bim"))
    step = max(1, n_all // target)
    keep = np.arange(0, n_all, step)
    n = len(ids); bpv = (n + 3) // 4
    shift = np.array([0, 2, 4, 6], dtype=np.uint8)
    lut = np.array([2., np.nan, 1., 0.], dtype=np.float32)  # 00,01,10,11
    X = np.empty((n, len(keep)), dtype=np.float32)
    with open(f"{prefix}.bed", "rb") as f:
        assert f.read(3) == b"\x6c\x1b\x01", "not a SNP-major PLINK .bed"
        for j, s in enumerate(keep):
            f.seek(3 + int(s) * bpv)
            b = np.frombuffer(f.read(bpv), dtype=np.uint8)
            X[:, j] = lut[((b[:, None] >> shift) & 3).ravel()[:n]]
    return ids, X, n_all, len(keep)

if os.path.exists(CACHE):
    z = np.load(CACHE, allow_pickle=True)
    gids, X, n_all, n_used = list(z["ids"]), z["X"], int(z["n_all"]), int(z["n_used"])
    print(f"genotype cache: {X.shape[0]} lines x {n_used} markers (of {n_all})")
else:
    gids, X, n_all, n_used = read_bed_thinned(BED, TARGET_SNP)
    np.savez_compressed(CACHE, ids=np.array(gids, dtype=object), X=X,
                        n_all=n_all, n_used=n_used)
    print(f"genotype: {X.shape[0]} lines x {n_used} markers thinned from {n_all}")

X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
X = X[:, np.nanstd(X, axis=0) > 0]
Z = (X - X.mean(0)) / X.std(0)
U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
npc = min(NPC, U.shape[1]); PC = (U[:, :npc] * S[:npc]).astype(np.float64)
print(f"markers used {X.shape[1]}, PC1-{npc} explain "
      f"{(S**2/np.sum(S**2))[:npc].sum()*100:.1f}%")

pos = {g: i for i, g in enumerate(gids)}
before = ph.k.nunique()
ph = ph[ph.k.isin(pos)].copy(); ph["gi"] = ph.k.map(pos)
print(f"matched genotypes: {ph.k.nunique()} of {before} phenotyped lines")
kept = ph.groupby("Env").k.nunique()
print("genotypes per environment after matching:\n" + kept.to_string())
assert (kept >= 25).all(), "an environment falls below the 25-genotype floor"

# ------------------------------------------------- environmental covariates
wx = pd.read_csv(f"{SP}/TableS13_weather.csv")
wx.columns = ["site", "tmp_h", "tmp_l", "daylen"]
wx["site"] = wx.site.replace({"HB": "HeB"})
hm = wx.daylen.astype(str).str.split(":", expand=True).astype(float)
wx["daylen_h"] = hm[0] + hm[1] / 60 + hm[2] / 3600
wx["gdd"] = np.clip((wx.tmp_h + wx.tmp_l) / 2 - 10, 0, None)
ec = wx.groupby("site").agg(tmax=("tmp_h", "mean"), tmin=("tmp_l", "mean"),
                            trange=("tmp_h", "std"), daylen=("daylen_h", "mean"),
                            gdd=("gdd", "sum"), ndays=("gdd", "size"))
ecz = (ec - ec.mean()) / ec.std()
print(f"environmental covariates: {ec.shape[1]} per site, sites {list(ec.index)}")

# ---------------------------------------------------------------- the panel
def ridge(Xtr, ytr, Xte, lam):
    Xtr = np.c_[np.ones(len(Xtr)), Xtr]; Xte = np.c_[np.ones(len(Xte)), Xte]
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); A[0, 0] -= lam
    return Xte @ np.linalg.solve(A, Xtr.T @ ytr)

rng = np.random.default_rng(0)
genos = ph.k.unique(); fold = pd.Series(rng.integers(0, 5, len(genos)), index=genos)
ph["fold"] = ph.k.map(fold)
preds = {}
# Environment index: each fold's test genotypes get the mean of that fold's TRAINING genotypes
# in the environment (leave-genotypes-out, CV1). An earlier version averaged the five
# training-fold means, which put every test genotype's own phenotype into its index (four
# folds in five) -- a leak into the environment level of the predictions. Fold-specific
# indices differ slightly within an environment; that is what CV1 predictions look like.
for f in range(5):
    tr = ph[ph.fold != f]; te = ph[ph.fold == f]
    em = tr.groupby("Env").y.mean(); mu = tr.y.mean()
    y_tr = tr.y.to_numpy()
    def F(d, n, e=em):
        return np.c_[d.Env.map(e).fillna(mu).to_numpy(), PC[d.gi.to_numpy()][:, :n]]
    for n in (5, 15, 30, 60):
        for lam, tag in ((1e0, "lo"), (1e2, "hi")):
            preds.setdefault(f"ridge_pc{n}_{tag}", []).append(
                (te.index, ridge(F(tr, n), y_tr, F(te, n), lam)))
    def FE(d, n, e=em):
        ev = np.array([ecz.loc[e_].to_numpy() if e_ in ecz.index else np.zeros(ecz.shape[1])
                       for e_ in d.Env])
        P = PC[d.gi.to_numpy()][:, :n]
        return np.c_[d.Env.map(e).fillna(mu).to_numpy(), P, ev, P[:, :5] * ev[:, [0]]]
    preds.setdefault("reaction_norm_EC", []).append(
        (te.index, ridge(FE(tr, 30), y_tr, FE(te, 30), 1e1)))
    A, B = F(tr, 60), F(te, 60)
    for name, md in (("rf", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                  n_jobs=-1, random_state=0)),
                     ("gbm", HistGradientBoostingRegressor(max_depth=4, max_iter=300,
                                                           learning_rate=.06, random_state=1)),
                     ("knn10", KNeighborsRegressor(n_neighbors=10)),
                     ("mlp", MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=500,
                                          random_state=2, early_stopping=True))):
        md.fit(A, y_tr); preds.setdefault(name, []).append((te.index, md.predict(B)))
    print(f"  fold {f}: train {len(tr)} test {len(te)}")

# Genomic mixed models (mixed_models.py): GBLUP and the marker x environment GBLUP, on every
# principal component of the thinned markers; each fold's own environment effects (no averaging).
sys.path.insert(0, "analysis/crosscrop/code"); import mixed_models as mm
WF = mm.scale_features(PC)   # same npc=60 features as the ridge baselines, not the untruncated basis
gen = {"gblup": [], "gblup_gxe": []}
for f in range(5):
    tr = ph[ph.fold != f]; te = ph[ph.fold == f]
    for name, gxe in (("gblup", False), ("gblup_gxe", True)):
        fit = mm.fit(tr.y.to_numpy(), tr.Env.to_numpy(), WF[tr.gi.to_numpy()],
                     tr.Env.to_numpy() if gxe else None)
        be = pd.Series(mm.env_effects(fit))
        lvl = te.Env.map(be).fillna(be.mean()).to_numpy()
        gen[name].append((te.index, lvl + mm.genetic(fit, WF[te.gi.to_numpy()],
                                                     te.Env.to_numpy() if gxe else None)))
        print(f"  fold {f} {name}: s2u/s2e {fit['d0']:.3f}, s2v/s2e {fit['d1']:.3f}")
for name in gen:
    preds[name] = gen[name]

rows = []
for m, parts in preds.items():
    idx = np.concatenate([p[0] for p in parts]); val = np.concatenate([p[1] for p in parts])
    s = pd.Series(val, index=idx).sort_index()
    d = ph.loc[s.index, ["Env", "k", "y"]].copy(); d["p"] = s.to_numpy(); d["method"] = m
    rows.append(d)
base = pd.concat(rows, ignore_index=True)
for a, b in (("ens_lin", ["ridge_pc30_lo", "ridge_pc60_hi", "reaction_norm_EC"]),
             ("ens_ml", ["rf", "gbm", "mlp"])):
    g = base[base.method.isin(b)].groupby(["Env", "k", "y"], as_index=False).p.mean()
    g["method"] = a; base = pd.concat([base, g], ignore_index=True)

# restore the calibration axis (see widen_panel.py); recipe as in panel_rice.py
r2 = np.random.default_rng(7); extra = []
for s in ["ridge_pc30_lo", "rf", "gbm", "reaction_norm_EC", "mlp", "knn10", "ens_ml"]:
    d = base[base.method == s]; p = d.p.to_numpy(); y = d.y.to_numpy()
    for tag, q in (("shrunk", p.mean() + .25 * (p - p.mean())),
                   ("inflated", p.mean() + 3 * (p - p.mean())),
                   ("biased", p + .8 * y.std()),
                   ("noise", p + r2.normal(0, p.std(), len(p))),
                   ("shuffled", None)):
        if q is None:
            q = p.copy(); m = r2.random(len(q)) < .35; q[m] = r2.permutation(q[m])
        e = d.copy(); e["p"] = q; e["method"] = f"{s}__{tag}"; extra.append(e)
W = pd.concat([base] + extra, ignore_index=True)
W.to_csv(f"{OUT}/china_panel_wide.csv", index=False)
print(f"\nchina (CUBIC) panel: {W.method.nunique()} methods x {W.Env.nunique()} "
      f"environments, {len(W)} rows -> {OUT}/china_panel_wide.csv")
