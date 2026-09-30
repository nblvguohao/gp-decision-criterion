"""GPverdict Atlas: every public multi-environment dataset and trait we can analyse, one design.

For each dataset x trait with at least MIN_ENVS environments holding >= min_n genotypes with
markers, a panel of 17 base methods is built with the design of the paper's panels
(analysis/crosscrop/code/panel_design.py: five-fold leave-genotypes-out cross-validation; ridge on
an environment index and marker principal components; marker-only ridge; random forest, gradient
boosting, two k-nearest-neighbour learners, a multilayer perceptron; GBLUP and marker x
environment GBLUP; an unweighted ensemble). No miscalibration variants: the atlas describes the
methods as a breeder would run them.

Each panel then goes through GPverdict (tool/gpverdict) exactly as a user's upload would, and
through the out-of-sample comparison of analysis/crosscrop/code/tie_ensemble.py.

Selection direction: the higher value is selected, except for lodging and disease scores, which
are negated so that the lower value is selected (DIRECTION below). For phenology and height the
direction a programme selects is its own choice; the within-environment correlations and the
reversal rates do not depend on it, the outcome test does.

Usage (from the repository root):
    python analysis/atlas/atlas.py list            # the dataset x trait combinations and their sizes
    python analysis/atlas/atlas.py run [ID ...]    # build panels and verdicts (all by default; resumable)
    python analysis/atlas/atlas.py collect         # analysis/atlas/atlas.csv and tool/atlas/atlas.{csv,json}

Panels are cached in analysis/atlas/panels/<ID>.csv.gz and per-combination results in
analysis/atlas/results/<ID>.json. CUBIC panels contain predictions derived from CC BY-NC-ND
phenotypes: they stay local and are never published; the atlas publishes summary results only.
"""
import gzip, json, os, re, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ROOT = os.getcwd()
sys.path.insert(0, "analysis/crosscrop/code"); sys.path.insert(0, "tool")
import panel_design
import gpverdict as gv

RAW = os.environ.get("GP_DATA", "data/raw")
A = "analysis/atlas"
MIN_ENVS = 5
NPC = 80
DIRECTION = {"lodging": -1, "Lodging": -1, "Chlorosis": -1, "SDS": -1}


def pcs(X, npc=NPC):
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    X = X[:, np.nanstd(X, axis=0) > 0]
    Z = (X - X.mean(0)) / X.std(0)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    n = min(npc, U.shape[1]); return (U[:, :n] * S[:n]).astype(np.float64)


def top_of(path, stop):
    """Run a panel script's own data preparation (everything before `stop`) and return its namespace."""
    src = open(path).read(); ns = {"__name__": "atlas_loader"}
    exec(compile(src[:src.index(stop)], path, "exec"), ns)
    return ns


# ---------------------------------------------------------------- loaders: (ph[Env,k,y,gi], PC)
def load_cubic(trait):
    """panel_china.py's own preparation, run once (its marker PCA is the slow step) and cached."""
    cache = f"{A}/cache/cubic_pc.npz"
    if not os.path.exists(cache):
        os.environ["CUBIC_TRAIT"] = "EW"
        ns = top_of("analysis/crosscrop/code/panel_china.py", "# ------------------------------------------------- environmental covariates")
        np.savez_compressed(cache, ids=np.array(ns["gids"], dtype=object), PC=ns["PC"])
    z = np.load(cache, allow_pickle=True); pos = {str(g): i for i, g in enumerate(z["ids"])}
    sites = ["JL", "LN", "BJ", "HeB", "HN"]
    ph = pd.read_csv(f"{RAW}/cubic/TableS12_phenotypes.csv"); ph = ph.rename(columns={ph.columns[0]: "k"})
    ph = ph[["k"] + [f"{trait}_{s}" for s in sites]].melt(id_vars="k", var_name="Env", value_name="y").dropna()
    ph["Env"] = ph.Env.str.split("_").str[-1]; ph["k"] = ph.k.astype(str)
    ph = ph[ph.k.isin(pos)].copy(); ph["gi"] = ph.k.map(pos)
    return ph, z["PC"]


def load_bean(col):
    src = open("analysis/crosscrop/code/panel_bean.py").read().replace('"Yd_BLUE"', f'"{col}"')
    ns = {"__name__": "atlas_loader"}; exec(compile(src[:src.index("def ridge(")], "panel_bean.py", "exec"), ns)
    return ns["ph"], ns["PC"]


def load_ursn(col):
    src = open("analysis/crosscrop/code/panel_ursn.py").read().replace('"DIS"', f'"{col}"')
    ns = {"__name__": "atlas_loader"}; exec(compile(src[:src.index("def ridge(")], "panel_ursn.py", "exec"), ns)
    return ns["ph"], ns["PC"]


_NUST = {}
def load_nust(trait):
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    if "PC" not in _NUST:
        code = {"0/0": 0., "0/1": 1., "1/0": 1., "1/1": 2., "./.": np.nan}; rows = []; ids = None
        with gzip.open(f"{RAW}/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", "rt") as fh:
            for line in fh:
                if line.startswith("##"): continue
                f = line.rstrip("\n").split("\t")
                if line.startswith("#CHROM"): ids = [norm(s) for s in f[9:]]; continue
                rows.append([code.get(g.split(":")[0], np.nan) for g in f[9:]])
        X = np.asarray(rows, dtype=np.float64).T
        keep = ~pd.Index(ids).duplicated()
        _NUST["ids"] = list(np.array(ids)[keep]); _NUST["PC"] = pcs(X[keep])
        _NUST["ph"] = pd.read_csv(f"{RAW}/nust/Phenotype_Measures_Final_Master_NUST_1993_2020_years28.csv.gz", low_memory=False)
    ph = _NUST["ph"]; ph = ph[ph.Phenotype == trait].copy()
    ph["y"] = pd.to_numeric(ph.Value, errors="coerce"); ph = ph.dropna(subset=["y"])
    ph["Env"] = ph.Location.astype(str) + "_" + ph.Experiment.str.extract(r"_(\d{4})$")[0]
    ph["k"] = ph.GermplasmId.map(norm)
    ph = ph.groupby(["Env", "k"], as_index=False).y.mean()
    pos = {g: i for i, g in enumerate(_NUST["ids"])}
    ph = ph[ph.k.isin(pos)].copy(); ph["gi"] = ph.k.map(pos)
    return ph, _NUST["PC"]


def load_processed(name, trait):
    z = np.load(f"{A}/cache/{name}_pc.npz", allow_pickle=True)
    pos = {str(g): i for i, g in enumerate(z["ids"])}
    ph = pd.read_csv(f"{A}/cache/{name}_pheno.csv.gz")
    ph = ph[ph.trait_id == trait].rename(columns={"genotype_id": "k", "environment_id": "Env", "value": "y"})
    ph["k"] = ph.k.astype(str); ph = ph[ph.k.isin(pos)].copy(); ph["gi"] = ph.k.map(pos)
    return ph[["Env", "k", "y", "gi"]], z["PC"]


def cubic_traits():
    cols = pd.read_csv(f"{RAW}/cubic/TableS12_phenotypes.csv", nrows=0).columns
    sites = ["JL", "LN", "BJ", "HeB", "HN"]
    return sorted({c.rsplit("_", 1)[0] for c in cols if "_" in c and c.rsplit("_", 1)[1] in sites
                   and all(f"{c.rsplit('_', 1)[0]}_{s}" in cols for s in sites)})


LIC = {"G2F": "ODC PDDL (CyVerse doi:10.25739/tq5e-ak26)", "CUBIC": "phenotypes CC BY-NC-ND 4.0 (summary results only); genotypes CropGS-Hub",
       "VEF": "CC BY 4.0", "URSN": "CC0", "NUST": "CC BY 4.0 (Wartha et al. 2025) / SoyBase (USDA-ARS)",
       "DROPS": "phenotypes CC BY-SA 4.0 (doi:10.15454/IASSTN); statgenGWAS data, GPL-3",
       "SoyNAM": "SoyBase (USDA-ARS); cite Xavier et al. 2018, G3"}
SRC = {"G2F": "Washburn et al. 2025 (G2F 2022 competition); five verified submission files", "CUBIC": "Liu et al. 2020 Genome Biol; Jin et al. 2023 Plant Commun; CropGS-Hub",
       "VEF": "Keller et al. 2020 Front Plant Sci; doi:10.7910/DVN/XCD67U",
       "URSN": "Uniform Regional Scab Nursery 1995-2024; doi:10.5061/dryad.wstqjq2z0",
       "NUST": "Wartha et al. 2025 Crop Sci; SoyBase",
       "DROPS": "Millet et al. 2016 Plant Physiol; statgenGWAS 1.0.14",
       "SoyNAM": "Xavier et al. 2018 G3; SoyBase"}


def catalogue():
    C = [dict(id="g2f_yield", crop="maize", dataset="G2F", trait="grain yield (5 verified competition submissions)",
              loader=("g2f", "yield"), min_n=20)]
    for t in cubic_traits():
        C.append(dict(id=f"cubic_{t}", crop="maize", dataset="CUBIC", trait=t, loader=("cubic", t), min_n=25))
    for col, t in (("Yd_BLUE", "seed yield"), ("100SdW_BLUE", "100-seed weight"), ("DF_BLUE", "days to flowering"),
                   ("DPM_BLUE", "days to physiological maturity")):
        C.append(dict(id=f"vef_{col.split('_')[0]}", crop="common bean", dataset="VEF", trait=t, loader=("bean", col), min_n=25))
    for col, t in (("DIS", "FHB resistance (100 - disease score)"), ("VSK", "100 - visually scabby kernels (%)")):
        C.append(dict(id=f"ursn_{col}", crop="wheat", dataset="URSN", trait=t, loader=("ursn", col), min_n=12))
    for t in ("YieldBuA", "SeedSize", "Maturity", "Lodging", "Height", "Protein", "Oil", "SeedQuality", "Chlorosis", "SDS"):
        C.append(dict(id=f"nust_{t}", crop="soybean", dataset="NUST", trait=t, loader=("nust", t), min_n=25))
    for t in ("yield", "grain_number", "grain_weight_mg", "anthesis_d20C", "silking_d20C", "plant_height_cm"):
        C.append(dict(id=f"drops_{t}", crop="maize", dataset="DROPS", trait=t, loader=("drops", t), min_n=25))
    for t in ("yield", "days_to_maturity", "lodging", "plant_height_cm", "protein", "oil", "seed_weight_100_g"):
        C.append(dict(id=f"soynam_{t}", crop="soybean", dataset="SoyNAM", trait=t, loader=("soynam", t), min_n=25))
    return C


def load(entry):
    kind, t = entry["loader"]
    if kind == "g2f":
        import decision_swap as ds
        ph = ds.maize_long()[["Env", "k", "y", "p", "method"]].copy(); ph["k"] = ph.k.astype(str)
        n = ph.groupby("Env").k.nunique(); return ph[ph.Env.isin(n[n >= entry["min_n"]].index)].reset_index(drop=True), None
    if kind == "cubic": ph, PC = load_cubic(t)
    elif kind == "bean": ph, PC = load_bean(t)
    elif kind == "ursn": ph, PC = load_ursn(t)
    elif kind == "nust": ph, PC = load_nust(t)
    else: ph, PC = load_processed(kind, t)
    ph = ph.copy(); ph["y"] = DIRECTION.get(t, 1) * ph.y
    n = ph.groupby("Env").k.nunique(); ph = ph[ph.Env.isin(n[n >= entry["min_n"]].index)]
    return ph[["Env", "k", "y", "gi"]].reset_index(drop=True), PC


def run_one(entry):
    import tie_ensemble as te
    out = f"{A}/results/{entry['id']}.json"
    if os.path.exists(out):
        return json.load(open(out))
    t0 = time.time()
    ph, PC = load(entry)
    if ph.Env.nunique() < MIN_ENVS:
        res = dict(entry, skipped=f"only {ph.Env.nunique()} environments with >= {entry['min_n']} genotypes")
        json.dump(res, open(out, "w"), default=str); return res
    pan = f"{A}/panels/{entry['id']}.csv.gz"
    if entry["loader"][0] == "g2f":                   # the competition's own submissions, not a built panel
        P = ph.assign(method=ph.method)
    else:
        P = pd.read_csv(pan) if os.path.exists(pan) else panel_design.build(ph, PC, pan, variants=False, verbose=False)
    P["k"] = P.k.astype(str)
    v = gv.verdict(P[["Env", "k", "y", "p", "method"]], frac=0.10, min_n=entry["min_n"], B=200, check_invariance=False)
    o = v["outcome"] or {}
    rec = o.get("recovery", {})
    tie = te.run(P[["Env", "k", "y", "p", "method"]], entry["min_n"], entry["id"], n_splits=200, n_boot=100, boot_splits=10) \
        if ph.Env.nunique() >= 8 else {}
    res = dict({k: v_ for k, v_ in entry.items() if k != "loader"},
               n_env=v["data"]["environments"], n_genotypes=v["data"]["genotypes"], n_methods=v["data"]["methods"],
               cells=v["data"]["cells"], leader=v["leader"], pearson_pick=v["pearson_pick"], rmse_pick=v["rmse_pick"],
               tied=v["resolution"]["tied_with_leader"], gap=v["resolution"]["gap"],
               leader_r=float(v["summary"].loc[v["leader"], "spearman"]),
               best_pearson=float(v["summary"].pearson.max()),
               reversal=v["reversal_rmse_pearson"], method_pairs=v["method_pairs"],
               outcome_design=o.get("design"), reliability=o.get("reliability"),
               recovery_spearman=rec.get("spearman"), recovery_pearson=rec.get("pearson"), recovery_rmse=rec.get("rmse"),
               tie=tie, seconds=round(time.time() - t0))
    json.dump(res, open(out, "w"), default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"{entry['id']}: {res['n_env']} env, {res['n_genotypes']} geno, leader {res['leader']}, "
          f"{len(res['tied'])} tied, reversal {res['reversal']:.2f}, {res['seconds']} s", flush=True)
    return res


def collect():
    rows = [json.load(open(f"{A}/results/{e['id']}.json")) for e in catalogue() if os.path.exists(f"{A}/results/{e['id']}.json")]
    ok = [r for r in rows if "skipped" not in r]
    for r in ok:
        r["source"] = SRC[r["dataset"]]; r["licence"] = LIC[r["dataset"]]
        cs = [x for x in (r["recovery_spearman"], r["recovery_pearson"]) if x is not None]
        r["recovery_corr"] = float(np.mean(cs)) if cs else None
    T = pd.DataFrame([{k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items() if k != "tie"} for r in ok])
    T.to_csv(f"{A}/atlas.csv", index=False)
    os.makedirs("tool/atlas", exist_ok=True)
    pub = [c for c in T.columns if c not in ("seconds",)]
    T[pub].to_csv("tool/atlas/atlas.csv", index=False)
    intro = (f"{len(ok)} dataset × trait combinations from {T.dataset.nunique()} public multi-environment datasets in "
             f"{T.crop.nunique()} crops, each run through the same 17 prediction methods (five-fold cross-validation with "
             "genotypes held out; G2F: the five verified competition submissions) and the same checks as the Analyse page. Summary results only; no phenotypes or "
             "predictions are published. Select a row for details.")
    def clean(v):                                     # browsers reject NaN in JSON: write null
        v = v.item() if hasattr(v, "item") else v
        return None if isinstance(v, float) and not np.isfinite(v) else v
    json.dump(dict(intro=intro, rows=[{k: ([clean(x) for x in v] if isinstance(v, list) else clean(v)) for k, v in r.items()
                                       if k not in ("tie", "seconds")} for r in ok]),
              open("tool/atlas/atlas.json", "w"), allow_nan=False)
    skipped = [r for r in rows if "skipped" in r]
    print(f"atlas: {len(ok)} combinations, {len(skipped)} skipped: " + "; ".join(f"{r['id']} ({r['skipped']})" for r in skipped))


if __name__ == "__main__":
    os.makedirs(f"{A}/panels", exist_ok=True); os.makedirs(f"{A}/results", exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        for e in catalogue(): print(e["id"], e["crop"], e["dataset"], e["trait"])
    elif cmd == "run":
        want = set(sys.argv[2:])
        for e in catalogue():
            if want and e["id"] not in want and e["dataset"] not in want: continue
            try: run_one(e)
            except SystemExit as x: print(f"{e['id']}: skipped ({x})", flush=True)
            except Exception as x: print(f"{e['id']}: FAILED {type(x).__name__}: {x}", flush=True)
    elif cmd == "collect":
        collect()
