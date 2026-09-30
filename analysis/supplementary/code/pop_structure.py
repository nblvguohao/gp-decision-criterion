"""Population structure of the three constructed panels.

For the genotypes that enter each panel's analysis, the markers the panel reads are standardised and
decomposed exactly as the panel scripts do (panel_bean.py, panel_ursn.py, prep_nust.py): mean
imputation, monomorphic markers dropped, SVD of the standardised matrix. Reported: genotypes, markers,
the share of marker variance on PC1 and PC2, the number of PCs needed for 50 % and 80 %, and the
median off-diagonal VanRaden (method 1) relationship; Fig. S4 plots PC1 against PC2.

CUBIC is described from its source publication rather than plotted (its phenotypes are CC BY-NC-ND and
its genotypes come from a separate portal); the G2F competition analysis uses no markers.

Output (new files): analysis/supplementary/results/pop_structure.csv (summary). PC scores are not written
(derived data); Fig. S4 is drawn by the figure scripts.
"""
import os, sys, re, gzip
import numpy as np, pandas as pd
GPDC = os.environ.get("GPDC", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
RAWROOT = os.environ.get("GP_DATA", os.path.join(GPDC, "data", "raw"))
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
OUT = os.path.join(BASE, "results", "pop_structure.csv")
os.chdir(GPDC)
X_ = "analysis/crosscrop/results"


def exec_top(path, stop):
    src = open(path).read(); ns = {"__name__": "pop_loader"}
    os.environ.setdefault("GP_DATA", RAWROOT)
    exec(compile(src[:src.index(stop)], path, "exec"), ns); return ns


def bean():
    ns = exec_top("analysis/crosscrop/code/panel_bean.py", "def ridge(")
    keep = sorted(set(ns["ph"].gi)); return ns["X"][keep], len(keep)


def ursn():
    ns = exec_top("analysis/crosscrop/code/panel_ursn.py", "def ridge(")
    keep = sorted(set(ns["ph"].gi)); return ns["X"][keep], len(keep)


def soybean():
    norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
    code = {"0/0": 0., "0/1": 1., "1/0": 1., "1/1": 2., "./.": np.nan}; rows = []; ids = None
    with gzip.open(f"{RAWROOT}/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("##"): continue
            f = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"): ids = [norm(s) for s in f[9:]]; continue
            rows.append([code.get(g.split(":")[0], np.nan) for g in f[9:]])
    X = np.asarray(rows, dtype=np.float64).T
    X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
    used = set(pd.read_csv(f"{X_}/nust_panel_wide.csv.gz", usecols=["k"]).k.astype(str).map(norm))
    first = {}
    for i, g in enumerate(ids):
        if g in used and g not in first: first[g] = i
    keep = sorted(first.values()); return X[keep], len(keep)


def summarise(X):
    X = X[:, np.nanstd(X, axis=0) > 0]
    Z = (X - X.mean(0)) / X.std(0)
    U, S, _ = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    ev = S ** 2 / np.sum(S ** 2); cum = np.cumsum(ev)
    p = X.mean(0) / 2; W = X - 2 * p; G = W @ W.T / (2 * np.sum(p * (1 - p)))
    off = G[np.triu_indices(len(G), 1)]
    return dict(markers=X.shape[1], pc1=ev[0], pc2=ev[1], pcs_50=int(np.searchsorted(cum, 0.5) + 1),
                pcs_80=int(np.searchsorted(cum, 0.8) + 1), grm_offdiag_median=float(np.median(off)),
                grm_offdiag_p95=float(np.quantile(off, 0.95))), U[:, :2] * S[:2]


def main():
    if os.path.exists(OUT) and os.environ.get("OVERWRITE") != "1":
        sys.exit(f"{os.path.relpath(OUT, GPDC)} exists; new results go to new files")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows, scores = [], {}
    for lab, fn in (("common bean", bean), ("spring wheat", ursn), ("soybean", soybean)):
        X, n = fn(); s, sc = summarise(X); rows.append(dict(dataset=lab, genotypes=n, **s)); scores[lab] = sc
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows); T.to_csv(OUT, index=False)
    print(T.round(4).to_string(index=False)); print(f"-> {os.path.relpath(OUT, GPDC)}")


if __name__ == "__main__":
    main()
