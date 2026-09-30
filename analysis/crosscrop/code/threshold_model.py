"""The surrogate threshold: how large an accuracy gap before the more accurate method
reliably selects better material.

For every pair of methods, dr is the mean difference in within-environment Pearson r and
dg the mean difference in realised selection differential (top f of genotypes, in
phenotypic SD) over the environments both methods cover; each pair is oriented so dr >= 0
and "agree" means the more accurate method also has the larger dg.

Model. Relabelling the two methods of a pair leaves their joint distribution unchanged, so
at dr = 0 the probability of agreement is exactly 1/2 whatever the quality of accuracy as a
surrogate. The probability is therefore modelled as a logistic curve through that point,
P(agree | dr) = 1 / (1 + exp(-beta * dr)), fitted by maximum likelihood, and the threshold
is the dr at which it reaches 0.95: dr* = logit(0.95) / beta. A monotone (isotonic) fit is
reported as a check on the logistic shape.

Pairs tied on accuracy or on outcome carry no information and are dropped (scope.ACC_TIE,
scope.OUT_TIE). Pairs need at least MIN_ENV shared environments. Intervals resample methods
and environments together (two-way cluster bootstrap); the out-of-sample variant takes dr from
one random half of the environments and dg from the other.
"""
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import pearsonr
import scope

LOGIT95 = np.log(0.95 / 0.05)
MIN_ENV = 8


def build(P, frac, min_n):
    """methods, environments, R and G (methods x environments, NaN where skipped) and the
    selected sets (for shortlist overlap)."""
    meths = sorted(P.method.unique())
    envs = sorted(P.Env.unique())
    ei = {e: j for j, e in enumerate(envs)}
    R = np.full((len(meths), len(envs)), np.nan); G = R.copy(); TOP = {}
    for i, m in enumerate(meths):
        for e, g in P[P.method == m].groupby("Env"):
            if len(g) < min_n or g.p.nunique() < 2 or g.y.nunique() < 2:
                continue
            y, p = g.y.to_numpy(), g.p.to_numpy(); k = max(1, int(round(frac * len(g))))
            top = np.argsort(-p, kind="mergesort")[:k]
            R[i, ei[e]] = pearsonr(y, p)[0]
            G[i, ei[e]] = (y[top].mean() - y.mean()) / y.std()
            TOP[(i, ei[e])] = frozenset(g.k.to_numpy()[top]) if "k" in g else None
    return meths, envs, R, G, TOP


def pairs(R, G, w_r=None, w_g=None, midx=None, min_env=MIN_ENV, eligible_on_full=False):
    """Oriented, tie-free (dr, dg) for all pairs of the methods in midx (rows of R, G, with
    repeats allowed), dr weighted by environment weights w_r and dg by w_g. A pair needs
    min_env shared environments: among the weighted ones, or, with eligible_on_full, in the
    full data (so that a bootstrap draw does not change which pairs are eligible)."""
    M, E = R.shape
    midx = np.arange(M) if midx is None else np.asarray(midx)
    w_r = np.ones(E) if w_r is None else np.asarray(w_r, float)
    w_g = w_r if w_g is None else np.asarray(w_g, float)
    Rs, Gs = R[midx], G[midx]
    ok = np.isfinite(Rs) & np.isfinite(Gs)
    R0, G0 = np.where(ok, Rs, 0.0), np.where(ok, Gs, 0.0)
    i, j = np.triu_indices(len(midx), 1)
    keep = midx[i] != midx[j]
    i, j = i[keep], j[keep]
    both = ok[i] & ok[j]
    nr = (both * w_r).sum(1); ng = (both * w_g).sum(1)
    nshared = both.sum(1) if eligible_on_full else np.minimum((both * (w_r > 0)).sum(1), (both * (w_g > 0)).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        dr = ((R0[j] - R0[i]) * both * w_r).sum(1) / nr
        dg = ((G0[j] - G0[i]) * both * w_g).sum(1) / ng
    good = (nshared >= min_env) & np.isfinite(dr) & np.isfinite(dg)
    dr, dg = dr[good], dg[good]
    s = np.where(dr < 0, -1.0, 1.0)
    dr, dg = dr * s, dg * s
    untied = (np.abs(dr) >= scope.ACC_TIE) & (np.abs(dg) >= scope.OUT_TIE)
    return dr[untied], dg[untied]


def fit_beta(dr, dg):
    y = (dg > 0).astype(float)
    if len(y) < 10:
        return np.nan

    def nll(b):
        z = b * dr
        return np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y))
    r = minimize_scalar(nll, bounds=(1e-6, 1e5), method="bounded")
    return float(r.x)


def threshold(dr, dg):
    """dr at which the fitted agreement reaches 0.95; +inf when that lies beyond the largest
    observed dr (the data then bound the threshold from below only)."""
    b = fit_beta(dr, dg)
    if not np.isfinite(b) or b <= 0:
        return np.nan
    t = LOGIT95 / b
    return t if t <= dr.max() else np.inf


def isotonic_crossing(dr, dg, level=0.95):
    """Smallest dr at which a non-decreasing fit of agreement on dr reaches `level`."""
    if len(dr) < 10:
        return np.nan
    o = np.argsort(dr); x, y = dr[o], (dg[o] > 0).astype(float)
    val, wt = list(y), [1.0] * len(y); blocks = [[k] for k in range(len(y))]
    k = 0
    while k < len(val) - 1:                      # pool adjacent violators
        if val[k] > val[k + 1]:
            tot = wt[k] + wt[k + 1]
            val[k] = (val[k] * wt[k] + val[k + 1] * wt[k + 1]) / tot; wt[k] = tot
            blocks[k] += blocks[k + 1]
            del val[k + 1], wt[k + 1], blocks[k + 1]
            k = max(k - 1, 0)
        else:
            k += 1
    for v, b in zip(val, blocks):
        if v >= level:
            return float(x[b[0]])
    return np.nan


def families(meths):
    fam = pd.Series([m.split("__")[0] for m in meths])
    return [np.flatnonzero((fam == f).to_numpy()) for f in pd.unique(fam)]


def boot_threshold(R, G, meths, rng, B=500):
    """Two-way cluster bootstrap: resample method families and environments together."""
    F = families(meths); E = R.shape[1]; out = []
    for _ in range(B):
        midx = np.concatenate([F[k] for k in rng.integers(0, len(F), len(F))])
        w = np.bincount(rng.integers(0, E, E), minlength=E).astype(float)
        out.append(threshold(*pairs(R, G, w, w, midx, eligible_on_full=True)))
    out = np.array(out)
    return out


def oos_threshold(R, G, rng, splits=200):
    """dr from one random half of the environments, dg from the other; pairs pooled."""
    E = R.shape[1]; DR, DG = [], []
    for _ in range(splits):
        e = rng.permutation(E); wa = np.zeros(E); wb = np.zeros(E)
        wa[e[:E // 2]] = 1; wb[e[E // 2:]] = 1
        dr, dg = pairs(R, G, wa, wb, min_env=max(2, MIN_ENV // 2))
        DR.append(dr); DG.append(dg)
    return threshold(np.concatenate(DR), np.concatenate(DG))


def overlap_at_equal_accuracy(R, TOP, meths, cut=0.01, min_env=MIN_ENV):
    """Mean shortlist overlap for untied pairs whose |dr| < cut."""
    M, E = R.shape; ov = []
    for a in range(M):
        for b in range(a + 1, M):
            sh = [e for e in range(E) if np.isfinite(R[a, e]) and np.isfinite(R[b, e])]
            if len(sh) < min_env:
                continue
            dr = abs(np.mean([R[b, e] - R[a, e] for e in sh]))
            if dr < scope.ACC_TIE or dr >= cut:
                continue
            o = [len(TOP[(a, e)] & TOP[(b, e)]) / max(len(TOP[(a, e)]), 1) for e in sh
                 if TOP.get((a, e)) is not None and TOP.get((b, e)) is not None]
            if o:
                ov.append(np.mean(o))
    return (float(np.mean(ov)), len(ov)) if ov else (np.nan, 0)
