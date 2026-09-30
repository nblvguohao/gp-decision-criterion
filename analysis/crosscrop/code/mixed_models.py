"""Genomic mixed models for the method panels: GBLUP and a marker x environment GBLUP.

Both models carry environment fixed effects and genomic random effects whose covariance
is WW', with W the genotype principal components supplied by the caller. With every
principal component, WW' is proportional to the realised relationship G of the
standardised markers and the model is equivalent to ridge regression on every marker
(rrBLUP). The panels supply only their leading components (80 in common bean and spring
wheat, 100 in soybean, 60 in Chinese maize), so the models they fit are reduced-rank
GBLUP: G truncated to those components. Variance components are estimated by REML.

    gblup      y = X b + W u + e                    u ~ N(0, s2u I)
    gblup_gxe  y = X b + W u + W_h v_h + e          v_h ~ N(0, s2v I) for each group h

In the G x E model every level h of a grouping carries its own genomic effect v_h, the
marker x environment model of Lopez-Cruz et al. (2015): the grouping is the environment in
the leave-genotypes-out panels and the location in the forward-year soybean panel, whose
test environments are new years at known locations.

REML is evaluated after projecting out the environment fixed effects (centring y and W
within environment), which gives the restricted likelihood exactly. The group-specific
effects are integrated out group by group through the eigendecomposition of each group's
kernel, so one evaluation costs one q x q Cholesky factorisation, q the number of
principal components.
"""
import numpy as np
from scipy.optimize import minimize


def scale_features(PC):
    """Scale genotype principal components so that the mean diagonal of WW' is one."""
    PC = np.asarray(PC, dtype=np.float64)
    return PC / np.sqrt(np.mean(np.sum(PC ** 2, axis=1)))


def _centre(v, env_codes, n_env):
    """Subtract the environment mean from every row of v (vector or matrix)."""
    cnt = np.bincount(env_codes, minlength=n_env).astype(float)
    if v.ndim == 1:
        m = np.bincount(env_codes, weights=v, minlength=n_env) / cnt
        return v - m[env_codes]
    m = np.zeros((n_env, v.shape[1]))
    np.add.at(m, env_codes, v)
    m /= cnt[:, None]
    return v - m[env_codes]


def fit(y, env, W, group=None):
    """REML fit. y (n,), env (n,) labels, W (n, q) features of each row's genotype,
    group (n,) labels for the G x E term or None for plain GBLUP.
    Returns a dict with the environment effects, the genomic effects and the variance ratios."""
    y = np.asarray(y, dtype=np.float64); W = np.asarray(W, dtype=np.float64)
    envs, ec = np.unique(np.asarray(env), return_inverse=True)
    n, q = W.shape; px = len(envs)
    yt = _centre(y, ec, px); Wt = _centre(W, ec, px)
    dof = n - px

    if group is None:                                    # one variance ratio: eigen trick
        mu, Q = np.linalg.eigh(Wt.T @ Wt); mu = np.clip(mu, 0, None)
        c = Q.T @ (Wt.T @ yt); yy = yt @ yt

        def f(ld0):
            d0 = np.exp(ld0); s = 1 / d0 + mu
            quad = yy - np.sum(c ** 2 / s)
            return dof * np.log(quad) + np.sum(np.log(s)) + q * ld0

        ld0 = minimize(lambda t: f(t[0]), x0=[0.0], method="L-BFGS-B", bounds=[(-12, 12)]).x[0]
        d0 = np.exp(ld0)
        u = Q @ (c / (1 / d0 + mu))
        b = np.bincount(ec, weights=y - W @ u, minlength=px) / np.bincount(ec, minlength=px)
        return dict(envs=envs, b=b, u=u, v={}, d0=d0, d1=0.0)

    groups, gc = np.unique(np.asarray(group), return_inverse=True)
    blocks = []                                         # per group: eigenvalues, V'P_h, V'y_h
    for h in range(len(groups)):
        r = np.flatnonzero(gc == h)
        P = Wt[r]; lam, V = np.linalg.eigh(P @ P.T); lam = np.clip(lam, 0, None)
        blocks.append((r, lam, V, V.T @ P, V.T @ yt[r]))

    def parts(d0, d1):
        S = np.eye(q) / d0; c = np.zeros(q); yRy = 0.0; ld = 0.0
        for _, lam, _, G, gy in blocks:
            w = 1 / (1 + d1 * lam)
            S += G.T @ (w[:, None] * G); c += G.T @ (w * gy)
            yRy += np.sum(w * gy ** 2); ld += np.sum(np.log1p(d1 * lam))
        return S, c, yRy, ld

    def f(t):
        d0, d1 = np.exp(t)
        S, c, yRy, ld = parts(d0, d1)
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError:
            return 1e300
        z = np.linalg.solve(L, c)
        quad = yRy - z @ z
        if quad <= 0:
            return 1e300
        return dof * np.log(quad) + ld + q * t[0] + 2 * np.sum(np.log(np.diag(L)))

    best = min((minimize(f, x0=x0, method="Nelder-Mead", options=dict(xatol=1e-3, fatol=1e-4, maxiter=400))
                for x0 in ([0.0, -2.0], [-2.0, 0.0])), key=lambda o: o.fun)
    d0, d1 = np.exp(np.clip(best.x, -12, 12))
    S, c, _, _ = parts(d0, d1)
    u = np.linalg.solve(S, c)
    v = {}
    fitted_g = W @ u
    for h, (r, lam, V, G, gy) in enumerate(blocks):
        rr = V @ ((gy - G @ u) / (1 + d1 * lam))        # R^-1 (y~ - W~ u) in group h
        v[groups[h]] = d1 * (Wt[r].T @ rr)
        fitted_g[r] += W[r] @ v[groups[h]]
    b = np.bincount(ec, weights=y - fitted_g, minlength=px) / np.bincount(ec, minlength=px)
    return dict(envs=envs, b=b, u=u, v=v, d0=d0, d1=d1)


def genetic(model, W, group=None):
    """Genomic part of the prediction for rows with features W (and G x E group labels)."""
    g = np.asarray(W, dtype=np.float64) @ model["u"]
    if group is not None and model["v"]:
        group = np.asarray(group)
        for h, vh in model["v"].items():
            r = group == h
            if r.any():
                g[r] += W[r] @ vh
    return g


def env_effects(model):
    """Estimated environment fixed effects as a dict."""
    return dict(zip(model["envs"], model["b"]))
