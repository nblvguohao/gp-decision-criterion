"""Fig. S4: population structure of the three constructed panels (first two marker principal components).

The PCA is the one of analysis/supplementary/code/pop_structure.py (imported, not re-implemented): for the genotypes
each panel analyses, the markers it reads are mean-imputed, monomorphic markers dropped, standardised and
decomposed by SVD. The summary is checked against analysis/supplementary/results/pop_structure.csv before plotting.
Crop, genotype and marker counts are given in the caption, not in the figure.

    python analysis/tcj_figures/figS4.py      -> analysis/tcj_figures/FigS4.{pdf,png}
"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "analysis/supplementary/code"); sys.path.insert(0, "analysis/tcj_figures")
import pop_structure as ps
import style
style.apply()
import matplotlib.pyplot as plt

rows, scores = [], {}
for lab, fn in (("common bean", ps.bean), ("spring wheat", ps.ursn), ("soybean", ps.soybean)):
    X, n = fn(); s, sc = ps.summarise(X); rows.append(dict(dataset=lab, genotypes=n, **s)); scores[lab] = sc
T = pd.DataFrame(rows)
REF = pd.read_csv("analysis/supplementary/results/pop_structure.csv")
assert list(REF.dataset) == list(T.dataset) and (REF.genotypes.values == T.genotypes.values).all()
assert np.allclose(REF[["pc1", "pc2"]].values, T[["pc1", "pc2"]].values, rtol=1e-9, atol=1e-12)

col = {"common bean": style.SPECIES["bean"], "spring wheat": style.SPECIES["spring wheat"], "soybean": style.SPECIES["soybean"]}
W = style.DOUBLE
fig, axs = plt.subplots(1, 3, figsize=(W, W / 3.2))
for ax, (lab, sc), L in zip(axs, scores.items(), "ABC"):
    r = T.set_index("dataset").loc[lab]
    ax.scatter(sc[:, 0], sc[:, 1], s=2.5, lw=0, alpha=.55, color=col[lab])
    ax.set_xlabel(f"PC1 ({100 * r.pc1:.1f} %)"); ax.set_ylabel(f"PC2 ({100 * r.pc2:.1f} %)")
    style.panel(ax, L)
fig.tight_layout(w_pad=1.2)
print("FigS4 width mm", round(style.save(fig, "FigS4", style.DOUBLE), 1))
