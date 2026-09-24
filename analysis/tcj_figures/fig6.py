"""Fig. 6: GPverdict, the criterion as a tool, and its verdict for Chinese maize (CUBIC).

(a) what the tool asks and answers; (b) the CUBIC verdict: 17 methods ranked by the admissible
metric, with the gap the trial resolves; (c) planning a trial: genotype-environment cells
needed to resolve an accuracy gain, with the paper's datasets and the leaderboard's top gap.
Reads analysis/crosscrop/results/gpverdict_cubic.json and gpverdict_cells.csv (written by
analysis/crosscrop/code/gpverdict_applications.py). No external artwork.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "analysis/tcj_figures"); sys.path.insert(0, "tool")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from style import *
import gpverdict as gv

apply()
R = "analysis/crosscrop/results"
J = json.load(open(f"{R}/gpverdict_cubic.json")); CELLS = pd.read_csv(f"{R}/gpverdict_cells.csv")
NAMES = {"ens_ml": "Ensemble (RF + GBM + MLP)", "rf": "Random forest", "gbm": "Gradient boosting",
         "gblup_gxe": "GBLUP × environment", "gblup": "GBLUP", "knn10": "k-nearest neighbours",
         "mlp": "Multilayer perceptron", "ens_lin": "Linear ensemble", "reaction_norm_EC": "Reaction norm (covariates)"}
def label(m):
    if m in NAMES: return NAMES[m]
    d, lam = m.replace("ridge_pc", "").split("_")
    return f"Ridge, {d} PCs, λ = {'1' if lam == 'lo' else '100'}"

W, H = 190, 104
fig = plt.figure(figsize=(W * MM, H * MM))

# ---------------------------------------------------------------- (a) workflow
ax = mm_axes(fig, 0, 70, 190, 32); ax.set_xlim(0, 190); ax.set_ylim(0, 32); ax.axis("off")
ax.text(1, 30.5, "a", fontsize=9, fontweight="bold", va="top")
def box(x, w, title, body, fc, ec):
    ax.add_patch(FancyBboxPatch((x, 4), w, 23, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=.7))
    ax.text(x + w / 2, 24.3, title, ha="center", va="top", fontsize=7.3, fontweight="bold")
    ax.text(x + w / 2, 18.6, body, ha="center", va="top", fontsize=7, linespacing=1.25)
box(3, 27, "Input", "cross-validated\npredictions by\nenvironment,\ngenotype, method", "#f2f2f2", RULE)
Q = [("1  Metric check", "which metrics can\nrank methods for\nthe decision"),
     ("2  Ranking", "leader and rank\nintervals; methods\nwithin k/√N are tied"),
     ("3  RMSE check", "what RMSE would\npick; out-of-sample\ngain of each rule"),
     ("4  Trial size", "cells needed to\nresolve a given\naccuracy gain")]
x0, w, gp = 37, 27.5, 2.0
for i, (tt, b) in enumerate(Q):
    box(x0 + i * (w + gp), w, tt, b, "#fbeaea" if i == 0 else "#eef3f9", C_ADM if i == 0 else C_BLUE)
xv = x0 + 4 * (w + gp) - gp + 6.5
box(xv, 187 - xv, "Verdict", "which method to use,\nwhich are tied,\nwhat a larger trial\nwould resolve", "#e9f4ec", C_GREEN)
for xa, xb in ((30.4, x0 - 0.6), (x0 + 4 * (w + gp) - gp + 0.4, xv - 0.6)):
    ax.annotate("", xy=(xb, 15.5), xytext=(xa, 15.5), arrowprops=dict(arrowstyle="-|>", lw=.8, color=INK))
ax.text(95, 0.6, "GPverdict runs in a web browser, with the data staying on the user's computer, or as a Python package",
        ha="center", va="bottom", fontsize=7, color=MUTED, style="italic")

# ---------------------------------------------------------------- (b) CUBIC verdict
ax = mm_axes(fig, 44, 7, 62, 58)
rank = J["ranking"]; names = [m for m, _ in rank]; vals = np.array([v for _, v in rank])
y = np.arange(len(vals))[::-1]
lead = vals[0]; gap = J["gap"]
ax.axvspan(lead - gap, lead, color=C_ADM, alpha=.10, lw=0)
ax.axvline(lead - gap, color=C_ADM, lw=.7, ls="--")
tied = set(J["tied"])
for yi, m, v in zip(y, names, vals):
    ax.plot([0, v], [yi, yi], color=RULE, lw=.8, zorder=1)
    ax.scatter([v], [yi], s=16, color=C_ADM if m in tied else C_INADM, zorder=3, lw=0)
ax.set_yticks(y); ax.set_yticklabels([label(m) for m in names], fontsize=7)
for tl, m in zip(ax.get_yticklabels(), names):
    if m in tied: tl.set_color(C_ADM); tl.set_fontweight("bold")
ax.set_xlim(0, 0.46); ax.set_ylim(-0.8, len(vals) - 0.2)
ax.set_xlabel("mean within-environment rank correlation")
ax.text(lead + 0.006, len(vals) - 4.6, f"gap the trial\nresolves:\n{gap:.3f}", ha="left", va="top", fontsize=7, color=C_ADM)
ax.text(0.348, 3.0, f"CUBIC ear weight,\n17 methods, 5 sites\n\nfirst two {J['margin']:.4f}\napart: ≈{J['cells_to_resolve_margin']/1e6:.0f} million\ncells to resolve",
        fontsize=7, va="center", ha="left", color=INK)
ax.text(-0.72, 1.035, "b", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

# ---------------------------------------------------------------- (c) planning a trial
ax = mm_axes(fig, 133, 7, 55, 58)
g = np.logspace(np.log10(0.002), np.log10(0.2), 200)
for f, ls in ((0.05, ":"), (0.10, "-"), (0.20, "--")):
    ax.plot(g, gv.cells_needed(g, f), color=INK, lw=1.0 if f == 0.10 else .8, ls=ls, label=f"top {f:.0%}")
cols = {"maize (G2F 2022 test set)": SPECIES["maize"], "common bean": SPECIES["bean"], "spring wheat": SPECIES["spring wheat"], "soybean": SPECIES["soybean"]}
for _, r in CELLS.iterrows():
    ax.scatter([r.resolvable_gap], [r.cells], s=18, color=cols[r.dataset], zorder=3, lw=0)
ax.scatter([J["gap"]], [J["cells"]], s=18, color=C_GREEN, zorder=3, lw=0)
lab = {"maize (G2F 2022 test set)": "G2F test set", "common bean": "common bean", "spring wheat": "spring wheat", "soybean": "soybean"}
from matplotlib.lines import Line2D
hs = [Line2D([], [], ls="", marker="o", ms=4, color=cols[r.dataset], label=lab[r.dataset]) for _, r in CELLS.sort_values("cells", ascending=False).iterrows()]
hs.insert(2, Line2D([], [], ls="", marker="o", ms=4, color=C_GREEN, label="CUBIC"))
top12 = 0.003; n12 = gv.cells_needed(top12, 0.10)
ax.scatter([top12], [n12], s=20, marker="D", color=C_ADM, zorder=3, lw=0)
ax.annotate("G2F 1st vs 2nd team,\n0.003 apart:\n≈1 million cells", xy=(top12, n12), xytext=(0.0052, 2.6e6), fontsize=7, color=C_ADM,
            va="bottom", ha="left", arrowprops=dict(arrowstyle="-", lw=.6, color=C_ADM, shrinkA=0, shrinkB=3))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.002, 0.2); ax.set_ylim(80, 6e7)
ax.set_xticks([0.003, 0.01, 0.03, 0.1]); ax.set_xticklabels(["0.003", "0.01", "0.03", "0.1"])
ax.set_xlabel("accuracy gain to resolve (Δr)"); ax.set_ylabel("genotype–environment cells needed")
leg = ax.legend(loc="lower left", fontsize=7, handlelength=2.2, title="selection", title_fontsize=7)
ax.add_artist(leg)
ax.legend(handles=hs, loc="upper right", fontsize=7, handletextpad=0.2, borderaxespad=0.2, title="resolvable at\neach design", title_fontsize=7)
ax.text(-0.30, 1.035, "c", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

save(fig, "Fig6", DOUBLE, tight=False)
print("wrote analysis/tcj_figures/Fig6.pdf/.png")
