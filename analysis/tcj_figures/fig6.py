"""Fig. 6: GPverdict, the criterion as a tool, and its verdict for Chinese maize (CUBIC).

(A) screenshots of the web page running its bundled example; (b) the CUBIC verdict: 17 methods ranked by the admissible
metric, with the gap the trial resolves; (c) planning a trial: genotype-environment cells
needed to resolve an accuracy gain inside the design grid (extrapolation shaded), with the
paper's datasets and the leaderboard's top gap.
Reads analysis/crosscrop/results/gpverdict_cubic.json and gpverdict_cells.csv (written by
analysis/crosscrop/code/gpverdict_applications.py) and the two screenshots in assets/.
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

W, H = 190, 136
fig = plt.figure(figsize=(W * MM, H * MM))

# ---------------------------------------------------------------- (a) the web page, as it runs
# screenshots of the published page running its bundled example (assets/capture_web.py)
from PIL import Image
IMG_IN = Image.open("analysis/tcj_figures/assets/gpverdict_web_input.png").convert("RGB")
IMG_REP = Image.open("analysis/tcj_figures/assets/gpverdict_web_report.png").convert("RGB")
IMG_REP = IMG_REP.crop((0, 0, IMG_REP.width, 1200))          # the report down to the end of its verdict
HA = 58                                                       # image height, mm
wi = HA * IMG_IN.width / IMG_IN.height; wr = HA * IMG_REP.width / IMG_REP.height
xi = 3; xr = W - 3 - wr
for x0, img, cap in ((xi, IMG_IN, "Input: the page loads Python in the browser; data are not uploaded"),
                     (xr, IMG_REP, "Report for the bundled spring wheat example (verdict section)")):
    ax = mm_axes(fig, x0, 70, img.width * HA / img.height, HA)
    ax.imshow(np.asarray(img), interpolation="lanczos"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(RULE); sp.set_linewidth(.6)
    mm_text(fig, x0, 70 + HA + 1.2, cap, fontsize=7, color=MUTED, va="bottom", ha="left")
mm_text(fig, 1, H - 1, "A", fontsize=9, fontweight="bold", va="top")
ax = mm_axes(fig, xi + wi, 70, xr - xi - wi, HA); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.annotate("", xy=(.92, .5), xytext=(.08, .5), arrowprops=dict(arrowstyle="-|>", lw=.9, color=INK))

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
ax.text(0.348, 3.0, "CUBIC ear weight,\n17 methods, 5 sites\n\nrank and Pearson\ncorrelations mark\nthe same three",
        fontsize=7, va="center", ha="left", color=INK)
ax.text(-0.72, 1.035, "B", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

# ---------------------------------------------------------------- (c) planning a trial
ax = mm_axes(fig, 133, 7, 55, 58)
g = np.logspace(np.log10(0.002), np.log10(0.2), 200)
GRID_MIN, GRID_MAX = 25 * 10, 400 * 200                       # cells spanned by the design grid (Table S11)
ax.axhspan(GRID_MAX, 6e7, color="#f1f1f1", lw=0, zorder=0)
ax.text(0.0023, 3.2e7, "beyond the design grid: extrapolated", fontsize=7, color=MUTED, ha="left", va="center")
for f, ls in ((0.05, ":"), (0.10, "-"), (0.20, "--")):
    n = gv.cells_needed(g, f); inside = (n >= GRID_MIN) & (n <= GRID_MAX)
    ax.plot(np.where(inside, g, np.nan), n, color=INK, lw=1.0 if f == 0.10 else .8, ls=ls, label=f"top {f:.0%}")
    ax.plot(np.where(n >= GRID_MAX, g, np.nan), n, color=C_INADM, lw=.8, ls=ls)
    ax.plot(np.where(n <= GRID_MIN, g, np.nan), n, color=C_INADM, lw=.8, ls=ls)
cols = {"maize (G2F 2022 test set)": SPECIES["maize"], "common bean": SPECIES["bean"], "spring wheat": SPECIES["spring wheat"], "soybean": SPECIES["soybean"]}
for _, r in CELLS.iterrows():
    ax.scatter([r.resolvable_gap], [r.cells], s=18, color=cols[r.dataset], zorder=3, lw=0)
ax.scatter([J["gap"]], [J["cells"]], s=18, color=C_GREEN, zorder=3, lw=0)
lab = {"maize (G2F 2022 test set)": "G2F maize", "common bean": "common bean", "spring wheat": "spring wheat", "soybean": "soybean"}
from matplotlib.lines import Line2D
hs = [Line2D([], [], ls="", marker="o", ms=4, color=cols[r.dataset], label=lab[r.dataset]) for _, r in CELLS.sort_values("cells", ascending=False).iterrows()]
hs.insert(2, Line2D([], [], ls="", marker="o", ms=4, color=C_GREEN, label="CUBIC"))
top12 = 0.003
ax.axvline(top12, color=C_ADM, lw=.8, ls=(0, (1.2, 1.4)), zorder=1)
ax.text(top12 * 1.1, 2.2e4, "G2F 1st vs 2nd\nteam: 0.003", fontsize=7, color=C_ADM, ha="left", va="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(0.002, 0.2); ax.set_ylim(80, 6e7)
ax.set_xticks([0.003, 0.01, 0.03, 0.1]); ax.set_xticklabels(["0.003", "0.01", "0.03", "0.1"])
ax.set_xlabel("accuracy gain to resolve (Δr)"); ax.set_ylabel("genotype–environment cells needed")
leg = ax.legend(loc="lower left", fontsize=7, handlelength=2.2, title="selection", title_fontsize=7)
ax.add_artist(leg)
ax.legend(handles=hs, loc="upper right", bbox_to_anchor=(1.0, 0.9), fontsize=7, handletextpad=0.2, borderaxespad=0.2, title="Gaussian lower bound\nat each design", title_fontsize=7)
ax.text(-0.30, 1.035, "C", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

save(fig, "Fig6", DOUBLE, tight=False)
print("wrote analysis/tcj_figures/Fig6.pdf/.png")
