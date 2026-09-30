"""Fig. 6: GPverdict, the criterion as a tool, and its verdict for Chinese maize (CUBIC).

(A) what the tool does: its input format, with the first rows of its bundled spring wheat example, and
the verdict it returns for that example (vector schematic; the screenshots of the web page are Fig. S5);
(B) the CUBIC verdict: 17 methods ranked by the admissible metric, with the gap the trial resolves;
(C) planning a trial: genotype-environment cells needed to resolve an accuracy gain inside the design grid
(extrapolation shaded), with the paper's datasets and the leaderboard's top gap; (D) all 23 CUBIC traits.
Reads analysis/crosscrop/results/gpverdict_cubic.json and gpverdict_cells.csv (written by
analysis/crosscrop/code/gpverdict_applications.py) and assets/gpverdict_example_verdict.json (written by
assets/example_verdict.py: GPverdict run on its example with the web page's settings). No number of
panel A is typed in here; all of them are read from that file.
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "analysis/tcj_figures"); sys.path.insert(0, "tool")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
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


# ---------------------------------------------------------------- (A) what the tool does: content
# every number comes from GPverdict's own output on its bundled example (assets/example_verdict.py)
EX = json.load(open("analysis/tcj_figures/assets/gpverdict_example_verdict.json"))


def verdict_lines(EX):
    """Title, table and the four verdict lines of panel A, all taken from the tool's output."""
    T = EX["tool_output"]; d = T["data"]; res = T["resolution"]
    tier = {r["key"]: r["tier"] for r in T["invariance"]}           # the tool's invariance check on these predictions
    assert tier["spearman"] == "I" and tier["pearson"] == "II" and all(tier[k] == "III" for k in ("rmse", "pooled_pearson", "pooled_rmse"))
    lead, lp = T["leader"], T["pearson_pick"]; assert lead != lp
    others = [m for m in res["tied_with_leader"] if m not in (lead, lp)]
    plan = {round(r["gap"], 3): r["cells"] for r in EX["planning_table"]}
    head = [f"{d['methods']} methods, {d['environments']} environments, {d['genotypes']} genotypes;",
            f"top {T['settings']['frac']:.0%} selected within each environment"]
    lines = [
        "Both within-environment correlations (rank correlation and Pearson r) can rank the methods; "
        "RMSE and pooled metrics cannot.",
        f"{lead} leads by the rank correlation, {lp} by Pearson r. With {d['cells']:,} cells per method the "
        f"smallest gap the trial resolves is {res['gap']:.3f}, so it cannot separate "
        f"{', '.join(others[:-1])} and {others[-1]} from them.",
        f"An RMSE ranking would pick {T['rmse_pick']}; RMSE and Pearson r order "
        f"{100 * T['reversal_rmse_pearson']:.0f}% of the {T['method_pairs']} method pairs oppositely.",
        f"Resolving a gain of 0.03 would need at least {plan[0.03]:,.0f} cells (Gaussian lower bound).",
    ]
    cols = EX["example_rows"]["columns"]
    rows = [[f"{float(c):.1f}" if j in (2, 3) else c for j, c in enumerate(r)] for r in EX["example_rows"]["rows"]]
    note = f"{EX['source']['rows']:,} rows in the example file"
    return head, lines, cols, rows, note


def text_mm(s, fs, bold=False):
    return TextPath((0, 0), s, size=fs, prop=FontProperties(family="Arial", weight=700 if bold else 400)).get_extents().width * 25.4 / 72


def wrap(s, fs, width, bold=False):
    out, cur = [], ""
    for w_ in s.split(" "):
        t = (cur + " " + w_).strip()
        if cur and text_mm(t, fs, bold) > width: out.append(cur); cur = w_
        else: cur = t
    return out + [cur]


def panel_a_layout(L):
    """Wrap the verdict for a box L['box'] mm wide; returns the wrapped text and the height of the band (mm)."""
    head, lines, cols, rows, note = verdict_lines(EX)
    inner = L["box"][1] - L["box"][0] - 2 * L["pad"] - L["indent"]
    wl = [wrap(s, L["fs"], inner) for s in lines]
    wh = [s for h in head for s in wrap(h, L["fs"], L["box"][1] - L["box"][0] - 2 * L["pad"], bold=True)]
    lh = L["fs"] * L["lead"] * 25.4 / 72                             # line height, mm
    box_h = 2 * L["pad"] + len(wh) * lh + L["gap"] + sum(len(w) for w in wl) * lh + (len(wl) - 1) * L["gap"]
    table_h = (len(rows) + 2) * L["row"] + 1 + lh
    colw = [max([text_mm(c, L["fs"], bold=True)] + [text_mm(r[j], L["fs"]) for r in rows]) + L["cpad"] for j, c in enumerate(cols)]
    return dict(head=wh, lines=wl, cols=cols, rows=rows, note=note, lh=lh, box_h=box_h, colw=colw,
                band=L["title"] + max(box_h, table_h))


def draw_panel_a(fig, A, L, y0):
    """Panel A in the band from y0 (mm) upwards: input table -> GPverdict -> verdict."""
    fs = L["fs"]; lh = A["lh"]; top = y0 + A["band"] - L["title"]            # top of table, arrow zone and box
    ax = mm_axes(fig, 0, y0, fig.get_size_inches()[0] * 25.4, A["band"]); ax.axis("off")
    ax.set_xlim(0, fig.get_size_inches()[0] * 25.4); ax.set_ylim(y0, y0 + A["band"])
    # input table
    x = L["table_x"]; cw = A["colw"]; xs = np.cumsum([x] + cw); right = xs[-1]
    ty = top - (A["band"] - L["title"] - ((len(A["rows"]) + 2) * L["row"] + 1 + lh)) / 2   # centred on the box
    ax.text(x, top + 1.2, L["t_input"], fontsize=fs, color=MUTED, ha="left", va="bottom")
    ax.add_patch(plt.Rectangle((x, ty - L["row"]), right - x, L["row"], facecolor="#ececec", edgecolor="none"))
    for j, c in enumerate(A["cols"]):
        ax.text(xs[j] + 1, ty - L["row"] / 2, c, fontsize=fs, fontweight="bold", ha="left", va="center_baseline")
    for i, r in enumerate(A["rows"] + [["…"] * len(A["cols"])]):
        yc = ty - (i + 1.5) * L["row"]
        for j, c in enumerate(r):
            num = j in (2, 3)
            ax.text(xs[j + 1] - 1 if num and c != "…" else xs[j] + 1, yc, c, fontsize=fs,
                    ha="right" if num and c != "…" else "left", va="center_baseline", color=INK if c != "…" else MUTED)
        ax.plot([x, right], [yc - L["row"] / 2] * 2, color=RULE, lw=.5)
    ax.plot([x, right], [ty, ty], color=INK, lw=.7)
    ax.plot([x, right], [ty - (len(A["rows"]) + 2) * L["row"]] * 2, color=INK, lw=.7)
    ax.text(x, ty - (len(A["rows"]) + 2) * L["row"] - 1, A["note"], fontsize=fs, color=MUTED, ha="left", va="top")
    # the tool
    ym = top - A["box_h"] / 2; a0, a1 = right + L["agap"], L["box"][0] - L["agap"]
    ax.annotate("", xy=(a1, ym), xytext=(a0, ym), arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.25", lw=1.1, color=INK, shrinkA=0, shrinkB=0))
    ax.text((a0 + a1) / 2, ym + 1.4, "GPverdict", fontsize=fs + 1, fontweight="bold", ha="center", va="bottom")
    ax.text((a0 + a1) / 2, ym - 1.4, L["t_tool"], fontsize=fs, ha="center", va="top", color=INK, linespacing=1.2)
    # the verdict
    b0, b1 = L["box"]
    ax.add_patch(FancyBboxPatch((b0, top - A["box_h"]), b1 - b0, A["box_h"], boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor="#eef5ee", edgecolor=C_GREEN, lw=.7))
    ax.text(b0, top + 1.2, "Verdict for the bundled spring wheat example", fontsize=fs, color=MUTED, ha="left", va="bottom")
    y = top - L["pad"]
    for s in A["head"]:
        ax.text(b0 + L["pad"], y, s, fontsize=fs, fontweight="bold", ha="left", va="top"); y -= lh
    y -= L["gap"]
    for wl in A["lines"]:
        ax.text(b0 + L["pad"], y, "•", fontsize=fs, ha="left", va="top", color=C_GREEN)
        for s in wl:
            ax.text(b0 + L["pad"] + L["indent"], y, s, fontsize=fs, ha="left", va="top"); y -= lh
        y -= L["gap"]
    return ax


LA = dict(fs=FS, lead=1.28, pad=2.2, indent=2.6, gap=1.0, row=3.9, title=5.0,
          table_x=6, cpad=2.0, agap=1.2, box=(101, 188.5),
          t_input="Input: a CSV of cross-validated predictions",
          t_tool="browser or Python;\ndata stay local")
A_ = panel_a_layout(LA)

W = 190
DY = 78                                                     # panel D sits below the original A-C; 8 mm more clearance under B/C
YA = 7 + DY + 58 + 8                                        # panel A band starts 8 mm above the top of B and C
H = round(YA + A_["band"] + 1.5)
fig = plt.figure(figsize=(W * MM, H * MM))

# ---------------------------------------------------------------- (A) what the tool does
draw_panel_a(fig, A_, LA, YA)
mm_text(fig, 1, H - 1, "A", fontsize=9, fontweight="bold", va="top")

# ---------------------------------------------------------------- (b) CUBIC verdict
ax = mm_axes(fig, 44, 7 + DY, 62, 58)
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
mm_text(fig, 1, 7 + DY + 58 * 1.035, "B", fontsize=9, fontweight="bold", va="bottom", ha="left")   # was 0.6 mm off the page

# ---------------------------------------------------------------- (c) planning a trial
ax = mm_axes(fig, 133, 7 + DY, 55, 58)
g = np.logspace(np.log10(0.002), np.log10(0.2), 200)
GRID_MIN, GRID_MAX = 25 * 10, 400 * 200                       # cells spanned by the design grid (Table S11)
ax.axhspan(GRID_MAX, 6e7, color="#f1f1f1", lw=0, zorder=0)
ax.text(0.0023, 1.8e7, "beyond the design grid: extrapolated", fontsize=7, color=MUTED, ha="left", va="center")
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

# ---------------------------------------------------------------- (D) all 23 CUBIC traits
# reversal between the two official metrics for every trait of Table S12 (china_trait_sweep.py,
# fixed in advance), with and without the miscalibration variants; ear weight, the trait of B, marked
TS = pd.read_csv("analysis/crosscrop/results/china_trait_sweep.csv")
TS = TS[TS.status == "analysed"].sort_values("reversal").reset_index(drop=True)
ax = mm_axes(fig, 20, 9, 104, 58)
yy = np.arange(len(TS))
ew = int(np.flatnonzero(TS.trait == "EW")[0])
ax.axhspan(ew - .5, ew + .5, color=C_ADM, alpha=.08, lw=0, zorder=0)
ax.hlines(yy + .16, TS.ci_lo, TS.ci_hi, color=C_ADM, lw=.8, zorder=2)
ax.scatter(TS.reversal, yy + .16, s=14, color=C_ADM, lw=0, zorder=3, label="all 52 methods (with miscalibration variants)")
ax.hlines(yy - .16, TS.ci_lo_base, TS.ci_hi_base, color=C_INADM, lw=.8, zorder=2)
ax.scatter(TS.reversal_base, yy - .16, s=14, marker="s", facecolor="white", edgecolor=INK, lw=.7, zorder=3, label="17 base methods")
ax.set_yticks(yy); ax.set_yticklabels(TS.trait, fontsize=FS)
ax.get_yticklabels()[ew].set_fontweight("bold"); ax.get_yticklabels()[ew].set_color(C_ADM)
ax.set_ylim(-.7, len(TS) - .3); ax.set_xlim(0, .62)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{100*v:.0f}%"))
ax.set_xlabel("method pairs that RMSE and Pearson r order oppositely (95% interval)")
ax.legend(loc="lower left", bbox_to_anchor=(1.03, 0.0), fontsize=FS, handletextpad=.3, borderaxespad=0, frameon=False)
mm_text(fig, 1, 9 + 58 + 1.2, "D", fontsize=9, fontweight="bold", va="bottom", ha="left")   # same left edge as A and B
mm_text(fig, 132, 60, "CUBIC: all 23 traits of\nSupplementary Table S19,\nsame 52-method panel,\nrule fixed before the run", fontsize=FS, va="top", ha="left", color=INK)
mm_text(fig, 132, 36, "ear weight (EW, panel B)\nhighlighted", fontsize=FS, va="top", ha="left", color=C_ADM)

print("Fig6 width mm", round(save(fig, "Fig6", DOUBLE, tight=False), 1))
print("wrote analysis/tcj_figures/Fig6.pdf/.png")
