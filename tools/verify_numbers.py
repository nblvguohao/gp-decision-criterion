#!/usr/bin/env python
"""Numerical traceability check for the TCJ manuscript.

Every registered quantitative statement is located in the manuscript with a
regular expression (the printed value), recomputed or read from the result file
that produces it (the expected value), and compared at the printed precision.

Pre-submission gate: on the current manuscript both 不一致 and 未找到 should be 0.
A 未找到 means a sentence was rewritten: update its entry in build_registry().

    python tools/verify_numbers.py                      # reads output/manuscript_TCJ.md
    python tools/verify_numbers.py --manuscript PATH    # any other copy
    python tools/verify_numbers.py --heavy              # also the slow recomputations
    python tools/verify_numbers.py --only-problems      # print only non-OK rows
    python tools/verify_numbers.py --csv out.csv        # also write the table

Run from anywhere; paths are resolved against the repository root. Nothing in the
repository is written (the optional --csv target is the only output file).

Statuses
    一致      printed value agrees with the source at printed precision
    不一致    it does not (or the stated rule/range is contradicted by the source)
    无来源    no result file or script in the repository produces the value
    未复算    the source needs a recomputation that was not run (slow, or --heavy off)
    未找到    the registered pattern no longer matches the manuscript: update the registry

Exit status: 1 if any 不一致, else 2 if any 未找到, else 0.

Comparison rule: a printed number with d decimals agrees if
|printed - expected| <= 0.5 * 10^-d (so both round-half-up and round-half-even
are accepted); scientific notation uses the mantissa's decimals; number words are
exact; a few claims carry an explicit tolerance or are bounds (<=, >).
"""
import argparse, functools, gzip, itertools, json, math, os, re, sys, time, warnings
sys.dont_write_bytecode = True          # importing the analysis modules must not write __pycache__
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis/crosscrop/code"))

G = "analysis/g2f_leaderboard/results"
X = "analysis/crosscrop/results"
CC = "analysis/crosscrop/code"
GC = "analysis/g2f_leaderboard/code"
RAW = os.environ.get("GP_DATA", "data/raw")

# --------------------------------------------------------------------------- parsing
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
         "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
         "nineteen": 19, "twenty": 20, "thirty": 30, "fifty": 50, "twenty-two": 22,
         "twenty-one": 21, "two hundred": 200, "first": 1, "second": 2, "third": 3,
         "seventh": 7, "fourfold": 4, "three fifths": 0.6, "half": 0.5, "no": 0, "none": 0, "a third": 1 / 3, "top-ranked method": 1,
         "only within a single family": 1, "indistinguishable from zero": 1}
N = r"([-+]?\d[\d,]*(?:\.\d+)?(?:\s*×\s*10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?)"   # a printed number
W = r"([A-Za-z][A-Za-z\- ]*?)"                                     # a number word


def parse(s):
    """printed string -> (value, half-unit tolerance, is_word)"""
    s0 = s.strip()
    if s0.lower() in WORDS:
        return float(WORDS[s0.lower()]), 0.0, True
    s = s0.replace(",", "").replace(" ", "")
    m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)×10([-0-9]+)", s.translate(SUP))
    if m:
        mant, ex = m.group(1), int(m.group(2))
        d = len(mant.split(".")[1]) if "." in mant else 0
        return float(mant) * 10 ** ex, 0.5 * 10 ** (-d) * 10 ** ex, False
    d = len(s.split(".")[1]) if "." in s else 0
    return float(s), 0.5 * 10 ** (-d), False


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "是" if v else "否"
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "NaN"
    v = float(v)
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        return f"{v:.3g}"
    if float(v).is_integer():
        return f"{int(v)}"
    return f"{v:.4g}" if abs(v) >= 1 else f"{v:.4f}".rstrip("0")


# --------------------------------------------------------------------------- data
V5 = ["KernelOfTruth_sub244906", "KernelOfTruth_sub244944", "KernelOfTruth_sub244953",
      "EnBiSys_sub243568", "NicheSquad_sub244985"]
PANELS = [("common bean", "bean_panel_wide.csv", 25), ("spring wheat", "ursn_panel_wide.csv", 12),
          ("soybean", "nust_panel_wide.csv", 25)]
PANEL_OF = {k: (f"{X}/{f}", mn) for k, f, mn in PANELS}
NONMAIZE = [k for k, _, _ in PANELS]
ALLD = ["maize"] + NONMAIZE
DIHK = {"common bean": "COMMON BEAN", "spring wheat": "SPRING WHEAT (FHB)", "soybean": "SOYBEAN",
        "maize": "MAIZE (5 verified G2F submissions)"}
ERRWORDS = ("RMSE", "MAE", "r2_score")


class Data:
    """Lazily loaded result files and derived quantities (each computed once)."""

    def __init__(self, heavy=False):
        self.heavy = heavy
        self._panel = {}

    def panel(self, ds):
        if ds not in self._panel:
            self._panel[ds] = pd.read_csv(PANEL_OF[ds][0])
        return self._panel[ds]

    # ---- maize, published values
    @functools.cached_property
    def S1(self):
        d = pd.read_csv(f"{G}/S1_best_per_team.csv")
        return d[d["Team Name"].notna()].reset_index(drop=True)

    @functools.cached_property
    def by_r(self):
        return self.S1.sort_values("mean_pearson_r", ascending=False).reset_index(drop=True)

    @property
    def r_top(self):
        return self.by_r.mean_pearson_r.to_numpy()

    @property
    def gap12(self):
        return self.r_top[0] - self.r_top[1]

    @property
    def gap13(self):
        return self.r_top[0] - self.r_top[2]

    def within(self, t):
        return int((self.r_top > self.r_top[0] - t).sum())

    def pairs_closer(self, t):
        """team pairs across the whole board whose mean_pearson_r differ by less than t"""
        r = self.r_top; d = np.abs(r[:, None] - r[None, :])[np.triu_indices(len(r), 1)]
        return int((d < t).sum()), len(d)

    @functools.cached_property
    def inv(self):
        return pd.read_csv(f"{G}/invariance_test.csv").set_index("metric")["median"]

    @functools.cached_property
    def mono(self):
        return pd.read_csv(f"{G}/monotone_test.csv").set_index("metric")["median"]

    @property
    def err_metrics(self):
        return [m for m in self.inv.index if any(w in m for w in ERRWORDS)]

    # ---- the partition (partition.py)
    @functools.cached_property
    def pm(self):
        return pd.read_csv(f"{X}/partition_metrics.csv")

    @functools.cached_property
    def pmz(self):
        return self.pm[self.pm.dataset == "maize"].set_index("metric")

    def tier(self, t):
        return sorted(self.pmz.index[self.pmz.tier == t])

    @functools.cached_property
    def part(self):
        return pd.read_csv(f"{X}/partition_by_dataset.csv").set_index("dataset")

    @functools.cached_property
    def summ(self):
        return pd.read_csv(f"{X}/cross_dataset_summary.csv").set_index("dataset")

    @functools.cached_property
    def rc(self):
        return pd.read_csv(f"{X}/rank_census.csv").set_index("dataset")

    @functools.cached_property
    def tierrank(self):
        return pd.read_csv(f"{G}/tierI_ranking.csv")

    @functools.cached_property
    def rv(self):
        return pd.read_csv(f"{G}/reversal_by_metric_pair.csv")

    @functools.cached_property
    def rt(self):
        return pd.read_csv(f"{G}/reversal_tests.csv")

    @property
    def rt_off(self):
        r = self.rt
        return r[((r.metric_a == "mean_RMSE") & (r.metric_b == "mean_pearson_r")) |
                 ((r.metric_a == "mean_pearson_r") & (r.metric_b == "mean_RMSE"))].iloc[0]

    @functools.cached_property
    def rbc(self):
        return pd.read_csv(f"{G}/reversal_by_class.csv").set_index("group")

    @functools.cached_property
    def perm_null(self):
        """reversal.py's permutation null, same seed and replicate count."""
        a = -self.S1.mean_RMSE.to_numpy(); b = self.S1.mean_pearson_r.to_numpy(); n = len(a)
        iu = np.triu_indices(n, 1); rng = np.random.default_rng(20260908)
        def rr(x, y):
            return ((x[:, None] - x[None, :])[iu] * (y[:, None] - y[None, :])[iu] < 0).mean()
        return float(np.mean([rr(rng.permutation(a), rng.permutation(b)) for _ in range(5000)]))

    @functools.cached_property
    def mcp(self):
        d = pd.read_csv(f"{G}/model_class_permutation.csv")
        return d[d.scope == "27 labelled teams"]

    @functools.cached_property
    def rr1(self):
        return pd.read_csv(f"{X}/unaugmented_panels.csv")

    def unaug(self, ds, col="rate"):
        r = self.rr1
        return float(r[(r.dataset == ds) & (r.panel == "base")][col].iloc[0])

    @functools.cached_property
    def rob(self):
        return pd.read_csv(f"{X}/robustness.csv").set_index("dataset")

    # ---- outcome test
    @functools.cached_property
    def dih(self):
        return json.load(open(f"{X}/does_it_help.json"))

    @functools.cached_property
    def rel(self):
        return json.load(open(f"{X}/does_it_help_reliability.json"))

    def raw(self, ds, rule, unaug=False):
        d = self.dih[DIHK[ds] + (" (unaugmented)" if unaug else "")]
        return d[[k for k in d if k.startswith(rule)][0]]

    def recov(self, ds, rule, unaug=False):
        d = self.dih[DIHK[ds] + (" (unaugmented)" if unaug else "")]; k = [k for k in d if k.startswith(rule)][0]
        return 100 * (d[k] - d["random"]) / (d["oracle"] - d["random"])

    def fam(self, ds, family):
        rules = {"corr": ["mean_pearson_r", "mean_spearman_r"], "err": ["mean_RMSE", "mean_MAE", "mean_r2_score"]}[family]
        v = [self.recov(ds, r) for r in rules]
        return min(v), max(v)

    @functools.cached_property
    def rds(self):
        return pd.read_csv(f"{X}/decision_swap.csv")

    def paired(self, ds, dec, col="paired_pearson_minus_rmse"):
        r = self.rds
        return float(r[(r.dataset.str.startswith(ds)) & (r.decision.str.startswith(dec))][col].iloc[0])

    # ---- rank intervals
    @functools.cached_property
    def ris(self):
        return pd.read_csv(f"{G}/rank_interval_sensitivity.csv").set_index("metric_set")

    @functools.cached_property
    def rri(self):
        return pd.read_csv(f"{G}/rank_interval_transfer.csv")

    # ---- selection differential against i·r
    @functools.cached_property
    def gap(self):
        return pd.read_csv(f"{X}/breeder_gap_se.csv")

    def gaps(self, f):
        g = self.gap[self.gap.frac == f].set_index("team")
        return [g.loc[t, "gap"] for t in V5]

    @property
    def main10(self):
        return self.gap[self.gap.frac.isin([0.10, 0.20])]

    @functools.cached_property
    def bgp(self):
        return pd.read_csv(f"{X}/breeder_gap_panels.csv")

    def bg(self, ds, f, col="gap", methods="all methods"):
        b = self.bgp
        return float(b[(b.dataset == ds) & (b.frac == f) & (b.methods == methods)][col].iloc[0])

    @functools.cached_property
    def cop(self):
        return pd.read_csv(f"{X}/copula_bridge.csv")

    @property
    def cop10(self):
        return self.cop[self.cop.frac.isin([0.10, 0.20])]

    @property
    def overstate(self):
        c = self.cop.assign(rel=(self.cop.bench_N - self.cop.bench_C0) / self.cop.bench_N)
        return c.groupby("team").rel.mean()

    @functools.cached_property
    def fin_explained(self):
        """% of the infinite-population gap removed by the finite-population benchmark,
        per selection intensity: 1 - |mean gap_N| / |mean gap_inf|."""
        out = []
        for f in (0.05, 0.10, 0.20):
            gi = self.gap[self.gap.frac == f].gap.mean(); gn = self.cop[self.cop.frac == f].gap_N.mean()
            out.append(100 * (1 - abs(gn) / abs(gi)))
        return out

    @functools.cached_property
    def n_per_env_maize(self):
        obs = pd.read_csv(f"{G}/Final_Observed_Yield.csv").rename(columns={"Yield_Mg_ha": "y"})
        med = []
        for t in V5:
            p = pd.read_csv(f"{G}/{t}.csv").rename(columns={"Yield_Mg_ha": "p"})
            d = obs.merge(p, on=["Env", "Hybrid"]).dropna(subset=["y", "p"])
            med.append(d.groupby("Env").size().median())
        return float(np.median(med))

    @functools.cached_property
    def inventory(self):
        return pd.read_csv(f"{G}/INVENTORY_metrics.csv")

    # ---- the threshold
    @functools.cached_property
    def te(self):
        return pd.read_csv(f"{X}/threshold_estimates.csv")

    def tpar(self, ds, col, frac=0.10, methods="parent methods"):
        t = self.te
        return float(t[(t.dataset == ds) & (t.methods == methods) & (t.frac == frac)][col].iloc[0])

    @functools.cached_property
    def tpairs(self):
        return pd.read_csv(f"{X}/threshold_pairs.csv")

    @functools.cached_property
    def ttheory(self):
        return pd.read_csv(f"{X}/threshold_theory.csv").set_index("dataset")

    @property
    def tmz(self):
        return self.ttheory.loc["maize (G2F design)"]

    @property
    def tratio(self):
        return self.ttheory.ratio_lam05.dropna()

    def tdesign(self, n, e, f=0.10):
        d = pd.read_csv(f"{X}/threshold_design.csv")
        return float(d[(d.f == f) & (d.n_per_env == n) & (d.n_env == e)].dr_star.iloc[0])

    @functools.cached_property
    def hb(self):
        return pd.read_csv(f"{X}/heritability_ceiling_A8_bounds.csv")

    def hq(self, ds, q, col="value"):
        b = self.hb
        return float(b[(b.dataset == ds) & (b.quantity == q)][col].iloc[0])

    def hh(self, q, col):
        b = self.hb
        return float(b[(b.dataset == "headline") & (b.quantity.str.startswith(q))][col].iloc[0])

    @functools.cached_property
    def dmi(self):
        return pd.read_csv(f"{G}/decision_metrics_invariance.csv")

    @functools.cached_property
    def pinv(self):
        return pd.read_csv(f"{G}/paired_invariance.csv")

    @functools.cached_property
    def ce(self):
        return pd.read_csv(f"{X}/counterexample_ursn.csv")

    @property
    def ce_best(self):
        c = self.ce[~self.ce.reversed.astype(bool)]
        return c.sort_values("mean_r2_pearson").iloc[-1]

    @property
    def ce_rev(self):
        c = self.ce[self.ce.reversed.astype(bool)]
        return c[c.method == self.ce_best.method].iloc[0]

    @functools.cached_property
    def n_gaps_at_floor(self):
        """gaps between consecutive teams that reach the Gaussian design value, the number of
        blocks that leaves, the size of the top block, and all block sizes."""
        r = self.r_top; thr = float(self.tmz.theory_lam05)
        cut = [i for i in range(len(r) - 1) if r[i] - r[i + 1] >= thr]
        sizes = []; st = 0
        for c in cut + [len(r) - 1]:
            sizes.append(c - st + 1); st = c + 1
        return len(cut), len(cut) + 1, sizes[0], sizes

    @functools.cached_property
    def china(self):
        return pd.read_csv(f"{X}/china_summary.csv").iloc[0]

    @functools.cached_property
    def nust_names(self):
        norm = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
        with gzip.open(f"{RAW}/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", "rt") as fh:
            for line in fh:
                if line.startswith("#CHROM"):
                    ids = line.rstrip("\n").split("\t")[9:]; break
        names = set(pd.read_csv(f"{RAW}/nust/nust_ge_means.csv").GermplasmId.astype(str))
        nid = {norm(s) for s in ids}
        return len(names), len(names & set(ids)), sum(norm(n) in nid for n in names)

    @functools.cached_property
    def train_plots(self):
        t = pd.read_csv(f"{RAW}/g2f/1_Training_Trait_Data_2014_2021.csv", usecols=["Yield_Mg_ha"])
        return int(t.Yield_Mg_ha.notna().sum())

    # ---- heavy recomputations (minutes)
    @functools.cached_property
    def markers(self):
        """Marker counts, replicating only the genotype-loading blocks of panel_bean/ursn.py
        and prep_nust.py."""
        def count(X):
            X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
            return int((np.nanstd(X, axis=0) > 0).sum())
        def vcf(path, code):
            rows = []
            with gzip.open(path, "rt") as fh:
                for line in fh:
                    if line.startswith("#"): continue
                    f = line.rstrip("\n").split("\t"); rows.append([code.get(g.split(":")[0], np.nan) for g in f[9:]])
            return np.asarray(rows, dtype=np.float64).T
        cb = {"0/0": 0., "0|0": 0., "0/1": 1., "1/0": 1., "0|1": 1., "1|0": 1., "1/1": 2., "1|1": 2.}
        cn = {"0/0": 0., "0/1": 1., "1/0": 1., "1/1": 2., "./.": np.nan}
        Gu = pd.read_csv(f"{RAW}/ursn/data/URSN_3K_1968-2024_formatted_imputed_005mafFilt_gdose.tsv", sep="\t", index_col=0)
        return {"common bean": count(vcf(f"{RAW}/vef/VEF_Pvulgaris_genotypic_data.vcf.gz", cb)),
                "spring wheat": count(Gu.to_numpy(dtype=float)),
                "soybean": count(vcf(f"{RAW}/nust/NUST_geno_Wm82.a2_v1_Filt_KNNimp.vcf.gz", cn))}


def code_int(path, pattern, group=1):
    m = re.search(pattern, open(path, encoding="utf-8").read())
    return float(m.group(group)) if m else float("nan")


def code_has(path, *snippets):
    s = open(path, encoding="utf-8").read()
    return all(x in s for x in snippets)


def npc_all():
    v = [code_int(f"{CC}/panel_bean.py", r"NPC=min\((\d+)"), code_int(f"{CC}/panel_ursn.py", r"NPC=min\((\d+)"),
         code_int(f"{CC}/prep_nust.py", r"NPC=(\d+)")]
    return min(v), max(v)


def panel_methods():
    """base (un-augmented) method names of each panel"""
    return {ds: sorted(m for m in pd.read_csv(PANEL_OF[ds][0], usecols=["method"]).method.unique() if "__" not in m)
            for ds, _, _ in PANELS}


def every_panel_ok():
    """§2.3: every panel has ridge_pc<d>_{lo,hi} at four d, and rf, gbm, knn, mlp, ens members."""
    bad = []
    for ds, v in panel_methods().items():
        dims = {m.split("_")[1] for m in v if m.startswith("ridge_pc")}
        tags = {m.split("_")[2] for m in v if m.startswith("ridge_pc")}
        miss = [c for c in ("rf", "gbm", "knn", "mlp", "ens") if not any(m.startswith(c) for m in v)]
        if len(dims) != 4 or tags != {"lo", "hi"} or miss:
            bad.append(f"{ds}: 维度{len(dims)} 惩罚{sorted(tags)} 缺{miss}")
    return bad


# --------------------------------------------------------------------------- registry
# Each entry: sec, desc, pat (regex, one group per printed value, matched against the
# normalised text: '−'→'-', '*' removed), exp (callable -> list), src, and optionally kind
# ('num' default | 'le' | 'ge' | 'gt' | 'bool'), tol (list or scalar), status ('NOSRC' |
# 'NOTRUN'), heavy (bool), note, note_fn.
D = None
R = []


def cells_law(f):
    """Gaussian design grid: fit log dr* on log(genotype-environment cells); return slope, R2, k = geo-mean of dr*·sqrt(N)."""
    d = pd.read_csv(f"{X}/threshold_design.csv"); d = d[d.f == f].dropna(subset=["dr_star"])
    c = d.n_per_env * d.n_env; y = np.log(d.dr_star); b, a = np.polyfit(np.log(c), y, 1)
    r2 = 1 - np.sum((y - (a + b * np.log(c))) ** 2) / np.sum((y - y.mean()) ** 2)
    return b, r2, float(np.exp(np.mean(np.log(d.dr_star * np.sqrt(c)))))


def soy_gblup_calibration():
    """Soybean base methods: within-env slopes of gblup/gblup_gxe, max slope of the rest, % mean_RMSE above the lowest."""
    m = pd.read_csv(f"{X}/nust_metrics_wide.csv"); b = m[~m.method.str.contains("__")].set_index("method")
    g = ["gblup", "gblup_gxe"]; lo = b.mean_RMSE.min()
    return [b.loc["gblup", "mean_lineRegressSlope"], b.loc["gblup_gxe", "mean_lineRegressSlope"],
            b.drop(g).mean_lineRegressSlope.max(), 100 * (b.loc[g, "mean_RMSE"].max() / lo - 1)]


def reg(sec, desc, pat, exp=None, src="", **kw):
    R.append(dict(sec=sec, desc=desc, pat=pat, exp=exp, src=src, **kw))


def build_registry():
    S1f, INV, MON = f"{G}/S1_best_per_team.csv", f"{G}/invariance_test.csv", f"{G}/monotone_test.csv"
    SUM, RC, TIER, RV = f"{X}/cross_dataset_summary.csv", f"{X}/rank_census.csv", f"{G}/tierI_ranking.csv", f"{G}/reversal_by_metric_pair.csv"
    DIH, RDS, GAP, COP = f"{X}/does_it_help.json", f"{X}/decision_swap.csv", f"{X}/breeder_gap_se.csv", f"{X}/copula_bridge.csv"
    HB, TT, TD, TE = f"{X}/heritability_ceiling_A8_bounds.csv", f"{X}/threshold_theory.csv", f"{X}/threshold_design.csv", f"{X}/threshold_estimates.csv"
    PM, PB, RT, RBC = f"{X}/partition_metrics.csv", f"{X}/partition_by_dataset.csv", f"{G}/reversal_tests.csv", f"{G}/reversal_by_class.csv"
    BGP, RR1, ROB = f"{X}/breeder_gap_panels.csv", f"{X}/unaugmented_panels.csv", f"{X}/robustness.csv"
    s_ = lambda k, c: float(D.summ.loc[k, c])
    s = s_
    nT = lambda t: len(D.tier(t))
    fmeans = lambda: [-D.gap[D.gap.frac == f].gap.mean() for f in (0.05, 0.10, 0.20)]
    lam = lambda: [D.tmz.theory_lam0, D.tmz.theory_lam05, D.tmz.theory_lam08]
    fin_src = f"{GAP} gap (i_inf·r) + {COP} gap_N (i_fin·r)"

    # ================= Abstract
    reg("摘要", "竞赛设计下的高斯值、迁移范围、第1–2名差距、闭合队伍对比例",
        r"a Gaussian model puts the accuracy gap needed before the more accurate method reliably selects better material at " + N + r" in within-environment correlation, and calibration to the three panels gives " + N + "–" + N + r"; the first two teams differ by " + N + r", and (\d+) % of all team pairs fall below it",
        lambda: [D.tmz.theory_lam05, D.tmz.transfer_lo, D.tmz.transfer_hi, D.gap12,
                 100 * D.pairs_closer(D.tmz.theory_lam05)[0] / D.pairs_closer(D.tmz.theory_lam05)[1]], TT + "; " + S1f)
    reg("摘要", "官方指标数与各档数",
        r"this criterion admits (one) of the (twenty-two) official metrics, the mean within-environment rank correlation; the within-environment Pearson correlation is unmoved by calibration only, and (twenty) — including the metric that decided the 2022 ranking — fail",
        lambda: [nT("I"), len(D.pmz), nT("III")], PM)
    reg("摘要", "两个官方指标间反转率及CI；名次跨度",
        r"reverses (\d+) % of team pairs \(95 % CI (\d+)–(\d+) %\), and team rank spans a median of (\d+) of (\d+) places",
        lambda: [100 * s_("maize", "reversal"), 100 * s_("maize", "ci_lo"), 100 * s_("maize", "ci_hi"),
                 D.rc.loc["maize", "census_median_range"], D.rc.loc["maize", "methods"]], SUM + "; " + RC)
    reg("摘要", "缺口均值范围 (三个选择强度)",
        r"verified submissions, by a mean of " + N + "–" + N + " phenotypic standard deviations",
        lambda: [min(fmeans()), max(fmeans())], GAP)
    reg("摘要", "五份已核实提交都低于 i·r", r"fell short of the projection i·r in (all five) verified submissions",
        lambda: [bool((D.gap.gap < 0).all()) and D.gap.team.nunique() == 5], GAP, kind="bool")
    reg("表1注", "大豆分析所用环境数 (≥25 个基因型) 与基因型数",
        r"the soybean analyses use the (\d+) of its environments that carry at least (\d+) genotypes, with ([\d,]+) genotypes",
        lambda: [(lambda o: [int((o.groupby("Env").k.nunique() >= 25).sum()), 25,
                             o[o.Env.isin(o.groupby("Env").k.nunique().loc[lambda c: c >= 25].index)].k.nunique()])(
                     D.panel("soybean")[D.panel("soybean").method == D.panel("soybean").method.iloc[0]])][0],
        PANEL_OF["soybean"][0])
    reg("摘要", "优势族随决策而定: D1 相关族在大豆/春小麦领先 (仅大豆显著), D2 误差族在春小麦/菜豆领先 (仅春小麦显著)",
        r"correlation rankings lead for within-environment selection in (soybean and spring wheat), error-magnitude rankings for an absolute target, which reads calibration, in spring wheat and common bean, each significantly in one panel",
        lambda: [all(D.paired(d, "D1", c) > 0 for d in ("soybean", "spring wheat") for c in ("paired_pearson_minus_rmse", "paired_spearman_minus_rmse"))
                 and all(D.paired(d, "D2", c) < 0 for d in ("spring wheat", "common bean") for c in ("paired_pearson_minus_rmse", "paired_spearman_minus_rmse"))
                 and D.paired("soybean", "D1", "paired_ci_lo") > 0 and D.paired("soybean", "D1", "paired_spearman_ci_lo") > 0
                 and D.paired("spring wheat", "D1", "paired_ci_lo") < 0 and D.paired("spring wheat", "D1", "paired_spearman_ci_lo") < 0
                 and D.paired("spring wheat", "D2", "paired_ci_hi") < 0 and D.paired("spring wheat", "D2", "paired_spearman_ci_hi") < 0
                 and D.paired("common bean", "D2", "paired_ci_hi") > 0 and D.paired("common bean", "D2", "paired_spearman_ci_hi") > 0],
        RDS, kind="bool")
    reg("§2.2", "数据集数/物种数/性状类别数", r"(Four) multi-environment datasets spanning (four) species and (two) trait classes",
        lambda: [len(D.summ), 4, 2], SUM + " 行数; §2.2 (玉米、菜豆、小麦、大豆; 产量与病害)")
    reg("§2.2", "G2F 队伍数/环境数", r"(Thirty) teams were scored on a 2022 test set of (\d+) environments",
        lambda: [len(D.S1), pd.read_csv(f"{G}/Final_Observed_Yield.csv").Env.nunique()], S1f + "; Final_Observed_Yield.csv")
    reg("§2.2", "可恢复提交文件数", r"(Seven) submission files were recoverable",
        lambda: [len(D.inventory)], f"{G}/INVENTORY_metrics.csv")
    reg("§2.2", "复现官方值的提交数", r"(five) reproduce the official values to within",
        lambda: [int((D.inventory["diff"] < 1e-3).sum())], f"{G}/INVENTORY_metrics.csv diff")
    reg("§2.2", "复现精度上界 (印刷值须≥最大偏差)", r"reproduce the official values to within " + N + ", the precision of the (six) decimals published",
        lambda: [D.inventory[D.inventory["diff"] < 1e-3]["diff"].max(), 6], f"{G}/INVENTORY_metrics.csv diff", kind="le")
    reg("§2.2", "已发表表中的提交数与指标齐全的份数", r"published in Supplemental Table S1 of that paper: (\d+) submissions, (\d+) of them carrying all 22 metrics",
        lambda: [len(pd.read_csv(f"{G}/S1_parsed.csv")), len(pd.read_csv(f"{G}/S1_parsed.csv").dropna(subset=list(D.inv.index)))],
        f"{G}/S1_parsed.csv")
    reg("§2.2", "每队取 2022 官方指标下最优的提交", r"a team is represented by (its best submission under the 2022 official metric)",
        lambda: [code_has(f"{GC}/build_inputs.py", 'sort_values("mean_RMSE")', 'groupby("Team Name"')], f"{GC}/build_inputs.py best_per_team", kind="bool")
    reg("§2.2", "五份提交来自的队伍数/同一队伍份数", r"They come from (three) teams — (three) from Kernel of Truth",
        lambda: [len({t.split("_")[0] for t in V5}), sum(t.startswith("KernelOfTruth") for t in V5)], "V5 (breeder_gap_se.py 清单)")
    reg("§2.2", "只有 EnBiSys 243568 是上榜提交", r"EnBiSys \(243568, (that team's leaderboard entry)\)",
        lambda: [set(D.S1.id.astype(int)) & {int(t.split("sub")[1]) for t in V5} == {243568}], S1f + " id 列", kind="bool")
    reg("§2.2", "被排除的提交数", r"(Two) files match none of their team",
        lambda: [int((D.inventory["diff"] >= 1e-3).sum())], f"{G}/INVENTORY_metrics.csv diff")

    for ds, pat in [("common bean", r"(\d+) lines, [\d,]+ SNPs, (\d+) trials in Colombia"),
                    ("spring wheat", r"data deposit \[\d+\]; (\d+) lines, [\d,]+ markers, (\d+) environments")]:
        reg("§2.2", f"{ds} 品系数/环境数", pat,
            (lambda ds=ds: [D.panel(ds).k.nunique(), D.panel(ds).Env.nunique()]), PANEL_OF[ds][0] + " k/Env 唯一数")
    for ds, pat in [("common bean", r"467 lines, ([\d,]+) SNPs"), ("spring wheat", r"384 lines, ([\d,]+) markers"),
                    ("soybean", r"2003–2020, ([\d,]+) SNPs")]:
        reg("§2.2", f"{ds} 标记数", pat, (lambda ds=ds: [D.markers[ds]]), f"{RAW}/… 原始基因型", heavy=True)
    reg("§2.2", "大豆全系列 基因型/环境/年份", r"Northern Uniform Soybean Tests \[\d+\]; ([\d,]+) genotypes, ([\d,]+) environments, (\d+)–(\d+)",
        lambda: [pd.read_csv(f"{X}/nust_ge.csv").k.nunique(), pd.read_csv(f"{X}/nust_ge.csv").Env.nunique(),
                 pd.read_csv(f"{X}/nust_ge.csv").Year.min(), pd.read_csv(f"{X}/nust_ge.csv").Year.max()], f"{X}/nust_ge.csv")
    reg("§2.2", "大豆评估子集 环境/基因型", r"evaluated subset reported in Table 1 is (\d+) environments and ([\d,]+) genotypes",
        lambda: [D.panel("soybean").Env.nunique(), D.panel("soybean").k.nunique()], PANEL_OF["soybean"][0])

    # ================= 2.3
    reg("§2.3", "环境最少基因型数 (春小麦放宽) 与春小麦中位数", r"fewer than (\d+) genotypes were skipped, the threshold being relaxed to (\d+) in spring wheat, whose nursery design gives a median of (\d+) lines per environment, and set to (\d+) for the maize",
        lambda: [PANEL_OF["common bean"][1], PANEL_OF["spring wheat"][1], s("spring wheat", "genos_per_env"),
                 code_int(f"{CC}/partition.py", r'SETS = \[\("maize", None, (\d+)\)')], f"{CC}/build_summary.py SETS; {CC}/partition.py SETS; " + SUM)
    reg("§2.3", "每个 panel 的基础方法数范围", r"a panel of (\d+)–(\d+) base methods was constructed for each",
        lambda: [min(len(v) for v in panel_methods().values()), max(len(v) for v in panel_methods().values())], "各 *_panel_wide.csv 去掉 '__' 变体")
    reg("§2.3", "每个 panel 都含岭回归(四维度×两惩罚)/RF/GBM/kNN/MLP/集成",
        r"Every panel contains ridge regression on an environment index and marker principal components at (four) dimensionalities crossed with (two) penalties",
        lambda: [every_panel_ok() == [], True], "各 *_panel_wide.csv 基础方法名", kind="bool",
        note_fn=lambda: ("不符: " + "; ".join(every_panel_ok())) if every_panel_ok() else "")
    reg("§2.3", "保留主成分数范围", r"retaining (\d+)–(\d+) principal components depending on dataset",
        lambda: list(npc_all()), f"{CC}/panel_bean.py, panel_ursn.py NPC; {CC}/prep_nust.py NPC")
    reg("§2.3", "大豆名称匹配: 原始名称/归一化基因型/原样匹配", r"recovers all ([\d,]+) raw names \(([\d,]+) genotypes after normalisation\) against (\d+) by raw matching",
        lambda: [D.nust_names[2], pd.read_csv(f"{X}/nust_ge.csv").k.nunique(), D.nust_names[1]], f"{RAW}/nust/nust_ge_means.csv + VCF 表头; {X}/nust_ge.csv")
    reg("§2.3", "G2F mean RMSE 范围与CV", r"spans mean RMSE " + N + "–" + N + r" \(CV = " + N,
        lambda: [D.S1.mean_RMSE.min(), D.S1.mean_RMSE.max(), D.S1.mean_RMSE.std() / D.S1.mean_RMSE.mean()], S1f + " mean_RMSE")
    reg("§2.3", "未增广大豆 panel mean RMSE 范围与CV", r"unmodified soybean panel spans " + N + "–" + N + r" \(CV = " + N,
        lambda: [pd.read_csv(f"{X}/nust_metrics.csv").mean_RMSE.min(), pd.read_csv(f"{X}/nust_metrics.csv").mean_RMSE.max(),
                 pd.read_csv(f"{X}/nust_metrics.csv").mean_RMSE.std() / pd.read_csv(f"{X}/nust_metrics.csv").mean_RMSE.mean()], f"{X}/nust_metrics.csv")
    reg("§2.3", "增广变体参数 shrunk/inflated/biased", r"shrunk \(×" + N + r" about the mean\), inflated \(×" + N + r"\), biased \(\+" + N + " SD",
        lambda: [code_int(f"{CC}/widen_panel.py", r"p\.mean\(\)\+([\d.]+)\*\(p-p\.mean\(\)\)\)\s+# dispersion collapse"),
                 code_int(f"{CC}/widen_panel.py", r"p\.mean\(\)\+([\d.]+)\*\(p-p\.mean\(\)\)\)\s+# dispersion blow-up"),
                 code_int(f"{CC}/widen_panel.py", r"p\+([\d.]+)\*y\.std\(\)")], f"{CC}/widen_panel.py")

    # ================= 2.4
    reg("§2.4", "单调检验: 分段线性, 六个内节点, 平均秩", r"a random strictly increasing (piecewise-linear) map through (six) sorted random interior knots, applied to (average) within-environment ranks",
        lambda: [code_has(f"{CC}/monotone_test_panels.py", "np.interp("), code_has(f"{CC}/monotone_test_panels.py", "rng.uniform(0,1,6)"),
                 code_has(f"{CC}/monotone_test_panels.py", 'rankdata(p,method="average")')], f"{CC}/monotone_test_panels.py mono()", kind="bool")
    reg("§2.4", "单调检验重复数与提交数", r"\((\d+) replicates on each of the (five) verified submissions\)",
        lambda: [code_int(f"{CC}/monotone_test_panels.py", r"for _ in range\((\d+)\):"), len(V5)], f"{CC}/monotone_test_panels.py")
    reg("§2.4", "仿射检验 a 区间与重复数; b 的标准差为 SD(y_e)", r"aₑ ~ U\(" + N + ", " + N + r"\) and bₑ ~ N\(0, SD\(yₑ\)²\), (\d+) replicates",
        lambda: [code_int(f"{GC}/invariance.py", r"rng\.uniform\(([\d.]+),"), code_int(f"{GC}/invariance.py", r"rng\.uniform\([\d.]+,([\d.]+),"),
                 code_int(f"{GC}/invariance.py", r"for rep in range\((\d+)\)")], f"{GC}/invariance.py",
        note_fn=lambda: "" if code_has(f"{GC}/invariance.py", "rng.normal(0,1.0,") and code_has(f"{GC}/invariance.py", "*sd") else "b_e 不再按环境 SD 缩放")
    reg("§2.4", "通过者: 单调漂移为零", r"the drift is the floating-point residual of an identity — (zero under the monotone test)",
        lambda: [float(D.mono[D.tier("I")].max()) == 0.0], MON + " Tier I", kind="bool")
    reg("§2.4", "通过者: 仿射漂移上限", r"and at most " + N + " under the affine test; among",
        lambda: [D.inv[D.tier("I") + D.tier("II")].max()], INV + " Tier I–II")
    reg("§2.4", "序敏感但失败者的最小漂移 (仿射/单调)", r"among the order-sensitive metrics that fail, the smallest drift is " + N + " under the affine test and " + N + r" under the monotone test \(" + N + "–" + N + " across datasets in the panel tests\)",
        lambda: [D.inv[[m for m in D.pmz.index if D.pmz.loc[m, "order_sensitive"] and D.pmz.loc[m, "tier"] == "III"]].min(),
                 D.mono[[m for m in D.pmz.index if D.pmz.loc[m, "order_sensitive"] and D.pmz.loc[m, "tier"] != "I"]].min(),
                 D.part.min_drift_tierIII_affine.min(), D.part.min_drift_tierIII_affine.max()], INV + "; " + MON + "; " + PM + "; " + PB)
    reg("§2.4", "面板上的检验重复数", r"base methods of each constructed panel \((\d+) replicates\)",
        lambda: [code_int(f"{CC}/partition.py", r"REPS = (\d+)")], f"{CC}/partition.py REPS")
    reg("§2.4", "并列阈值 10⁻⁹", r"\(\|Δ\| < (10⁻⁹)\) cannot reverse",
        lambda: [code_int(f"{CC}/scope.py", r"ACC_TIE = ([\de.-]+)") == 1e-9], f"{CC}/scope.py ACC_TIE", kind="bool")
    reg("§2.4", "聚类bootstrap重复数与宽度倍数", r"in the panels \(([\d,]+) replicates\); these are " + N + "–" + N + " times wider than binomial intervals",
        lambda: [code_int(f"{CC}/analyse_species.py", r"def reversal_ci\(T,rng,B=(\d+)"), D.rr1.width_ratio.min(), D.rr1.width_ratio.max()],
        f"{CC}/analyse_species.py reversal_ci; " + RR1)
    reg("§2.4", "指标对数/team bootstrap 重复数", r"For the (\d+) metric pairs in maize, each rate was tested .*?\(([\d,]+) replicates\)",
        lambda: [len(D.rt), code_int(f"{GC}/reversal_tests.py", r"B, H0 = (\d+),")], RT + "; reversal_tests.py B", flags=re.S)
    reg("§2.4", "环境重抽重复数", r"under the fixed metric `mean_pearson_r` \((\d+) replicates\)",
        lambda: [code_int(f"{CC}/robustness.py", r"for _ in range\((\d+)\):\s*\n\s*e=rng\.choice")], f"{CC}/robustness.py")
    reg("§2.4", "阈值: 共同环境数下限", r"For every pair of base methods sharing at least (eight) environments",
        lambda: [code_int(f"{CC}/threshold_model.py", r"MIN_ENV = (\d+)")], f"{CC}/threshold_model.py MIN_ENV")
    reg("§2.4", "阈值: 两向 bootstrap 重复数", r"two-way cluster bootstrap, (\d+) replicates",
        lambda: [code_int(f"{CC}/threshold_model.py", r"def boot_threshold\(R, G, meths, rng, B=(\d+)\)")], f"{CC}/threshold_model.py")
    reg("§2.4", "玉米五份提交的方法对数", r"whose five verified submissions give (ten) pairs",
        lambda: [len(V5) * (len(V5) - 1) // 2], "C(5,2)")
    reg("§2.4", "结果检验: 前10%, 400次分割",
        r"top-ranked method \(the mean over methods tied for first\), selecting the top (\d+) % of genotypes, was measured on half B, over (\d+) splits",
        lambda: [100 * code_int(f"{CC}/does_it_help.py", r"FRAC=([\d.]+)"), code_int(f"{CC}/does_it_help.py", r"def run\(P,label,min_n,B=(\d+)")],
        f"{CC}/does_it_help.py FRAC/B", tol=[0.5, 0.5])
    reg("§2.4", "结果检验: 并列方法取均值", r"top-ranked method \((the mean over methods tied for first)\)",
        lambda: [code_has(f"{CC}/does_it_help.py", "scope.tied_best_mean(")], f"{CC}/does_it_help.py", kind="bool")
    reg("§2.4", "保留环境数与地块数 / 训练文件地块数", r"on the (\d+) environments of the G2F 2014–2021 training data that pass the retention rules of Supplementary Section S6 \(([\d,]+) of its ([\d,]+) plots\)",
        lambda: [len(pd.read_csv(f"{X}/heritability_ceiling_A8.csv").query("dataset == 'maize_G2F'")),
                 pd.read_csv(f"{X}/heritability_ceiling_A8.csv").query("dataset == 'maize_G2F'").plots.sum(), D.train_plots],
        f"{X}/heritability_ceiling_A8.csv; {RAW}/g2f/1_Training_Trait_Data_2014_2021.csv")
    reg("§2.4", "噪声零分布: 信号/噪声倍数与面板数", r"a noise null \(disagreement between two metrics is " + N + "–" + N + " times one metric's sampling variability\)",
        lambda: [D.rob.signal_over_noise.min(), D.rob.signal_over_noise.max()], ROB)

    # ================= 3.1
    reg("§3.1", "标题: 22 个中 1 个", r"### 3\.1 Which metrics can represent the decision: (one) of (twenty-two)",
        lambda: [nT("I"), len(D.pmz)], PM)
    reg("§3.1", "mean_pearson_r 仿射/单调漂移", r"`mean_pearson_r` is invariant under the affine test \(drift " + N + r"\) but moves by " + N + " under the monotone test",
        lambda: [D.inv["mean_pearson_r"], D.mono["mean_pearson_r"]], INV + "; " + MON)
    reg("§3.1", "春小麦: 最佳方法与其反序得分相同; 两者选择差",
        r"it gives the best of the (seventeen) base methods and the same method with its predictions reversed within each environment the same score, " + N + ", although the first selects a top decile " + N + " phenotypic standard deviations above the environment mean and the second one " + N + " below",
        lambda: [D.part.loc["spring wheat", "n_methods"], D.ce_best.mean_r2_pearson, D.ce_best.SG_f10, -D.ce_rev.SG_f10],
        f"{X}/counterexample_ursn.csv ({CC}/counterexample_ursn.py); " + PB,
        note_fn=lambda: (f"最佳方法 {D.ce_best.method}; r² 同分: {abs(D.ce_best.mean_r2_pearson - D.ce_rev.mean_r2_pearson) < 1e-12}; "
                         f"Pearson 也最高: {D.ce_best.method == D.ce[~D.ce.reversed.astype(bool)].sort_values('mean_pearson_r').method.iloc[-1]}"))
    reg("§3.1", "误差族漂移范围 (单调/仿射)", r"the fourteen error-magnitude metrics move by " + N + " to " + N + " under the monotone test and by " + N + " to " + N + " under the affine test",
        lambda: [D.mono[D.err_metrics].min(), D.mono[D.err_metrics].max(), D.inv[D.err_metrics].min(), D.inv[D.err_metrics].max()], MON + "; " + INV + " (14个误差族)")
    reg("§3.1", "合并 Spearman 单调漂移", r"pooled Spearman, computed across environments rather than within them, moves by " + N,
        lambda: [D.mono["spearman_r"]], MON)
    reg("§3.1", "划分在五个数据集相同 (1/1/20)", r"it recurs in every dataset — (one, one and twenty) in maize, common bean, spring wheat, soybean and a fifth dataset",
        lambda: [len(D.part) == 5 and bool(((D.part.tier_I == 1) & (D.part.tier_II == 1) & (D.part.tier_III == 20)).all())], PB, kind="bool")
    reg("§3.1", "配对排名改变概率 (失败者)", r"reorders them in (\d+)–(\d+) % of replicates",
        lambda: [100 * D.pinv[(D.pinv.tie_convention == "tie-safe") & (D.pinv.p_ranking_changed > 0)].p_ranking_changed.min(),
                 100 * D.pinv[(D.pinv.tie_convention == "tie-safe")].p_ranking_changed.max()], f"{G}/paired_invariance.csv tie-safe")
    reg("§3.1", "决策型指标在三数据集两检验下的最大漂移", r"are invariant under both tests in maize, common bean and soybean \(drift at most " + N + r"\)",
        lambda: [float(D.dmi[D.dmi.metric.isin(["SG", "hit", "NDCG", "kendall"])][["affine", "monotone"]].max().max())],
        f"{G}/decision_metrics_invariance.csv", kind="le")
    reg("§3.1", "RMSE 仿射/单调漂移范围", r"RMSE on the same predictions moves by " + N + "–" + N + " under the affine test and " + N + "–" + N + " under the monotone test",
        lambda: [D.dmi[D.dmi.metric == "RMSE"].affine.min(), D.dmi[D.dmi.metric == "RMSE"].affine.max(),
                 D.dmi[D.dmi.metric == "RMSE"].monotone.min(), D.dmi[D.dmi.metric == "RMSE"].monotone.max()], f"{G}/decision_metrics_invariance.csv")

    # ================= 3.2
    reg("§3.2", "标题: 十二/三十", r"### 3\.2 How far the choice moves a ranking: a quarter of team pairs, (twelve) of (thirty) places",
        lambda: [D.rc.loc["maize", "census_median_range"], D.rc.loc["maize", "methods"]], RC)
    reg("§3.2", "官方指标间反转 计数/比例/CI", r"(\d+) of (\d+) team pairs reverse order — " + N + r" %, team-bootstrap 95 % CI \[" + N + r" %, " + N + r" %\]",
        lambda: [round(s("maize", "reversal") * 435), s("maize", "methods") * (s("maize", "methods") - 1) / 2, 100 * s("maize", "reversal"),
                 100 * s("maize", "ci_lo"), 100 * s("maize", "ci_hi")], SUM + " maize")
    reg("§3.2", "前十名反转率与CI", r"Among the top ten teams the rate is " + N + r" % \[" + N + r" %, " + N + r" %\]",
        lambda: [100 * pd.read_csv(f"{G}/top10_reversal.csv").iloc[0][c] for c in ("rate", "ci_lo", "ci_hi")], f"{G}/top10_reversal.csv")
    reg("§3.2", "置换零分布均值", r"against " + N + " for unrelated rankings",
        lambda: [D.perm_null], S1f + " (reversal.py 种子 20260908, 5000 次)")
    reg("§3.2", "名次变化中位数/最大", r"Median rank change is " + N + " places, the largest (fifteen)",
        lambda: [(D.tierrank.official_2022 - D.tierrank.official_2024).abs().median(), (D.tierrank.official_2022 - D.tierrank.official_2024).abs().max()], TIER)
    reg("§3.2", "前三名集合不同", r"and the (top-three set differs) between the two metrics",
        lambda: [set(D.tierrank[D.tierrank.official_2022 <= 3].team) != set(D.tierrank[D.tierrank.official_2024 <= 3].team)], TIER, kind="bool")
    reg("§3.2", "231对指标反转率中位数", r"Across all (\d+) pairs of the 22 metrics the median reversal rate is " + N + " %",
        lambda: [len(D.rv), 100 * D.rv.rate.median()], RV)
    reg("§3.2", "BH 显著数 (5 %, 1 %) 与官方对校正P", r"and (\d+) pairs reverse more than (\d+) % of team pairs under Benjamini–Hochberg control at FDR (\d+) % \((\d+) at 1 %\); for the two official metrics the adjusted P is " + N,
        lambda: [int((D.rt.q_wald < .05).sum()), 5, 5, int((D.rt.q_wald < .01).sum()), D.rt_off.q_wald], RT + " q_wald")
    reg("§3.2", "类内/类间反转", r"The two Tier I–II metrics reverse " + N + r" % of team pairs against each other, and the (fourteen) error-magnitude metrics a median of " + N + r" % \(" + N + "–" + N + r" %\) among themselves, but the two groups reverse a median of " + N + r" % \(" + N + "–" + N + r" %\)",
        lambda: [100 * D.rbc.loc["Tier I-II pair", "median"], len(D.err_metrics),
                 100 * D.rbc.loc["within error family", "median"], 100 * D.rbc.loc["within error family", "lo"], 100 * D.rbc.loc["within error family", "hi"],
                 100 * D.rbc.loc["Tier I-II vs error family", "median"], 100 * D.rbc.loc["Tier I-II vs error family", "lo"], 100 * D.rbc.loc["Tier I-II vs error family", "hi"]], RBC)
    reg("§3.2", "Tier-I 排名 vs 2022/2024 官方", r"differs from the 2022 official leaderboard by a median of (\d+) places \(maximum (\d+)\) but from the 2024 one by (\d+) \(maximum (\d+)\)",
        lambda: [(D.tierrank.official_2022 - D.tierrank.tier_I).abs().median(), (D.tierrank.official_2022 - D.tierrank.tier_I).abs().max(),
                 (D.tierrank.official_2024 - D.tierrank.tier_I).abs().median(), (D.tierrank.official_2024 - D.tierrank.tier_I).abs().max()], TIER)
    reg("§3.2", "模型类别数/平均名次变化", r"with the (four) model classes labelled in the competition paper \(mean " + N + " to " + N + " places",
        lambda: [D.mcp.model_class.nunique(), D.mcp.mean_move.min(), D.mcp.mean_move.max()], f"{G}/model_class_permutation.csv")
    reg("§3.2", "模型类别置换P下界", r"places, permutation P > " + N + r"\)",
        lambda: [D.mcp.perm_p.min()], f"{G}/model_class_permutation.csv perm_p", kind="gt")
    reg("§3.2", "面板反转率 (未增广/增广)", r"(\d+)–(\d+) % of base-method pairs reverse, and (\d+)–(\d+) % once the miscalibrated variants are added",
        lambda: [100 * min(D.unaug(d) for d in NONMAIZE), 100 * max(D.unaug(d) for d in NONMAIZE),
                 100 * D.summ.loc[NONMAIZE].reversal.min(), 100 * D.summ.loc[NONMAIZE].reversal.max()], SUM + "; " + RR1)
    reg("§3.2", "玉米值位于两者之间", r"the maize rate from 30 independent submissions lies (between the two)",
        lambda: [max(D.unaug(d) for d in NONMAIZE) < s("maize", "reversal") < D.summ.loc[NONMAIZE].reversal.min()], SUM + "; " + RR1, kind="bool")
    reg("§3.2", "玉米名次跨度/占比/冠军", r"a G2F team's rank spans a median of (\d+) of (\d+) places \((\d+) %\), and the winning team's spans first to (seventh)",
        lambda: [D.rc.loc["maize", "census_median_range"], D.rc.loc["maize", "methods"], 100 * D.rc.loc["maize", "census_range_frac"],
                 float(json.loads(D.rc.loc["maize", "leader_observed"])[1])], RC)
    reg("§3.2", "区间半宽/中位宽/冠军区间", r"has a half-width of (nine) places, giving a median width of (\d+) places \((\d+) %\) after clipping at the ends of the board and \[(\d+), (\d+)\] for the winner",
        lambda: [D.rc.loc["maize", "conformal_q"], D.rc.loc["maize", "conformal_median_width"], 100 * D.rc.loc["maize", "conformal_width_frac"],
                 *json.loads(D.rc.loc["maize", "leader_conformal"])], RC)
    reg("§3.2", "留一指标覆盖率", r"when one metric is left out \(coverage " + N + " %",
        lambda: [100 * float(D.rri[D.rri.quantity == "loo_metric_coverage_mean"].value.iloc[0])], f"{G}/rank_interval_transfer.csv")
    reg("§3.2", "误差族校准→相关族覆盖率", r"it covers the correlation metrics' ranks (\d+) % of the time",
        lambda: [100 * float(D.rri[(D.rri.calibrate_on == "error-magnitude (14)") & (D.rri.predict == "correlation (6)")].coverage.iloc[0])], f"{G}/rank_interval_transfer.csv")
    reg("§3.2", "三个面板: 观察跨度/区间宽度", r"In the three panels the observed range spans (\d+)–(\d+) % of the board and the interval (\d+)–(\d+) %",
        lambda: [100 * D.rc.loc[NONMAIZE].census_range_frac.min(), 100 * D.rc.loc[NONMAIZE].census_range_frac.max(),
                 100 * D.rc.loc[NONMAIZE].conformal_width_frac.min(), 100 * D.rc.loc[NONMAIZE].conformal_width_frac.max()], RC)
    reg("§3.2", "去冗余后宽度/单类内宽度", r"leaves the maize width at (\d+) places; only restricting the panel to one class narrows it, to (\d+) places within the error-magnitude family and (\d+) for the two Tier I–II metrics",
        lambda: [D.ris.loc["six distinct quantities, within environment", "median_width"], D.ris.loc["error-magnitude family only", "median_width"],
                 D.ris.loc["Tier I and II metrics only", "median_width"]], f"{G}/rank_interval_sensitivity.csv")
    reg("§3.2", "环境重抽区间占比", r"gives median intervals of (\d+)–(\d+) % of the board in the three panels",
        lambda: [100 * (D.rob.boot_width / D.rob.methods).min(), 100 * (D.rob.boot_width / D.rob.methods).max()], ROB + " boot_width/methods")
    reg("§3.2", "环境重抽宽度为指标宽度的五分之一到一半", r"(a fifth to a half) of the width across metrics",
        lambda: [bool((D.rob.loc[NONMAIZE].boot_width / D.rc.loc[NONMAIZE].conformal_median_width).between(0.18, 0.5).all())], ROB + "; " + RC, kind="bool",
        note_fn=lambda: ", ".join(f"{k} {v:.2f}" for k, v in (D.rob.loc[NONMAIZE].boot_width / D.rc.loc[NONMAIZE].conformal_median_width).items()))

    # ================= 3.3
    reg("§3.3", "结果信度 (三个面板) 与未增广三面板", r"split-half reliability " + N + " in common bean, " + N + " in spring wheat, " + N + r" in soybean\), and stays above the 0\.5 criterion without the added variants \(" + N + ", " + N + ", " + N + r"\)",
        lambda: [s("common bean", "reliability"), s("spring wheat", "reliability"), s("soybean", "reliability"),
                 D.rel["COMMON BEAN (unaugmented)"], D.rel["SPRING WHEAT (FHB) (unaugmented)"], D.rel["SOYBEAN (unaugmented)"]],
        SUM + "; does_it_help_reliability.json")
    reg("§3.3", "大豆两族回收范围与实际选择差之差", r"the correlation family recovers (\d+)–(\d+) % of the available selection gain against (\d+)–(\d+) % for the error-magnitude family, and the method ranked first by `mean_pearson_r` selected material " + N + " phenotypic SD better than that ranked first by `mean_RMSE`",
        lambda: [*D.fam("soybean", "corr"), *D.fam("soybean", "err"), D.raw("soybean", "mean_pearson_r") - D.raw("soybean", "mean_RMSE")], DIH)
    reg("§3.3", "仅大豆的配对区间排除 0", r"With environments resampled that gap excludes zero for both correlation rules — `mean_pearson_r` " + N + r" \[" + N + ", " + N + r"\] percentage points against `mean_RMSE`, `mean_spearman_r` " + N + r" \[" + N + ", " + N + r"\] — the only panel in which the paired contrast excludes zero",
        lambda: [100 * D.paired("soybean", "D1", c) for c in ("paired_pearson_minus_rmse", "paired_ci_lo", "paired_ci_hi")]
                + [100 * D.paired("soybean", "D1", c) for c in ("paired_spearman_minus_rmse", "paired_spearman_ci_lo", "paired_spearman_ci_hi")],
        RDS + " D1",
        note_fn=lambda: "" if all(D.paired(d, "D1", lo) < 0 < D.paired(d, "D1", hi) for d in ("common bean", "spring wheat")
                     for lo, hi in (("paired_ci_lo", "paired_ci_hi"), ("paired_spearman_ci_lo", "paired_spearman_ci_hi")))
                     and all(D.paired("soybean", "D1", c) > 0 for c in ("paired_ci_lo", "paired_spearman_ci_lo"))
                     else "菜豆/春小麦仍有区间排除0, 或大豆下限未过0!")
    reg("§3.3", "春小麦: 两族回收范围与两条配对对比 (D1)",
        r"In spring wheat the correlation family also leads, " + N + "–" + N + " % against " + N + "–" + N + r" %, but less securely \(" + N + r" \[" + N + ", " + N + r"\] and " + N + r" \[" + N + ", " + N + r"\] points\)",
        lambda: [*D.fam("spring wheat", "corr"), *D.fam("spring wheat", "err"),
                 100 * D.paired("spring wheat", "D1", "paired_pearson_minus_rmse"), 100 * D.paired("spring wheat", "D1", "paired_ci_lo"), 100 * D.paired("spring wheat", "D1", "paired_ci_hi"),
                 100 * D.paired("spring wheat", "D1", "paired_spearman_minus_rmse"), 100 * D.paired("spring wheat", "D1", "paired_spearman_ci_lo"), 100 * D.paired("spring wheat", "D1", "paired_spearman_ci_hi")],
        DIH + "; " + RDS + " D1")
    reg("§3.3", "菜豆: 两族回收范围", r"in common bean the two families choose methods of comparable value \(" + N + "–" + N + " % and " + N + "–" + N + r" %\)",
        lambda: [*D.fam("common bean", "corr"), *D.fam("common bean", "err")], DIH)
    reg("§3.3", "春小麦: 优势族随决策反转 (D1 相关族领先, D2 误差族领先)", r"In spring wheat the family that selects better therefore (reverses with the decision)",
        lambda: [D.paired("spring wheat", "D1", "paired_pearson_minus_rmse") > 0 and D.paired("spring wheat", "D1", "paired_spearman_minus_rmse") > 0
                 and D.paired("spring wheat", "D2", "paired_ci_hi") < 0 and D.paired("spring wheat", "D2", "paired_spearman_ci_hi") < 0], RDS, kind="bool")
    reg("§3.3", "除 NDCG 外每条规则增广前后选出的材料价值相同 (且 NDCG 确实不同)", r"every rule except NDCG@10 % (selects material of identical realised value) with or without them",
        lambda: [all(abs(D.raw(d, r) - D.raw(d, r, unaug=True)) < 1e-9 for d in NONMAIZE
                     for r in ("mean_pearson_r", "mean_spearman_r", "mean_RMSE", "mean_MAE", "mean_r2_score"))
                 and any(abs(D.raw(d, "mean_NDCG@10%") - D.raw(d, "mean_NDCG@10%", unaug=True)) > 1e-9 for d in NONMAIZE)], DIH, kind="bool")
    reg("§3.3", "绝对阈值: RMSE 在菜豆春小麦的领先幅度范围", r"`mean_RMSE` leads both correlation rules in common bean and spring wheat, by " + N + "–" + N + r" percentage points, with the interval excluding zero in spring wheat and, at this sample size, not quite in common bean",
        lambda: [-100 * max(D.paired(d, "D2", c) for d in ("common bean", "spring wheat") for c in ("paired_pearson_minus_rmse", "paired_spearman_minus_rmse")),
                 -100 * min(D.paired(d, "D2", c) for d in ("common bean", "spring wheat") for c in ("paired_pearson_minus_rmse", "paired_spearman_minus_rmse"))], RDS + " D2",
        note_fn=lambda: "" if D.paired("spring wheat", "D2", "paired_ci_hi") < 0 and D.paired("spring wheat", "D2", "paired_spearman_ci_hi") < 0
                        and D.paired("common bean", "D2", "paired_ci_hi") > 0 else "春小麦/菜豆的区间符号与描述不符!")
    reg("§3.3", "大豆 GBLUP 校准: 两个环境内斜率、其余方法最大斜率、mean_RMSE 高出最低值的百分比",
        r"within-environment slopes " + N + " and " + N + ", other base methods at most " + N + "; mean RMSE within " + N + " % of the lowest",
        lambda: soy_gblup_calibration(), f"{X}/nust_metrics_wide.csv")
    DGM = f"{X}/decision_genotype_means.csv"
    _dg = lambda: pd.read_csv(DGM).drop_duplicates("dataset").set_index("dataset")
    reg("§3.3", "跨环境基因型均值决策: 菜豆/大豆 mean_spearman_r 与基因型均值秩相关之差 [CI]",
        r"where the outcome is reliable — " + N + r" \[" + N + ", " + N + r"\] percentage points in common bean and " + N + r" \[" + N + ", " + N + r"\] in soybean",
        lambda: [100 * _dg().loc[d, c] for d in ("common bean", "soybean")
                 for c in ("spearman_minus_gm_spearman", "spearman_minus_gm_spearman_lo", "spearman_minus_gm_spearman_hi")], DGM)
    reg("§3.3", "跨环境基因型均值决策: 只比较信度 ≥ 0.5 的数据集 (菜豆、大豆), 且两处区间含 0",
        r"(no detectable difference) between `mean_spearman_r` and that metric where the outcome is reliable",
        lambda: [bool(_dg().loc["common bean", "outcome_reliability"] >= .5 and _dg().loc["soybean", "outcome_reliability"] >= .5
                      and not _dg().loc["spring wheat", "outcome_reliability"] >= .5
                      and all(_dg().loc[d, "spearman_minus_gm_spearman_lo"] < 0 < _dg().loc[d, "spearman_minus_gm_spearman_hi"] for d in ("common bean", "soybean")))], DGM, kind="bool")
    reg("§3.3", "跨环境基因型均值决策: 大豆两条秩相关规则领先 mean_RMSE 的幅度",
        r"in soybean both lead `mean_RMSE`, by " + N + " and " + N + r" percentage points",
        lambda: [100 * _dg().loc["soybean", "spearman_minus_rmse"], 100 * _dg().loc["soybean", "gm_spearman_minus_rmse"]], DGM)
    reg("§3.3", "跨环境基因型均值决策: 大豆两处区间排除 0",
        r"percentage points, (with intervals excluding zero) \(Supplementary Section S15\)",
        lambda: [bool(_dg().loc["soybean", "spearman_minus_rmse_lo"] > 0 and _dg().loc["soybean", "gm_spearman_minus_rmse_lo"] > 0)], DGM, kind="bool")
    reg("§3.3", "Spearman 与 Pearson 回收之差: 菜豆/大豆都在 2 个百分点以内",
        r"`mean_spearman_r` recovers (within two percentage points) of `mean_pearson_r` in common bean and soybean",
        lambda: [max(abs(D.recov(d, "mean_pearson_r") - D.recov(d, "mean_spearman_r")) for d in ("common bean", "soybean")) < 2], DIH, kind="bool")
    reg("§3.3", "Spearman 比 Pearson 在春小麦少回收的百分点",
        r"in common bean and soybean and (\d+) points less in spring wheat",
        lambda: [D.recov("spring wheat", "mean_pearson_r") - D.recov("spring wheat", "mean_spearman_r")], DIH)
    reg("§3.3", "NDCG@10% 回收范围 (春小麦最低, 大豆最高)", r"the top-weighted NDCG@10 % ranges from " + N + " % in spring wheat to " + N + " % in soybean",
        lambda: [D.recov("spring wheat", "mean_NDCG"), D.recov("soybean", "mean_NDCG")], DIH,
        note_fn=lambda: "" if min(D.recov(d, "mean_NDCG") for d in NONMAIZE) == D.recov("spring wheat", "mean_NDCG") and max(D.recov(d, "mean_NDCG") for d in NONMAIZE) == D.recov("soybean", "mean_NDCG") else "端点并非春小麦/大豆!")
    reg("§3.3", "玉米各规则回收", r"`mean_spearman_r` recovers (\d+) %, `mean_pearson_r` (\d+) %, `mean_RMSE` (\d+) % and `mean_r2_score` " + N + " %",
        lambda: [D.recov("maize", r) for r in ("mean_spearman_r", "mean_pearson_r", "mean_RMSE", "mean_r2_score")], DIH)
    reg("§3.3", "玉米绝对阈值: Pearson 领先而 Spearman 不", r"under the absolute target `mean_pearson_r` leads `mean_RMSE` there, " + N + r" \[" + N + ", " + N + r"\], while `mean_spearman_r` does not, " + N + r" \[" + N + ", " + N + r"\]",
        lambda: [100 * D.paired("maize", "D2", c) for c in ("paired_pearson_minus_rmse", "paired_ci_lo", "paired_ci_hi")]
                + [100 * D.paired("maize", "D2", c) for c in ("paired_spearman_minus_rmse", "paired_spearman_ci_lo", "paired_spearman_ci_hi")], RDS + " D2 maize")

    # ================= 3.4
    reg("§3.4", "15 个比较全部为负; 三个强度的均值; f=0.10 五个缺口",
        r"falls below iₖ · r in all (\d+) comparisons: by a mean of " + N + ", " + N + " and " + N + " phenotypic standard deviations at f = 0\.05, 0\.10 and 0\.20, and at f = 0\.10 by " + N + ", " + N + ", " + N + ", " + N + " and " + N + " SD",
        lambda: [int((D.gap.gap < 0).sum())] + fmeans() + [-g for g in D.gaps(0.10)], GAP,
        note_fn=lambda: f"比较总数 {len(D.gap)}")
    reg("§3.4", "环境数/SE/|t|/P≤0.001 计数", r"Standard errors over the (\d+)–(\d+) environments are " + N + "–" + N + " SD, \|t\| is " + N + "–" + N + ", and P ≤ 0\.001 in (\d+) of the (\d+)",
        lambda: [D.gap.n_env.min(), D.gap.n_env.max(), D.gap.se.min(), D.gap.se.max(), D.gap.t.abs().min(), D.gap.t.abs().max(),
                 int((D.gap.p_two_sided <= .001).sum()), len(D.gap)], GAP)
    reg("§3.4", "前10%低于环境均值的提交数", r"(one) submission's top decile performed below its environment mean",
        lambda: [int((D.gap[D.gap.frac == 0.10].mean_observed < 0).sum())], GAP + " mean_observed (f=0.10)")
    reg("§3.4", "队伍数与其 2022 官方名次", r"come from (three) teams that the 2022 leaderboard placed (\d+)(?:st|nd|rd|th), (\d+)(?:st|nd|rd|th) and (\d+)(?:st|nd|rd|th) of (\d+)",
        lambda: [len({t.split("_")[0] for t in V5})] + sorted(
            float(D.tierrank.set_index("team").loc[t, "official_2022"]) for t in ("Kernel of Truth", "EnBiSys", "Niche Squad")) + [len(D.tierrank)],
        TIER + " official_2022")
    reg("§3.4", "copula 前后平均缺口 (有限群体基准, f=0.10, 0.20)", r"the mean shortfall over f = 0\.10 and 0\.20 falls from " + N + " against the finite-population benchmark to " + N + " SD",
        lambda: [-D.cop10.gap_N.mean(), -D.cop10.gap_C1.mean()], COP + " gap_N, gap_C1")
    reg("§3.4", "copula 缺口处处为负", r"stays (negative) for every submission at every intensity",
        lambda: [bool((D.cop.gap_C1 < 0).all())], COP, kind="bool")
    reg("§3.4", "copula 残差 |t|", r"with residual \|t\| of " + N + "–" + N,
        lambda: [(D.cop.gap_C1 / D.cop.se_C1).abs().min(), (D.cop.gap_C1 / D.cop.se_C1).abs().max()], COP)
    reg("§3.4", "正态与观测边缘之差", r"normal and observed margins differ by " + N + " SD",
        lambda: [D.cop10.margin_effect.abs().mean()], COP + " (f=0.10,0.20)")
    reg("§3.4", "Pearson 高估/低估秩一致性 (五份全列)", r"overstates the rank agreement in (four) of the five submissions — by " + N + " %, (\d+) %, (\d+) % and (\d+) % — and understates it by (\d+) % in the fifth",
        lambda: [int((D.overstate > 0).sum())] + sorted(100 * v for v in D.overstate if v > 0) + [-100 * D.overstate.min()],
        COP + " (bench_N − bench_C0)/bench_N 按队均值", tol=[None, 0.05, None, None, None, None])
    reg("§3.4", "每环境基因型数中位数", r"with a median of (\d+) genotypes per environment, replacing iₖ",
        lambda: [D.n_per_env_maize], "Final_Observed_Yield.csv × 五份提交")
    reg("§3.4", "有限群体校正: 只改第三位小数, 解释为零", r"changes the projection in the (third decimal and explains none) of the gap",
        lambda: [0.0005 <= (pd.read_csv(f"{G}/analytic_bridge_check.csv").set_index(["team", "frac"]).SG_analytic - D.cop.set_index(["team", "frac"]).bench_N).abs().max() < 0.005,
                 max(D.fin_explained) < 0.5], f"analytic_bridge_check.csv; " + fin_src, kind="bool",
        note_fn=lambda: "有限群体解释比例: " + ", ".join(f"{v:.2f}%" for v in D.fin_explained))
    reg("§3.4", "并列破缺对缺口的影响上限", r"resolving every tie in the selected set's favour moves the mean shortfall by at most " + N + " SD",
        lambda: [pd.read_csv(f"{X}/tie_effect_on_gap.csv").eval("gap_best_case - gap_as_analysed").max()],
        f"{X}/tie_effect_on_gap.csv ({CC}/tie_effect_on_gap.py)", kind="le")
    reg("§3.4", "面板缺口, 有限群体基准 (春小麦/大豆 f=0.10, 0.20; 菜豆 f=0.10)",
        r"it is " + N + r" SD at f = 0\.10 in spring wheat \(95 % CI " + N + "–" + N + r"\) and " + N + r" SD in soybean \(" + N + " to " + N + r"\), and " + N + r" \(" + N + "–" + N + r"\) and " + N + r" \(" + N + "–" + N + r"\) at f = 0\.20; in common bean, with (ten) environments, it is " + N + r" \(" + N + " to " + N + r"\) at f = 0\.10",
        lambda: [-D.bg("spring wheat", .1, "gap_fin"), -D.bg("spring wheat", .1, "ci_hi_fin"), -D.bg("spring wheat", .1, "ci_lo_fin"),
                 -D.bg("soybean", .1, "gap_fin"), -D.bg("soybean", .1, "ci_hi_fin"), -D.bg("soybean", .1, "ci_lo_fin"),
                 -D.bg("spring wheat", .2, "gap_fin"), -D.bg("spring wheat", .2, "ci_hi_fin"), -D.bg("spring wheat", .2, "ci_lo_fin"),
                 -D.bg("soybean", .2, "gap_fin"), -D.bg("soybean", .2, "ci_hi_fin"), -D.bg("soybean", .2, "ci_lo_fin"),
                 D.bgp[D.bgp.dataset == "common bean"].n_env.iloc[0],
                 -D.bg("common bean", .1, "gap_fin"), -D.bg("common bean", .1, "ci_hi_fin"), -D.bg("common bean", .1, "ci_lo_fin")], BGP + " all methods, gap_fin")

    # ================= 3.5
    reg("§3.5", "Δr<0.01 时一致比例 (菜豆/大豆) 与春小麦对数", r"among base-method pairs with Δr < " + N + " it is " + N + " % and " + N + r" % in common bean and soybean \(spring wheat has (seven) such pairs\)",
        lambda: [0.01, 100 * D.tpar("common bean", "p_agree_below_001"), 100 * D.tpar("soybean", "p_agree_below_001"), D.tpar("spring wheat", "n_below_001")], TE)
    reg("§3.5", "Δr<0.01 时候选名单差异 (菜豆/大豆)", r"even below Δr = 0\.01, (\d+)–(\d+) % of the two methods' shortlists differ in those two panels",
        lambda: [100 * (1 - max(D.tpar(d, "overlap_below_001") for d in ("common bean", "soybean"))), 100 * (1 - min(D.tpar(d, "overlap_below_001") for d in ("common bean", "soybean")))], TE)
    reg("§3.5", "logistic 阈值与 isotonic 检查", r"the logistic estimate of Δr is " + N + " in common bean, " + N + " in spring wheat and " + N + r" in soybean \(isotonic check " + N + ", " + N + " and " + N,
        lambda: [D.tpar(d, "threshold") for d in NONMAIZE] + [D.tpar(d, "isotonic") for d in NONMAIZE], TE)
    reg("§3.5", "bootstrap 下限; 三个面板上限都超出观测范围; 未达 0.95 的比例", r"Two-way bootstrap lower limits are " + N + ", " + N + " and " + N + r"; the upper limit lies beyond the observed range in all three panels, where " + N + " %, " + N + " % and " + N + r" % of resamples do not reach 0\.95",
        lambda: [D.tpar(d, "ci_lo") for d in NONMAIZE] + [100 * D.tpar(d, "boot_unbounded") for d in NONMAIZE], TE,
        note_fn=lambda: "" if all(not np.isfinite(D.tpar(d, "ci_hi")) for d in NONMAIZE) else "有面板的上限是有限值!")
    reg("§3.5", "样本外阈值 (菜豆/春小麦/大豆); f=0.20 阈值", r"disjoint halves of the environments gives " + N + ", " + N + " and " + N + r", and relaxing selection to f = 0\.20 gives " + N + ", " + N + " and " + N,
        lambda: [D.tpar("common bean", "oos"), D.tpar("spring wheat", "oos"), D.tpar("soybean", "oos")] + [D.tpar(d, "threshold", 0.20) for d in NONMAIZE], TE)
    reg("§3.5", "匹配高斯阈值与倍数", r"give " + N + ", " + N + " and " + N + ", and the observed values are " + N + ", " + N + " and " + N + " times these",
        lambda: [D.ttheory.loc[d, "theory_lam05"] for d in NONMAIZE] + [D.ttheory.loc[d, "ratio_lam05"] for d in NONMAIZE], TT)
    reg("§3.5", "竞赛设计参数", r"— (\d+) methods, (\d+) genotypes per environment, (\d+) environments, accuracies spanning the teams'",
        lambda: [D.tmz.M, D.tmz.n_per_env, D.tmz.n_env], TT + " maize (G2F design)")
    reg("§3.5", "竞赛高斯值/λ 范围/迁移范围", r"the Gaussian value is " + N + r" \(" + N + "–" + N + r" as method errors go from independent to (\d+) % shared\), and scaling by the panels' ratios gives " + N + "–" + N,
        lambda: [D.tmz.theory_lam05, min(lam()), max(lam()), 80, D.tmz.transfer_lo, D.tmz.transfer_hi], TT)
    reg("§3.5", "第1–2名差距", r"First and second place, " + N + " apart in reported `mean_pearson_r`",
        lambda: [D.gap12], S1f)
    reg("§3.5", "第1–2名差距低于高斯值一个数量级且低于所有下限", r"are (an order of magnitude below the Gaussian value and below every panel's lower limit)",
        lambda: [D.gap12 < 0.2 * D.tmz.theory_lam05 and D.gap12 < min(D.tpar(d, "ci_lo") for d in NONMAIZE)], S1f + "; " + TT + "; " + TE, kind="bool",
        note_fn=lambda: f"高斯值/差距 = {D.tmz.theory_lam05 / D.gap12:.1f}")
    reg("§3.5", "前三名数值与差距", r"CLAC " + N + ", CGM " + N + ", UCD_MegaLMM " + N + ", spread " + N,
        lambda: [float(D.S1.set_index("Team Name").loc[t, "mean_pearson_r"]) for t in ("CLAC", "CGM", "UCD_MegaLMM")] + [D.gap13], S1f)
    reg("§3.5", "前三名差距在高斯值之内", r"(lie within the Gaussian value) of one another",
        lambda: [D.gap13 < D.tmz.theory_lam05], S1f + "; " + TT, kind="bool")
    reg("§3.5", "并列队伍数 (0.03, 0.13)", r"(three) teams at " + N + ", (seven) at " + N,
        lambda: [D.within(D.tmz.theory_lam05), 0.03, D.within(D.tmz.transfer_hi), D.tmz.transfer_hi], S1f + "; " + TT)
    reg("§3.5", "按 2024 官方指标择优时并列榜首的队伍数", r"(four) teams lie within 0\.03 of the leader if each is represented by its best `mean_pearson_r` instead",
        lambda: [int(pd.read_csv(f"{G}/selection_rule_sensitivity.csv").set_index("rule").loc["best mean_pearson_r (2024 official)", "within_floor"])],
        f"{G}/selection_rule_sensitivity.csv", note="印刷值为 'four'", tol=None,
        note_fn=lambda: "四种替代规则的并列队伍数: " + ", ".join(str(int(v)) for v in pd.read_csv(f"{G}/selection_rule_sensitivity.csv").within_floor))
    reg("§3.5", "全表中差距小于高斯值/迁移上限的队伍对", r"Across the whole board, (\d+) of the (\d+) team pairs \((\d+) %\) lie closer than the Gaussian value and (\d+) \((\d+) %\) closer than the transferred upper value",
        lambda: [D.pairs_closer(D.tmz.theory_lam05)[0], D.pairs_closer(D.tmz.theory_lam05)[1],
                 100 * D.pairs_closer(D.tmz.theory_lam05)[0] / D.pairs_closer(D.tmz.theory_lam05)[1],
                 D.pairs_closer(D.tmz.transfer_hi)[0], 100 * D.pairs_closer(D.tmz.transfer_hi)[0] / D.pairs_closer(D.tmz.transfer_hi)[1]],
        "S1_best_per_team.csv mean_pearson_r; " + TT + " theory_lam05, transfer_hi")
    reg("§3.5", "设计网格: 阈值≈k/√N (斜率, R², 500 与 8,000–10,000 单元处的值, 三个选择比例的常数)",
        r"At f = 0\.10 it falls as about " + N + r"/√N \(log–log slope " + N + r", R² = " + N + r"\), from " + N + "–" + N + " at " + N + " cells to " + N + " at " + N + "–" + N + r"; the constant is " + N + r" at f = 0\.20 and " + N + r" at f = 0\.05",
        lambda: [cells_law(0.10)[2], cells_law(0.10)[0], cells_law(0.10)[1],
                 min(D.tdesign(25, 20), D.tdesign(50, 10)), max(D.tdesign(25, 20), D.tdesign(50, 10)), 500,
                 max(D.tdesign(400, 20), D.tdesign(50, 200), D.tdesign(100, 100)), 8000, 10000,
                 cells_law(0.20)[2], cells_law(0.05)[2]], TD)
    reg("§3.5", "面板需要的倍数", r"the three panels here needed " + N + "–" + N + " times it",
        lambda: [D.tratio.min(), D.tratio.max()], TT)
    reg("§3.5", "天花板 (玉米 n=1, n=2; 菜豆)", r"the ceiling √H²\(n\) is " + N + r" \(95 % CI " + N + "–" + N + r"\) for a single plot and " + N + r" \(" + N + "–" + N + r"\) for a two-plot mean in the G2F training data \(raw common-bean plots, " + N + " and " + N,
        lambda: [D.hq("maize_G2F", "ceiling_n1"), D.hq("maize_G2F", "ceiling_n1", "lo"), D.hq("maize_G2F", "ceiling_n1", "hi"),
                 D.hq("maize_G2F", "ceiling_n2"), D.hq("maize_G2F", "ceiling_n2", "lo"), D.hq("maize_G2F", "ceiling_n2", "hi"),
                 D.hq("bean_VEF_rawplots", "ceiling_n1"), D.hq("bean_VEF_rawplots", "ceiling_n2")], HB)
    reg("§3.5", "冠军/第1–3差距占天花板", r"the winning team reached (\d+)–(\d+) % of that bound, and the first-to-third gap is " + N + "–" + N + " % of the ceiling",
        lambda: [100 * D.hh("n=2", "winner_frac_of_ceiling"), 100 * D.hh("n=1  single", "winner_frac_of_ceiling"),
                 100 * D.hh("n=2", "gap13_frac_of_ceiling"), 100 * D.hh("n=1  single", "gap13_frac_of_ceiling")], HB)
    reg("§3.5", "i·r 下的相对增益", r"the first-to-second gap is worth " + N + " % of the winner's projected selection differential, the first-to-third " + N + " % and the transferred threshold (\d+)–(\d+) %",
        lambda: [100 * D.gap12 / D.r_top[0], 100 * D.gap13 / D.r_top[0], 100 * D.tmz.transfer_lo / D.r_top[0], 100 * D.tmz.transfer_hi / D.r_top[0]], S1f + "; " + TT)

    # ================= 3.6
    reg("§3.6", "反转率范围 (全部; 仅未增广)", r"reversal of (\d+)–(\d+) % of method pairs when the official metric changes, (\d+)–(\d+) % counting only unaugmented methods",
        lambda: [100 * D.summ.reversal.min(), 100 * D.summ.reversal.max(), 100 * min(D.unaug(d) for d in NONMAIZE),
                 100 * max([D.unaug(d) for d in NONMAIZE] + [s("maize", "reversal")])], SUM + "; " + RR1)
    reg("§3.6", "设计范围 (环境数, 每环境基因型数)", r"designs from (\d+) to (\d+) environments and (\d+) to (\d+) genotypes each",
        lambda: [D.summ.envs.min(), D.summ.envs.max(), D.summ.genos_per_env.min(), D.summ.genos_per_env.max()], SUM)
    reg("§3.6", "名次跨度范围", r"a rank spanning (\d+)–(\d+) % of the board",
        lambda: [100 * D.rc.census_range_frac.min(), 100 * D.rc.census_range_frac.max()], RC)
    reg("§3.6", "10 % 选择下每个数据集都有缺口; 三处区间排除 0",
        r"a realised selection differential (short of its Gaussian projection at 10 % selection everywhere, with an interval excluding zero in maize and spring wheat)",
        lambda: [bool((D.gap[D.gap.frac == 0.10].gap < 0).all()) and all(D.bg(d, .1, "gap_fin") < 0 for d in NONMAIZE)
                 and bool((D.gap[D.gap.frac == 0.10].p_two_sided < .05).all()) and D.bg("spring wheat", .1, "ci_hi_fin") < 0
                 and D.bg("soybean", .1, "ci_hi_fin") > 0 and D.bg("common bean", .1, "ci_hi_fin") > 0], GAP + "; " + BGP + " gap_fin", kind="bool")
    reg("§3.6", "阈值下限范围且高于第1–2名差距", r"a surrogate threshold whose lower confidence limit, " + N + "–" + N + ", exceeds the gap between the competition's first two places",
        lambda: [min(D.tpar(d, "ci_lo") for d in NONMAIZE), max(D.tpar(d, "ci_lo") for d in NONMAIZE)], TE,
        note_fn=lambda: "" if min(D.tpar(d, "ci_lo") for d in NONMAIZE) > D.gap12 else "下限不高于第1–2名差距!")
    reg("§3.6", "CUBIC 反转率", r"gives the same partition and a reversal of (\d+) %",
        lambda: [100 * D.china.reversal], f"{X}/china_summary.csv")

    # ================= 4
    reg("§4.1", "划分 22 → 1/1/20", r"partitions the (twenty-two) metrics a major competition reported into (one) that qualifies, (one) that calibration cannot move but order-preserving distortion can, and (twenty) that qualify under neither",
        lambda: [len(D.pmz), nT("I"), nT("II"), nT("III")], PM)
    reg("§4.2", "指标选择/环境重抽 倍数", r"moves a rank " + N + " to " + N + " times as far as resampling environments does",
        lambda: [(D.rc.loc[NONMAIZE].conformal_median_width / D.rob.loc[NONMAIZE].boot_width).min(),
                 (D.rc.loc[NONMAIZE].conformal_median_width / D.rob.loc[NONMAIZE].boot_width).max()], RC + "; " + ROB)
    reg("§4.3", "规划: 分辨 0.05 / 0.03 所需单元数 (k/Δ)², 取两位有效数字; 竞赛测试集单元数",
        r"about " + N + " cells for 0\.05 and " + N + r", the size of the competition's test set, for 0\.03",
        lambda: [round((cells_law(0.10)[2] / 0.05) ** 2, -2), round((cells_law(0.10)[2] / 0.03) ** 2, -3)], TD)
    reg("§4.3", "竞赛测试集约 10,000 单元 (445 × 23)", r"(10,000), the size of the competition's test set",
        lambda: [round(445 * 23, -3)], "§3.5 竞赛设计参数 445 基因型/环境 × 23 环境")
    reg("§4.3", "天花板与冠军值", r"the ceiling is " + N + "–" + N + " against the winning team's " + N,
        lambda: [D.hq("maize_G2F", "ceiling_n1"), D.hq("maize_G2F", "ceiling_n2"), D.r_top[0]], HB + "; " + S1f)
    reg("§4.4", "每份提交都被高估, 有限群体校正解释为零", r"overstated the realised selection differential of (every) verified submission, and (neither finite-population correction nor the tied predictions) explain it",
        lambda: [bool((D.gap.gap < 0).all()), max(D.fin_explained) < 0.5], GAP + "; " + fin_src, kind="bool")
    reg("§4.6", "会排除的指标数", r"excludes (twenty) of the (twenty-two) metrics here",
        lambda: [nT("III"), len(D.pmz)], PM)
    reg("§4.6", "合并相关系数的漂移", r"it moves by " + N + " under transformations that leave every such decision unchanged",
        lambda: [D.mono["spearman_r"]], MON)

    # ================= 5
    reg("§5", "Tier III 数", r"and the other (twenty), including the metric that decided the 2022 ranking, fail",
        lambda: [nT("III")], PM)
    reg("§5", "阈值: 高斯值/迁移范围/第1–2名差距", r"must reach about " + N + " in within-environment correlation at the competition's design before the more accurate method reliably selects better, and " + N + "–" + N + " once calibrated to real panels, against " + N + " between the first two places",
        lambda: [D.tmz.theory_lam05, D.tmz.transfer_lo, D.tmz.transfer_hi, D.gap12], TT + "; " + S1f)

    # ================= figure legends
    reg("图1注", "RMSE 仿射/单调漂移 (玉米)", r"RMSE moves by " + N + " under rescaling and " + N + " under the monotone transform",
        lambda: [float(D.dmi[(D.dmi.metric == "RMSE") & D.dmi.dataset.str.startswith("maize")][c].iloc[0]) for c in ("affine", "monotone")], f"{G}/decision_metrics_invariance.csv")
    reg("图2注", "变动≥10名的队伍数", r"the (eight) teams moving (ten) or more places are drawn in colour",
        lambda: [int(((D.S1.mean_RMSE.rank(method="min") - D.S1.mean_pearson_r.rank(ascending=False, method="min")).abs() >= 10).sum()), 10], S1f + " (Fig. 2a 的 rank 规则)")
    reg("图2注", "队伍对数/前十名对数", r"for all (\d+) team pairs and for the (\d+) pairs among the top ten",
        lambda: [len(D.S1) * (len(D.S1) - 1) / 2, 45], S1f)
    reg("图3注", "玉米高斯值与迁移范围", r"the Gaussian value at the competition's design \(diamond, " + N + r"\) and the range after scaling by the panels' observed-to-Gaussian ratios \(" + N + "–" + N,
        lambda: [D.tmz.theory_lam05, D.tmz.transfer_lo, D.tmz.transfer_hi], TT)
    reg("图4注", "观察跨度/区间宽度/冠军名次区间", r"The median observed range is (\d+) of (\d+) places \(interval, (\d+)\), and the first-placed team takes ranks (\d+) to (\d+)",
        lambda: [D.rc.loc["maize", "census_median_range"], D.rc.loc["maize", "methods"], D.rc.loc["maize", "conformal_median_width"],
                 *json.loads(D.rc.loc["maize", "leader_observed"])], RC)
    reg("图5注", "方法对数 (菜豆/春小麦/大豆)", r"common bean \((\d+) pairs\), spring wheat \((\d+)\) and soybean \((\d+)\)",
        lambda: [int((D.tpairs.dataset == d).sum()) for d in NONMAIZE], f"{X}/threshold_pairs.csv",
        note_fn=lambda: "threshold_estimates n_pairs: " + ", ".join(str(int(D.tpar(d, "n_pairs"))) for d in NONMAIZE))
    reg("图5注", "高斯曲线达到 0.95 的位置", r"which reaches 0\.95 at " + N,
        lambda: [D.tmz.theory_lam05], TT)
    reg("图5注", "高斯值与迁移上限", r"within the Gaussian value \(" + N + r"\) of the winner; orange, within the transferred upper value \(" + N,
        lambda: [D.tmz.theory_lam05, D.tmz.transfer_hi], TT)
    reg("图5注", "天花板与地块数", r"— " + N + " if the scored value is a single plot and " + N + r" if a two-plot mean, estimated from ([\d,]+) plots",
        lambda: [D.hq("maize_G2F", "ceiling_n1"), D.hq("maize_G2F", "ceiling_n2"),
                 pd.read_csv(f"{X}/heritability_ceiling_A8.csv").query("dataset == 'maize_G2F'").plots.sum()],
        HB + f"; {X}/heritability_ceiling_A8.csv")

    # ================= Table 1
    t1 = {"G2F 2022": "maize", "VEF": "common bean", "URSN": "spring wheat", "NUST": "soybean"}
    for lab, ds in t1.items():
        reg("表1", f"{ds}: 方法数/环境数/每环境基因型数", r"\| " + re.escape(lab) + r" \|[^|]*\| (\d+) [^|]*\| (\d+) \| (\d+) \|",
            (lambda ds=ds: [s(ds, "methods"), s(ds, "envs"), s(ds, "genos_per_env")]), SUM + " methods/envs/genos_per_env")
    for lab, ds in list(t1.items())[1:]:
        reg("表1", f"{ds}: 基础方法数", r"\| " + re.escape(lab) + r" \|[^|]*\| \d+ methods \((\d+) base\) \|",
            (lambda ds=ds: [len(panel_methods()[ds])]), PANEL_OF[ds][0])
        reg("表1", f"{ds}: 标记数", r"\| " + re.escape(lab) + r" \|[^|]*\|[^|]*\|[^|]*\|[^|]*\| ([\d,]+) \|",
            (lambda ds=ds: [D.markers[ds]]), f"{RAW}/… 原始基因型", heavy=True)
    reg("表1", "大豆两种口径的每环境基因型数", r"the median over the method–environment cells that carry at least (\d+) genotypes, which in soybean is (\d+)",
        lambda: [PANEL_OF["soybean"][1], D.ttheory.loc["soybean", "n_per_env"]], f"{CC}/build_summary.py SETS; " + f"{X}/threshold_theory.csv n_per_env")
    reg("表1", "G2F 已验证提交文件数", r"30 competition teams \((\d) verified submission files\)", lambda: [len(V5)], "V5")

    # ================= Table 2 (maize, common bean, spring wheat, soybean)
    reg("表2", "Tier I/II/III (四列)", r"\| Official metrics in Tier I / II / III \|" + r" (\d+) / (\d+) / (\d+) \|" * 4,
        lambda: [int(D.part.loc[d, c]) for d in ALLD for c in ("tier_I", "tier_II", "tier_III")], PB)
    reg("表2", "反转率 (四列)", r"\| Reversal between the two official metrics \|" + (r" " + N + r" % \|") * 4,
        lambda: [100 * s(d, "reversal") for d in ALLD], SUM)
    reg("表2", "反转率 CI (四列)", r"\| 95 % CI \(teams or method families resampled\) \|" + (r" " + N + "–" + N + r" \|") * 4,
        lambda: [v for d in ALLD for v in (100 * s(d, "ci_lo"), 100 * s(d, "ci_hi"))], SUM)
    reg("表2", "未增广反转率与 CI (三列)", r"\| Reversal without the added variants \[95 % CI\] \| — \|" + (r" " + N + r" % \[" + N + ", " + N + r"\] \|") * 3,
        lambda: [v for d in NONMAIZE for v in (100 * D.unaug(d), 100 * D.unaug(d, "ci_lo"), 100 * D.unaug(d, "ci_hi"))], RR1)
    reg("表2", "名次跨度 (四列)", r"\| Median rank range across the official metrics \|" + r" (\d+) % \|" * 4,
        lambda: [100 * D.rc.loc[d, "census_range_frac"] for d in ALLD], RC)
    reg("表2", "区间宽度 (四列)", r"\| Median 95 % rank-interval width \|" + r" (\d+) % \|" * 4,
        lambda: [100 * D.rc.loc[d, "conformal_width_frac"] for d in ALLD], RC)
    reg("表2", "结果信度 (三列)", r"\| Outcome reliability \| —ᵃ \|" + (r" " + N + r" \|") * 3,
        lambda: [s(d, "reliability") for d in NONMAIZE], SUM)
    reg("表2", "相关族回收 (四列)", r"\| Selection gain recovered, correlation family \| (\d+)–(\d+) %ᵃ \|" + (r" " + N + "–" + N + r" % \|") * 3,
        lambda: [v for d in ALLD for v in D.fam(d, "corr")], DIH)
    reg("表2", "误差族回收 (四列)", r"\| Selection gain recovered, error-magnitude family \| " + N + r" to " + N + r" %ᵃ \|" + (r" " + N + "–" + N + r" % \|") * 3,
        lambda: [v for d in ALLD for v in D.fam(d, "err")], DIH)
    reg("表2", "f=0.10 缺口 (玉米均值; 面板)", r"\| Shortfall below the Gaussian projection at f = 0\.10 \(SD\)† \| " + N + r"ᵇ \|" + (r" " + N + r" \|") * 3,
        lambda: [-np.mean(D.gaps(0.10))] + [-D.bg(d, .1, "gap_fin") for d in NONMAIZE], GAP + "; " + BGP + " gap_fin")
    reg("表2", "阈值 (玉米高斯值与迁移; 面板点估计与下限)", r"\| Threshold Δr at f = 0\.10 \[lower limit\] \| " + N + r" \(" + N + "–" + N + r"\)ᶜ \|" + (r" " + N + r" \[" + N + r"\] \|") * 3,
        lambda: [D.tmz.theory_lam05, D.tmz.transfer_lo, D.tmz.transfer_hi] + [v for d in NONMAIZE for v in (D.tpar(d, "threshold"), D.tpar(d, "ci_lo"))], TT + "; " + TE)


# --------------------------------------------------------------------------- engine
def normalize(t):
    for a, b in (("−", "-"), (" ", " "), (" ", " "), (" ", " "), ("**", ""), ("*", "")):
        t = t.replace(a, b)
    return t


def table_cells(groups, split=None, strip=None):
    """Table-2 cells -> printed strings (None for '—', 'n.a.', 'not reached')."""
    out = []
    for g in groups:
        g = g.strip()
        if strip: g = g.replace(strip, "").strip()
        if g in ("—", "n.a.", "not reached", "not estimable", "-"):
            out += [None] * (2 if split else 1); continue
        out += [x.strip() for x in g.split(split)] if split else [g]
    return out


def check(c, text):
    flags = c.get("flags", 0)
    m = re.search(c["pat"], text, flags)
    if not m:
        return dict(status="未找到", printed="", expected="", note="正则未匹配，正文可能已改写")
    printed = list(m.groups())
    if c.get("table"):
        printed = table_cells(printed, c.get("split"), c.get("strip"))
    ptxt = " / ".join("—" if p is None else p for p in printed)
    if c.get("status") == "NOSRC":
        note = c.get("note", "")
        if "note_fn" in c:
            try: note = c["note_fn"]()
            except Exception as e: note = f"(注释计算失败: {e})"
        return dict(status="无来源", printed=ptxt, expected="—", note=note)
    if c.get("status") == "NOTRUN" or (c.get("heavy") and not D.heavy):
        return dict(status="未复算", printed=ptxt, expected="—", note="需 --heavy" if c.get("heavy") else c.get("note", "耗时>10分钟"))
    try:
        exp = c["exp"]()
    except Exception as e:
        return dict(status="未复算", printed=ptxt, expected="—", note=f"计算失败: {type(e).__name__}: {e}")
    if not isinstance(exp, (list, tuple)): exp = [exp]
    kind = c.get("kind", "num")
    ok = True
    etxt = []
    if kind == "bool":
        ok = all(bool(e) for e in exp); etxt = ["成立" if bool(e) else "不成立" for e in exp]
    elif kind == "text":
        ok = False; etxt = [str(e) for e in exp]
    else:
        if len(exp) != len(printed):
            return dict(status="不一致", printed=ptxt, expected=" / ".join(fmt(e) for e in exp), note="个数不符")
        tols = c.get("tol")
        if not isinstance(tols, (list, tuple)): tols = [tols] * len(exp)
        for p, e, t in zip(printed, exp, tols):
            if p is None or e is None:
                etxt.append("—" if e is None else fmt(e))
                if (p is None) != (e is None or (isinstance(e, float) and not np.isfinite(e))): ok = False
                continue
            try:
                v, tol, _ = parse(p)
            except ValueError:
                ok = False; etxt.append(fmt(e) if e is not None else "—"); continue
            if t is not None: tol = t
            e = float(e); etxt.append(fmt(e))
            if kind == "le":
                ok &= e <= v * (1 + 1e-9)
            elif kind == "ge":
                ok &= e >= v * (1 - 1e-9)
            elif kind == "gt":
                ok &= e > v
            else:
                ok &= np.isfinite(e) and abs(v - e) <= tol + 1e-9 * max(1, abs(e))
    note = c.get("note", "")
    if "note_fn" in c:
        try:
            n2 = c["note_fn"]()
            note = (note + "；" + n2) if note and n2 else (n2 or note)
        except Exception as e:
            note = f"(注释计算失败: {e})"
    return dict(status="一致" if ok else "不一致", printed=ptxt, expected=" / ".join(etxt), note=note)


def main():
    global D
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manuscript", default="output/manuscript_TCJ.md")
    ap.add_argument("--heavy", action="store_true", help="also run the slow recomputations (≈5 min)")
    ap.add_argument("--only-problems", action="store_true")
    ap.add_argument("--csv", default=None, help="write the table to this CSV path")
    a = ap.parse_args()
    D = Data(heavy=a.heavy)
    build_registry()
    text = normalize(open(a.manuscript, encoding="utf-8").read())
    t0 = time.time(); rows = []
    for c in R:
        r = check(c, text)
        rows.append(dict(章节=c["sec"], 陈述=c["desc"], 印刷值=r["printed"], 期望值=r["expected"],
                         来源=c["src"], 结果=r["status"], 说明=r["note"]))
    T = pd.DataFrame(rows)
    show = T[T.结果 != "一致"] if a.only_problems else T
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 90); pd.set_option("display.max_rows", 500)
    print(f"manuscript: {a.manuscript}   registered statements: {len(T)}   heavy={a.heavy}   ({time.time()-t0:.0f} s)\n")
    for _, r in show.iterrows():
        print(f"[{r.结果}] {r.章节} | {r.陈述}\n    印刷 {r.印刷值}  |  期望 {r.期望值}\n    来源 {r.来源}" + (f"\n    说明 {r.说明}" if r.说明 else ""))
    print("\n" + T.结果.value_counts().to_string())
    if a.csv:
        T.to_csv(a.csv, index=False, encoding="utf-8-sig"); print(f"\n-> {a.csv}")
    sys.exit(1 if (T.结果 == "不一致").any() else 2 if (T.结果 == "未找到").any() else 0)


if __name__ == "__main__":
    main()
