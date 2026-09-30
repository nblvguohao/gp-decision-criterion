"""CUBIC: every trait of the source phenotype table (Jin et al. 2023, Supplemental Table S12) through the existing method panel
(the analysis of Supplementary Table S18, repeated for each trait).

Rules fixed before any result (see china_trait_sweep_RUN_STARTED.txt):
  R1  All traits of TableS12_phenotypes.csv with a column at each of the five sites (23), no more, no fewer.
  R2  Environment floor 25 genotypes, as everywhere in the paper. A site below 25 for a trait is dropped and
      recorded; a trait left with fewer than 2 sites is dropped and recorded. The floor is never lowered.
  R3  Quality gate: the strongest base method's mean within-environment Pearson r >= 0.15. Recorded for every
      trait, pass or fail; nothing is removed on it.
  R4  The output lists all 23 traits, dropped ones included. No claim that a result holds "for every trait" is
      made here; one would need the Benjamini-Hochberg treatment of Supplementary Section S4.

Nothing is re-implemented. The panel is panel_china.py itself, run once per trait with CUBIC_TRAIT set; the
statistics are the functions summary_china.py uses: analyse_species.metric_table / reversal_ci /
rank_interval_width, does_it_help.outcome_reliability, threshold_model.build and MIN_ENV.

panel_china.py asserts that every site keeps >= 25 genotypes and writes a fixed file name. The driver therefore
(i) counts, before the run, the genotypes per site that have a phenotype and a genotype record (the same join
panel_china.py makes); a site below 25 is dropped by giving panel_china.py a copy of the phenotype table with that
site's column emptied (all other inputs are links to the originals) -- the algorithm is unchanged; (ii) records
any non-zero return code and its message instead of stopping; (iii) moves china_panel_wide.csv to
china_panel_wide_<trait>.csv after each run. Those per-trait panels are derived from CC BY-NC-ND data: they stay
local (.gitignore) and are never committed, packed or sent. Only china_trait_sweep.csv (summary numbers) is.

    GP_DATA=<data root> python analysis/crosscrop/code/china_trait_sweep.py --selftest   # EW only, vs Supplementary Table S18
    GP_DATA=<data root> python analysis/crosscrop/code/china_trait_sweep.py              # all traits
"""
import os, sys, shutil, subprocess, tempfile, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "analysis/crosscrop/code")
from scipy.stats import spearmanr
from analyse_species import metric_table, reversal_ci, rank_interval_width
from does_it_help import outcome_reliability
import threshold_model as tm

RAW = os.environ.get("GP_DATA", "data/raw")
SP = f"{RAW}/cubic"
RES = "analysis/crosscrop/results"
SITES = ["JL", "LN", "BJ", "HeB", "HN"]
MN, GATE = 25, 0.15
EXPECT = dict(reversal=0.472, ci_lo=0.369, ci_hi=0.539, reversal_base=0.140, ci_lo_base=0.008, ci_hi_base=0.326,
              width=43, methods=52, reliability=0.893)


def traits():
    cols = pd.read_csv(f"{SP}/TableS12_phenotypes.csv", nrows=0).columns
    return sorted({c.rsplit("_", 1)[0] for c in cols if "_" in c and c.rsplit("_", 1)[1] in SITES
                   and all(f"{c.rsplit('_', 1)[0]}_{s}" in cols for s in SITES)})


def site_counts(trait):
    ph = pd.read_csv(f"{SP}/TableS12_phenotypes.csv"); ph = ph.rename(columns={ph.columns[0]: "k"})
    ids = set(map(str, np.load(f"{SP}/cubic_geno_thinned.npz", allow_pickle=True)["ids"]))
    ph["k"] = ph.k.astype(str); ph = ph[ph.k.isin(ids)]
    return {s: int(ph[f"{trait}_{s}"].notna().sum()) for s in SITES}


def build_panel(trait, drop_sites):
    env = dict(os.environ, CUBIC_TRAIT=trait, GP_DATA=RAW)
    tmp = None
    if drop_sites:                      # R2: the same inputs, with the dropped sites' columns emptied
        tmp = tempfile.mkdtemp(prefix="cubic_sweep_"); os.makedirs(f"{tmp}/cubic")
        ph = pd.read_csv(f"{SP}/TableS12_phenotypes.csv")
        for s in drop_sites: ph[f"{trait}_{s}"] = np.nan
        ph.to_csv(f"{tmp}/cubic/TableS12_phenotypes.csv", index=False)
        for f in os.listdir(SP):
            if f != "TableS12_phenotypes.csv": os.symlink(os.path.abspath(f"{SP}/{f}"), f"{tmp}/cubic/{f}")
        env["GP_DATA"] = tmp
    t0 = time.time()
    r = subprocess.run([sys.executable, "analysis/crosscrop/code/panel_china.py"], env=env, capture_output=True, text=True)
    if tmp: shutil.rmtree(tmp)
    return r, time.time() - t0


def summarise(P):
    """summary_china.py's statistics, verbatim."""
    T = metric_table(P, MN)
    pt, lo, hi, M = reversal_ci(T, np.random.default_rng(0))
    base = P[~P.method.str.contains("__")]
    Tb = metric_table(base, MN)
    bpt, blo, bhi, bM = reversal_ci(Tb, np.random.default_rng(0))
    w, _ = rank_interval_width(T)
    rel = outcome_reliability(P, MN, np.random.default_rng(0))
    one = P[P.method == P.method.iloc[0]]
    meths, envs, R, G, _ = tm.build(base, 0.10, MN)
    best = float(Tb["mean_pearson_r"].max())
    return dict(methods=M, methods_base=bM, envs=one.Env.nunique(), genos_per_env=int(one.groupby("Env").k.nunique().median()),
                best_mean_pearson_r=best, best_base_method=Tb["mean_pearson_r"].idxmax(), passes_gate=bool(best >= GATE),
                reversal=pt, ci_lo=lo, ci_hi=hi, reversal_base=bpt, ci_lo_base=blo, ci_hi_base=bhi,
                width=w, width_frac=w / M, reliability=rel, threshold_estimable=bool(len(envs) >= tm.MIN_ENV))


def run_trait(trait):
    counts = site_counts(trait)
    drop = [s for s, n in counts.items() if n < MN]
    row = dict(trait=trait, **{f"n_{s}": n for s, n in counts.items()}, sites_dropped=";".join(drop))
    if len(SITES) - len(drop) < 2:
        return dict(row, status="dropped", reason=f"fewer than 2 sites with >= {MN} genotypes ({counts})")
    r, secs = build_panel(trait, drop)
    row["seconds"] = round(secs)
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["no message"])[-1]
        return dict(row, status="dropped", reason=f"panel_china.py exit {r.returncode}: {msg}")
    dst = f"{RES}/china_panel_wide_{trait}.csv"
    shutil.move(f"{RES}/china_panel_wide.csv", dst)
    row.update(summarise(pd.read_csv(dst)), status="analysed",
               reason=(f"sites dropped (< {MN} genotypes): {';'.join(drop)}" if drop else ""))
    return row


if __name__ == "__main__":
    print(f"GP_DATA={RAW}")
    if "--selftest" in sys.argv:
        row = run_trait("EW")
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
        got = dict(reversal=round(row["reversal"], 3), ci_lo=round(row["ci_lo"], 3), ci_hi=round(row["ci_hi"], 3),
                   reversal_base=round(row["reversal_base"], 3), ci_lo_base=round(row["ci_lo_base"], 3),
                   ci_hi_base=round(row["ci_hi_base"], 3), width=round(row["width"]), methods=row["methods"],
                   reliability=round(row["reliability"], 3))
        bad = {k: (got[k], v) for k, v in EXPECT.items() if got[k] != v}
        print(f"best_mean_pearson_r = {row['best_mean_pearson_r']:.3f} ({row['best_base_method']}); expected about 0.341")
        print("SELFTEST", "PASS" if not bad else f"FAIL {bad}", f"| wall clock {row.get('seconds')} s panel")
        sys.exit(0 if not bad else 1)
    TR = traits()
    assert len(TR) == 23, f"R1: expected 23 traits in Table S12, found {len(TR)}: {TR}"
    rows = []
    for t in TR:
        t0 = time.time(); r = run_trait(t); r["total_seconds"] = round(time.time() - t0); rows.append(r)
        print(f"{t}: {r['status']} {r.get('reason','')} "
              + (f"rev {r['reversal']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}] best r {r['best_mean_pearson_r']:.3f}" if r["status"] == "analysed" else ""), flush=True)
        pd.DataFrame(rows).to_csv(f"{RES}/china_trait_sweep.csv", index=False)
    S = pd.DataFrame(rows); A = S[S.status == "analysed"]; G = A[A.passes_gate]
    print("\n==================== disclosure ====================")
    print(f"attempted {len(S)} | dropped {int((S.status == 'dropped').sum())} | analysed {len(A)} | passing gate (best r >= {GATE}) {len(G)}")
    print(f"reversal interval excluding zero: {int((A.ci_lo > 0).sum())} of {len(A)} (all methods); "
          f"{int((A.ci_lo_base > 0).sum())} of {len(A)} (base methods)")
    for lab, D in (("all analysed", A), ("passing gate", G)):
        if len(D):
            print(f"{lab:>14}: reversal median {D.reversal.median():.3f} (range {D.reversal.min():.3f}-{D.reversal.max():.3f}); "
                  f"base median {D.reversal_base.median():.3f} (range {D.reversal_base.min():.3f}-{D.reversal_base.max():.3f})")
    if len(A) > 2:
        rho, p = spearmanr(A.reversal, A.best_mean_pearson_r)
        rb, pb = spearmanr(A.reversal_base, A.best_mean_pearson_r)
        print(f"Spearman(reversal, best_mean_pearson_r) = {rho:.3f} (P = {p:.3g}, n = {len(A)}); base methods {rb:.3f} (P = {pb:.3g})")
    print(f"wrote {RES}/china_trait_sweep.csv")
