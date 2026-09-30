"""Which constructed method panels the paper uses.

Rice (IRRI) and wheat (CAIGE) were removed on 2026-09-23. Every method in those two
panels ranks genotypes within an environment at or near chance (best mean
within-environment Pearson r 0.08 and 0.11; 24 of 51 rice methods negative), so they
cannot test claims about ranking methods: their reversal rates, rank ranges and
thresholds were driven by noise in one of the two metrics being compared. Their panel scripts and data
are kept for the record; `kept` drops them from every cross-dataset analysis.
"""
REMOVED_PANELS = ("rice_panel_wide", "wheat_panel_wide")


def keep(entry):
    """True unless the entry (a dataset tuple, dict item or path) refers to a removed panel."""
    return not any(tag in str(entry) for tag in REMOVED_PANELS)


def kept(sets):
    return [s for s in sets if keep(s)]


# ---------------------------------------------------------------- ties
# The miscalibration variants of Section 2.3 (shrunk, inflated, biased) are global
# affine maps of their parent method, so within every environment they rank genotypes
# exactly as the parent does: their accuracy differs from the parent's only by
# floating-point residue and they select the same material. Such pairs say nothing
# about which of two methods is better, and counting them as disagreements (or reading
# a winner off the sign of a 1e-16 residue) biases every pair-based statistic.
# Residues are <= 4e-16 and the smallest genuine accuracy difference is >= 1.3e-4
# (verify_B.md, M2a), so any tolerance between 1e-15 and 1e-5 gives the same pairs.
ACC_TIE = 1e-9    # |difference in a metric| below this is a tie
OUT_TIE = 1e-12   # |difference in realised selection differential| below this is a tie


def drop_ties(D, acc="dr", out="dg"):
    """Remove method pairs tied on accuracy or on outcome."""
    return D[(D[acc].abs() >= ACC_TIE) & (D[out].abs() >= OUT_TIE)].copy()


def untied(d1, d2):
    """Mask of pairs that are not tied on either of two score differences."""
    import numpy as np
    d1, d2 = np.asarray(d1, float), np.asarray(d2, float)
    return np.isfinite(d1) & np.isfinite(d2) & (np.abs(d1) >= ACC_TIE) & (np.abs(d2) >= ACC_TIE)


def tied_best_mean(scores, outcomes):
    """Outcome of the method a rule picks, averaged over methods tied for the best score."""
    import numpy as np
    s, o = np.asarray(scores, float), np.asarray(outcomes, float)
    ok = np.isfinite(s)
    if not ok.any():
        return np.nan
    top = np.nanmax(s[ok])
    tied = ok & (np.abs(s - top) <= ACC_TIE * max(1.0, abs(top)))
    return float(np.nanmean(o[tied]))
