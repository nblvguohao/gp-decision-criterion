#!/usr/bin/env bash
# Reproduce every result in the manuscript, in dependency order.
# Run from the repository root:   bash run_all.sh
#
# Primary data are read from $GP_DATA (default: data/raw). See DATA_SOURCES.md.
set -euo pipefail
PY="${PY:-python}"
export GP_DATA="${GP_DATA:-data/raw}"

run () {  # run <script> <description>
  printf '\n\033[1m== %s\033[0m  (%s)\n' "$2" "$1"
  "$PY" "$1"
}

echo "python : $($PY -c 'import sys;print(sys.version.split()[0])')"
echo "GP_DATA: $GP_DATA"

# ---------------------------------------------------------------- stage 1
# The maize competition. Uses only values published by the organisers plus the
# five verified submission files. Seconds to run, needs no primary data beyond
# analysis/g2f_leaderboard/results/.
run analysis/g2f_leaderboard/code/build_inputs.py   "rebuild derived leaderboard inputs"
run analysis/g2f_leaderboard/code/invariance.py     "affine invariance of the 22 official metrics -> Fig. 1B"
run analysis/g2f_leaderboard/code/reversal.py       "pairwise reversal, all 231 metric pairs -> Fig. 2B, Fig. S3"
run analysis/g2f_leaderboard/code/reversal_tests.py "team-clustered tests of the 231 pairs    -> Section 3.2, S4"
run analysis/g2f_leaderboard/code/rank_intervals.py "rank intervals across the 22 metrics  -> Fig. 4"
run analysis/g2f_leaderboard/code/interval_sensitivity.py "rank-interval width vs metric panel   -> Table S8"
run analysis/g2f_leaderboard/code/leaderboard.py    "official ranking vs decision endpoint"
run analysis/g2f_leaderboard/code/model_class.py    "model class vs rank movement, permutation -> Section 3.2"
run analysis/g2f_leaderboard/code/paired_invariance.py "paired rank invariance, five submissions -> Table S13"
run analysis/g2f_leaderboard/code/ties_and_coverage.py    "ties; leave-one-family-out coverage      -> Tables S15-S16"

# ---------------------------------------------------------------- stage 2
# Method panels for the non-competition datasets. This is the slow stage:
# it refits every method in every fold. Skip it if the panels in
# analysis/crosscrop/results/*_panel_wide.csv are already present.
if [ "${SKIP_PANELS:-0}" != "1" ]; then
  # panel_rice.py and panel_wheat.py are kept for the record but not run: those two panels
  # were removed from the paper (analysis/crosscrop/code/scope.py).
  run analysis/crosscrop/code/panel_bean.py   "bean panel   (VEF Andean)"
  run analysis/crosscrop/code/panel_ursn.py   "spring wheat panel (URSN)"
  run analysis/crosscrop/code/prep_nust.py    "soybean prep (NUST genotypes + G×E means)"
  run analysis/crosscrop/code/panel_nust.py   "soybean panel, forward-year"
  run analysis/crosscrop/code/widen_panel.py  "add miscalibration variants to the soybean panel"
  # Chinese maize (CUBIC, Supplementary Section S11): needs the downloads described in
  # analysis/crosscrop/results/china_DATA_SOURCE.md, and must run before the statistics below.
  if [ "${RUN_CUBIC:-0}" = "1" ]; then
    run analysis/crosscrop/code/panel_china.py       "Chinese maize, CUBIC panel"
  fi
fi
# The two soybean panel files ship gzipped in the public repository; when the
# panel stage is skipped, restore them in place.
for f in analysis/crosscrop/results/nust_panel_wide.csv analysis/crosscrop/results/nust_panel_predictions.csv; do
  [ -f "$f" ] || { [ -f "$f.gz" ] && gunzip -k "$f.gz"; }
done

# ---------------------------------------------------------------- stage 3
# Cross-dataset statistics. Every one of these reads the panels above.
run analysis/crosscrop/code/threshold_estimate.py    "surrogate thresholds, logistic estimator -> Table 2, S5, Fig. 5"
run analysis/crosscrop/code/build_summary.py          "the cross-dataset table          -> Table 2, Fig. 3"
run analysis/crosscrop/code/analyse_species.py       "the canonical statistics, demonstrated on common bean"
run analysis/crosscrop/code/heritability.py          "ceiling on replicated cells only (superseded by the next line; kept for the record)"
run analysis/crosscrop/code/heritability_ceiling_A8.py "ceiling as a function of replication  -> Table S12, Fig. 5C"
run analysis/crosscrop/code/threshold.py             "surrogate threshold, coarse grid (kept for the record)"
run analysis/crosscrop/code/threshold_ci.py          "threshold, window estimator (superseded; kept for the record)"
run analysis/crosscrop/code/copula_bridge.py     "copula benchmark for the breeder's equation -> Table S9"
run analysis/crosscrop/code/threshold_theory.py  "normal-theory threshold and design grid   -> Tables S10-S11, Fig. 5A"
run analysis/g2f_leaderboard/code/selection_rule_sensitivity.py "which submission stands for a team -> Table S19"
run analysis/crosscrop/code/does_it_help.py          "does scoring on an admissible metric help?"
run analysis/crosscrop/code/composition_sensitivity.py "outcome test under changes to panel composition -> Section 3.3, Table S22"
run analysis/crosscrop/code/unaugmented_panels.py  "clustered reversal intervals and unaugmented panels -> Tables S4, 2"
run analysis/crosscrop/code/monotone_test_panels.py        "monotone test, 22 metrics        -> Fig. 1B"
run analysis/crosscrop/code/partition.py             "admissibility partition, every dataset -> Table 2, Fig. 3A"
run analysis/crosscrop/code/breeder_gap_se.py        "shortfall below i*r with environment SEs -> Section 3.4, Table S9"
run analysis/crosscrop/code/tie_effect_on_gap.py     "tied predictions and the shortfall      -> Section 3.4, Table S20"
run analysis/crosscrop/code/rank_census.py           "observed rank range across the 22 metrics -> Table 2, Table S15, Fig. 4"
run analysis/crosscrop/code/decision_swap.py "outcome test under a changed decision -> Table S14"
run analysis/crosscrop/code/decision_genotype_means.py "outcome test for selection on genotype means across environments -> Table S21"
run analysis/crosscrop/code/gpverdict_applications.py "GPverdict: validation and the CUBIC verdict -> Section 3.7, Tables S23-S24"
if [ -f analysis/crosscrop/results/china_panel_wide.csv ]; then
  run analysis/crosscrop/code/summary_china.py       "Chinese maize, CUBIC                  -> Table S17"
fi
run analysis/crosscrop/code/robustness.py            "noise null and the environment bootstrap -> Tables S3, S6"
run analysis/crosscrop/code/between_env_share.py     "between-environment variance share   -> Section S3"
run analysis/crosscrop/code/replicate.py             "independent replication on soybean"
run analysis/crosscrop/code/constructive.py          "decision-oriented alternatives   -> Fig. 1C"
run analysis/crosscrop/code/counterexample_ursn.py   "a ranking and its reverse under r^2 -> Fig. 1D"
run analysis/crosscrop/code/build_tableS2.py         "Supplementary Table S2 as markdown"

# ---------------------------------------------------------------- stage 4
# Claims that were tested and withdrawn. Kept so the falsification is visible.
run analysis/crosscrop/code/scope_conditions.py      "[withdrawn] scope conditions"
run analysis/crosscrop/code/connectivity_test.py     "[withdrawn] genotype connectivity as a cause"
# Exploratory analyses of 9 September 2026, not carried into the paper (Supplementary Section S16).
run analysis/easygese/code/collapse_test.py          "[exploratory] reversal after averaging genotypes over environments"
if [ -f "${GP_DATA:-data/raw}/easygese/datasets/results_raw.csv" ]; then
  run analysis/easygese/code/reorder.py              "[exploratory] Pearson r against RMSE on EasyGeSe"
fi

# ---------------------------------------------------------------- stage 5
run analysis/tcj_figures/fig1.py    "Fig. 1"
run analysis/tcj_figures/figs2_5.py "Fig. 2-5"
run analysis/tcj_figures/fig6.py    "Fig. 6 (GPverdict)"
run analysis/tcj_figures/supp_figs.py "Fig. S1-S3"

printf '\n\033[1mdone.\033[0m Figures are in analysis/tcj_figures/ as 600 dpi PNG and vector PDF.\n'
