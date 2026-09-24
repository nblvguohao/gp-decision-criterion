# Code and derived data for: *A decision-based criterion and an open tool for choosing genomic prediction methods across environments*

Guohao Lv, Lichuan Gu

School of Artificial Intelligence, Anhui Agricultural University, Hefei 230036, China
Anhui Province Key Laboratory of Intelligent Agricultural Technology and Equipment,
Anhui Agricultural University, Hefei 230036, China

Corresponding author: Lichuan Gu (glc@ahau.edu.cn)

Truncation selection within an environment depends only on the order of the
predictions, so a metric that represents it must be unchanged by any
order-preserving transformation of them and must change when that order is
reversed. This archive applies that criterion to the 22 official metrics of the
Genomes to Fields (G2F) 2022 maize prediction competition and to method panels for
common bean, spring wheat and soybean, with Chinese maize (CUBIC) as a fifth
dataset in the Supplementary Material, and reproduces every number, table and
figure in the manuscript from public data.

---

## What the paper reports, and which script establishes it

| Result | Script | Output |
|---|---|---|
| Of 22 official metrics, 1 is admissible (Tier I), 1 is calibration-invariant only (Tier II) and 20 are neither, in every dataset | `analysis/crosscrop/code/partition.py`, `analysis/g2f_leaderboard/code/invariance.py`, `analysis/crosscrop/code/monotone_test_panels.py` | `partition_metrics.csv`, `invariance_test.csv`, `monotone_test.csv` → Fig. 1, Table 2 |
| Switching official metric reverses 26.9 % of team pairs (team-bootstrap CI 18.0–37.1 %) | `analysis/crosscrop/code/build_summary.py` | `cross_dataset_summary.csv` → Fig. 2, Table 2 |
| 129 of the 231 metric pairs reverse more than 5 % of team pairs (BH, FDR 5 %) | `analysis/g2f_leaderboard/code/reversal_tests.py` | `reversal_tests.csv`, `reversal_by_class.csv` |
| A published rank spans a median of 12 of 30 places across the official metrics | `analysis/crosscrop/code/rank_census.py`, `analysis/g2f_leaderboard/code/rank_intervals.py` | `rank_census.csv`, `rank_intervals.csv` → Fig. 4 |
| The method panels include GBLUP and marker × environment GBLUP, fitted by REML on the same marker principal components as the ridge baselines | `analysis/crosscrop/code/mixed_models.py`, `panel_bean.py`, `panel_ursn.py`, `panel_nust.py`, `panel_china.py` | `*_panel_wide.csv` |
| Without the two GBLUP models the correlation family still leads in soybean and spring wheat; leaving out one base method at a time changes the soybean result in no panel | `analysis/crosscrop/code/composition_sensitivity.py` | `composition_sensitivity.csv` → Table S22 |
| GPverdict reproduces the spring-wheat results exactly and, for Chinese maize (CUBIC), finds three methods the trial cannot separate | `analysis/crosscrop/code/gpverdict_applications.py` (tool in `tool/`) | `gpverdict_validation.csv`, `gpverdict_cubic.json` → Section 3.7, Tables S23–S24 |
| Which metric family selects better material follows the decision | `analysis/crosscrop/code/does_it_help.py`, `decision_swap.py` | `does_it_help.json`, `decision_swap.csv` → Tables S7, S14 |
| For selection on genotype means across environments, the mean within-environment rank correlation shows no detectable difference from that decision's own admissible metric where the outcome is reliable (common bean, soybean) | `analysis/crosscrop/code/decision_genotype_means.py` | `decision_genotype_means.csv` → Table S21 |
| Realised selection differentials fall 0.13–0.31 SD below i·r in the verified submissions | `analysis/crosscrop/code/breeder_gap_se.py`, `copula_bridge.py` | `breeder_gap_se.csv`, `breeder_gap_panels.csv` (panels also against the finite-population expectation, `gap_fin`), `copula_bridge.csv` |
| The accuracy gap a trial resolves is 0.066–0.213 in the panels, 0.028 under the Gaussian model at the competition's design and 0.04–0.12 after calibration to the panels; across the Gaussian design grid it falls as about 3/√N in genotype–environment cells N | `analysis/crosscrop/code/threshold_model.py`, `threshold_estimate.py`, `threshold_theory.py` | `threshold_estimates.csv`, `threshold_theory.csv`, `threshold_design.csv` → Fig. 5 |
| The ceiling set by measurement error is 0.61 (single plot) to 0.74 (two-plot mean); the winner reaches 48–58 % of it | `analysis/crosscrop/code/heritability_ceiling_A8.py` | `heritability_ceiling_A8*.csv` |

Every number in the paper was generated on macOS 26.6.2 (Apple silicon) with
Python 3.11.16, numpy 2.4.6, scipy 1.17.1, pandas 2.3.3, matplotlib 3.11.1,
scikit-learn 1.9.1, statsmodels 0.14.6 and openpyxl 3.1.5 (see
`requirements.txt` and `environment.yml`).

## Running it

**All scripts are run from the repository root**, not from their own directory:

```bash
bash run_all.sh          # everything, in dependency order
```

Primary data are read from `$GP_DATA`, default `data/raw`. This repository
carries the pipeline, the derived results and every redistributable primary file
the scripts read; [DATA_SOURCES.md](DATA_SOURCES.md) records where each comes from
and its licence. Not included are the Chinese maize (CUBIC) phenotypes and the
panel derived from them, whose NoDerivatives licence makes redistributing an adapted
table questionable, and the CAIGE wheat data of a withdrawn panel, which carry no
redistribution licence.
The two soybean method-panel files ship gzipped
(`nust_panel_wide.csv.gz`, `nust_panel_predictions.csv.gz`); stage 2 rebuilds
them (`prep_nust.py`, `panel_nust.py`, `widen_panel.py`), and
`SKIP_PANELS=1` unpacks the shipped copies instead.

Stage 1 (the maize leaderboard) needs no primary data beyond what is in
`analysis/g2f_leaderboard/results/` and runs in seconds. Stage 2 refits the
method panels and is the slow part; once the panels exist,
`SKIP_PANELS=1 bash run_all.sh` reuses them. The Chinese maize panel needs the
downloads described in `analysis/crosscrop/results/china_DATA_SOURCE.md` and is
built with `RUN_CUBIC=1`.

In the leave-genotypes-out panels (common bean, spring wheat, Chinese maize), the
environment index of a fold's test genotypes is the mean of that fold's training
genotypes in the environment, and the two mixed models use that fold's own
environment effects, so no test phenotype enters a prediction. An earlier version
averaged the five training-fold means, which put every test genotype's own phenotype
into its index in four folds of five; it was corrected on 2026-09-24 and every
downstream result was recomputed.

## Layout

```
analysis/
  g2f_leaderboard/     the maize competition: leaderboard, invariance,
    code/              reversal, rank intervals, ties
    results/           Supplemental Table S1, 5 verified submissions, derived CSVs
  crosscrop/           the method panels and all cross-dataset statistics
    code/
    results/           method panels (Env, k, y, p, method) and per-dataset output
  tcj_figures/         Fig. 1-5 and Fig. S1-S3, as 600 dpi PNG and vector PDF
  easygese/            two exploratory analyses (Supplementary Section S16), not used for any result
data/raw/              primary data, by dataset
tools/                 manuscript support: reference database and number check
```

### GPverdict, the tool

The criterion, the resolution threshold and the outcome test are released as GPverdict, a
Python package with a command line and an in-browser web version: source and tests at
https://github.com/nblvguohao/gpverdict (v1.0.0), web version at
https://nblvguohao.github.io/gpverdict/. A copy of the package used for the paper is in
`tool/`.

### The generic runner

`analysis/crosscrop/code/analyse_species.py` takes any table with columns
`Env, k, y, p, method` and reports the metric invariance checks, the pairwise
reversal rate with its interval, the rank intervals and their leave-one-metric-out
coverage, and the agreement of accuracy with realised selection at near-equal
accuracy. The outcome test (`does_it_help.py`, `decision_swap.py`,
`decision_genotype_means.py`) and the logistic threshold (`threshold_estimate.py`)
read the same table format; adding a dataset means producing that table and adding
it to their dataset lists.

### No number is typed in by hand

Every value in the tables and figures is computed by a script and written to a
file that the figure or table then reads. The reversal rate, the rank-interval
width and the outcome reliability each have one canonical implementation, in
`analyse_species.py` and `does_it_help.py`, which the scripts that report them
import; `replicate.py` re-implements the 22 metrics independently for soybean as a
cross-check. `tools/verify_numbers.py` checks each quantitative
statement of the manuscript against the file that produces it (it needs the
manuscript text, passed with `--manuscript PATH`).

### Rebuilding the derived leaderboard inputs

`analysis/g2f_leaderboard/code/build_inputs.py` regenerates `S1_parsed.csv`,
`S1_best_per_team.csv`, `coverage_by_metric_class.csv` and `tierI_ranking.csv`
from the published Supplemental Table S1. A team's submission is the one
minimising `mean_RMSE`, the 2022 official metric, among submissions for which the
organisers computed all 22 metrics; `selection_rule_sensitivity.py` repeats the
leaderboard analyses under four alternative rules.

## Licence

Code: MIT ([LICENSE-CODE](LICENSE-CODE)). Derived data in `analysis/*/results/`:
CC BY 4.0 ([LICENSE-DATA](LICENSE-DATA)), except the third-party files that
LICENSE-DATA lists (the competition answer key, team prediction files and published
tables). Primary data keep the licences of their original deposits, recorded file by
file in [DATA_SOURCES.md](DATA_SOURCES.md).

## Citing

See [CITATION.cff](CITATION.cff). Please also cite the primary data sources you
use — they are the reason this analysis was possible.
