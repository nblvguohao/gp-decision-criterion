# Analysis plan: metric disagreement and the genetic architecture of the trait

The plan below was fixed before the genetic parameters were computed; the tests were run as specified.
Results: Supplementary Section S21 and Tables S28–S29 of the paper; files in `analysis/genetics/results/`.

## 1. Question

In a dataset × trait combination, is the reversal rate between the rankings of methods by RMSE and by the
mean within-environment Pearson *r* related to the genetic architecture of the trait? Three aspects are
considered: genomic heritability, the G×E share of genomic variance and the between-environment correlation.

Across the 23 CUBIC traits the reversal rate had already been found to correlate −0.004 (Spearman) with
predictability (Supplementary Table S19), so a null result was a possible outcome in advance.

## 2. Units of analysis

- The 52 combinations of the Atlas (`analysis/atlas/atlas.csv`): CUBIC 23, VEF 4, URSN 2, NUST 10, DROPS 6,
  SoyNAM 7.
- The G2F row is excluded: it carries five competition submissions, no 17-method panel and no markers.
- Each combination keeps the Atlas's own data preparation: the same loader, environment floor (`min_n`) and
  marker principal components W (`NPC = 80` in `atlas.py`; CUBIC uses the cache of `panel_china.py`).
- The Atlas panels and results are not rerun.

## 3. Outcomes (read from `atlas.csv`, not recomputed)

| Code | Outcome | Column |
|---|---|---|
| O1 (primary) | reversal rate between RMSE and Pearson *r* over method pairs | `reversal` |
| O2 (secondary) | number of methods tied with the leader | size of `tied` |
| O3 (secondary) | difference in selection gain recovered by the correlation family and by the RMSE pick | `recovery_corr − recovery_rmse`, combinations with `reliability ≥ 0.5` only |

## 4. Predictors

For each combination, one fit of `mixed_models.fit(y, Env, W, group=Env)` — the panels' marker × environment
GBLUP, with environments as fixed effects and W scaled by `scale_features` — on all the data. No
cross-validation: the fit only estimates variance components. With σ²_u, σ²_v and σ²_e the main-effect, G×E
and residual variances, the function returns d0 = σ²_u/σ²_e and d1 = σ²_v/σ²_e.

| Code | Predictor | Definition |
|---|---|---|
| P1 | genomic heritability within an environment | h²_g = (d0 + d1)/(d0 + d1 + 1) |
| P2 | G×E share of genomic variance | s_GE = d1/(d0 + d1) |
| P3 | between-environment phenotypic correlation (model-free) | median Pearson correlation of genotype values over environment pairs sharing ≥ 25 genotypes |

Notes:

- Under this compound-symmetry model the between-environment genetic correlation is
  r_g = σ²_u/(σ²_u + σ²_v) = 1 − P2, so it is not a separate predictor; r_g is reported with P2.
- P3 depends on both heritability and r_g and is descriptive.
- The variance components come from the kernel of the panels' leading marker principal components.

## 5. Non-genetic covariate (secondary model)

C1: the spread of calibration across the 17 base methods — the standard deviation across methods of their mean
within-environment slope of observed on predicted values. It checks whether an association with a genetic
parameter merely stands in for calibration differences between methods.

## 6. Statistical analysis

The combinations are nested in six datasets, whose differences (crop, design, number of environments) would
confound a pooled analysis, so the primary analysis is within datasets.

- **Primary tests** (P1, P2 and P3, each against O1): within each dataset, outcome and predictor are ranked and
  centred on the dataset mean; the pooled Pearson correlation of the centred ranks is ρ_w. P-values come from
  10,000 permutations of the predictor within datasets (seed 20260929), two-sided, with Holm correction over the
  three tests at α = 0.05.
- **Power**: with about 52 − 6 = 46 within-dataset degrees of freedom, the design has 80 % power at α = 0.05 for
  |ρ_w| ≥ 0.40; smaller associations cannot be detected.
- **Secondary analyses** (uncorrected, descriptive): O2 and O3 against P1–P3; the coefficient of P in the
  within-dataset rank regression O1 ~ P + C1; the pooled Spearman correlation ignoring datasets.
- **Reporting rules**: a significant result is reported as "within datasets of this Atlas, the reversal rate
  rises with X", together with the C1 model, and not as a cause; a null result is reported in text and table
  with the detectable effect size, without a scatter plot.

## 7. Fixed elements

- The Atlas methods, folds, selection fraction (0.10), environment floor, number of principal components and
  GPverdict settings are unchanged.
- New results are written to new files under `analysis/genetics/results/`; no existing result is overwritten.
- CUBIC phenotypes carry a CC BY-NC-ND licence, so per-combination variance components are released only as a
  summary table.

## 8. Outputs

- `analysis/genetics/genetic_params.py`: P1–P3 and C1 → `results/genetic_params.csv`
- `analysis/genetics/genetic_tests.py`: the tests of Section 6 → `results/genetic_tests.csv`
