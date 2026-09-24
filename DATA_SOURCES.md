# Data sources, licences and redistribution

Every primary dataset used in this study is public. This file records, for each
one, where it came from, which licence it carries, which scripts consume it, and
whether the files are included in this archive or must be downloaded.

`GP_DATA` is the root under which the primary data live; it defaults to
`data/raw` relative to the repository root and can be overridden:

```bash
export GP_DATA=/path/to/your/data
```

---

## 1. Maize — Genomes to Fields 2022 G×E prediction competition

| | |
|---|---|
| Primary deposit | doi:10.25739/tq5e-ak26 (CyVerse Data Commons) |
| Licence | **ODC PDDL** (Public Domain Dedication and Licence), per the DataCite record |
| Results paper | Washburn et al. (2025) *Genetics* 229:iyae195, doi:10.1093/genetics/iyae195 |
| In this archive | `data/raw/g2f/`, `analysis/g2f_leaderboard/results/` |

Three distinct things are used, and they are not interchangeable:

**(a) The official leaderboard.** Supplemental Table S1 of the *Genetics* paper —
124 successful submissions × 30 teams × 22 metrics computed by the organisers.
`analysis/g2f_leaderboard/results/Supplemental_Table_S1_GENETICS-2024-307594.tsv`
is the published supplementary file converted from `.xlsx` to TSV, one sheet per
`##### SHEET:` marker. Every maize leaderboard number in the paper comes from
these published values, so those results contain no modelling choice of ours.
The *Genetics* article is open access and deposited in PubMed Central as a
**public-domain** US-government work (PMC12054733).

**(b) Five verified team submissions.** Per-genotype prediction files recovered
from the public team repositories named in the paper's data-availability
statement, used wherever per-genotype predictions are needed (the invariance
test, the selection-differential analysis). Seven files were recovered, and five
reproduce their official metric values to
within 5 × 10⁻⁷ (the precision of the six published decimals) and are the ones used:
`EnBiSys_sub243568.csv`, `KernelOfTruth_sub244906/244944/244953.csv` and
`NicheSquad_sub244985.csv`. The other two, `CLAC_repo_submission5.csv` and
`MPB_Group_repo_prediction.csv`, match none of their team's scored submissions and
were excluded as post-hoc versions; they are kept for transparency.

> **Licence note.** The depositing teams did not declare a licence on these
> individual prediction files. They are included here for verification, with
> attribution to the source repositories. If you would rather not redistribute
> them, delete them; every leaderboard result still reproduces. They are used by
> Fig. 1b–c, Sections 3.1, 3.3 and 3.4 of the paper and Supplementary Tables S2, S9,
> S13, S14, S16, S20 and S21.

**(c) Observed yields and training phenotypes.** `Final_Observed_Yield.csv`
(10,290 rows, 23 environments), the observed test-set yields against which the 2022
submissions were scored, was obtained from the public GitHub repository
github.com/alenxav/Lectures (folder `MGC_2023`), the same repository that holds
`CLAC_repo_submission5.csv`. It carries the same values as
`Test_Set_Observed_Values_ANSWER.csv` in the later CyVerse release of the competition
data, which is ODC PDDL. `1_Training_Trait_Data_2014_2021.csv` is from the CyVerse
deposit above (ODC PDDL). Both are redistributed here.

**(d) The 2024 edition leaderboard.** Retrieved from the public EvalAI API,
challenge 2376, phase split 5893, on 2026-09-08. Provenance and the raw JSON are
in `analysis/g2f_leaderboard/results/edition2024/`. EvalAI stores a single
metric (`Mean_r`) for this edition and no submissions are public, which is why
the cross-metric analysis cannot be repeated on it.

**(e) Meta, soil, weather and environmental-covariate tables.**
`2_Training_Meta_Data_2014_2021.csv`, `3_Training_Soil_Data_2015_2021.csv`,
`4_Training_Weather_Data_2014_2021.csv` and `6_Training_EC_Data_2014_2021.csv`,
from the same CyVerse deposit as (c) — same licence, redistributed here for
completeness. **No script in this archive reads them and no claim in the
manuscript depends on them**; they are not part of the reproducibility
pipeline in `run_all.sh`. Hashes: `analysis/g2f_leaderboard/results/g2f_meta_weather_ec_soil_sources.json`.

---

## 2. Rice — IRRI elite breeding lines

| | |
|---|---|
| Source | Supplementary files of Nguyen et al. (2023) *Rice* 16:7, doi:10.1186/s12284-023-00623-6 |
| Licence | **CC BY 4.0** (Crossref licence record) |
| Files | `12284_2023_623_MOESM3_ESM.txt` (markers), `..._MOESM4_ESM.csv`, `..._MOESM5_ESM.csv` (phenotypes) |
| In this archive | yes — `data/raw/rice/` |
| Consumed by | `analysis/crosscrop/code/panel_rice.py` (not run by `run_all.sh`) |

107 lines, 882 SNPs, 15 environments across Asia and Africa (2018–2020), grain yield.
**Withdrawn from the paper** on 2026-09-23: every method ranks lines within an
environment at or near chance (best mean within-environment Pearson r 0.081; 24 of 51
methods negative), so the panel cannot test claims about ranking methods
(`analysis/crosscrop/code/scope.py`; Supplementary Section S16). The files and script
are kept for the record.

---

## 3. Wheat — CAIGE (CIMMYT Australia ICARDA Germplasm Evaluation)

| | |
|---|---|
| Source | https://www.caigeproject.org.au/data-compilations/ — 2017 and 2018 shipments |
| Reference | Trethowan et al. (2024) *Front. Plant Sci.* 15:1435837 |
| Licence | **not declared** by the data provider |
| In this archive | **NO — download it yourself**; no file derived from it is included either |
| Consumed by | `analysis/crosscrop/code/panel_wheat.py` (not run by `run_all.sh`) |

**Withdrawn from the paper** for the same reason as rice (best mean within-environment
Pearson r 0.114; 10 of 44 methods negative; `scope.py`, Supplementary Section S16).

CAIGE publishes its compilations for research use but states no redistribution
licence, so the files are not included here. Download the 2017 and 2018 wheat
data compilations and
`CAIGE_Bread-Wheat_ICARDA-Line_2015-20_GenotypicData.xlsx` into
`$GP_DATA/caige/` and `panel_wheat.py` will run unchanged. 121 lines with
matched genotypes (4,982 markers), 13 site-years, grain yield.

---

## 4. Common bean — VEF Andean elite panel

| | |
|---|---|
| Deposit | doi:10.7910/DVN/XCD67U (Harvard Dataverse) |
| Licence | **CC BY 4.0** (dataset custom terms) |
| Reference | Keller et al. (2020) *Front. Plant Sci.* 11:1001 |
| In this archive | yes — `data/raw/vef/` |
| Consumed by | `analysis/crosscrop/code/panel_bean.py`, `heritability.py`, `heritability_ceiling_A8.py` (raw plots, `VEF_raw_phenotypic_data.csv`) |

467 lines, 5,820 SNPs, 10 trials in Colombia under drought, irrigated and
phosphorus regimes, seed yield. The Dataverse download is behind a guestbook
form, so an automated fetch is not possible; the files are redistributed here
under CC BY 4.0 with attribution to Keller et al.

---

## 5. Spring wheat — Uniform Regional Scab Nursery, 1995–2024

| | |
|---|---|
| Deposit | doi:10.5061/dryad.wstqjq2z0 (Dryad) |
| Licence | **CC0 1.0** (public domain dedication) |
| Reference | Brault et al. (2025) *Plant Methods* 21:114, doi:10.1186/s13007-025-01418-0 |
| In this archive | yes — `data/raw/ursn/` |
| Consumed by | `analysis/crosscrop/code/panel_ursn.py` |

384 lines, 2,302 markers, 109 environments, Fusarium head blight severity,
analysed as resistance = 100 − DIS.

---

## 6. Soybean — Northern Uniform Soybean Tests

| | |
|---|---|
| Source | SoyBase (https://www.soybase.org) and the supplementary data of Wartha et al. (2025) *Crop Sci.* 65:e70138, doi:10.1002/csc2.70138 |
| Licence | **CC BY 4.0** on the accepted manuscript (Crossref); SoyBase is a USDA-ARS resource |
| In this archive | yes — `data/raw/nust/` |
| Consumed by | `analysis/crosscrop/code/prep_nust.py` → `panel_nust.py` |
| Derived here | `nust_ge_means.csv`, the genotype × location-year yield means, is computed from the phenotype master table by `data/raw/nust/stat_nust.py` (run inside that folder; it reproduces the shipped file byte for byte) |

2,513 genotypes, 5,158 SNPs, 626 environments 2003–2020, yield, evaluated by
forward-year prediction for each of 2016–2020. The phenotype tables ship as R
`.RData`/`.rds` serialisations; `analysis/crosscrop/code/rdata.py` is a
from-scratch pure-Python reader for R serialization formats 2 and 3 (XDR), so
**no R installation is required** anywhere in this archive.

---

## What is included, and what is not

This archive redistributes **only the primary files the code actually reads**,
not full mirrors of the source deposits. Each of those deposits already has its
own DOI and its own archive; mirroring them here would duplicate storage without
adding provenance. Where a dataset is not included, the section above says where to
get it.

---

## Chinese maize — CUBIC (fifth dataset, Supplementary Section S11 only)

| | |
|---|---|
| Phenotypes | Jin et al. (2023) *Plant Commun.* 4:100473, Supplemental Table S12, doi:10.1016/j.xplc.2022.100473 — **CC BY-NC-ND 4.0** |
| Genotypes | CropGS-Hub population GSTP004, Chen et al. (2024) *Nucleic Acids Res.* 52:D1519 — no licence declared |
| Population | Liu et al. (2020) *Genome Biol.* 21:20, doi:10.1186/s13059-020-1930-x |
| In this archive | **no** — neither the phenotypes nor the method panel derived from them, because the NoDerivatives clause makes redistribution of an adapted table questionable |
| Consumed by | `analysis/crosscrop/code/panel_china.py` (run with `RUN_CUBIC=1 bash run_all.sh`) |

Download instructions, file hashes and the site-label mapping are in
`analysis/crosscrop/results/china_DATA_SOURCE.md`. The summary statistics the
paper reports are included: `china_summary.csv` (Table S17), `gpverdict_cubic.json` and
`gpverdict_cubic_methods.csv` (Section 3.7, Table S23), and `china_trait_sweep.csv`,
the exploratory run of all 23 traits through an earlier version of the panel that
Supplementary Section S11 summarises. `china_invariance.csv`, from the same earlier
version, is not reported and is not included.

## EasyGeSe — exploratory only (Supplementary Section S16)

| | |
|---|---|
| Source | Quesada-Traver et al. (2025) *BMC Genomics* 26:953, doi:10.1186/s12864-025-12129-0; data doi:10.5281/zenodo.15348871 |
| Licence | **CC BY 4.0** (Zenodo record) |
| Files | `datasets/results_raw.csv` — the benchmark's own per-split Pearson *r* and RMSE for 10 models on 93 traits in 10 species |
| In this archive | yes — `data/raw/easygese/datasets/results_raw.csv` |
| Consumed by | `analysis/easygese/code/reorder.py` |

An exploratory check, reported in Supplementary Section S16 and used for no result in the paper.

---

## Summary

| Dataset | Licence | Redistributed here |
|---|---|---|
| Maize G2F (CyVerse) | ODC PDDL | yes |
| Maize Supplemental Table S1 | public domain (US government work) | yes |
| Maize team submissions | not declared | yes, with the caveat above |
| Rice IRRI | CC BY 4.0 | yes |
| Wheat CAIGE | not declared | **no — fetch yourself** |
| Common bean VEF | CC BY 4.0 | yes |
| Spring wheat URSN | CC0 1.0 | yes |
| Soybean NUST | CC BY 4.0 / USDA | yes |
| EasyGeSe (exploratory, Supplementary Section S16) | CC BY 4.0 | yes (the results table) |
| Chinese maize CUBIC (Supplementary Section S11) | phenotypes CC BY-NC-ND 4.0; genotypes not declared | **no — fetch yourself** |

Licences were read from the DataCite, Crossref and Dataverse records on
2026-09-09. Where a table says "not declared", that is a statement about what
the depositor published, not a judgement that reuse is prohibited.
