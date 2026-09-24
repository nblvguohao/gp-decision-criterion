# Data source — China maize, CUBIC population (fifth dataset, Supplementary Section S11)

Provenance, licence, accession and download date for the dataset consumed by
`analysis/crosscrop/code/panel_china.py`. `DATA_SOURCES.md` summarises this record;
`run_all.sh` builds the panel when run with `RUN_CUBIC=1`.

All downloads: **2026-09-10** (UTC). Files staged under `data/raw/cubic/`,
which is git-ignored by repository convention (`.gitignore`: third-party
primary data, "not ours to redistribute", recoverable from the recorded DOIs).

---

## What the dataset is

CUBIC — Complete-diallel plus Unbalanced Breeding-like Inter-Cross — is a
synthetic maize population of 1,404 recombinant inbred lines derived from **24
Chinese elite maize inbred founders**, developed and phenotyped in China. The
lines were grown at **five Chinese sites in one season**: JL (Gongzhuling,
Jilin), LN (Shenyang, Liaoning), BJ (Beijing), HeB (Shijiazhuang, Hebei) and
HN (Xinxiang, Henan). Twenty-three agronomic traits were scored at every site.

The trait used in the paper (Supplementary Section S11, Table S17) is **ear weight
(EW)**, the yield trait of the panel. An early exploratory run of all 23 traits through
the pre-GBLUP version of the panel is kept as `china_trait_sweep.csv`; it is not
reported in the paper and was not regenerated.

---

## 1. Phenotypes — per-site raw measurements

| | |
|---|---|
| Source | Jin et al. (2023) *Plant Communications* 4:100473, Supplemental Table S12, "Raw measurements and BLUP of 23 maize agronomic traits investigated for the CUBIC population at 5 locations" |
| DOI | doi:10.1016/j.xplc.2022.100473 |
| Obtained from | Europe PMC supplementary-files endpoint for **PMC10203269** (`https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10203269/supplementaryFiles`) |
| Licence | **CC BY-NC-ND 4.0** (the article's own licence) |
| Files | `xplc2022_100473_supplement.zip` (sha256 `5b490424…f8601`), `mmc2.xlsx` extracted from it (sha256 `b10df316…e53b21`) |
| Derived | `TableS12_phenotypes.csv` (sha256 `e5f996c8…a38f8e`) — sheet `Table S12` of `mmc2.xlsx`, `header=1`, exported verbatim to CSV. 1,404 rows × 139 columns (`ID`, then for each of 23 traits five per-site columns and one BLUP column). |
| Also derived | `TableS13_weather.csv` (sha256 `e53b901f…c94ae132`) — sheet `Table S13`, site-specific daily temperature and day length, aggregated by `panel_china.py` into the six environmental covariates used by the reaction-norm method. Note the site label `HB` in Table S13 corresponds to `HeB` in Table S12; `panel_china.py` maps it. |

**Licence caveat, stated plainly.** The ND ("NoDerivatives") clause of CC
BY-NC-ND 4.0 makes redistribution of an *adapted* form of this table
questionable. `data/raw/cubic/` is git-ignored, and the public repository carries no
phenotype value and no per-genotype quantity derived from these values: only the
aggregate statistics reported in the paper (`china_summary.csv`) and those of the
exploratory trait sweep (`china_trait_sweep.csv`). The CSV above is a verbatim sheet export
kept locally so the pipeline can run, and it is recoverable by anyone from the
DOI in two steps. If the public archive is ever extended to include this
dataset, the phenotype file must **not** be shipped — only this record.

An alternative source that may carry a more permissive licence is the
**MaizeCUBIC** database (Wang et al. (2020) *Database* 2020:baaa044,
doi:10.1093/database/baaa044, host `cubicmaize.hzau.edu.cn`), whose describing
article is CC BY. **This was not checked**: the host is not on the analysis
sandbox's network allowlist and no access was requested for it, so whether it
serves the same per-site values, and under what terms, is unverified.

## 2. Genotypes — PLINK binary set

| | |
|---|---|
| Source | CropGS-Hub, population **GSTP004** ("GSTP004-Maize-1404") |
| Database paper | Chen et al. (2024) *Nucleic Acids Res.* 52:D1519–D1529, doi:10.1093/nar/gkad1062 (article CC BY 4.0) |
| URL | `https://iagr.genomics.cn/static/gstool/data/GSTP004/population/GSTP004.bed.tar.gz` |
| Archive | 1,264,127,693 bytes, sha256 `13d422057f65f43575d8d2e83eb36c34ad615b57f53501ce2a9b3f136c4e302b` |
| Contents | `cubic_1404_hmp2plink_maf0.02.bed` (4,150,585,533 bytes), `.bim` (402,861,836 bytes), `.fam` (31,297 bytes, sha256 `8b628320…511714`) — 1,404 samples × **11,825,030** markers, MAF ≥ 0.02, positions on B73 AGPv3 |
| Licence | **None declared.** CropGS-Hub states a citation request and no data licence; the underlying deposits (NCBI BioProject PRJNA597703; CNCB-NGDC GSA CRA000171) carry no licence statement either. Same situation as the CAIGE entry in `DATA_SOURCES.md`. |
| Upstream primary | Liu et al. (2020) *Genome Biol.* 21:20, doi:10.1186/s13059-020-1930-x (CC BY 4.0). The primary deposit is **raw sequencing reads only**, so it is not a usable substitute for a genotype matrix. |

**Not kept in the repository.** The 4.15 GB `.bed` and 403 MB `.bim` are not
staged under `data/raw/cubic/`; only the 31 KB `.fam` is, together with the
decoded thinned matrix below. Re-create them with

```bash
curl -L -o GSTP004.bed.tar.gz \
  https://iagr.genomics.cn/static/gstool/data/GSTP004/population/GSTP004.bed.tar.gz
tar xzf GSTP004.bed.tar.gz     # -> cubic_1404_hmp2plink_maf0.02.{bed,bim,fam}
```

then point `CUBIC_BED_PREFIX` at the unpacked prefix for one run of
`panel_china.py`. The server is slow and drops connections; `curl -C -` in a
retry loop was needed to finish the transfer.

| | |
|---|---|
| Derived | `cubic_geno_thinned.npz` (sha256 `20accddd…cfff9f`, 36.8 MB) — every 118th marker of the PLINK set, **100,213 markers × 1,404 lines**, decoded to allele-1 dosage 0/1/2 with `01` read as missing. Built once by `panel_china.py`; thereafter the `.bed` is not needed. |
| | `GSTP004.pheno` (sha256 `b4f7902e…e6580`) and `GSTP004_info.xlsx` (sha256 `7c86cadb…f4ecd5`) — CropGS-Hub's own BLUP phenotypes and sample table, downloaded for the identifier cross-check below. |

## 3. Identifier join, verified

The 1,404 line identifiers (`MG_49`, `MG_50`, …) are **identical sets** across
the CropGS-Hub `.fam`, the CropGS-Hub `.pheno`, `GSTP004_info.xlsx` and Table
S12; the intersection is 1,404 in every pairwise comparison. Stronger: for 21
of the 23 traits the BLUP column of Table S12 and the corresponding column of
CropGS-Hub's `GSTP004.pheno` are **bit-identical** (maximum absolute difference
0, or ≤ 3.6 × 10⁻¹⁵ for two traits), so the two sources describe the same lines
with the same identifiers. The remaining two (`TBN`, `TL`) could not be
compared because the column names do not align between the files; nothing in
the analysis uses them beyond the trait sweep.

After matching, 1,400 of the 1,404 lines have an EW measurement; every
environment retains 1,308–1,378 genotypes, far above the 25-genotype floor of
Methods 2.3.

## 4. Deviations from the repository's declared toolchain

1. **Toolchain.** Since 2026-09-24 the panel is built under the same interpreter as every other
   script (Python 3.11.16, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, pandas 2.3.3).
   An earlier build used a separate environment because scikit-learn was then missing from
   the declared one; that build is superseded.
2. **Marker thinning.** Genomic principal components come from 100,213 evenly
   spaced markers of the 11,825,030 available (every 118th), not all of them.
   PC1–60 explain 48.6 % of the marker variance. The PCs are used as features
   by the ridge and machine-learning methods and are the basis of the two
   reduced-rank GBLUP models, so the thinning applies to them too. Set
   `GENO_TARGET_SNP` to change it.
