# -*- coding: utf-8 -*-
"""Reference database. Entries were verified against Crossref, PubMed or arXiv on 2026-09-09
and rechecked on 2026-09-24, when four titles or author lists were corrected."""

REFS = {
# --- already in the manuscript -------------------------------------------
"blondel2015":  "M. Blondel, A. Onogi, H. Iwata, N. Ueda, A ranking approach to genomic selection, PLoS One 10 (2015) e0128570. https://doi.org/10.1371/journal.pone.0128570.",
"maierhein2018":"L. Maier-Hein, M. Eisenmann, A. Reinke, S. Onogur, M. Stankovic, P. Scholz, et al., Why rankings of biomedical image analysis competitions should be interpreted with care, Nat. Commun. 9 (2018) 5217. https://doi.org/10.1038/s41467-018-07619-7.",
"maierhein2024":"L. Maier-Hein, A. Reinke, P. Godau, M.D. Tizabi, F. Buettner, E. Christodoulou, et al., Metrics reloaded: recommendations for image analysis validation, Nat. Methods 21 (2024) 195–212. https://doi.org/10.1038/s41592-023-02151-z.",
"wang2025":     "F. Wang, M.J. Feldmann, D.E. Runcie, Do not benchmark phenomic prediction against genomic prediction accuracy, Plant Phenome J. 8 (2025) e70029. https://doi.org/10.1002/ppj2.70029.",
"g2fdata":      "Genomes to Fields 2022 maize genotype-by-environment prediction competition data, CyVerse Data Commons. https://doi.org/10.25739/tq5e-ak26.",
"washburn2025": "J.D. Washburn, J.I. Varela, A. Xavier, Q. Chen, D. Ertl, J.L. Gage, et al., Global genotype by environment prediction competition reveals that diverse modeling strategies can deliver satisfactory maize yield estimates, Genetics 229 (2025) iyae195. https://doi.org/10.1093/genetics/iyae195.",
"nguyen2023":   "V.H. Nguyen, R.I.Z. Morantte, V. Lopena, H. Verdeprado, R. Murori, A. Ndayiragije, et al., Multi-environment genomic selection in rice elite breeding lines, Rice 16 (2023) 7. https://doi.org/10.1186/s12284-023-00623-6.",
"trethowan2024":"R.M. Trethowan, J.M. Nicol, A. Singh, R.P. Singh, W. Tadesse, V. Govidan, et al., The CIMMYT Australia ICARDA Germplasm Evaluation concept: a model for international cooperation and impact, Front. Plant Sci. 15 (2024) 1435837. https://doi.org/10.3389/fpls.2024.1435837. Data compilations: https://www.caigeproject.org.au/data-compilations/.",
"keller2020":   "B. Keller, D. Ariza-Suarez, J. de la Hoz, J.S. Aparicio, A.E. Portilla-Benavides, H.F. Buendia, et al., Genomic prediction of agronomic traits in common bean (Phaseolus vulgaris L.) under environmental stress, Front. Plant Sci. 11 (2020) 1001. https://doi.org/10.3389/fpls.2020.01001.",
"beandata":     "Replication data for: Genomic prediction of agronomic traits in common bean under environmental stress, Harvard Dataverse. https://doi.org/10.7910/DVN/XCD67U.",
"brault2025":   "C. Brault, E.J. Conley, A.C. Read, A.J. Green, K.D. Glover, J.P. Cook, et al., Improving genomic prediction for plant disease using environmental covariates, Plant Methods 21 (2025) 114. https://doi.org/10.1186/s13007-025-01418-0.",
"ursndata":     "Data from: Improving genomic prediction for plant disease using environmental covariates, Dryad. https://doi.org/10.5061/dryad.wstqjq2z0.",
"wartha2025":   "C.A. Wartha, B.W. Campbell, V. Ramasubramanian, L. Nice, A. Brock, G. Cai, et al., Genomic analysis and predictive modeling in the Northern Uniform Soybean Tests, Crop Sci. 65 (2025) e70138. https://doi.org/10.1002/csc2.70138. Data: https://www.soybase.org.",
"arxivgrn":     "I. Kendiukhov, Quantifying ranking instability across evaluation protocol axes in gene regulatory network benchmarking, arXiv:2603.03493 (2026) (preprint).",
"kendall1938":  "M.G. Kendall, A new measure of rank correlation, Biometrika 30 (1938) 81–93. https://doi.org/10.1093/biomet/30.1-2.81.",
"arxivrank":    "B. Neuhof, Y. Benjamini, Rank intervals for leaderboards: a hierarchical framework for model evaluation, arXiv:2606.08679 (2026) (preprint).",

# --- genomic selection foundations ----------------------------------------
"meuwissen2001":"T.H.E. Meuwissen, B.J. Hayes, M.E. Goddard, Prediction of total genetic value using genome-wide dense marker maps, Genetics 157 (2001) 1819–1829. https://doi.org/10.1093/genetics/157.4.1819.",
"heffner2009":  "E.L. Heffner, M.E. Sorrells, J.L. Jannink, Genomic selection for crop improvement, Crop Sci. 49 (2009) 1–12. https://doi.org/10.2135/cropsci2008.08.0512.",
"deloscampos2013":"G. de los Campos, J.M. Hickey, R. Pong-Wong, H.D. Daetwyler, M.P.L. Calus, Whole-genome regression and prediction methods applied to plant and animal breeding, Genetics 193 (2013) 327–345. https://doi.org/10.1534/genetics.112.143313.",
"crossa2017":   "J. Crossa, P. Pérez-Rodríguez, J. Cuevas, O. Montesinos-López, D. Jarquín, G. de los Campos, et al., Genomic selection in plant breeding: methods, models, and perspectives, Trends Plant Sci. 22 (2017) 961–975. https://doi.org/10.1016/j.tplants.2017.08.011.",
"hickey2017":   "J.M. Hickey, T. Chiurugwi, I. Mackay, W. Powell, Genomic prediction unifies animal and plant breeding programs to form platforms for biological discovery, Nat. Genet. 49 (2017) 1297–1303. https://doi.org/10.1038/ng.3920.",
"vossfels2019": "K.P. Voss-Fels, M. Cooper, B.J. Hayes, Accelerating crop genetic gains with genomic selection, Theor. Appl. Genet. 132 (2019) 669–686. https://doi.org/10.1007/s00122-018-3270-8.",
"gianola2009":  "D. Gianola, G. de los Campos, W.G. Hill, E. Manfredi, R. Fernando, Additive genetic variability and the Bayesian alphabet, Genetics 183 (2009) 347–363. https://doi.org/10.1534/genetics.109.103952.",
"endelman2011": "J.B. Endelman, Ridge regression and other kernels for genomic selection with R package rrBLUP, Plant Genome 4 (2011) 250–255. https://doi.org/10.3835/plantgenome2011.08.0024.",

# --- genotype-by-environment ----------------------------------------------
"burgueno2012": "J. Burgueño, G. de los Campos, K. Weigel, J. Crossa, Genomic prediction of breeding values when modeling genotype × environment interaction using pedigree and dense molecular markers, Crop Sci. 52 (2012) 707–719. https://doi.org/10.2135/cropsci2011.06.0299.",
"jarquin2014":  "D. Jarquín, J. Crossa, X. Lacaze, P. Du Cheyron, J. Daucourt, J. Lorgeou, et al., A reaction norm model for genomic selection using high-dimensional genomic and environmental data, Theor. Appl. Genet. 127 (2014) 595–607. https://doi.org/10.1007/s00122-013-2243-1.",
"lopezcruz2015":"M. Lopez-Cruz, J. Crossa, D. Bonnett, S. Dreisigacker, J. Poland, J.L. Jannink, et al., Increased prediction accuracy in wheat breeding trials using a marker × environment interaction genomic selection model, G3 5 (2015) 569–582. https://doi.org/10.1534/g3.114.016097.",
"heslot2014":   "N. Heslot, D. Akdemir, M.E. Sorrells, J.L. Jannink, Integrating environmental covariates and crop modeling into the genomic selection framework to predict genotype by environment interactions, Theor. Appl. Genet. 127 (2014) 463–480. https://doi.org/10.1007/s00122-013-2231-5.",
"costaneto2021":"G. Costa-Neto, R. Fritsche-Neto, J. Crossa, Nonlinear kernels, dominance, and envirotyping data increase the accuracy of genome-based prediction in multi-environment trials, Heredity 126 (2021) 92–106. https://doi.org/10.1038/s41437-020-00353-1.",
"envrtype2021": "G. Costa-Neto, G. Galli, H.F. Carvalho, J. Crossa, R. Fritsche-Neto, EnvRtype: a software to interplay enviromics and quantitative genomics in agriculture, G3 11 (2021) jkab040. https://doi.org/10.1093/g3journal/jkab040.",
"rogers2022":   "A.R. Rogers, J.B. Holland, Environment-specific genomic prediction ability in maize using environmental covariates depends on environmental similarity to training data, G3 12 (2022) jkab440. https://doi.org/10.1093/g3journal/jkab440.",
"finlay1963":   "K.W. Finlay, G.N. Wilkinson, The analysis of adaptation in a plant-breeding programme, Aust. J. Agric. Res. 14 (1963) 742–754. https://doi.org/10.1071/AR9630742.",

# --- accuracy, validation, design -----------------------------------------
"daetwyler2013":"H.D. Daetwyler, M.P.L. Calus, R. Pong-Wong, G. de los Campos, J.M. Hickey, Genomic prediction in animals and plants: simulation of data, validation, reporting, and benchmarking, Genetics 193 (2013) 347–365. https://doi.org/10.1534/genetics.112.147983.",
"legarra2018":  "A. Legarra, A. Reverter, Semi-parametric estimates of population accuracy and bias of predictions of breeding values and future phenotypes using the LR method, Genet. Sel. Evol. 50 (2018) 53. https://doi.org/10.1186/s12711-018-0426-6.",
"estaghvirou2013":"S.B. Ould Estaghvirou, J.O. Ogutu, T. Schulz-Streeck, C. Knaak, M. Ouzunova, A. Gordillo, et al., Evaluation of approaches for estimating the accuracy of genomic prediction in plant breeding, BMC Genomics 14 (2013) 860. https://doi.org/10.1186/1471-2164-14-860.",
"rincent2012":  "R. Rincent, D. Laloë, S. Nicolas, T. Altmann, D. Brunel, P. Revilla, et al., Maximizing the reliability of genomic selection by optimizing the calibration set of reference individuals, Genetics 192 (2012) 715–728. https://doi.org/10.1534/genetics.112.141473.",
"isidro2015":   "J. Isidro, J.L. Jannink, D. Akdemir, J. Poland, N. Heslot, M.E. Sorrells, Training set optimization under population structure in genomic selection, Theor. Appl. Genet. 128 (2015) 145–158. https://doi.org/10.1007/s00122-014-2418-4.",
"piepho2007":   "H.P. Piepho, J. Möhring, Computing heritability and selection response from unbalanced plant breeding trials, Genetics 177 (2007) 1881–1888. https://doi.org/10.1534/genetics.107.074229.",
"cullis2006":   "B.R. Cullis, A.B. Smith, N.E. Coombes, On the design of early generation variety trials with correlated data, J. Agric. Biol. Environ. Stat. 11 (2006) 381–393. https://doi.org/10.1198/108571106X154443.",

# --- realised gain and crop-growth modelling -------------------------------
"rutkoski2019": "J.E. Rutkoski, Estimation of realized rates of genetic gain and indicators for breeding program assessment, Crop Sci. 59 (2019) 981–993. https://doi.org/10.2135/cropsci2018.09.0537.",
"seck2023":     "F. Seck, G. Covarrubias-Pazaran, T. Gueye, J. Bartholomé, Realized genetic gain in rice: achievements from breeding programs, Rice 16 (2023) 61. https://doi.org/10.1186/s12284-023-00677-6.",
"technow2015":  "F. Technow, C.D. Messina, L.R. Totir, M. Cooper, Integrating crop growth models with whole genome prediction through approximate Bayesian computation, PLoS One 10 (2015) e0130855. https://doi.org/10.1371/journal.pone.0130855.",
"cooper2009":   "M. Cooper, F.A. van Eeuwijk, G.L. Hammer, D.W. Podlich, C. Messina, Modeling QTL for complex traits: detection and context for plant breeding, Curr. Opin. Plant Biol. 12 (2009) 231–240. https://doi.org/10.1016/j.pbi.2009.01.006.",

# --- machine learning in genomic prediction --------------------------------
"mlopez2018b":  "A. Montesinos-López, O.A. Montesinos-López, D. Gianola, J. Crossa, C.M. Hernández-Suárez, Multi-environment genomic prediction of plant traits using deep learners with dense architecture, G3 8 (2018) 3813–3828. https://doi.org/10.1534/g3.118.200740.",
"perezenciso2019":"M. Pérez-Enciso, L.M. Zingaretti, A guide on deep learning for complex trait genomic prediction, Genes 10 (2019) 553. https://doi.org/10.3390/genes10070553.",
"washburn2020": "J.D. Washburn, M.B. Burch, J.A.V. Franco, Predictive breeding for maize: making use of molecular phenotypes, machine learning, and physiological crop models, Crop Sci. 60 (2020) 622–638. https://doi.org/10.1002/csc2.20052.",

# --- evaluation and statistics ---------------------------------------------
"jarvelin2002": "K. Järvelin, J. Kekäläinen, Cumulated gain-based evaluation of IR techniques, ACM Trans. Inf. Syst. 20 (2002) 422–446. https://doi.org/10.1145/582415.582418.",
"lei2018":      "J. Lei, M. G'Sell, A. Rinaldo, R.J. Tibshirani, L. Wasserman, Distribution-free predictive inference for regression, J. Am. Stat. Assoc. 113 (2018) 1094–1111. https://doi.org/10.1080/01621459.2017.1307116.",
"efron1979":    "B. Efron, Bootstrap methods: another look at the jackknife, Ann. Stat. 7 (1979). https://doi.org/10.1214/aos/1176344552.",
"dietterich1998":"T.G. Dietterich, Approximate statistical tests for comparing supervised classification learning algorithms, Neural Comput. 10 (1998) 1895–1923. https://doi.org/10.1162/089976698300017197.",
"norel2011":    "R. Norel, J.J. Rice, G. Stolovitzky, The self-assessment trap: can we all be better than average?, Mol. Syst. Biol. 7 (2011). https://doi.org/10.1038/msb.2011.70.",
"alkhalifah2018":"N. AlKhalifah, D.A. Campbell, C.M. Falcon, J.M. Gardiner, N.D. Miller, M.C. Romay, et al., Maize Genomes to Fields: 2014 and 2015 field season genotype, phenotype, environment, and inbred ear image datasets, BMC Res. Notes 11 (2018) 452. https://doi.org/10.1186/s13104-018-3508-1.",

# --- measurement theory and scoring-function theory ------------------------
"stevens1946":  "S.S. Stevens, On the theory of scales of measurement, Science 103 (1946) 677–680. https://doi.org/10.1126/science.103.2684.677.",
"hand1996":     "D.J. Hand, Statistics and the theory of measurement, J. R. Stat. Soc. A 159 (1996) 445–492. https://doi.org/10.2307/2983326.",
"gneiting2011": "T. Gneiting, Making and evaluating point forecasts, J. Am. Stat. Assoc. 106 (2011) 746–762. https://doi.org/10.1198/jasa.2011.r10138.",
"fissler2019":  "T. Fissler, J.F. Ziegel, Order-sensitivity and equivariance of scoring functions, Electron. J. Stat. 13 (2019) 1166–1211. https://doi.org/10.1214/19-ejs1552.",
"ferrante2021": "M. Ferrante, N. Ferro, N. Fuhr, Towards meaningful statements in IR evaluation: mapping evaluation measures to interval scales, IEEE Access 9 (2021) 136182–136216. https://doi.org/10.1109/access.2021.3116857.",

# --- surrogate endpoints ---------------------------------------------------
"prentice1989": "R.L. Prentice, Surrogate endpoints in clinical trials: definition and operational criteria, Stat. Med. 8 (1989) 431–440. https://doi.org/10.1002/sim.4780080407.",
"buyse1998":    "M. Buyse, G. Molenberghs, Criteria for the validation of surrogate endpoints in randomized experiments, Biometrics 54 (1998) 1014–1029. https://doi.org/10.2307/2533853.",

# --- the closest neighbours in breeding ------------------------------------
"waldmann2019": "P. Waldmann, On the use of the Pearson correlation coefficient for model evaluation in genome-wide prediction, Front. Genet. 10 (2019) 899. https://doi.org/10.3389/fgene.2019.00899.",
"pan2024":      "S. Pan, Z. Liu, Y. Han, D. Zhang, X. Zhao, J. Li, et al., Using the Pearson's correlation coefficient as the sole metric to measure the accuracy of quantitative trait prediction: is it sufficient?, Front. Plant Sci. 15 (2024) 1480463. https://doi.org/10.3389/fpls.2024.1480463.",
"heinrich2025": "F. Heinrich, T.M. Lange, F. Ramzan, M. Gültas, A.O. Schmitt, Normalized cumulative gain as an alternative evaluation measure for genomic selection models, Genet. Sel. Evol. 57 (2025) 70. https://doi.org/10.1186/s12711-025-01022-9.",
# --- added 2026-09-23; verified against Crossref/PubMed ---
"ornella2014":  "L. Ornella, P. Pérez, E. Tapia, J.M. González-Camacho, J. Burgueño, X. Zhang, et al., Genomic-enabled prediction with classification algorithms, Heredity 112 (2014) 616–626. https://doi.org/10.1038/hdy.2013.144.",
"burzykowski2006": "T. Burzykowski, M. Buyse, Surrogate threshold effect: an alternative measure for meta-analytic surrogate endpoint validation, Pharm. Stat. 5 (2006) 173–186. https://doi.org/10.1002/pst.207.",
"villar2018":   "B.d.J. Villar-Hernández, S. Pérez-Elizalde, J. Crossa, P. Pérez-Rodríguez, F.H. Toledo, J. Burgueño, A Bayesian decision theory approach for genomic selection, G3 8 (2018) 3019–3037. https://doi.org/10.1534/g3.118.200430.",
"runcie2026":   "D.E. Runcie, The use of cross-validation has overestimated the value of genomic selection in plant breeding, bioRxiv (2026) 2026.01.05.697784. https://doi.org/10.64898/2026.01.05.697784 (preprint).",

# --- leaderboard uncertainty and the gain shortfall -------------------------
"wiesenfarth2021":"M. Wiesenfarth, A. Reinke, B.A. Landman, M. Eisenmann, L.A. Saiz, M.J. Cardoso, et al., Methods and open-source toolkit for analyzing and visualizing challenge results, Sci. Rep. 11 (2021) 2369. https://doi.org/10.1038/s41598-021-82017-6.",
"bijma2012":    "P. Bijma, Accuracies of estimated breeding values from ordinary genetic evaluations do not reflect the correlation between true and estimated breeding values in selected populations, J. Anim. Breed. Genet. 129 (2012) 345–358. https://doi.org/10.1111/j.1439-0388.2012.00991.x.",

# --- the 2024 competition, and the software that already groups by environment
"g2f2024":    "Q. Chen, J.D. Washburn, D.C. Lima, M.C. Romay, J.L. Gage, J.B. Holland, et al., Genomes to fields 2024 maize genotype by environment prediction competition, BMC Res. Notes 19 (2026) 113. https://doi.org/10.1186/s13104-026-07629-5.",
"skm2022":    "O.A. Montesinos López, B.A. Mosqueda González, A. Palafox González, A. Montesinos López, J. Crossa, A general-purpose machine learning R library for sparse kernels methods with an application for genome-based prediction, Front. Genet. 13 (2022) 887643. https://doi.org/10.3389/fgene.2022.887643.",
"g2p2023":    "Q. Wang, S. Jiang, T. Li, Z. Qiu, J. Yan, R. Fu, et al., G2P provides an integrative environment for multi-model genomic selection analysis to improve genotype-to-phenotype prediction, Front. Plant Sci. 14 (2023) 1207139. https://doi.org/10.3389/fpls.2023.1207139.",

"roostaei2014": "M. Roostaei, R. Mohammadi, A. Amri, Rank correlation among different statistical models in ranking of winter wheat genotypes, Crop J. 2 (2014) 154–163. https://doi.org/10.1016/j.cj.2014.02.002.",
"kusmec2026":   "A. Kusmec, K.L. Negus, J. Yu, Critical evaluation of the theory and practice of feed-forward neural networks for genomic prediction, G3 16 (2026) jkaf314. https://doi.org/10.1093/g3journal/jkaf314.",

# --- added 2026-09-22; each verified against Crossref / DataCite / arXiv / bioRxiv APIs
#     (collision_scan_2026H2_update.md, section 5)
"liu2026assay":  "Z. Liu, Assay concordance sets exact ceilings on what one biological score can predict, bioRxiv (2026) 2026.08.24.746774. https://doi.org/10.64898/2026.08.24.746774 (preprint).",
"schneider2026": "A. Schneider, T. Rochussen, J. Stiller, V. Fortuin, Decision-aligned evaluation of uncertainty quantification, arXiv:2606.26990 (2026) (preprint).",
"scavino2026":   "V. Scavino, Decision kernels for quantum error mitigation: why accuracy gains need not improve downstream decisions, arXiv:2607.02888 (2026) (preprint).",
"kondratev2026": "A.Y. Kondratev, E. Ianovski, E. Voronina, J. Crossa, An axiomatic approach to cultivar ranking in multi-environment trials, bioRxiv (2026) 2026.06.27.734959. https://doi.org/10.64898/2026.06.27.734959 (preprint).",
"fieldwelsh2007":"C.A. Field, A.H. Welsh, Bootstrapping clustered data, J. R. Stat. Soc. B 69 (2007) 369–390. https://doi.org/10.1111/j.1467-9868.2007.00593.x.",
"viglione2026": "V. Viglione, L. Paleari, A. Tondelli, C. Marchetti, R. Confalonieri, Accuracy, robustness and the Occam’s razor in model-aided genomic prediction, Agric. For. Meteorol. 390 (2026) 111483. https://doi.org/10.1016/j.agrformet.2026.111483.",
# --- added 2026-09-24 (verified against Crossref on that date) --------------
"schrauf2021":  "M.F. Schrauf, G. de los Campos, S. Munilla, Comparing genomic prediction models by means of cross validation, Front. Plant Sci. 12 (2021) 734512. https://doi.org/10.3389/fpls.2021.734512.",
"hamblin1986":  "J. Hamblin, M.J. de O. Zimmermann, Breeding common bean for yield in mixtures, Plant Breed. Rev. 4 (1986) 245–272. https://doi.org/10.1002/9781118061015.ch8.",
"caballero2026":"E. Caballero, J. Garcia-Abadillo, D. Jarquin, Missing comparability: when genomic selection faces field variability. A case study in soybeans, Plant Genome 19 (2026) e70264. https://doi.org/10.1002/tpg2.70264.",
"yan2026bench": "M. Yan, W. Wang, Y. Zhang, H. Guo, Z. Xue, X. Wang, et al., Standardizing benchmarks for plant genomic prediction, Agronomy 16 (2026) 1131. https://doi.org/10.3390/agronomy16121131.",
"eckhoff2026":  "W. Eckhoff, F. Parat, G. Bracho-Mujica, C. Flamm, D. Bustos-Korts, H.P. Piepho, Tailoring AI and ML models for genotype-by-environment prediction leveraging environmental covariates: a European rye example, Theor. Appl. Genet. 139 (2026) 206. https://doi.org/10.1007/s00122-026-05280-z.",
}
