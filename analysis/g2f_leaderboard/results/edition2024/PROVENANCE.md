# 2024 edition (second edition): official leaderboard

Source: EvalAI public API, no authentication required
- Challenge page: https://eval.ai/web/challenges/challenge-page/2376/overview
- Leaderboard: `https://eval.ai/api/jobs/challenge_phase_split/5893/leaderboard/?page_size=1000`
- Retrieved: 2026-09-08

**Official metric (EvalAI schema, verbatim)**: `Mean_r` = "Average of within-environment Pearson r scores"

For comparison, the official metric of the 2022 edition (Genetics 229(2):iyae195) was the per-environment RMSE
averaged over environments: the organisers changed the ranking metric between the two editions.

## Leaderboard summary
- 64 teams (best submission per team); Mean_r from −0.0245 to 0.4501
- 1st `Naaa` (0.4501), 2nd `NJUST_KMG1` (0.4440)
- 3rd is `Model 2022 (not competing)` (0.4377), a reference model entered by the organisers
- 8th `GxE4GoodY` (0.4143) has public code (github.com/ldcesilva/GxE4GoodY) but depends on commercial JMP/SAS
  software and was not rerun

## Why this edition is not analysed across metrics
EvalAI stores only `Mean_r`; no results paper or supplementary table has been published for the 2024 edition,
and the submissions are not public. A cross-metric analysis like that of the 2022 edition would need the teams'
prediction files.
