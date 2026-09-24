# 2024 届（第二届）官方排行榜

来源：EvalAI 公开 API，无需认证
- 挑战页：https://eval.ai/web/challenges/challenge-page/2376/overview
- 排行榜：`https://eval.ai/api/jobs/challenge_phase_split/5893/leaderboard/?page_size=1000`
- 抓取日期：2026-09-08

**官方指标（EvalAI schema 原文）**：`Mean_r` = "Average of within-environment Pearson r scores"

对比：2022 届（Genetics 229(2):iyae195）官方指标 = 逐环境 RMSE 再平均。
**组委会在两届之间更换了排名指标。**

## 排行榜要点
- 64 支队伍（每队最佳提交），Mean_r 范围 −0.0245 ~ 0.4501
- 第 1 名 `Naaa` (0.4501)，第 2 `NJUST_KMG1` (0.4440)
- **第 3 名是 `Model 2022 (not competing)` (0.4377)** —— 组委会把 2022 届模型作为参照跑入，64 支中排第 3
- 第 8 名 `GxE4GoodY` (0.4143) 代码公开（github.com/ldcesilva/GxE4GoodY），但依赖 JMP/SAS 商业软件，本机不可重跑

## 本届为何无法做跨指标分析
EvalAI 只存 `Mean_r` 一个指标；2024 届未发表结果论文，无补充表；参赛提交未公开。
要复现 2022 届的跨指标重排分析，需要各队原始预测文件。
