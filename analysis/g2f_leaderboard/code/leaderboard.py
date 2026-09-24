# -*- coding: utf-8 -*-
"""G2F 2022 官方排行榜 vs 决策终点排行榜：真实参赛提交"""
import numpy as np, pandas as pd, glob, os
from scipy import stats
rng=np.random.default_rng(11)
obs=pd.read_csv('analysis/g2f_leaderboard/results/Final_Observed_Yield.csv').rename(columns={'Yield_Mg_ha':'y'})
SUBS=['CLAC_repo_submission5','MPB_Group_repo_prediction','KernelOfTruth_sub244906',
      'KernelOfTruth_sub244944','KernelOfTruth_sub244953','EnBiSys_sub243568','NicheSquad_sub244985']
files={k:f'analysis/g2f_leaderboard/results/{k}.csv' for k in SUBS}
name={'CLAC_repo_submission5':'CLAC (官方第1)','MPB_Group_repo_prediction':'ML_APT/MPB',
      'KernelOfTruth_sub244906':'KernelOfTruth-a','KernelOfTruth_sub244944':'KernelOfTruth-b',
      'KernelOfTruth_sub244953':'KernelOfTruth-c','EnBiSys_sub243568':'EnBiSys','NicheSquad_sub244985':'NicheSquad'}
D=obs.copy()
for k,f in sorted(files.items()):
    s=pd.read_csv(f).rename(columns={'Yield_Mg_ha':k})
    D=D.merge(s[['Env','Hybrid',k]],on=['Env','Hybrid'],how='left')
subs=[k for k in sorted(files) if k in D.columns]
D=D.dropna(subset=subs).reset_index(drop=True)
print(f"合并后 {len(D)} 条，{D.Env.nunique()} 个环境，{len(subs)} 份真实提交\n")

def per_env(col,q):
    out=[]
    for e,g in D.groupby('Env'):
        o=g.y.to_numpy(); p=g[col].to_numpy(); n=len(o); k=max(1,int(np.ceil(q*n)))
        out.append(o[np.argsort(-p)[:k]].mean()-o.mean())
    return np.array(out)
def per_env_rmse(col):
    return np.array([np.sqrt(((g[col]-g.y)**2).mean()) for _,g in D.groupby('Env')])
def per_env_r(col):
    return np.array([(np.corrcoef(g[col],g.y)[0,1] if g[col].std()>1e-12 else np.nan) for _,g in D.groupby('Env')])

M={}
for k in subs:
    M[k]={'meanRMSE':per_env_rmse(k).mean(),'r':np.nanmean(per_env_r(k))}
    for q in [0.01,0.05,0.10,0.20]:
        M[k][f'SG{int(q*100)}']=per_env(k,q).mean()
T=pd.DataFrame(M).T
T.index=[name.get(i,i) for i in T.index]
T=T.sort_values('meanRMSE')
print(T.round(4).to_string())

print("\n=== 各口径下的排名（1 = 最好）===")
ranks=pd.DataFrame({'官方指标 meanRMSE':T.meanRMSE.rank(),'环境内 r':(-T.r).rank(),
                    'SG@1%':(-T.SG1).rank(),'SG@5%':(-T.SG5).rank(),
                    'SG@10%':(-T.SG10).rank(),'SG@20%':(-T.SG20).rank()})
print(ranks.to_string())

print("\n=== 与官方排名的一致性（Kendall τ）===")
off=T.meanRMSE.rank().to_numpy()
for c in ranks.columns[1:]:
    tau,p=stats.kendalltau(off,ranks[c].to_numpy())
    flag='  ← τ<0.8，重排显著' if tau<0.8 else ''
    print(f"  官方 vs {c:<14} τ = {tau:+.3f}  (p={p:.3f}){flag}")

print("\n=== 名次变动最大的提交 ===")
for c in ['SG@1%','SG@5%','SG@10%','SG@20%']:
    d=(ranks[c]-ranks['官方指标 meanRMSE'])
    j=d.abs().idxmax()
    print(f"  {c:<7}: {j} 从官方第 {int(ranks.loc[j,'官方指标 meanRMSE'])} 名 -> 第 {int(ranks.loc[j,c])} 名 （变动 {int(d[j]):+d}）")
T.to_csv('analysis/g2f_leaderboard/results/leaderboard_metrics.csv')
ranks.to_csv('analysis/g2f_leaderboard/results/leaderboard_ranks.csv')
