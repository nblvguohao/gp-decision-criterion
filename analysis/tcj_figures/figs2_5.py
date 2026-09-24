import sys, itertools; sys.path.insert(0,"analysis/tcj_figures")
import numpy as np, pandas as pd
from style import *
apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R="analysis/g2f_leaderboard/results"
G=pd.read_csv(f"{R}/S1_best_per_team.csv"); G=G[G["Team Name"].notna()].copy()
nm=G["Team Name"].to_numpy(); M=len(G)
r22=G["mean_RMSE"].rank(ascending=True,method="min").to_numpy()
r24=G["mean_pearson_r"].rank(ascending=False,method="min").to_numpy()

# ---------------------------------------------------------------- Fig 2
fig=plt.figure(figsize=(ONEHALF,ONEHALF*0.60))
gs=fig.add_gridspec(1,2,width_ratios=[1.25,1],wspace=.50)
ax=fig.add_subplot(gs[0]); mv=r22-r24


def spread(y,gap):
    """move label positions apart until neighbours are at least `gap` ranks apart"""
    y=np.asarray(y,float); o=np.argsort(y); z=y[o].copy()
    for _ in range(200):
        moved=False
        for j in range(1,len(z)):
            d=z[j]-z[j-1]
            if d<gap:
                z[j-1]-=(gap-d)/2; z[j]+=(gap-d)/2; moved=True
        if not moved: break
    out=np.empty_like(z); out[o]=z; return out


lab=[i for i in range(M) if abs(mv[i])>=10 or min(r22[i],r24[i])<=3]
for i in range(M):
    big=abs(mv[i])>=10
    c=(C_ADM if mv[i]<0 else C_BLUE) if big else "#cfd4d9"
    ax.plot([0,1],[r22[i],r24[i]],color=c,lw=1.3 if big else .6,
            alpha=1 if big else .8,zorder=3 if big else 1)
GAP_R=1.4
for side,rk,x,ha in [(0,r22,-.05,"right"),(1,r24,1.05,"left")]:
    yl=spread([rk[i] for i in lab],GAP_R)
    for i,yy in zip(lab,yl):
        big=abs(mv[i])>=10; cc=((C_ADM if mv[i]<0 else C_BLUE) if big else INK)
        ax.text(x+(-.06 if side==0 else .06),yy,nm[i],ha=ha,va="center",fontsize=FS,color=cc)
        ax.plot([side,x+(-.05 if side==0 else .05)],[rk[i],yy],color=cc,lw=.5,alpha=.7)
ax.set_xlim(-1.5,2.05); ax.set_ylim(M+1.2,-.6); ax.set_xticks([0,1])
ax.set_xticklabels(["2022 metric\nper-env RMSE","2024 metric\nwithin-env $r$"],fontsize=FS)
ax.set_ylabel("leaderboard rank"); ax.set_yticks([1,10,20,30])
ax.spines["bottom"].set_visible(False); ax.tick_params(axis="x",length=0)
panel(ax,"a",dx=-0.30,dy=1.02)
ax=fig.add_subplot(gs[1])
SUM=pd.read_csv("analysis/crosscrop/results/cross_dataset_summary.csv").set_index("dataset")
mz=SUM.loc["maize"]
TOP=pd.read_csv(f"{R}/top10_reversal.csv").iloc[0]
rows=[("all 30 teams",mz.reversal,mz.ci_lo,mz.ci_hi,C_BLUE),
      ("top 10 teams",TOP.rate,TOP.ci_lo,TOP.ci_hi,C_ADM)]
for i,(l,v,lo,hi,c) in enumerate(rows):
    ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt="o",color=c,ms=5,capsize=3,lw=1.3,zorder=3)
    ax.text(v,i+.24,f"{v*100:.0f}%",ha="center",fontsize=7,color=c,fontweight="bold")
ax.axvline(.5,color=INK,ls=(0,(3.5,2)),lw=.85)
ax.text(.49,1.62,"unrelated\nrankings",fontsize=FS,ha="right",color=MUTED)
ax.axvline(0,color=C_GREEN,ls=":",lw=.85)
ax.text(.01,1.62,"perfect\nagreement",fontsize=FS,ha="left",color=C_GREEN)
ax.set_yticks([0,1]); ax.set_yticklabels([r[0] for r in rows]); ax.set_ylim(-.6,1.95)
ax.set_xlim(-.05,.60); ax.set_xlabel("fraction of team pairs\nreordered")
panel(ax,"b",dx=-0.40,dy=1.02)
print("Fig2 width mm", round(save(fig,"Fig2",ONEHALF),1)); plt.close(fig)

# ------------------------------------------- Fig 4 (rank distribution across the 22 metrics)
# Blob plot in the sense of Wiesenfarth et al. 2021: the area of each blob is the number of
# official metrics that put the team at that rank, so the distribution is shown directly and
# no interval has to be constructed. The pale band behind is the interval of Section 2.4,
# kept only so that the reader can see how much wider it is than what the metrics do.
RM=pd.read_csv("analysis/crosscrop/results/rank_matrix_maize.csv",index_col=0)
IV=pd.read_csv(f"{R}/rank_intervals.csv").set_index("team")
_CEN=pd.read_csv("analysis/crosscrop/results/rank_census.csv").set_index("dataset")
order=RM.median(axis=1).sort_values(kind="mergesort").index
fig,ax=plt.subplots(figsize=(SINGLE,SINGLE*1.32))
for y,t in enumerate(order):
    iv=IV.loc[t]
    ax.plot([iv["lo"],iv["hi"]],[y,y],color=C_BLUE,lw=5.2,alpha=.13,solid_capstyle="round",zorder=1)
    v=RM.loc[t].value_counts()
    ax.plot([v.index.min(),v.index.max()],[y,y],color=C_BLUE,lw=.7,alpha=.55,zorder=2)
    ax.scatter(v.index,[y]*len(v),s=v.values*5.2,color=C_BLUE,alpha=.85,lw=0,zorder=3)
    ax.plot(RM.loc[t].median(),y,"x",color=C_ADM,ms=3.6,mew=1.0,zorder=4)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order,fontsize=FS)
ax.invert_yaxis(); ax.set_xlim(0,31); ax.set_xticks([1,10,20,30])
ax.set_xlabel("leaderboard rank"); ax.grid(axis="x",alpha=.18,lw=.5)
ax.set_ylim(len(order)-.4,-.6)

ax.legend(handles=[Line2D([],[],marker="o",ls="",color=C_BLUE,ms=3.2,alpha=.85,label="1 metric"),
                   Line2D([],[],marker="o",ls="",color=C_BLUE,ms=6.4,alpha=.85,label="4 metrics"),
                   Line2D([],[],marker="x",ls="",color=C_ADM,ms=3.6,mew=1.0,label="median rank"),
                   Line2D([],[],ls="-",color=C_BLUE,alpha=.25,lw=5.2,label="95% interval")],
          loc="upper center",bbox_to_anchor=(.42,-.095),ncol=2,handletextpad=.5,
          columnspacing=1.2,labelspacing=.5,borderpad=.2)
print("Fig4 width mm", round(save(fig,"Fig4",SINGLE),1)); plt.close(fig)
print("Fig2, Fig4 written")


# ------------------------------------------- Fig 3 (four datasets)
_S=pd.read_csv("analysis/crosscrop/results/cross_dataset_summary.csv").set_index("dataset")
_PT=pd.read_csv("analysis/crosscrop/results/partition_by_dataset.csv").set_index("dataset")
_TE=pd.read_csv("analysis/crosscrop/results/threshold_estimates.csv")
_TE=_TE[(_TE.methods=="parent methods")&(_TE.frac==.10)].set_index("dataset")
_TT=pd.read_csv("analysis/crosscrop/results/threshold_theory.csv").set_index("dataset")
_RR=pd.read_csv("analysis/crosscrop/results/unaugmented_panels.csv")
_RR=_RR[_RR.panel=="base"].set_index("dataset")          # panels without the added variants
_MZ=_TT.loc["maize (G2F design)"]
FLOOR=float(_MZ.theory_lam05); TR_LO,TR_HI=float(_MZ.transfer_lo),float(_MZ.transfer_hi)   # design value, not a bound
_LAB={"maize":"maize","common bean":"bean","spring wheat":"spring wheat","soybean":"soybean"}
D=pd.DataFrame([
  (_LAB[k], k, int(r.methods), int(r.envs), int(r.genos_per_env), r.reversal, r.ci_lo, r.ci_hi, r.width_frac,
   int(_PT.loc[k,"tier_I"]), int(_PT.loc[k,"tier_II"]))
  for k,r in _S.iterrows()],
 columns=["sp","ds","M","E","GpE","rev","lo","hi","ivl","t1","t2"])
fig=plt.figure(figsize=(DOUBLE,DOUBLE*0.44))
gs=fig.add_gridspec(2,3,hspace=.95,wspace=.55)
y=np.arange(len(D)); YL=(len(D)-.35,-.65)
def _rows(ax):
    ax.set_yticks(y); ax.set_yticklabels(D.sp,fontsize=FS); ax.set_ylim(*YL)
    for t,c in zip(ax.get_yticklabels(),[SPECIES[x] for x in D.sp]): t.set_color(c)
ax=fig.add_subplot(gs[0,0])
for i,r in D.iterrows():
    ax.barh(i,22,height=.62,color=SPECIES[r.sp],alpha=.12)
    ax.barh(i,r.t1,height=.62,color=SPECIES[r.sp])
    ax.barh(i,r.t2,left=r.t1,height=.62,color=SPECIES[r.sp],alpha=.45)
    ax.text(r.t1+r.t2+.7,i,f"{r.t1} + {r.t2}",va="center",fontsize=FS,color=SPECIES[r.sp],fontweight="bold")
_rows(ax); ax.set_xlim(0,23)
ax.set_xlabel("of 22 official metrics:\nTier I + Tier II")
panel(ax,"a",dx=-0.46,dy=1.05)
ax=fig.add_subplot(gs[0,1:])
for i,r in D.iterrows():
    c=SPECIES[r.sp]
    if r.ds in _RR.index:                                 # the same panel without the variants
        b=_RR.loc[r.ds]
        ax.errorbar(b.rate,i+.26,xerr=[[b.rate-b.ci_lo],[b.ci_hi-b.rate]],fmt="o",color=c,ms=3.6,
                    capsize=2.2,lw=.9,mfc="white",zorder=3)
    ax.errorbar(r.rev,i,xerr=[[r.rev-r.lo],[r.hi-r.rev]],fmt="o",color=c,ms=4.6,capsize=2.6,lw=1.2,zorder=3)
    ax.text(r.hi+.012,i,f"{r.rev*100:.0f}%",ha="left",va="center",fontsize=FS,color=c,fontweight="bold",
            bbox=dict(fc="white",ec="none",pad=.2),zorder=5)
ax.axvline(.5,color=INK,ls=(0,(3.5,2)),lw=.85); ax.axvline(0,color=C_GREEN,ls=":",lw=.85)
_rows(ax); ax.set_xlim(-.04,.80)
ax.legend(handles=[Line2D([],[],marker="o",ls="",color=MUTED,ms=4.6,label="with the added variants"),
                   Line2D([],[],marker="o",ls="",color=MUTED,mfc="white",ms=3.6,label="base methods only")],
          loc="upper right",bbox_to_anchor=(1.02,1.08),handletextpad=.3)
ax.set_xlabel("method pairs reordered when the official metric changes")
panel(ax,"b",dx=-0.26,dy=1.05)
ax=fig.add_subplot(gs[1,0])
_CEN2=pd.read_csv("analysis/crosscrop/results/rank_census.csv").set_index("dataset")
for i,r in D.iterrows():
    ax.barh(i,1,height=.6,color="#e3e6e9"); ax.barh(i,r.ivl,height=.6,color=SPECIES[r.sp],alpha=.45)
    obs=float(_CEN2.loc[r.ds,"census_range_frac"])
    ax.barh(i,obs,height=.6,color=SPECIES[r.sp])
    ax.text(obs/2,i,f"{obs*100:.0f}%",ha="center",va="center",color="white",fontsize=FS,fontweight="bold")
    ax.text(r.ivl+.02,i,f"{r.ivl*100:.0f}%",va="center",fontsize=FS,color=SPECIES[r.sp])
_rows(ax); ax.set_xlim(0,1.14); ax.set_xticks([0,.5,1]); ax.set_xticklabels(["0","50%","all"])
ax.set_xlabel("rank span across the official metrics:\nobserved range (solid), interval (pale)")
panel(ax,"c",dx=-0.46,dy=1.05)
ax=fig.add_subplot(gs[1,1])
# what the design buys: the Gaussian value falls as the trial gets bigger (Table S11)
_TD=pd.read_csv("analysis/crosscrop/results/threshold_design.csv")
_TD=_TD[_TD.f==0.10]
_sh={25:"#9aa3ab",50:"#6b7a88",100:C_BLUE,400:INK}
for n,g in _TD.groupby("n_per_env"):
    g=g.sort_values("n_env")
    ax.plot(g.n_env,g.dr_star,"-o",color=_sh[n],ms=2.4,lw=1.0,zorder=2)
    ax.annotate(f"{n}",(g.n_env.iloc[-1],g.dr_star.iloc[-1]),xytext=(4,-1),
                textcoords="offset points",fontsize=FS,color=_sh[n],ha="left",va="center")
ax.plot(_MZ.n_env,_MZ.theory_lam05,"*",color=C_ADM,ms=8,zorder=4,mec="white",mew=.5)
ax.annotate(f"G2F design, {_MZ.theory_lam05:.3f}",(_MZ.n_env,_MZ.theory_lam05),xytext=(8.6,.0125),
            textcoords="data",fontsize=FS,color=C_ADM,ha="left",va="center",
            arrowprops=dict(arrowstyle="-",color=C_ADM,lw=.6,shrinkA=1,shrinkB=3))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8,560); ax.set_ylim(.0082,.25)
ax.set_xticks([10,20,50,100,200]); ax.set_xticklabels(["10","20","50","100","200"])
ax.set_yticks([.01,.02,.05,.1,.2]); ax.set_yticklabels(["0.01","0.02","0.05","0.1","0.2"])
ax.set_xlabel("environments"); ax.set_ylabel("Gaussian $\\Delta r^{*}$",labelpad=1)
ax.tick_params(axis="y",labelsize=FS); ax.tick_params(axis="x",labelsize=FS)
ax.text(225,.075,"genotypes\nper env.",fontsize=FS,color=MUTED,ha="left",va="bottom")
panel(ax,"d",dx=-0.44,dy=1.05)
ax=fig.add_subplot(gs[1,2]); XM=.45
for i,r in D.iterrows():
    c=SPECIES[r.sp]
    if r.ds=="maize":                                # five submissions: transferred, not estimated
        ax.barh(i,TR_HI-TR_LO,left=TR_LO,height=.42,color=c,alpha=.22,hatch="//////",edgecolor=c,lw=0)
        ax.plot(FLOOR,i,"D",color=c,ms=3.6,zorder=3)
        ax.text(TR_HI+.01,i,"transferred",va="center",fontsize=FS,color=c,style="italic")
        continue
    t=_TE.loc[r.ds]
    if np.isfinite(t.ci_hi):                         # upper limit identified: a closed interval
        ax.plot([t.ci_lo,t.ci_hi],[i,i],color=c,lw=1.0,zorder=2,solid_capstyle="butt")
        ax.plot([t.ci_hi]*2,[i-.13,i+.13],color=c,lw=1.0,zorder=2)
    else:                                            # upper limit beyond the observed range
        ax.annotate("",xy=(XM,i),xytext=(t.ci_lo,i),arrowprops=dict(arrowstyle="-|>",color=c,lw=1.0,
                    mutation_scale=6,shrinkA=0,shrinkB=0),zorder=2)
    ax.plot(t.threshold,i,"o",color=c,ms=4.4,zorder=3,mec="white",mew=.5)
    ax.text(t.threshold,i-.34,f"{t.threshold:.2f}",ha="center",fontsize=FS,color=c,fontweight="bold")
_rows(ax); ax.set_xlim(0,XM+.01)
ax.set_xlabel("threshold $\\Delta r^{*}$ (95% interval;\narrow: open above)")
panel(ax,"e",dx=-0.44,dy=1.05)
print("Fig3 width mm", round(save(fig,"Fig3",DOUBLE),1)); plt.close(fig)

# ---------------------------------------------------------------- Fig 5
TP=pd.read_csv("analysis/crosscrop/results/threshold_pairs.csv")
TC=pd.read_csv("analysis/crosscrop/results/threshold_theory_curve.csv")
_A8=pd.read_csv("analysis/crosscrop/results/heritability_ceiling_A8_bounds.csv")
_A8=_A8[_A8.dataset=="maize_G2F"].set_index("quantity").value
CEIL_LO,CEIL_HI=float(_A8["ceiling_n1"]),float(_A8["ceiling_n2"])   # single plot .. two-plot mean
FL_LO,FL_HI=float(min(_MZ.theory_lam0,_MZ.theory_lam05,_MZ.theory_lam08)),float(max(_MZ.theory_lam0,_MZ.theory_lam05,_MZ.theory_lam08))
_g=G.dropna(subset=["mean_pearson_r"]).sort_values("mean_pearson_r",ascending=False)
_r=_g["mean_pearson_r"].to_numpy()
WIN=float(_r[0]); GAP12=float(_r[0]-_r[1]); GAP13=float(_r[0]-_r[2])
fig=plt.figure(figsize=(DOUBLE,DOUBLE*0.34))
gs=fig.add_gridspec(1,3,width_ratios=[1,1.15,.85],wspace=.72)
ax=fig.add_subplot(gs[0])
bins=[0,.01,.02,.04,.07,.11,.16,.30]
_P={"common bean":"bean","spring wheat":"spring wheat","soybean":"soybean"}
for k,(ds,sp) in enumerate(_P.items()):
    s_=TP[TP.dataset==ds]; c=SPECIES[sp]; xs=[];ps=[];es=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        b=s_[(s_.dr>=lo)&(s_.dr<hi)]
        if len(b)<8: continue
        p=(b.dg>0).mean(); xs.append(b.dr.mean()); ps.append(p); es.append(np.sqrt(p*(1-p)/len(b)))
    ax.errorbar(np.array(xs)+(k-1)*.002,ps,yerr=es,fmt="o",color=c,ms=3.0,lw=.9,capsize=1.6,zorder=3)
    cv=TC[TC.dataset==ds]; ax.plot(cv.dr,cv.p_observed,color=c,lw=1.1,zorder=2,label=sp)
cv=TC[TC.dataset=="maize (G2F design)"]
ax.plot(cv.dr,cv.p_gaussian,ls=(0,(2.2,1.6)),color=INK,lw=1.15,zorder=2,label="Gaussian model,\nG2F design")
ax.axhline(.95,color=INK,ls=(0,(3.5,2)),lw=.7); ax.axhline(.5,color=MUTED,ls=":",lw=.85)
ax.text(.285,.915,"95%",fontsize=FS,color=MUTED,ha="right"); ax.text(.285,.51,"chance",fontsize=FS,color=MUTED,ha="right",va="bottom")
ax.set_xlim(0,.29); ax.set_ylim(.22,1.03)
ax.set_xlabel("accuracy gap $\\Delta r$ between two methods")
ax.set_ylabel("P(more accurate method\nselects better)")
ax.legend(loc="lower right",bbox_to_anchor=(1.03,-.02),handlelength=1.8,labelspacing=.3)
panel(ax,"a",dx=-0.30,dy=1.03)
ax=fig.add_subplot(gs[1])
GG=_g.head(15)
rr=GG["mean_pearson_r"].to_numpy(); nn=GG["Team Name"].to_numpy(); top=rr[0]
def _col(v): return C_ADM if v>top-FLOOR else (C_COND if v>top-TR_HI else "#cfd4d9")
for i,(v,n2) in enumerate(zip(rr,nn)):
    ax.plot([0,v],[i,i],color=_col(v),lw=1.1,alpha=.6)
    ax.plot(v,i,"o",color=_col(v) if v>top-TR_HI else C_BLUE,ms=4,zorder=3)
ax.axvspan(top-TR_HI,top+.02,color=C_COND,alpha=.07,zorder=0)
ax.axvspan(top-FLOOR,top+.02,color=C_ADM,alpha=.08,zorder=0)
ax.plot([top-FLOOR]*2,[-.6,14.6],color=C_ADM,lw=1.1)
ax.plot([top-TR_HI]*2,[-.6,14.6],color=C_COND,lw=1.0,ls=(0,(3,1.5)))
# resolution is a property of each pair of teams, not of consecutive gaps: count the
# pairs across the whole board that lie closer than the design's Gaussian value
_pd=np.abs(_r[:,None]-_r[None,:])[np.triu_indices(len(_r),1)]
ax.set_yticks(range(15)); ax.set_yticklabels([f"{i+1}. {n2}" for i,n2 in enumerate(nn)],fontsize=FS)
for i,t in enumerate(ax.get_yticklabels()): t.set_color(C_ADM if rr[i]>top-FLOOR else (C_ORANGE if rr[i]>top-TR_HI else MUTED))
ax.invert_yaxis(); ax.set_xlim(0,.40); ax.set_ylim(17.4,-.7)
ax.set_xlabel("reported mean within-environment $r$")
ax.text(.005,17.1,f"≤ {FLOOR:.3f} behind the leader (Gaussian value): {int((rr>top-FLOOR).sum())} teams\n"
        f"≤ {TR_HI:.2f} behind the leader (transferred): {int((_r>top-TR_HI).sum())} teams\n"
        f"team pairs closer than {FLOOR:.3f}: {int((_pd<FLOOR).sum())} of {len(_pd)}",
        fontsize=FS,color=INK,va="bottom")
panel(ax,"b",dx=-0.34,dy=1.03)
ax=fig.add_subplot(gs[2])
items=[("1st vs 2nd",GAP12,C_INADM),("1st vs 3rd",GAP13,C_INADM),
       ("Gaussian value\n(G2F design)",(FL_LO,FL_HI),C_ADM),("transferred\nthreshold",(TR_LO,TR_HI),C_COND),
       ("winner's score",WIN,C_BLUE),("ceiling\n(1–2 plots)",(CEIL_LO,CEIL_HI),C_GREEN)]
for i,(l,v,c) in enumerate(items):
    if isinstance(v,tuple):
        ax.barh(i,v[0],height=.6,color=c)
        ax.barh(i,v[1]-v[0],left=v[0],height=.6,color=c,alpha=.35)
        f=(lambda x:f"{x:.3f}") if v[1]<.1 else (lambda x:f"{x:.2f}")
        ax.text(v[1]+.012,i,f"{f(v[0])}–{f(v[1])}",va="center",fontsize=FS,color=c)
        continue
    ax.barh(i,v,height=.6,color=c)
    ax.text(v+.012,i,f"{v:.3f}",va="center",fontsize=FS,color=c if c!=C_INADM else INK)
ax.set_yticks(range(len(items))); ax.set_yticklabels([l for l,_,_ in items],fontsize=FS)
ax.invert_yaxis(); ax.set_xlim(0,.95); ax.set_xlabel("within-environment $r$")
panel(ax,"c",dx=-0.52,dy=1.03)
print("Fig5 width mm", round(save(fig,"Fig5",DOUBLE),1)); plt.close(fig)
print("Fig3, Fig5 written")
