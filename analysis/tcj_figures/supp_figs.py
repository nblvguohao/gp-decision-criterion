import sys, itertools; sys.path.insert(0,"analysis/tcj_figures")
import numpy as np, pandas as pd
from style import *
apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
R="analysis/g2f_leaderboard/results"
KEY="within-env x discrimination"
C_WD="#01665E"   # within-env discrimination: teal, so that red keeps its main-text meaning (Tier I)
COL={KEY:C_WD,"pooled x discrimination":"#80CDC1",
     "within-env x error-magnitude":C_BLUE,"pooled x error-magnitude":"#b9c0c7"}

# ---------------- Fig S1 : leave-one-metric-out coverage
C=pd.read_csv(f"{R}/coverage_by_metric_class.csv").sort_values("coverage").reset_index(drop=True)
fig,ax=plt.subplots(figsize=(ONEHALF,ONEHALF*0.62))
cols=[COL[c] for c in C.cell]
ax.barh(range(len(C)),C.coverage,color=cols,height=.72)
lo=C[C.cell==KEY].coverage.max(); hi=C[C.cell!=KEY].coverage.min()
ax.axvspan(lo,hi,color=C_WD,alpha=.08,zorder=0)
ax.text((lo+hi)/2,(C.cell==KEY).sum()/2-.5,"no metric\nlands here",fontsize=6,ha="center",va="center",zorder=4,
        color=C_WD,style="italic",rotation=90)
ax.axvline(.95,color=INK,ls=(0,(3.5,2)),lw=.85)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.metric,fontsize=5.8)
for t,c in zip(ax.get_yticklabels(),cols): t.set_color(c if c!="#b9c0c7" else MUTED)
ax.set_xlim(0,1.03); ax.set_ylim(-.8,len(C)-.3)
ax.set_xlabel("leave-one-metric-out coverage of the 95% rank interval")
i24=int(C.index[C.metric=="mean_pearson_r"][0])
ax.text(.015,i24,"official ranking metric of the 2024 edition",fontsize=6,color="white",fontweight="bold",
        va="center",ha="left")
ax.legend(handles=[Line2D([],[],marker="s",ls="",color=COL[k],ms=5,
          label=f"{k.replace(' x ',' × ')}  (n={int((C.cell==k).sum())})") for k in
          (KEY,"pooled x discrimination","within-env x error-magnitude","pooled x error-magnitude")],
          fontsize=5.8,loc="upper center",bbox_to_anchor=(.5,-.155),ncol=2)
fig.savefig("analysis/tcj_figures/FigS1.png"); fig.savefig("analysis/tcj_figures/FigS1.pdf"); plt.close(fig)

# ---------------- Fig S2 : the 2x2 taxonomy
fig,ax=plt.subplots(figsize=(SINGLE,SINGLE*0.85))
cells=[("pooled","error-magnitude"),("pooled","discrimination"),
       ("within-env","error-magnitude"),("within-env","discrimination")]
for a,t in cells:
    sub=C[(C.aggregation==a)&(C.type==t)]
    x=0 if t=="error-magnitude" else 1; y=0 if a=="pooled" else 1
    k=f"{a} x {t}"; key=(k==KEY)
    ax.add_patch(plt.Rectangle((x-.46,y-.42),.92,.84,facecolor=COL[k],
                 alpha=.95 if key else .28,edgecolor=INK,lw=.8))
    tc="white" if key else INK
    ax.text(x,y+.19,f"n = {len(sub)}",ha="center",fontsize=7,color=tc,fontweight="bold")
    ax.text(x,y-.02,f"coverage\n{sub.coverage.min():.2f} – {sub.coverage.max():.2f}",
            ha="center",fontsize=6.2,color=tc)
    if key: ax.text(x,y-.28,"what selection\nactually needs",ha="center",fontsize=5.8,
                    color="white",style="italic")
ax.set_xticks([0,1]); ax.set_xticklabels(["error magnitude\n(RMSE, MAE, $R^2_{score}$)",
                                          "discrimination\n($r$, $r^2$, $\\rho$, slope)"],fontsize=6.4)
ax.set_yticks([0,1]); ax.set_yticklabels(["pooled across\nenvironments","within\nenvironment"],fontsize=6.4)
ax.set_xlim(-.62,1.62); ax.set_ylim(-.62,1.62); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
fig.savefig("analysis/tcj_figures/FigS2.png"); fig.savefig("analysis/tcj_figures/FigS2.pdf"); plt.close(fig)

# ---------------- Fig S3 : reversal between all metric pairs
G=pd.read_csv(f"{R}/S1_best_per_team.csv"); G=G[G["Team Name"].notna()]
LOWER={"MAE","realative_MAE","normalized_MAE","RMSE","relative_RMSE","normalized_RMSE",
       "mean_MAE","mean_realative_MAE","mean_normalized_MAE","mean_RMSE","mean_relative_RMSE","mean_normalized_RMSE"}
SLOPE={"lineRegressSlope","mean_lineRegressSlope"}
MET=[c for c in G.columns if c not in ("id","Team Name","Submitted At")]
M=len(G); pairs=list(itertools.combinations(range(M),2))
def orient(c):
    v=G[c].astype(float).to_numpy()
    return -np.abs(v-1.) if c in SLOPE else (-v if c in LOWER else v)
S={m:orient(m) for m in MET}
X=np.zeros((len(MET),)*2)
for i,a in enumerate(MET):
    for j,b in enumerate(MET):
        if i<j:
            d1=np.array([S[a][p]-S[a][q] for p,q in pairs]); d2=np.array([S[b][p]-S[b][q] for p,q in pairs])
            ok=np.isfinite(d1)&np.isfinite(d2); X[i,j]=X[j,i]=(d1[ok]*d2[ok]<0).mean()
order=leaves_list(linkage(squareform(X,checks=False),method="average"))
lab=[MET[i] for i in order]
WITHIN={"mean_pearson_r","mean_spearman_r","mean_lineRegressSlope","mean_r2_pearson"}
fig,ax=plt.subplots(figsize=(ONEHALF,ONEHALF*0.92))
im=ax.imshow(X[np.ix_(order,order)],cmap="RdYlBu_r",vmin=0,vmax=.4)
ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
cc=[C_WD if l in WITHIN else MUTED for l in lab]
ax.set_xticklabels(lab,rotation=90,fontsize=5.2); ax.set_yticklabels(lab,fontsize=5.2)
for t,c in zip(ax.get_xticklabels(),cc): t.set_color(c)
for t,c in zip(ax.get_yticklabels(),cc): t.set_color(c)
cb=plt.colorbar(im,ax=ax,shrink=.6,pad=.02); cb.set_label("pairwise reversal rate",fontsize=6.4)
cb.ax.tick_params(labelsize=5.8)
ax.tick_params(length=1.5)
fig.savefig("analysis/tcj_figures/FigS3.png"); fig.savefig("analysis/tcj_figures/FigS3.pdf"); plt.close(fig)
print("FigS1-S3 written")
