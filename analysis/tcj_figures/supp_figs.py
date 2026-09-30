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
fig,ax=plt.subplots(figsize=(ONEHALF,ONEHALF*1.15))
cols=[COL[c] for c in C.cell]
ax.barh(range(len(C)),C.coverage,color=cols,height=.72)
lo=C[C.cell==KEY].coverage.max(); hi=C[C.cell!=KEY].coverage.min()
ax.axvspan(lo,hi,color=C_WD,alpha=.08,zorder=0)
ax.text((lo+hi)/2,(C.cell==KEY).sum()/2-.5,"no metric\nlands here",fontsize=FS,ha="center",va="center",zorder=4,
        color=C_WD,style="italic",rotation=90)
ax.axvline(.95,color=INK,ls=(0,(3.5,2)),lw=.85)
ax.set_yticks(range(len(C))); ax.set_yticklabels(C.metric,fontsize=FS)
for t,c in zip(ax.get_yticklabels(),cols): t.set_color(c if c!="#b9c0c7" else MUTED)
ax.set_xlim(0,1.03); ax.set_ylim(-.8,len(C)-.3)
ax.set_xlabel("leave-one-metric-out coverage of the 95% rank interval")
i24=int(C.index[C.metric=="mean_pearson_r"][0])
ax.text(.015,i24,"official ranking metric of the 2024 edition",fontsize=FS,color="white",fontweight="bold",
        va="center",ha="left")
ax.legend(handles=[Line2D([],[],marker="s",ls="",color=COL[k],ms=5,
          label=f"{k.replace(' x ',' × ')}  (n={int((C.cell==k).sum())})") for k in
          (KEY,"pooled x discrimination","within-env x error-magnitude","pooled x error-magnitude")],
          fontsize=FS,loc="upper center",bbox_to_anchor=(.5,-.155),ncol=2)
print("FigS1 width mm", round(save(fig,"FigS1",ONEHALF),1)); plt.close(fig)

# ---------------- Fig S2 : the 2x2 taxonomy
fig,ax=plt.subplots(figsize=(SINGLE,SINGLE*0.98))
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
            ha="center",fontsize=FS,color=tc)
    if key: ax.text(x,y-.28,"what selection\nactually needs",ha="center",fontsize=FS,
                    color="white",style="italic")
ax.set_xticks([0,1]); ax.set_xticklabels(["error magnitude\n(RMSE, MAE, $R^2_{score}$)",
                                          "discrimination\n($r$, $r^2$, $\\rho$, slope)"],fontsize=FS)
ax.set_yticks([0,1]); ax.set_yticklabels(["pooled across\nenvironments","within\nenvironment"],fontsize=FS)
ax.set_xlim(-.62,1.62); ax.set_ylim(-.62,1.62); ax.tick_params(length=0)
for sp in ax.spines.values(): sp.set_visible(False)
print("FigS2 width mm", round(save(fig,"FigS2",SINGLE),1)); plt.close(fig)

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
fig,ax=plt.subplots(figsize=(ONEHALF,ONEHALF*1.15))
im=ax.imshow(X[np.ix_(order,order)],cmap="RdYlBu_r",vmin=0,vmax=.4)
ax.set_xticks(range(len(lab))); ax.set_yticks(range(len(lab)))
cc=[C_WD if l in WITHIN else MUTED for l in lab]
ax.set_xticklabels([f"{i+1}  {l}" for i,l in enumerate(lab)],rotation=90,fontsize=FS); ax.set_yticklabels([str(i+1) for i in range(len(lab))],fontsize=FS)
for t,c in zip(ax.get_xticklabels(),cc): t.set_color(c)
for t,c in zip(ax.get_yticklabels(),cc): t.set_color(c)   # row i = column i (numbers in the x labels)
cb=plt.colorbar(im,ax=ax,shrink=.6,pad=.02); cb.set_label("pairwise reversal rate",fontsize=FS)
cb.ax.tick_params(labelsize=FS)
ax.tick_params(length=1.5)
print("FigS3 width mm", round(save(fig,"FigS3",ONEHALF),1)); plt.close(fig)
print("FigS1-S3 written")

# ---------------- Fig S5 : the GPverdict web page running its bundled example (formerly Fig. 6A)
# Screenshots of the published page (assets/capture_web.py), unaltered; the report is cropped to its
# verdict section, as it was in Fig. 6A. They are raster images, so their lettering is outside the 7 pt
# rule, which applies to drawn text; Fig. 6A now shows the same verdict as vector text. Stacked at the
# largest width that keeps the page within 234 mm; embedded at native resolution (interpolation "none").
from PIL import Image
IMG_IN=Image.open("analysis/tcj_figures/assets/gpverdict_web_input.png").convert("RGB")
IMG_REP=Image.open("analysis/tcj_figures/assets/gpverdict_web_report.png").convert("RGB")
IMG_REP=IMG_REP.crop((0,0,IMG_REP.width,1200))                 # the report down to the end of its verdict
W5,HMAX,GAP5,XL=190,234,4,6                                     # page width, height cap, gap, room for the letters (mm)
asp=[im.height/im.width for im in (IMG_IN,IMG_REP)]
wimg=min(W5-XL,(HMAX-GAP5-1)/sum(asp)); H5=wimg*sum(asp)+GAP5+1
fig=plt.figure(figsize=(W5*MM,H5*MM)); x0=XL+(W5-XL-wimg)/2; y=H5-.5
for L,im,a in zip("AB",(IMG_IN,IMG_REP),asp):
    ax=mm_axes(fig,x0,y-wimg*a,wimg,wimg*a)
    ax.imshow(np.asarray(im),interpolation="none",aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(True); sp.set_color(RULE); sp.set_linewidth(.6)
    mm_text(fig,x0-5,y,L,fontsize=9,fontweight="bold",va="top",ha="left")
    y-=wimg*a+GAP5
print("FigS5 width mm", round(save(fig,"FigS5",W5*MM,tight=False),1), "height mm", round(H5,1)); plt.close(fig)
