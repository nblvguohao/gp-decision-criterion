"""Fig 1 — the decision, the criterion, and what fails it.

(a) schematic of within-environment truncation selection and the two tests the criterion
    applies. Everything in it, including the field of plots, is drawn here from explicit
    values -- no external artwork is used; every bar is drawn so that each list is strictly sorted and the colours track
    genotype identity: dark red, the two genotypes the method advances; orange, the two it
    ranks last.
(b)-(d) the invariance and order tests on the 22 official metrics, on decision-oriented
    measures, and the counterexample of the squared correlation.
"""
import sys; sys.path.insert(0,"analysis/tcj_figures")
import numpy as np, pandas as pd
from style import *
apply()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AFF=pd.read_csv("analysis/g2f_leaderboard/results/invariance_test.csv")
aff={r.metric:r["median"] for _,r in AFF.iterrows()}
MON=pd.read_csv("analysis/g2f_leaderboard/results/monotone_test.csv")
MONO={r.metric:r["median"] for _,r in MON.iterrows()}
LO=1e-18; TOL=1e-3
TCOL={"I":C_ADM,"II":C_COND,None:C_INADM}


def draw(ax,rows,xlabel,zero_label=True):
    n=len(rows); ys=np.arange(n)[::-1]; zeros=[]
    for i,(lab,a,m,t) in zip(ys,rows):
        col=TCOL[t]; A=max(a,LO); M=max(m,LO)
        if a==0 and m==0: zeros.append(i)
        if A<M*0.5:
            ax.plot([A,M],[i,i],color=col,lw=1.3,alpha=.5,zorder=2,solid_capstyle="round")
        off=.16 if abs(np.log10(max(M,LO)/max(A,LO)))<0.7 else 0.
        ax.plot(A,i-off,"o",color=col,ms=4.4,zorder=4,mec="white",mew=.5,clip_on=False)
        ax.plot(M,i+off,"o",color="white",ms=4.4,zorder=4,mec=col,mew=1.2,clip_on=False)
    ax.axvline(TOL,color=INK,ls=(0,(3.5,2)),lw=.85,zorder=1)
    ax.set_xscale("log"); ax.set_xlim(LO,3e2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows])
    for tk,(lab,a,m,t) in zip(ax.get_yticklabels(),rows):
        tk.set_color(C_ADM if t=="I" else (C_ORANGE if t=="II" else MUTED))
    ax.set_ylim(-.7,n-.3); ax.set_xlabel(xlabel)
    ax.set_xticks([1e-18,1e-14,1e-10,1e-6,1e-2,1e2])
    if zeros and zero_label:            # exactly zero under both tests: said once, beside the points
        ax.text(LO*40,(min(zeros)+max(zeros))/2,"exactly 0",fontsize=FS,color=C_ADM,
                va="center",ha="left",fontweight="bold")


W,H=190,190                                             # the page, in millimetres
fig=plt.figure(figsize=(W*MM,H*MM))
SA=64                                                   # height of the schematic strip

# ------------------------------------------------------------------ (a) schematic
ax=mm_axes(fig,0,H-SA,W,SA); ax.set_xlim(0,W); ax.set_ylim(0,SA); ax.axis("off")   # 1 unit = 1 mm
# The field: 20 plots (4 rows x 5), one genotype each, drawn here -- no external artwork.
fw=27; fh=fw*469/418
ax.add_patch(FancyBboxPatch((3,16),fw,fh,boxstyle="round,pad=0,rounding_size=1.5",
                            fc="#eee4cf",ec=RULE,lw=.7,zorder=1))
_rf=np.random.default_rng(11)
_nx,_ny,_g=5,4,1.2
_pw=(fw-_g*(_nx+1))/_nx; _pl=(fh-_g*(_ny+1))/_ny
_greens=["#8fbf6a","#7fb35c","#9cc877","#76a955","#a8cf85"]
for _ix in range(_nx):
    for _iy in range(_ny):
        _x=3+_g+_ix*(_pw+_g); _y=16+_g+_iy*(_pl+_g)
        ax.add_patch(plt.Rectangle((_x,_y),_pw,_pl,fc=_greens[_rf.integers(len(_greens))],
                                   ec="#5d8a40",lw=.4,zorder=2))
        for _k in range(3):                               # three rows of plants per plot
            _xx=_x+_pw*(_k+1)/4
            ax.plot([_xx,_xx],[_y+_pl*.14,_y+_pl*.86],color="#3f6b2a",lw=.6,zorder=3,
                    solid_capstyle="round")
ax.text(3+fw/2,13.5,"20 genotypes,\none plot each",ha="center",va="top",fontsize=FS,color=INK)
HEAD=dict(fontsize=7.5,fontweight="bold",va="bottom",ha="left",color=INK)
ax.text(6,59,"One environment",**HEAD)
ax.text(42,59,"Predict and rank",**HEAD)
ax.text(92,59,"Change the predictions",**HEAD)
ax.text(151.5,59,"The criterion",**HEAD)

n=20; v=0.25+0.75*(1-np.arange(n)/(n-1))**1.15          # predicted values, sorted, highest first
ident=np.array(["top"]*2+["mid"]*(n-4)+["low"]*2)        # genotype identity in that first order
COL={"top":C_ADM,"mid":C_BLUE,"low":C_COND}

def bars(x0,ytop,vals,ids,L,step,th,label=None):
    o=np.argsort(-vals,kind="mergesort")                 # every list is sorted by its own values
    for j,k in enumerate(o):
        y=ytop-j*step
        ax.add_patch(plt.Rectangle((x0,y-th/2),L*vals[k],th,color=COL[ids[k]],lw=0,zorder=2))
    ax.plot([x0,x0],[ytop+step*.6,ytop-(n-1)*step-step*.6],color=INK,lw=.7,zorder=3)
    if label:
        ax.text(x0+L*vals[o[0]]+1.2,ytop-step/2,label,fontsize=FS,va="center",ha="left",
                color=COL[ids[o[0]]])

bars(42,54,v,ident,24,2.5,1.85,"advanced\n(top 10%)")
# A: an order-preserving change (shift, rescale and bend): same order, different lengths
vA=0.18+0.62*v**2.4
bars(95,53,vA,ident,24,1.15,0.86,"advanced")
ax.text(95,54.6,"A  order kept (shift, rescale, bend)",fontsize=FS,va="bottom",color=INK)
# B: the ranking reversed
vB=1.18-v
bars(95,24.5,vB,ident,24,1.15,0.86,"advanced")
ax.text(95,26.1,"B  order reversed",fontsize=FS,va="bottom",color=INK)

def arrow(x0,y0,x1,y1):
    ax.add_patch(FancyArrowPatch((x0,y0),(x1,y1),arrowstyle="-|>",mutation_scale=8,
                                 color=INK,lw=.9,shrinkA=0,shrinkB=0,zorder=3))
arrow(32,30,39.5,30)
arrow(76,30,84,30); ax.plot([84,84],[13,41],color=INK,lw=.9)
arrow(84,41,92.5,41); arrow(84,13,92.5,13)

# verdict marks, drawn as strokes so that no symbol font is needed
def check(x,y,s=2.2,c=C_BLUE):
    ax.plot([x-s,x-s*.35,x+s*1.2],[y,y-s*.9,y+s*1.1],color=c,lw=2.2,solid_capstyle="round",zorder=4)
def cross(x,y,s=1.8,c=C_ADM):
    ax.plot([x-s,x+s],[y-s,y+s],color=c,lw=2.2,solid_capstyle="round",zorder=4)
    ax.plot([x-s,x+s],[y+s,y-s],color=c,lw=2.2,solid_capstyle="round",zorder=4)
check(137,44); ax.text(137,39.6,"same genotypes\nadvanced",ha="center",va="top",fontsize=FS)
cross(137,16.5); ax.text(137,12.2,"other genotypes\nadvanced",ha="center",va="top",fontsize=FS)

ax.add_patch(FancyBboxPatch((152,13),36,39,boxstyle="round,pad=0.6,rounding_size=2",
                            fc="#f6f6f6",ec=RULE,lw=.7,zorder=1))
ax.text(170,48.5,"A metric can represent\nthe decision only if it is",ha="center",va="top",fontsize=FS)
ax.text(155,37.5,"unchanged under A",ha="left",va="center",fontsize=FS,color=C_BLUE,fontweight="bold")
ax.text(155,32.0,"and",ha="left",va="center",fontsize=FS)
ax.text(155,26.5,"changed under B",ha="left",va="center",fontsize=FS,color=C_ADM,fontweight="bold")
ax.text(170,18.5,"(tested in b and c)",ha="center",va="center",fontsize=FS,color=MUTED)


# ------------------------------------------------------------------ (b) the 22 official metrics
ax=mm_axes(fig,40,27,62,H-SA-27-9)
PRETTY={"mean_spearman_r":"within-env Spearman $\\rho$","mean_pearson_r":"within-env Pearson $r$",
 "mean_r2_pearson":"within-env $r^{2}$","mean_lineRegressSlope":"within-env slope",
 "mean_RMSE":"within-env RMSE","mean_MAE":"within-env MAE","mean_r2_score":"within-env $R^{2}_{score}$",
 "mean_normalized_RMSE":"within-env normalised RMSE","mean_relative_RMSE":"within-env relative RMSE",
 "mean_normalized_MAE":"within-env normalised MAE","mean_realative_MAE":"within-env relative MAE",
 "spearman_r":"pooled Spearman $\\rho$","pearson_r":"pooled Pearson $r$","r2_pearson":"pooled $r^{2}$",
 "lineRegressSlope":"pooled slope","RMSE":"pooled RMSE","MAE":"pooled MAE","r2_score":"pooled $R^{2}_{score}$",
 "normalized_RMSE":"pooled normalised RMSE","relative_RMSE":"pooled relative RMSE",
 "normalized_MAE":"pooled normalised MAE","realative_MAE":"pooled relative MAE"}
# the tiers are those of partition.py (identical in every dataset); the squared correlations
# are unmoved by rescaling but score a ranking and its reverse alike, so they are Tier III
_PM=pd.read_csv("analysis/crosscrop/results/partition_metrics.csv")
_PM=_PM[_PM.dataset=="maize"].set_index("metric")
TIER={m:(t if t in ("I","II") else None) for m,t in _PM.tier.items()}
BLIND=set(_PM.index[~_PM.order_sensitive])
R1=[(PRETTY.get(m,m)+(" $^{\\dagger}$" if m in BLIND else ""),aff.get(m,np.nan),MONO[m],TIER.get(m))
    for m in MON.sort_values("median").metric]
draw(ax,R1,"median relative change in the metric")
ax.text(TOL*2.0,len(R1)-0.45,"tolerance",fontsize=FS,color=MUTED,va="center")


# ------------------------------------------------------------------ (c) decision-oriented measures
ax=mm_axes(fig,137,H-SA-9-27,49,27)
DEC=pd.read_csv("analysis/g2f_leaderboard/results/decision_metrics_invariance.csv")
DEC=DEC[DEC.dataset.str.startswith("maize")].set_index("metric")
_L={"SG":("selection differential","I"),"hit":("top-$k$ hit rate","I"),
    "NDCG":("NDCG@$k$","I"),"kendall":("Kendall's $\\tau$","I"),
    "pearson":("within-env Pearson $r$","II"),"RMSE":("RMSE",None)}
R2=[(_L[m][0],float(DEC.loc[m,"affine"]),float(DEC.loc[m,"monotone"]),_L[m][1])
    for m in ["SG","hit","NDCG","kendall","pearson","RMSE"]]
draw(ax,R2,"median relative change in the metric")


# ------------------------------------------------------------------ (d) the counterexample
ax=mm_axes(fig,122,27,64,50)
CE=pd.read_csv("analysis/crosscrop/results/counterexample_ursn.csv")
rev=CE.reversed.astype(bool)
ax.scatter(CE.mean_pearson_r[~rev],CE.mean_r2_pearson[~rev],s=26,color=C_BLUE,lw=0,alpha=.85,zorder=3)
ax.scatter(CE.mean_pearson_r[rev],CE.mean_r2_pearson[rev],s=26,color=MUTED,lw=0,alpha=.8,zorder=3)
x=np.linspace(-.55,.55,200)
ax.plot(x,x**2,ls=(0,(2.5,1.8)),color=MUTED,lw=.9,zorder=1)
w=CE[~rev].sort_values("mean_r2_pearson",ascending=False).iloc[0]
v=CE[rev & (CE.method==w.method)].iloc[0]
ax.plot([v.mean_pearson_r,w.mean_pearson_r],[w.mean_r2_pearson]*2,color=C_ADM,lw=.8,ls=(0,(1,1.5)),zorder=2)
for q in (w,v):
    ax.plot(q.mean_pearson_r,q.mean_r2_pearson,"o",mfc="none",mec=C_ADM,ms=9,mew=1.2,zorder=4)
ax.text(0,w.mean_r2_pearson+.012,(f"{w.method} and its reverse: same $r^{{2}}$ ({w.mean_r2_pearson:.2f})\n"
        f"gain {w.SG_f10:+.2f} SD against {v.SG_f10:+.2f} SD").replace("-", "\u2212"),
        fontsize=FS,color=C_ADM,va="bottom",ha="center")
ax.text(.30,.035,"as fitted:\nselect better\nthan average",color=C_BLUE,fontsize=FS,ha="left",va="center")
ax.text(-.30,.035,"reversed:\nselect worse\nthan average",color=MUTED,fontsize=FS,ha="right",va="center")
ax.axvline(0,color=INK,lw=.6,ls=":",zorder=0)
ax.set_xlabel("within-environment Pearson $r$",labelpad=1)
ax.set_ylabel("within-environment $r^{2}$",labelpad=1)
ax.set_xlim(-.55,.55); ax.set_ylim(0,max(.27,w.mean_r2_pearson*1.35))
# shared keys under panel b: the two tests, then the tiers
fig.legend(handles=[Line2D([],[],marker="o",ls="",color=MUTED,ms=4.4,mec="white",label="affine rescaling"),
                    Line2D([],[],marker="o",ls="",color="white",ms=4.4,mec=MUTED,mew=1.2,label="any monotone transform"),
                    Line2D([],[],marker="s",ls="",color=C_ADM,ms=5,label="Tier I  admissible for the decision"),
                    Line2D([],[],marker="s",ls="",color=C_COND,ms=5,label="Tier II  unmoved by calibration only"),
                    Line2D([],[],marker="s",ls="",color=C_INADM,ms=5,label="Tier III  inadmissible"),
                    Line2D([],[],ls="",label="$^{\\dagger}$ scores a ranking and its reverse alike")],
           loc="lower center",bbox_to_anchor=(.5,0.0),ncol=3,handletextpad=.35,columnspacing=1.6)
for L,x,y in [("a",1,H-1),("b",1,H-SA-3),("c",108,H-SA-3),("d",108,H-SA-9-27-17)]:
    mm_text(fig,x,y,L,fontsize=9,fontweight="bold",va="top",ha="left")
print("Fig1 width mm", round(save(fig,"Fig1",DOUBLE,tight=False),1))
