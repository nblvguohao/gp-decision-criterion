"""Shared publication style for The Crop Journal figures.

The journal's guide for authors asks for Arial, Courier, Times New Roman or Symbol,
uniform lettering, artwork sized close to the published dimensions and colour that is
accessible with impaired colour vision; Elsevier's artwork sizes are 90 mm (single
column), 140 mm (1.5 column) and 190 mm (double column). So: Arial throughout, no text
below 7 pt at the printed size, and save() iterates the figure width until the exported
page is exactly the column width, so the file is never rescaled in production."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
MM=1/25.4
SINGLE, ONEHALF, DOUBLE = 90*MM, 140*MM, 190*MM
INK="#1a1a1a"; MUTED="#6b6b6b"; RULE="#d5d5d5"
# a restrained, colour-blind-safe set
C_ADM   = "#B2182B"   # admissible / highlighted
C_COND  = "#EF8A62"   # conditionally admissible
C_INADM = "#9aa3ab"   # inadmissible
C_BLUE  = "#2166AC"
C_GREEN = "#1B7837"
C_ORANGE= "#C26A1B"
C_PURPLE= "#762A83"
SPECIES = {"maize":C_ADM,"bean":C_PURPLE,"spring wheat":"#E08214","soybean":C_BLUE}
def apply():
    plt.rcParams.update({
        "font.family":"Arial","font.size":7,"mathtext.fontset":"custom","mathtext.rm":"Arial",
        "mathtext.it":"Arial:italic","mathtext.bf":"Arial:bold","mathtext.sf":"Arial",
        "axes.labelsize":7.5,"axes.titlesize":8,"axes.titleweight":"bold",
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,"pdf.fonttype":42,
        "axes.edgecolor":INK,"axes.linewidth":.7,
        "xtick.color":INK,"ytick.color":INK,"text.color":INK,"axes.labelcolor":INK,
        "xtick.major.width":.7,"ytick.major.width":.7,
        "xtick.major.size":2.6,"ytick.major.size":2.6,
        "axes.spines.top":False,"axes.spines.right":False,
        "figure.dpi":180,"savefig.dpi":600,"savefig.bbox":"tight",
        "savefig.pad_inches":0.02,"legend.frameon":False,
        "axes.grid":False,"lines.linewidth":1.1,
    })
def panel(ax,letter,title=None,dx=-0.13,dy=1.06):
    ax.text(dx,dy,letter,transform=ax.transAxes,fontsize=9,fontweight="bold",
            va="bottom",ha="left")
    if title: ax.set_title(title,loc="left",pad=6)

FS=7          # the smallest text on any figure, in points at the printed size


def save(fig,name,width,tol=0.2,tight=True):
    """Write analysis/tcj_figures/<name>.pdf/.png with the tight page exactly `width` inches wide.
    Text is in points, so widening the canvas gives the axes more room without rescaling any
    lettering; a few iterations make the tight bounding box land on the column width."""
    if not tight:                        # laid out in millimetres by the caller: keep the page as is
        for ext in ("pdf","png"):
            fig.savefig(f"analysis/tcj_figures/{name}.{ext}",bbox_inches=None)
        return fig.get_size_inches()[0]*25.4
    for _ in range(8):
        fig.canvas.draw()
        bb=fig.get_tightbbox(fig.canvas.get_renderer())
        got=bb.width+2*plt.rcParams["savefig.pad_inches"]
        if abs(got-width)*25.4<tol: break
        w,h=fig.get_size_inches(); fig.set_size_inches(w+(width-got),h)
    for ext in ("pdf","png"):
        fig.savefig(f"analysis/tcj_figures/{name}.{ext}")
    return got*25.4


def mm_axes(fig,x0,y0,w,h,**kw):
    """axes placed in millimetres from the lower-left corner of the page"""
    W,H=fig.get_size_inches()*25.4
    return fig.add_axes([x0/W,y0/H,w/W,h/H],**kw)


def mm_text(fig,x,y,s,**kw):
    W,H=fig.get_size_inches()*25.4
    return fig.text(x/W,y/H,s,**kw)
