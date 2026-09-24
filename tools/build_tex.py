# -*- coding: utf-8 -*-
"""Build an Elsevier elsarticle submission from the markdown master.

Citations and maths are swapped for alphanumeric placeholders before pandoc runs
and restored afterwards, so pandoc never gets the chance to escape a backslash or
to refuse a `$...$` that abuts a digit."""
import io, re, sys, subprocess, os
sys.path.insert(0, "tools")
from refs import REFS
from md2tex_prep import RULES, SUP

SRC = "output/manuscript_TCJ.md"
OUT = "submission"
PANDOC = os.path.expanduser("~/.claude-science/conda/envs/tex/bin/pandoc")
s = io.open(SRC, encoding="utf-8").read()

# ---- number -> key, recovered from the emitted reference list --------------
reflist = s.split("## References", 1)[1].split("\n## ", 1)[0]
num2key = {}
for m in re.finditer(r"^\[(\d+)\]\s+(.+?)\s*$", reflist, re.M):
    n, text = int(m.group(1)), m.group(2)
    hit = [k for k, v in REFS.items() if v[:60] == text[:60]]
    if len(hit) != 1: sys.exit(f"cannot map reference [{n}]: {text[:70]}")
    num2key[n] = hit[0]
order = [num2key[i] for i in sorted(num2key)]
print(f"recovered {len(order)} reference keys")

STASH = {}
def stash(latex):
    tok = f"ZQX{len(STASH):04d}QZX"
    STASH[tok] = latex
    return tok

def prep(t):
    # display-quality equations written as $...$ in the master pass through untouched
    # (pandoc renders the same spans as native equations in the Word file)
    t = re.sub(r"\$[^$\n]+?\$", lambda m: stash(m.group(0)), t)
    # citations -> placeholder
    def one(m):
        ks = []
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if re.match(r"^\d+[–-]\d+$", tok):
                a, b = re.split(r"[–-]", tok)
                ks += [num2key[i] for i in range(int(a), int(b) + 1)]
            elif tok.isdigit():
                ks.append(num2key[int(tok)])
            else:
                return m.group(0)
        return stash(r"\cite{" + ",".join(ks) + "}")
    t = re.sub(r"\[([\d,–-]+)\]", one, t)
    # maths -> placeholder
    for a, b in RULES:
        if a in t: t = t.replace(a, stash(b))
    def pw(m): return stash("$10^{" + "".join(SUP[c] for c in m.group(1)) + "}$")
    t = re.sub(r"10([⁻⁰¹²³⁴-⁹]+)", pw, t)
    t = re.sub(r"×\s*(\d)", lambda m: stash(r"$\times$") + " " + m.group(1), t)
    for ch, rep in [("×", r"$\times$"), ("≤", r"$\le$"), ("√", r"$\sqrt{\ }$"),
                    ("ₑ", "$_e$"), ("ᵢ", "$_i$"), ("ₜ", "$_t$"), ("ₖ", "$_k$"),
                    ("ₐ", "$_a$"), ("₁", "$_1$"), ("₂", "$_2$"),
                    ("Δ", r"$\Delta$"), ("σ", r"$\sigma$"), ("λ", r"$\lambda$"),
                    ("ρ", r"$\rho$"), ("τ", r"$\tau$"),
                    ("ŷ", r"$\hat{y}$"), ("ȳ", r"$\bar{y}$"), ("→", r"$\rightarrow$"),
                    ("²", "$^2$"), ("³", "$^3$"),
                    ("ᵃ", r"\textsuperscript{a}"), ("ᵇ", r"\textsuperscript{b}"), ("ᶜ", r"\textsuperscript{c}"),
                    ("₀", "$_0$"), ("ₘ", "$_m$"), ("≥", r"$\ge$"), ("β", r"$\beta$")]:
        t = t.replace(ch, stash(rep))
    t = t.replace("−", "--")
    return t

title = re.search(r"^# (.+)$", s, re.M).group(1)
abstract = prep(s.split("## Abstract", 1)[1].split("**Keywords:**", 1)[0].strip())
keywords = s.split("**Keywords:**", 1)[1].split("\n", 1)[0].strip()
body   = prep(s[s.index("## 1. Introduction"):s.index("## References")])
figs   = prep(s[s.index("## Figure legends"):s.index("## Tables")])
tables = prep(s[s.index("## Tables"):])

def topx(text, name):
    # drop the literal section numbers: elsarticle numbers the sections itself
    text = re.sub(r"^(#{2,4})\s+\d+(?:\.\d+)*\.?\s+", r"\1 ", text, flags=re.M)
    # table headings are captions, not numbered subsections
    text = re.sub(r"^###\s+(Table \d+)\s+(.*)$", r"**\1** \2", text, flags=re.M)
    f = f"{OUT}/_{name}.md"; io.open(f, "w", encoding="utf-8").write(text)
    r = subprocess.run([PANDOC, f, "-f", "markdown-tex_math_dollars", "-t", "latex",
                        "--wrap=preserve", "--shift-heading-level-by=-1"],
                       capture_output=True, text=True)
    if r.returncode: sys.exit(r.stderr[:900])
    out = r.stdout
    for tok, latex in STASH.items():          # restore, tolerating pandoc's escaping
        out = out.replace(tok, latex)
    # Let LaTeX size the r/c columns to their (short) natural content width, as before;
    # a bare unwrapped "l" column let a long label ("Selection gain recovered,
    # correlation family", "Reversal without the added variants [95 % CI]") push the
    # whole table past the page, so left/text columns become tabularx X columns instead,
    # which wrap and share whatever width the r/c columns leave. Both of this document's
    # tables are short enough to fit on one page, so longtable's page-break machinery
    # (\endhead etc.) is dropped along with it in favour of plain tabularx, which is what
    # actually supports X columns.
    def respec(m):
        block = m.group(0)
        cols = ""
        for kind in __import__("re").findall(r"\\(raggedright|raggedleft|centering)\\arraybackslash", block):
            cols += {"raggedright": r">{\raggedright\arraybackslash}X",
                     "raggedleft": "r", "centering": "c"}[kind]
        return r"\begin{tabularx}{\linewidth}{@{}" + (cols or "X") + r"@{}}"
    out = re.sub(r"\\begin\{longtable\}\[\]\{@\{\}.*?@\{\}\}", respec, out, flags=re.S)
    out = out.replace(r"\end{longtable}", r"\end{tabularx}")
    out = re.sub(r"\\end(?:first)?head\n|\\end(?:last)?foot\n", "", out)
    # pandoc wraps every header cell in a full-\linewidth minipage, which measures its
    # own width against the page rather than the (now narrower, and X-column-flexible)
    # cell, and breaks tabularx's natural-width pass; the column spec above already
    # carries each column's alignment, so the minipage wrapper is redundant - drop it
    # and keep the cell text.
    out = re.sub(r"\\begin\{minipage\}\[b\]\{\\linewidth\}\\ragged(?:right|left)\n(.*?)\n\\end\{minipage\}",
                 r"\1", out, flags=re.S)
    out = out.replace(r"\begin{tabularx}", r"{\footnotesize\begin{tabularx}")
    out = out.replace(r"\end{tabularx}", r"\end{tabularx}}")
    return out

atex, btex, ftex, ttex = (topx(x, n) for x, n in
                          ((abstract, "abstract"), (body, "body"), (figs, "figs"), (tables, "tables")))

# The figure legends were prose captions with no image: each \textbf{Fig. N.} paragraph
# now gets the actual artwork above it, at column width, so the PDF shows the figure a
# reader sees, not just its caption. FigN.pdf sits next to manuscript.tex (build_tex.py's
# caller copies analysis/tcj_figures/Fig?.pdf there); missing files are left as a visible
# TODO instead of a silently absent figure.
def embed(m):
    n = m.group(1)
    path = f"{OUT}/Fig{n}.pdf"
    art = (r"\begin{center}\includegraphics[width=\linewidth]{Fig" + n + r"}\end{center}"
           if os.path.exists(path) else
           r"\begin{center}\fbox{\parbox{.8\linewidth}{\centering\vspace{2em}"
           r"\textbf{[Fig. " + n + r" artwork missing --- run the figure scripts and "
           r"copy Fig?.pdf into submission/ before this file is used]}\vspace{2em}}}\end{center}")
    return art + "\n\n" + m.group(0)
ftex = re.sub(r"\\textbf\{Fig\.\s*(\d+)\.\}", embed, ftex)

bib = "\n".join(rf"\bibitem{{{k}}} {REFS[k]}" for k in order)
bib = bib.replace("&", r"\&").replace("_", r"\_").replace("–", "--")

tex = r"""\documentclass[review,3p,times]{elsarticle}
\usepackage{fontspec}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{longtable,booktabs,array,calc,tabularx}
\usepackage{lineno}
\usepackage[hidelinks]{hyperref}
\usepackage{url}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\journal{The Crop Journal}
\begin{document}
\begin{frontmatter}
\title{""" + title + r"""}
\author[1,2]{Guohao Lv}
\ead{lvguohao@stu.ahau.edu.cn}
\author[1,2]{Lichuan Gu\corref{cor1}}
\ead{glc@ahau.edu.cn}
\cortext[cor1]{Corresponding author. ORCID: Guohao Lv, 0009-0007-4334-9818; Lichuan Gu, 0000-0002-3768-8203.}
\address[1]{School of Artificial Intelligence, Anhui Agricultural University, Hefei 230036, China}
\address[2]{Anhui Province Key Laboratory of Intelligent Agricultural Technology and Equipment, Anhui Agricultural University, Hefei 230036, China}
\begin{abstract}
""" + atex + r"""
\end{abstract}
\begin{keyword}
""" + keywords.replace(";", r" \sep ") + r"""
\end{keyword}
\end{frontmatter}
\linenumbers

""" + btex + r"""

\begin{thebibliography}{99}
""" + bib + r"""
\end{thebibliography}

""" + ftex + "\n\n" + ttex + r"""
\end{document}
"""
io.open(f"{OUT}/manuscript.tex", "w", encoding="utf-8").write(tex)
print(f"wrote {OUT}/manuscript.tex ({len(tex.splitlines())} lines); "
      f"{len(STASH)} protected tokens restored")
