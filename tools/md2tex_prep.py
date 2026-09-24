# -*- coding: utf-8 -*-
"""Turn the Unicode maths in the manuscript into LaTeX math before pandoc sees it.
Rules are ordered: the most specific expression first, so that a composite is never
broken up by a single-character rule."""
import io, re, sys

RULES = [
 # composite expressions -------------------------------------------------
 ("*a*ₑŷ*ₑ* + *b*ₑ",            r"$a_e\hat{y}_e + b_e$"),
 ("ŷ*ₑ* → *a*ₑŷ*ₑ* + *b*ₑ",     r"$\hat{y}_e \rightarrow a_e\hat{y}_e + b_e$"),
 ("*g*(ŷ*ₑ*)",                  r"$g(\hat{y}_e)$"),
 ("ŷ*ₑ*",                       r"$\hat{y}_e$"),
 ("*a*ₑ ~ U(0.5, 2)",           r"$a_e \sim U(0.5, 2)$"),
 ("*b*ₑ ~ N(0, σ²_y)",          r"$b_e \sim N(0, \sigma^2_y)$"),
 ("*a*ₑ > 0",                   r"$a_e > 0$"),
 ("*a*ₑ",                       r"$a_e$"),
 ("*b*ₑ",                       r"$b_e$"),
 ("*R*²_score",                 r"$R^2_{\mathrm{score}}$"),
 ("*R*²_{score}",               r"$R^2_{\mathrm{score}}$"),
 ("σ²_g/(σ²_g + σ²_e/*n̄*)",     r"$\sigma^2_g/(\sigma^2_g + \sigma^2_e/\bar{n})$"),
 ("σ²_y",                       r"$\sigma^2_y$"),
 ("√*H*²",                      r"$\sqrt{H^2}$"),
 ("*H*²",                       r"$H^2$"),
 ("SE²",                        r"$\mathrm{SE}^2$"),
 ("Δ₁ · Δ₂ < 0",                r"$\Delta_1 \cdot \Delta_2 < 0$"),
 ("*r̂ᵢ* = median*ₜ r*ᵢₜ",       r"$\hat{r}_i = \mathrm{median}_t\, r_{it}$"),
 ("*sᵢₜ* = |*r*ᵢₜ − *r̂ᵢ*|",     r"$s_{it} = |r_{it} - \hat{r}_i|$"),
 ("⌈*fn*⌉",                     r"$\lceil fn \rceil$"),
 ("(ȳ_selected − ȳ)/SD(*y*)",   r"$(\bar{y}_{\mathrm{selected}} - \bar{y})/\mathrm{SD}(y)$"),
 ("*i*ₖ · *ρ*",                 r"$i_k \cdot \rho$"),
 ("*i*ₖ",                       r"$i_k$"),
 ("Δ*z* = *i ρ σ*ₐ",            r"$\Delta z = i\rho\sigma_a$"),
 ("Δ*r*",                       r"$\Delta r$"),
 ("Δ*g*",                       r"$\Delta g$"),
 ("*r*²",                       r"$r^2$"),
 ("*τ*",                        r"$\tau$"),
 ("*ρ*",                        r"$\rho$"),
 ("(λ = 1 and 100)",            r"($\lambda$ = 1 and 100)"),
]
SUP = {"⁻":"-","⁰":"0","¹":"1","²":"2","³":"3","⁴":"4","⁵":"5","⁶":"6","⁷":"7","⁸":"8","⁹":"9"}

def convert(s):
    for a,b in RULES:
        s = s.replace(a,b)
    # 10^-x powers, including "1.1 × 10⁻⁵⁰"
    def pw(m):
        return "$10^{" + "".join(SUP[c] for c in m.group(1)) + "}$"
    s = re.sub(r"10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", pw, s)
    s = s.replace("× $10^", r"$\times$ $10^")
    s = s.replace("×", r"$\times$").replace("≤", r"$\le$").replace("−", "--")
    s = s.replace("√", r"$\sqrt{\ }$")
    # any stray sub/superscript letters left over
    for ch, rep in [("ₑ","$_e$"),("ᵢ","$_i$"),("ₜ","$_t$"),("ₖ","$_k$"),("ₐ","$_a$"),
                    ("₁","$_1$"),("₂","$_2$"),("Δ",r"$\Delta$"),("σ",r"$\sigma$"),
                    ("λ",r"$\lambda$"),("ŷ",r"$\hat{y}$"),("ȳ",r"$\bar{y}$"),
                    ("→",r"$\rightarrow$")]:
        s = s.replace(ch, rep)
    s = re.sub(r"\$\$", "", s)          # collapse abutting math
    return s

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    t = convert(io.open(src, encoding="utf-8").read())
    io.open(dst, "w", encoding="utf-8").write(t)
    left = {c for c in t if ord(c) > 127 and c not in "—–’‘“”·áéíóúñäöëüÁÉÍÓÚÑ"}
    print(f"written {dst}")
    print("unicode still present:", sorted(left) if left else "only accented Latin (fine for XeTeX)")
