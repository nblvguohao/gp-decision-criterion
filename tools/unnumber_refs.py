# -*- coding: utf-8 -*-
"""Turn the numbered manuscript back into {{key}} markers, so the text can be
edited without touching numbers. Inverse of number_refs.py."""
import io, re, sys
sys.path.insert(0, "tools")
from refs import REFS
SRC="output/manuscript_TCJ.md"
s=io.open(SRC,encoding="utf-8").read()
reflist=s.split("## References",1)[1].split("\n## ",1)[0]
num2key={}
for m in re.finditer(r"^\[(\d+)\]\s+(.+?)\s*$", reflist, re.M):
    n,text=int(m.group(1)),m.group(2)
    hit=[k for k,v in REFS.items() if v[:55]==text[:55]]
    if len(hit)!=1:
        # entry whose database record was edited or removed: keep it verbatim
        num2key[n]=("__RAW__", text); continue
    num2key[n]=hit[0]
head,rest=s.split("## References",1)
tail="<<REFERENCES>>\n\n## "+rest.split("\n## ",1)[1]
def one(m):
    ks=[]
    for tok in m.group(1).split(","):
        tok=tok.strip()
        rng=re.match(r"^(\d+)[–-](\d+)$",tok)
        nums=range(int(rng.group(1)),int(rng.group(2))+1) if rng else ([int(tok)] if tok.isdigit() else None)
        if nums is None: return m.group(0)
        for n in nums:
            k=num2key.get(n)
            if k is None: return m.group(0)
            ks.append(k if isinstance(k,str) else "__RAW__")
    if "__RAW__" in ks: return m.group(0)
    return "".join("{{"+k+"}}" for k in ks)
head=re.sub(r"\[([\d,–-]+)\]", one, head)
io.open(SRC,"w",encoding="utf-8").write(head+tail)
left=re.findall(r"\[\d[\d,\u2013-]*\]",head)   # any bracket group, not just single numbers
print(f"converted to keys; numeric citations left in body: {sorted(set(left)) if left else 'none'}")
print("keys in text:", len(set(re.findall(r'\{\{(\w+)\}\}',head))))
