# -*- coding: utf-8 -*-
"""Replace {{key}} citation markers with [n] numbered in order of first appearance,
and write the reference list. Numbering can therefore never drift from the text."""
import io, re, sys
sys.path.insert(0, "tools")
from refs import REFS

SRC = "output/manuscript_TCJ.md"
s = io.open(SRC, encoding="utf-8").read()
MARK = "<<REFERENCES>>"
if MARK in s:
    head, tail = s.split(MARK, 1)
elif "## References" in s:
    head, rest = s.split("## References", 1)
    i = rest.index("\n## ")
    head, tail = head, "\n\n" + rest[i + 1:]
else:
    sys.exit("no reference-list anchor found")

order, seen = [], set()
for m in re.finditer(r"\{\{(\w+)\}\}", head):
    k = m.group(1)
    if k not in seen:
        seen.add(k); order.append(k)
missing = [k for k in order if k not in REFS]
if missing:
    sys.exit("keys with no reference entry: " + ", ".join(missing))
num = {k: i + 1 for i, k in enumerate(order)}

# collapse runs of adjacent citations: [3][4][5] -> [3–5]
head = re.sub(r"\{\{(\w+)\}\}", lambda m: f"\x00{num[m.group(1)]}\x00", head)
def collapse(m):
    ns = [int(x) for x in re.findall(r"\x00(\d+)\x00", m.group(0))]
    out, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        out.append(f"{ns[i]}–{ns[j]}" if j - i >= 2 else ",".join(str(x) for x in ns[i:j + 1]))
        i = j + 1
    return "[" + ",".join(out) + "]"
head = re.sub(r"(?:\x00\d+\x00)+", collapse, head)

refs = "## References\n\n" + "\n\n".join(f"[{num[k]}] {REFS[k]}" for k in order)
io.open(SRC, "w", encoding="utf-8").write(head + refs + tail)
print(f"{len(order)} references numbered in order of first citation")
