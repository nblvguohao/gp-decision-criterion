"""GPverdict's verdict on its bundled spring wheat example, for Fig. 6A.

Runs the package in tool/gpverdict on examples/spring_wheat_URSN.csv of the GPverdict repository
(https://github.com/nblvguohao/gpverdict) with the web page's settings: top 10 % within each
environment, B = 200 environment resamples, seed 0, minimum 10 genotypes per environment (the page's
default; its "Try the spring wheat example" button sets 12, which gives identical results because every
method has at least 12 genotypes in every environment of the example; checked below). Writes
gpverdict_example_verdict.json next to this file: the tool's own JSON output (report.to_json), its
verdict sentences (report.headline), its planning table (report.planning_table), and the first three
rows of the example file as they appear there. fig6.py reads every number of
panel A from that file.

    python analysis/tcj_figures/assets/example_verdict.py path/to/spring_wheat_URSN.csv
"""
import csv, hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "tool"))
import gpverdict as gv
from gpverdict.report import headline, planning_table

if len(sys.argv) < 2:
    sys.exit("usage: example_verdict.py path/to/spring_wheat_URSN.csv  (examples/ of https://github.com/nblvguohao/gpverdict, v1.1.1)")
CSV = sys.argv[1]
SET = dict(frac=0.10, min_n=10, B=200, seed=0)                     # index.html: verdict(csv, frac=FRAC, min_n=MINN, B=200)

v = gv.verdict(CSV, **SET)
v12 = gv.verdict(CSV, frac=SET["frac"], min_n=12, B=SET["B"], seed=SET["seed"], check_invariance=False)
same = (v12["data"] == v["data"] and v12["leader"] == v["leader"] and v12["resolution"]["tied_with_leader"] == v["resolution"]["tied_with_leader"]
        and v12["rmse_pick"] == v["rmse_pick"] and v12["reversal_rmse_pearson"] == v["reversal_rmse_pearson"]
        and v12["outcome"]["recovery"] == v["outcome"]["recovery"])
assert same, "min_n = 10 and 12 differ on the example"

with open(CSV, newline="") as fh:
    rows = list(csv.reader(fh))
out = dict(
    source=dict(file="examples/spring_wheat_URSN.csv (GPverdict repository)", sha256=hashlib.sha256(open(CSV, "rb").read()).hexdigest(),
                rows=len(rows) - 1, gpverdict_version=gv.__version__,
                settings=dict(SET, note="web page settings; min_n = 12 (the example button) gives identical results")),
    example_rows=dict(columns=rows[0], rows=rows[1:4]),
    headline=headline(v),
    planning_table=[dict(gap=g, cells=n) for g, n in planning_table(v)],
    tool_output=json.loads(gv.to_json(v)),
)
json.dump(out, open(os.path.join(HERE, "gpverdict_example_verdict.json"), "w"), indent=1)
print("\n".join(out["headline"]))
