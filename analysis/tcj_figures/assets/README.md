# Figure assets

`gpverdict_web_input.png` and `gpverdict_web_report.png` are screenshots of the published
GPverdict page (`tool/index.html`, version 1.1.1) running its bundled spring wheat example in
headless Chrome, captured by `capture_web.py` on 2026-09-26. The page is not altered;
`supp_figs.py` crops the report to its verdict section and shows both as Fig. S5 (they were Fig. 6A until 2026-09-29).
Fig. 6A is now a vector schematic whose numbers are read from `gpverdict_example_verdict.json`, written by
`example_verdict.py` (GPverdict run on its bundled example with the web page's settings). Every other figure element is
drawn from result files by the scripts in the parent folder; no external artwork is used.
