"""Screenshots of the GPverdict web page for Fig. 6A.

Drives the published page (tool/index.html) in headless Chrome through the DevTools protocol:
waits until Python has loaded in the browser, captures the input form, clicks "Try the spring
wheat example", waits for the analysis to finish and captures the report the page produced.
Nothing on the page is altered. Needs Google Chrome, websocket-client and the tool served
locally, e.g.  (cd tool && python3 -m http.server 8792)  then
    python capture_web.py <output-directory>
Writes web_input.png and web_report.png; fig6.py reads the copies in this folder
(gpverdict_web_input.png, gpverdict_web_report.png)."""
import json, subprocess, time, base64, urllib.request, sys, os
import websocket
S = sys.argv[1]
C = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
env = dict(os.environ, LANG="en_US.UTF-8", LANGUAGE="en")
p = subprocess.Popen([C, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--lang=en-US",
                      "--accept-lang=en-US", f"--user-data-dir={S}/prof2", "--remote-debugging-port=9333", "--remote-allow-origins=http://127.0.0.1:9333",
                      "--window-size=1000,1400", "-AppleLanguages", "(en-US)"], env=env,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(150):
        try:
            tabs = json.load(urllib.request.urlopen("http://127.0.0.1:9333/json")); break
        except Exception: time.sleep(.3)
    ws = websocket.create_connection([t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"], timeout=300, suppress_origin=True)
    n = [0]
    def call(method, **params):
        n[0] += 1; ws.send(json.dumps({"id": n[0], "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == n[0]: return m.get("result", m)
    def js(expr):
        r = call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return r.get("result", {}).get("value")
    call("Emulation.setEmulatedMedia", features=[{"name": "prefers-color-scheme", "value": "light"}])
    call("Emulation.setDeviceMetricsOverride", width=1000, height=1400, deviceScaleFactor=2, mobile=False)
    call("Page.enable"); call("Page.navigate", url="http://localhost:8792/index.html")
    t0 = time.time()
    while "Ready" not in (js("document.getElementById('status').textContent") or ""):
        if time.time() - t0 > 240: raise SystemExit("python did not load")
        time.sleep(1)
    print("ready after", round(time.time() - t0), "s")
    def clip_of(sel):
        return js(f"(()=>{{const r=document.querySelector('{sel}').getBoundingClientRect();"
                  f"return {{x:r.left+window.scrollX,y:r.top+window.scrollY,w:r.width,h:r.height}}}})()")
    def shot(name, c, pad=10):
        r = call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                 clip=dict(x=max(c["x"] - pad, 0), y=max(c["y"] - pad, 0), width=c["w"] + 2 * pad, height=c["h"] + 2 * pad, scale=1))
        open(f"{S}/{name}.png", "wb").write(base64.b64decode(r["data"]))
    # the form as a user first sees it: title, the four questions and the input card
    top = js("(()=>{const a=document.querySelector('h1').getBoundingClientRect();const b=document.querySelectorAll('section.card')[1].getBoundingClientRect();"
             "return {x:b.left,y:a.top+window.scrollY,w:b.width,h:b.bottom-a.top}})()")
    shot("web_input", top)
    js("document.getElementById('example').click()")
    t0 = time.time()
    while (js("document.getElementById('status').textContent") or "") != "Done.":
        if time.time() - t0 > 300: raise SystemExit("analysis did not finish: " + str(js("document.getElementById('status').textContent")))
        time.sleep(1)
    print("example done after", round(time.time() - t0), "s")
    time.sleep(1.5)
    # the report is the iframe's own document: render it on its own at the same width
    html = js("document.getElementById('report').srcdoc")
    open(f"{S}/report_from_app.html", "w").write(html)
    call("Page.setDocumentContent", frameId=call("Page.getFrameTree")["frameTree"]["frame"]["id"], html=html)
    time.sleep(1)
    h = js("document.documentElement.scrollHeight"); w = js("document.documentElement.scrollWidth")
    shot("web_report", dict(x=0, y=0, w=w, h=h), pad=0)
    print("report size", w, h)
finally:
    p.terminate()
