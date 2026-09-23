"""La vista Brio: presente, raggiungibile, e non rompe la chat accanto."""
import json, subprocess, sys, threading, time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

DIST = Path(__file__).resolve().parent.parent / "dist"

class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k): super().__init__(*a, directory=str(DIST), **k)
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        body = json.dumps({"answer":"request changes","entropy":0.121,"normalize":"mean",
            "choices":[{"option":"request changes","p":0.974,"logprob":-0.25,"mean_logprob":-0.12,"tokens":2},
                       {"option":"merge","p":0.023,"logprob":-4.0,"mean_logprob":-4.0,"tokens":1}],
            "usage":{"prompt_tokens":88,"completion_tokens":0,"read_tokens":4,"total_tokens":92}}).encode()
        self.send_response(200); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path.startswith("/v1/models"):
            b=json.dumps({"object":"list","data":[{"id":"m","object":"model","created":0,"owned_by":"c"}]}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        if self.path.startswith("/health"):
            b=json.dumps({"status":"ok"}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(b))); self.end_headers(); self.wfile.write(b); return
        super().do_GET()

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{srv.server_address[1]}"
errs = []
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width":1200,"height":800}, locale="en-US")
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.set_default_timeout(10000)
    pg.goto(base, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    # the endpoint is derived from the page origin, so only the probe is needed
    pg.get_by_role("button", name="Settings", exact=True).click()
    pg.locator(".settings-tabs").get_by_role("button", name="Connection", exact=True).click()
    pg.get_by_role("button", name="Probe server", exact=True).click()
    expect(pg.locator(".settings-card .connection-state")).to_have_text("Engine reachable")

    def navigate(name):
        pg.locator(".dock-handle").click()
        pg.locator(".navigation-dock nav").get_by_role("button", name=name, exact=True).click()

    navigate("Brio")
    pg.locator(".brio-card textarea").first.fill("The PR touches the engine and has no tests.")
    pg.locator(".brio-q-head input").first.fill("What should the reviewer do?")
    pg.locator(".brio-own").first.fill("merge\nrequest changes")
    pg.locator(".brio-run").click()
    expect(pg.locator(".brio-bar").first).to_be_visible(timeout=15000)
    # the chat is still mounted, but it must not be on screen behind brio
    expect(pg.locator(".chat-view .composer textarea")).to_be_hidden()
    expect(pg.locator(".brio-foot strong")).to_have_text("request changes")
    assert "0 generated" in pg.locator(".brio-foot em").inner_text()
    # torno in chat: la bozza deve essere ancora li, e brio non smontato
    navigate("Chat")
    expect(pg.locator(".chat-view .composer textarea")).to_be_visible()
    # and brio is still mounted behind it: a scoring run must survive the trip
    assert pg.locator(".brio-bar").count() > 0, "brio was unmounted by going to chat"
    assert not errs, errs[:2]
    b.close()
srv.shutdown()
print("PASS: brio view reachable, scores, keeps its state behind the chat. No JavaScript errors.")
