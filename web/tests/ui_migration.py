"""Browser regression check using a local, deterministic OpenAI-compatible fixture.

Run `npm run build`, then `python3 tests/ui_migration.py` (requires Playwright
and its Chromium browser). No model, API credentials or external server needed.
"""
import json
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
requests = []
cancelled = threading.Event()
health = {"status": "ok", "kv_slots": 2, "scheduler": {"active": 0, "capacity": 2, "queued": 0, "max_queue": 8, "completed": 12, "rejected": 0, "timed_out": 0, "cancelled": 0}, "tiers": {"vram": 12, "ram": 20, "disk": 30, "vram_gb": 1.2, "ram_gb": 2.3}, "hwinfo": {"cpu": "Fixture CPU", "cores": 8, "gpus": 1, "vram_total_gb": 24, "ram_total_gb": 64, "ram_avail_gb": 32}}
turn = {"wall_s": 4, "prompt_tokens": 50, "completion_tokens": 20, "expert_disk_s": 2, "expert_wait_s": .5, "expert_matmul_s": 1, "attention_s": .6, "lm_head_s": .4, "forwards": 10}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "dist"), **kwargs)

    def log_message(self, *_):
        pass

    def response(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        requests.append(("GET", self.path, self.headers.get("Authorization"), None))
        if self.path == "/v1/models":
            self.response({"data": [{"id": "fixture-model"}, {"id": "second-model"}]})
        elif self.path == "/health":
            self.response(health)
        elif self.path == "/experts":
            self.response({"rows": 3, "cols": 4, "map": "81823f404142808182000102", "hits": "0100", "seq": 1})
        elif self.path == "/profile":
            self.response({"seq": 1, "turns": [turn, {**turn, "wall_s": 5}]})
        else:
            super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append(("POST", self.path, self.headers.get("Authorization"), body))
        if body["messages"][-1]["content"] == "ERROR":
            self.response({"error": {"message": "Fixture unavailable"}}, 503)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("x-colibri-queue-wait-ms", "12")
        self.end_headers()
        slow = body["messages"][-1]["content"] == "SLOW"
        try:
            for value in (["Token "] * 80 if slow else ["Risposta ", "dal ", "server."]):
                self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": value}}]}) + "\n\n").encode())
                self.wfile.flush()
                time.sleep(.08 if slow else .12)
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70}}) + "\n\ndata: [DONE]\n\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            cancelled.set()


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_port}"
errors = []
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, locale="it-IT", reduced_motion="reduce")
        page.set_default_timeout(10000)
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        expect(page.locator(".model-button")).to_contain_text("fixture-model")
        page.evaluate("document.fonts.ready")
        assert page.locator(".rail .colibri-word").count() == 0

        def navigate(name):
            page.locator(".dock-handle").click()
            page.locator(".navigation-dock nav").get_by_role("button", name=name, exact=True).click()

        def settings(category):
            page.get_by_role("button", name="Impostazioni", exact=True).click()
            page.locator(".settings-tabs").get_by_role("button", name=category, exact=True).click()

        settings("Connessione")
        expect(page.locator("#endpoint")).to_have_value(url + "/v1")
        page.locator("#api-key").fill("test-only-key")
        page.get_by_role("button", name="Sonda il server").click()
        expect(page.locator(".settings-card .connection-state")).to_have_text("Motore raggiungibile")
        assert page.evaluate("localStorage.getItem('colibri.apiKey')") is None
        settings("Modello")
        page.locator("#temperature").fill("0.4")
        page.locator("#max-tokens").fill("512")
        page.get_by_role("button", name="Ragionamento", exact=True).click()
        navigate("Chat")
        for width, height in [(1440, 1000), (1366, 768), (390, 844), (360, 640)]:
            page.set_viewport_size({"width": width, "height": height})
            page.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
            logo, box, tips = page.evaluate("['.welcome-brand','.composer','.suggestions'].map(s => document.querySelector(s).getBoundingClientRect().toJSON())")
            page.screenshot(path=f"/tmp/colibri-integrated-layout-{width}.png")
            assert logo["y"] + logo["height"] < box["y"]
            assert box["y"] + box["height"] < tips["y"], (width, logo, box, tips)
            assert tips["y"] + tips["height"] < height - 65
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.set_viewport_size({"width": 1440, "height": 1000})
        page.screenshot(path="/tmp/colibri-integrated-chat.png")
        page.get_by_role("button", name="Esplora un’idea").click()
        expect(page.locator("#draft")).not_to_be_empty()
        page.locator("#draft").fill("Primo messaggio")
        page.locator("#draft").press("Shift+Enter")
        assert "\n" in page.locator("#draft").input_value()
        page.locator("#draft").press("Enter")
        expect(page.locator(".message.assistant .message-body")).to_have_text("Risposta dal server.")
        expect(page.get_by_role("button", name="Invia messaggio")).to_be_visible()
        body = [r[3] for r in requests if r[0] == "POST"][-1]
        assert body["cache_slot"] == 0 and body["temperature"] == .4 and body["max_completion_tokens"] == 512 and body["enable_thinking"] is True
        assert body["stream"] is True and body["stream_options"]["include_usage"] is True
        assert [r for r in requests if r[0] == "POST"][-1][2] == "Bearer test-only-key"
        assert page.locator(".composer").bounding_box()["y"] > 700
        page.get_by_role("button", name="Rigenera risposta").click()
        expect(page.get_by_role("button", name="Ferma la generazione")).to_be_visible()
        expect(page.get_by_role("button", name="Invia messaggio")).to_be_visible()
        assert page.locator(".message.user").count() == 1
        with page.expect_download() as download:
            page.get_by_role("button", name="Esporta conversazione").click()
        data = json.loads(Path(download.value.path()).read_text())
        assert data["messages"][-1]["content"] == "Risposta dal server."
        page.screenshot(path="/tmp/colibri-integrated-conversation.png")
        settings("Modello")
        page.locator("#kv-slot").select_option("1")
        navigate("Chat")
        expect(page.locator(".welcome-brand")).to_be_visible()
        page.locator("#draft").fill("Slot due")
        page.locator("#draft").press("Enter")
        expect(page.get_by_role("button", name="Invia messaggio")).to_be_visible()
        assert [r[3] for r in requests if r[0] == "POST"][-1]["cache_slot"] == 1
        settings("Modello")
        page.locator("#kv-slot").select_option("0")
        navigate("Chat")
        expect(page.locator(".message.user")).to_have_text("Primo messaggio")
        page.get_by_role("button", name="Nuova chat").click()
        expect(page.locator(".welcome-brand")).to_be_visible()
        page.get_by_role("button", name="Conversazioni", exact=True).click()
        page.locator(".history-search input").fill("Primo")
        page.locator(".history-results button").click()
        expect(page.locator(".message.user")).to_have_text("Primo messaggio")
        page.locator("#draft").fill("SLOW")
        page.locator("#draft").press("Enter")
        expect(page.locator(".message.assistant").last).to_contain_text("Token")
        expect(page.get_by_role("button", name="Nuova chat")).to_be_disabled()
        navigate("Profiling")
        expect(page.locator(".prof-table tbody tr")).to_have_count(2)
        page.screenshot(path="/tmp/colibri-integrated-profiling.png")
        navigate("Chat")
        page.get_by_role("button", name="Ferma la generazione").click()
        expect(page.get_by_role("button", name="Invia messaggio")).to_be_visible()
        assert cancelled.wait(3), "Stop must cancel the actual response stream"
        page.locator("#draft").fill("ERROR")
        page.locator("#draft").press("Enter")
        expect(page.get_by_role("alert")).to_have_text("Fixture unavailable")
        navigate("Brain")
        expect(page.locator(".cortex-regions button")).to_have_count(10)
        assert page.locator(".cortex-explorer canvas").bounding_box()["height"] == 1000
        page.screenshot(path="/tmp/colibri-integrated-brain.png")
        page.locator(".cortex-regions button").filter(has_text="Python").click()
        expect(page.locator(".cortex-details")).to_be_visible()
        page.get_by_label("Expert nella regione").select_option(index=15)
        expect(page.locator(".cortex-affinities>div")).to_have_count(4)
        import re
        selected_label = page.get_by_label("Expert nella regione").locator("option:checked").inner_text()
        layer, expert = re.findall(r"\d+", selected_label)
        entry = json.loads((ROOT / "public/experts.json").read_text())["experts"][f"{layer}:{expert}"]
        expected = [f"{v * 100:.1f}%" for _, v in sorted(entry["affinity"].items(), key=lambda item: -item[1])[:4]]
        assert page.locator(".cortex-affinities>div>span>b").all_text_contents() == expected
        page.screenshot(path="/tmp/colibri-integrated-expert.png")
        page.locator(".cortex-details").get_by_role("button", name="Chiudi").click()
        page.get_by_role("button", name="Routing live", exact=True).click()
        expect(page.locator(".brain-head")).to_contain_text("3 layer × 4 expert")
        assert any(r[1] == "/experts" and r[2] == "Bearer test-only-key" for r in requests)
        settings("Sistema")
        expect(page.locator(".settings-card")).to_contain_text("Fixture CPU")
        settings("Generale")
        page.get_by_role("button", name="Chiaro", exact=True).click()
        expect(page.locator("html")).to_have_attribute("data-theme", "light")
        page.locator(".language-field select").select_option("en")
        # :visible, not just .page-heading h1 — since brio mode the workspace
        # keeps more than one page mounted at a time (hidden, so a scoring run
        # survives a trip to chat), so a bare heading selector matches several.
        expect(page.locator(".page-heading h1:visible")).to_have_text("Settings")
        mobile = browser.new_context(viewport={"width": 360, "height": 640}, has_touch=True, is_mobile=True, locale="it-IT", reduced_motion="reduce")
        touch = mobile.new_page()
        touch.on("pageerror", lambda e: errors.append(str(e)))
        touch.goto(url)
        touch.locator(".dock-handle").tap()
        expect(touch.locator(".navigation-dock nav")).to_be_visible()
        touch.locator(".navigation-dock nav").get_by_role("button", name="Brain", exact=True).tap()
        expect(touch.locator(".navigation-dock nav")).to_be_hidden()
        expect(touch.locator(".cortex-regions button")).to_have_count(10)
        assert touch.locator(".cortex-explorer canvas").bounding_box()["height"] == 640
        touch.screenshot(path="/tmp/colibri-integrated-mobile-brain.png")
        touch.get_by_role("button", name="Argomenti e dettagli", exact=True).tap()
        expect(touch.locator("#brain-details")).to_be_visible()
        touch.get_by_label("Cerca un argomento").fill("Python")
        touch.locator(".cortex-topic-list button").tap()
        touch.get_by_label("Expert nella regione").select_option(index=4)
        layer, expert = re.findall(r"\d+", touch.get_by_label("Expert nella regione").locator("option:checked").inner_text())
        mobile_entry = json.loads((ROOT / "public/experts.json").read_text())["experts"][f"{layer}:{expert}"]
        expect(touch.locator(".cortex-affinities>div")).to_have_count(min(4, len(mobile_entry["affinity"])))
        touch.locator("#brain-details").get_by_role("button", name="Chiudi").tap()
        touch.screenshot(path="/tmp/colibri-integrated-mobile-expert.png")
        # Real touch swipe, then keyboard access to the same collapsed dock.
        # The dock hangs from the TOP edge now, so the gesture that opens it is a
        # pull DOWN. It used to sit at the bottom and open on a pull up; swiping
        # away from the screen edge the dock is attached to is the same motion,
        # mirrored.
        handle = touch.locator(".dock-handle").bounding_box()
        cdp = mobile.new_cdp_session(touch)
        x, y = handle["x"] + handle["width"] / 2, handle["y"] + handle["height"] / 2
        cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y + 30}]})
        cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        expect(touch.locator(".navigation-dock nav")).to_be_visible()
        touch.keyboard.press("Escape")
        touch.locator(".dock-handle").focus()
        touch.locator(".dock-handle").press("Enter")
        expect(touch.locator(".navigation-dock nav button").first).to_be_focused()
        touch.keyboard.press("Escape")
        expect(touch.locator(".navigation-dock nav")).to_be_hidden()
        mobile.close()
        # A generic OpenAI-compatible backend may omit /health and KV support.
        generic = browser.new_page(locale="it-IT")
        generic.on("pageerror", lambda e: errors.append(str(e)))
        generic.route("**/health", lambda route: route.fulfill(status=404, body="{}"))
        generic.goto(url)
        expect(generic.locator(".model-button")).to_contain_text("fixture-model")
        generic.locator("#draft").fill("Generic backend")
        generic.locator("#draft").press("Enter")
        expect(generic.locator(".message.assistant .message-body")).to_have_text("Risposta dal server.")
        assert "cache_slot" not in [r[3] for r in requests if r[0] == "POST"][-1]
        generic.close()
        assert not errors, errors
        assert not any(r[1] in ["/v1/health", "/v1/profile", "/v1/experts"] for r in requests)
        print("PASS: four layouts, connection/auth/model settings, streaming, cancellation, errors, regeneration, export, isolated KV sessions, archive restore, real profile/experts endpoints, immersive atlas, theme and locale. No JavaScript errors.")
        browser.close()
finally:
    server.shutdown()
    server.server_close()
