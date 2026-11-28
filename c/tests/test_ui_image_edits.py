"""Browser regression for the chat composer and the Qwen-Image gateway.

With Playwright, Chromium and Pillow installed, run from c/:
  COLI_TEST_BROWSER=1 python3 -m unittest discover -s tests -p test_ui_image_edits.py
"""
import base64
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


@unittest.skipUnless(os.environ.get("COLI_TEST_BROWSER") == "1", "optional browser test")
class ImageComposerTest(unittest.TestCase):
    def setUp(self):
        from playwright.sync_api import sync_playwright
        from test_qwen_image_gateway import GatewayHarness, image_engine

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ui"))
        try:
            from server import Handler
        finally:
            sys.path.pop(0)
        gateway = GatewayHarness(api_key=None, QWENIMAGE_STUB_EDITS="1")
        self.addCleanup(gateway.close)

        class PageHandler(Handler):
            backend = urlparse(f"http://127.0.0.1:{gateway.port}")

        server = ThreadingHTTPServer(("127.0.0.1", 0), PageHandler)
        PageHandler.port = server.server_port
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        pw = sync_playwright().start()
        self.addCleanup(pw.stop)
        browser = pw.chromium.launch()
        self.addCleanup(browser.close)
        self.page = browser.new_page()
        self.url = f"http://127.0.0.1:{server.server_port}/?lang=en"
        self.png = image_engine.encode_png(1408, 704, bytes((20, 60, 90, 127)) * (1408 * 704), 4)

    def attach(self, count=1):
        page = self.page
        page.locator("#plusBtn").click()
        with page.expect_file_chooser() as chooser:
            page.locator("[data-k=attach]").click()
        chooser.value.set_files([{"name": f"reference-{i}.png", "mimeType": "image/png", "buffer": self.png}
                                 for i in range(count)])
        page.locator("#cchips img").nth(count - 1).wait_for()

    def send_image(self, button, route):
        page = self.page
        with page.expect_response(lambda r: r.url.endswith("/v1/images/" + route)) as response:
            button.click()
        self.assertEqual(response.value.status, 200, response.value.status_text)
        page.locator("#send:not(.stop)").wait_for()
        page.locator(".genimg").last.wait_for()
        return response.value.request.post_data_json

    def test_send_and_regenerate_keep_original_bytes(self):
        p = self.page
        p.goto(self.url)
        p.locator("#pickerText").filter(has_text="Qwen").wait_for()
        self.attach()
        p.locator("#send").click()
        p.get_by_text("Describe the edit you want.", exact=True).wait_for()
        self.assertEqual(p.locator("#cchips img").count(), 1)
        instruction = "Give the fox a blue scarf"
        p.locator("#prompt").fill(instruction)
        self.attach()
        p.get_by_text("Attach one PNG, JPEG or WebP image.", exact=True).wait_for()
        self.assertEqual(p.locator("#cchips img").count(), 1)
        first = self.send_image(p.locator("#send"), "edits")
        self.assertEqual(first["prompt"], instruction)
        self.assertEqual(first["image"], "data:image/png;base64," + base64.b64encode(self.png).decode())
        with p.expect_response("**/v1/models"):
            p.reload()
        p.locator("#navToggle").click()
        p.locator(".navsession").filter(has_text=instruction).click()
        self.assertEqual(self.send_image(p.locator("[data-act=regen]"), "edits"), first)
        p.locator("#prompt").fill("A red fox in the snow")
        text_only = self.send_image(p.locator("#send"), "generations")
        self.assertNotIn("image", text_only)

    def test_generation_only_model_hides_attachment(self):
        def disable_edits(route):
            upstream = route.fetch()
            card = upstream.json()
            card["data"][0]["capabilities"] = ["image_generation"]
            route.fulfill(response=upstream, json=card)

        p = self.page
        p.route("**/v1/models", disable_edits)
        p.goto(self.url)
        p.locator("#pickerText").filter(has_text="Qwen").wait_for()
        p.locator("#plusBtn").click()
        self.assertTrue(p.locator("[data-k=attach]").is_hidden())
        p.locator("#plusBtn").click()
        p.locator("#prompt").fill("A red fox")
        self.assertNotIn("image", self.send_image(p.locator("#send"), "generations"))

    def test_vision_chat_keeps_multiple_resized_attachments(self):
        def vision_card(route):
            upstream = route.fetch()
            card = upstream.json()
            card["data"][0]["capabilities"] = ["chat"]
            route.fulfill(response=upstream, json=card)

        p = self.page
        p.route("**/v1/models", vision_card)
        p.route("**/v1/chat/completions", lambda r: r.fulfill(
            content_type="text/event-stream", body="data: [DONE]\n\n"))
        p.goto(self.url)
        p.locator("#pickerText").filter(has_text="Qwen").wait_for()
        self.attach(count=2)
        p.locator("#prompt").fill("Describe both pictures")
        with p.expect_request("**/v1/chat/completions") as request:
            p.locator("#send").click()
        content = request.value.post_data_json["messages"][-1]["content"]
        urls = [part["image_url"]["url"] for part in content if part["type"] == "image_url"]
        self.assertEqual(len(urls), 2)
        self.assertTrue(all(url.startswith("data:image/jpeg;base64,") for url in urls))
        self.assertEqual(p.locator(".msg-user img").first.evaluate("(im) => im.naturalWidth"), 1280)
