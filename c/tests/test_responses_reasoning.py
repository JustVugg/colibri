"""Responses reasoning effort reaches the real chat gateway over HTTP."""
import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
import openai_server as gw
from test_openai_server import ScriptedEngine, _spawn_test_server

spec = importlib.util.spec_from_file_location("responses_reasoning_shim", C.parent / "ui/responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


class ResponsesReasoningTest(unittest.TestCase):
    def test_effort_is_translated_to_the_chat_parameter(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort):
                chat, _ = shim.to_chat({"input": "hello", "reasoning": {"effort": effort}})
                self.assertEqual(chat.get("reasoning_effort"), effort)

    def test_omitted_effort_keeps_the_gateway_default(self):
        for reasoning in (None, {}, {"effort": None}, {"summary": "auto"}):
            with self.subTest(reasoning=reasoning):
                chat, _ = shim.to_chat({"input": "hello", "reasoning": reasoning})
                self.assertNotIn("reasoning_effort", chat)

    def start_proxy(self):
        engine = ScriptedEngine(chunks=("answer",))
        backend = urlsplit(_spawn_test_server(self, engine))

        class Proxy(BaseHTTPRequestHandler):
            def do_POST(self):
                shim.serve(self, backend)

            def log_message(self, *args):
                pass

        proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        thread = threading.Thread(target=proxy.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, timeout=2)
        self.addCleanup(proxy.server_close)
        self.addCleanup(proxy.shutdown)
        return engine, "http://127.0.0.1:%d/v1/responses" % proxy.server_port

    def request(self, endpoint, reasoning):
        body = {"input": "hello", "model": "test-model", "stream": False}
        if reasoning is not None:
            body["reasoning"] = reasoning
        req = Request(endpoint, data=json.dumps(body).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer secret"})
        with urlopen(req, timeout=5) as response:
            return json.load(response)

    def test_http_none_overrides_the_gateway_thinking_default(self):
        engine, endpoint = self.start_proxy()
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None), \
                patch.dict(os.environ, {"COLI_THINK": "1"}), \
                patch.object(gw, "render_chat_for_arch", wraps=gw.render_chat_for_arch) as render:
            self.assertEqual(self.request(endpoint, {"effort": "none"})["status"], "completed")
            self.assertEqual(render.call_args.args[1:3], (False, "none"))
            self.assertTrue(engine.calls[0][0].endswith("</think>"))

    def test_http_high_enables_thinking_when_the_gateway_default_is_off(self):
        engine, endpoint = self.start_proxy()
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None), \
                patch.dict(os.environ, {"COLI_THINK": "0"}), \
                patch.object(gw, "render_chat_for_arch", wraps=gw.render_chat_for_arch) as render:
            self.assertEqual(self.request(endpoint, {"effort": "high"})["status"], "completed")
            self.assertEqual(render.call_args.args[1:3], (True, "high"))
            self.assertTrue(engine.calls[0][0].endswith("<think>"))

    def test_http_missing_effort_preserves_the_gateway_default(self):
        _, endpoint = self.start_proxy()
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None), \
                patch.dict(os.environ, {"COLI_THINK": "1"}), \
                patch.object(gw, "render_chat_for_arch", wraps=gw.render_chat_for_arch) as render:
            self.assertEqual(self.request(endpoint, None)["status"], "completed")
            self.assertEqual(render.call_args.args[1:3], (True, "high"))


if __name__ == "__main__":
    unittest.main()
