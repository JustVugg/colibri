"""Responses request settings reach the real chat gateway over HTTP."""
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
from urllib.error import HTTPError

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

    def request(self, endpoint, reasoning=None, text=None):
        body = {"input": "hello", "model": "test-model", "stream": False}
        if reasoning is not None:
            body["reasoning"] = reasoning
        if text is not None:
            body["text"] = text
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


    def test_schema_format_preserves_metadata_in_the_chat_wrapper(self):
        fmt = {"type": "json_schema", "name": "codex_output_schema", "strict": True,
               "description": "Return a short answer", "schema": {"type": "object"}}
        chat, _ = shim.to_chat({"input": "hello", "text": {"format": fmt}})
        self.assertEqual(chat.get("response_format"), {"type": "json_schema",
                         "json_schema": {k: v for k, v in fmt.items() if k != "type"}})
        self.assertEqual(fmt["type"], "json_schema")

    def test_missing_format_keeps_the_gateway_default(self):
        for text in (None, {}, {"verbosity": "low"}):
            with self.subTest(text=text):
                chat, _ = shim.to_chat({"input": "hello", "text": text})
                self.assertNotIn("response_format", chat)

    def test_http_codex_schema_reaches_the_engine_grammar(self):
        engine, endpoint = self.start_proxy()
        schema = {"type": "object", "properties": {"answer": {"type": "string"}},
                  "required": ["answer"], "additionalProperties": False}
        fmt = {"type": "json_schema", "name": "codex_output_schema", "strict": True, "schema": schema}
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None):
            self.assertEqual(self.request(endpoint, text={"format": fmt})["status"], "completed")
            self.assertEqual(json.loads(engine.calls[0][5]), schema)

    def test_http_json_object_and_text_preserve_the_gateway_grammar_behavior(self):
        for kind, expected in (("json_object", gw.GENERIC_JSON_GBNF), ("text", None)):
            with self.subTest(kind=kind):
                engine, endpoint = self.start_proxy()
                with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None):
                    self.assertEqual(self.request(endpoint, text={"format": {"type": kind}})["status"], "completed")
                    self.assertEqual(engine.calls[0][5], expected)

    def test_http_invalid_schema_reaches_gateway_validation_before_generation(self):
        engine, endpoint = self.start_proxy()
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", None):
            with self.assertRaises(HTTPError) as caught:
                self.request(endpoint, text={"format": {"type": "json_schema", "name": "invalid", "schema": []}})
            self.assertEqual(caught.exception.code, 400)
            self.assertEqual(engine.calls, [])

if __name__ == "__main__":
    unittest.main()
