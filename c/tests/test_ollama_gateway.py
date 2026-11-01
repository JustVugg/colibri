"""coli gateway (ollama_gateway.py): the translators on their own, and the gateway attached to
an openai_server.APIServer with a scripted engine -- no model weights, no child process."""
import json
import os
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ollama_gateway as gw   # noqa: E402
from openai_server import APIServer   # noqa: E402
from test_openai_server import ScriptedEngine   # noqa: E402


class TranslateRequests(unittest.TestCase):
    def test_chat_basic(self):
        req, dropped = gw.ollama_chat_to_openai({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                                 "options": {"temperature": 0.2, "num_predict": 64, "top_k": 20,
                                                             "num_ctx": 8192, "stop": ["\n"]}, "stream": False}, "test-model")
        self.assertEqual(req["model"], "test-model")
        self.assertEqual(req["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(req["temperature"], 0.2)
        self.assertEqual(req["max_tokens"], 64)
        self.assertEqual(req["stop"], ["\n"])
        self.assertFalse(req["stream"])
        self.assertNotIn("stream_options", req)
        self.assertEqual(sorted(dropped), ["num_ctx", "top_k"])

    def test_chat_stream_asks_usage(self):
        req, _ = gw.ollama_chat_to_openai({"messages": [{"role": "user", "content": "hi"}]}, "m")
        self.assertTrue(req["stream"])
        self.assertEqual(req["stream_options"], {"include_usage": True})

    def test_num_predict_negative_is_default(self):
        req, _ = gw.ollama_chat_to_openai({"messages": [], "options": {"num_predict": -1}}, "m")
        self.assertNotIn("max_tokens", req)

    def test_images_become_data_uris(self):
        png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        req, _ = gw.ollama_chat_to_openai({"messages": [{"role": "user", "content": "what", "images": [png]}]}, "m")
        parts = req["messages"][0]["content"]
        self.assertEqual(parts[0], {"type": "text", "text": "what"})
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertTrue(parts[1]["image_url"]["url"].startswith("data:image/png;base64,iVBOR"))

    def test_tool_calls_and_tool_reply(self):
        msgs = [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Bonn"}}}]},
                {"role": "tool", "content": "{\"temp\": 12}", "tool_name": "get_weather"}]
        req, _ = gw.ollama_chat_to_openai({"messages": msgs, "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]}, "m")
        a = req["messages"][0]
        self.assertIsNone(a["content"])
        self.assertEqual(a["tool_calls"][0]["type"], "function")
        self.assertEqual(a["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(a["tool_calls"][0]["function"]["arguments"]), {"city": "Bonn"})
        self.assertEqual(req["messages"][1]["role"], "tool")
        self.assertEqual(req["messages"][1]["tool_call_id"], "get_weather")
        self.assertEqual(len(req["tools"]), 1)

    def test_format_and_think(self):
        req, _ = gw.ollama_chat_to_openai({"messages": [], "format": "json", "think": False}, "m")
        self.assertEqual(req["response_format"], {"type": "json_object"})
        self.assertFalse(req["enable_thinking"])
        self.assertEqual(req["reasoning_effort"], "none")
        req, _ = gw.ollama_chat_to_openai({"messages": [], "format": {"type": "object"}, "think": "high"}, "m")
        self.assertEqual(req["response_format"]["type"], "json_schema")
        self.assertEqual(req["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertTrue(req["enable_thinking"])
        self.assertEqual(req["reasoning_effort"], "high")
        with self.assertRaises(gw.GatewayError):
            gw.ollama_chat_to_openai({"messages": [], "think": "yes"}, "m")

    def test_generate(self):
        req, raw, _ = gw.ollama_generate_to_openai({"prompt": "Say hi", "system": "Be brief", "stream": False}, "m")
        self.assertFalse(raw)
        self.assertEqual(req["messages"], [{"role": "system", "content": "Be brief"}, {"role": "user", "content": "Say hi"}])
        req, raw, _ = gw.ollama_generate_to_openai({"prompt": "<raw>", "raw": True, "options": {"num_predict": 3}}, "m")
        self.assertTrue(raw)
        self.assertEqual(req["prompt"], "<raw>")
        self.assertEqual(req["max_tokens"], 3)

    def test_bad_bodies(self):
        with self.assertRaises(gw.GatewayError):
            gw.ollama_chat_to_openai([], "m")
        with self.assertRaises(gw.GatewayError):
            gw.ollama_chat_to_openai({"messages": "no"}, "m")
        with self.assertRaises(gw.GatewayError):
            gw.ollama_chat_to_openai({"messages": [], "options": 3}, "m")


class TranslateReplies(unittest.TestCase):
    def _sse(self, *chunks, done=True):
        lines = [f"data: {json.dumps(c)}" for c in chunks]
        if done:
            lines.append("data: [DONE]")
        return lines

    def test_stream_content_thinking_and_counts(self):
        conv = gw.SseToOllama("m", started=1.0)
        out = []
        for line in self._sse({"choices": [{"index": 0, "delta": {"role": "assistant"}}]},
                              {"choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}]},
                              {"choices": [{"index": 0, "delta": {"content": "Hel"}}]},
                              {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
                              {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 2}}):
            out += conv.feed(line)
        out += conv.finish()
        self.assertEqual([o["message"].get("thinking") for o in out[:1]], ["hmm"])
        self.assertEqual([o["message"]["content"] for o in out[1:3]], ["Hel", "lo"])
        self.assertTrue(all(o["done"] is False for o in out[:3]))
        last = out[-1]
        self.assertTrue(last["done"])
        self.assertEqual(last["done_reason"], "stop")
        self.assertEqual(last["prompt_eval_count"], 7)
        self.assertEqual(last["eval_count"], 2)
        for key in ("total_duration", "load_duration", "prompt_eval_duration", "eval_duration"):
            self.assertIsInstance(last[key], int)
        self.assertEqual(last["message"], {"role": "assistant", "content": ""})

    def test_stream_think_in_content(self):
        conv = gw.SseToOllama("m", started=1.0, think_in_content=True)
        out = []
        for line in self._sse({"choices": [{"index": 0, "delta": {"reasoning_content": "a"}}]},
                              {"choices": [{"index": 0, "delta": {"content": "b"}, "finish_reason": "stop"}]}):
            out += conv.feed(line)
        out += conv.finish()
        text = "".join(o["message"]["content"] for o in out if not o["done"])
        self.assertEqual(text, "<think>a</think>b")

    def test_stream_tool_calls_gathered(self):
        conv = gw.SseToOllama("m", started=1.0)
        out = []
        for line in self._sse({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                                                    "function": {"name": "f", "arguments": "{\"a\": "}}]}}]},
                              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]},
                                            "finish_reason": "tool_calls"}]}):
            out += conv.feed(line)
        out += conv.finish()
        calls = out[-2]["message"]["tool_calls"]
        self.assertEqual(calls, [{"function": {"name": "f", "arguments": {"a": 1}}}])
        self.assertEqual(out[-1]["done_reason"], "stop")

    def test_generate_stream(self):
        conv = gw.SseToOllama("m", started=1.0, generate=True)
        out = []
        for line in self._sse({"choices": [{"index": 0, "text": "x", "finish_reason": None}]},
                              {"choices": [{"index": 0, "text": "y", "finish_reason": "length"}]}):
            out += conv.feed(line)
        out += conv.finish()
        self.assertEqual([o["response"] for o in out[:2]], ["x", "y"])
        self.assertEqual(out[-1]["done_reason"], "length")
        self.assertIn("response", out[-1])

    def test_non_stream_reply(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "Hi", "reasoning_content": "t",
                                          "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
                              "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}
        out = gw.openai_reply_to_ollama(reply, "m", started=1.0)
        self.assertEqual(out["message"]["content"], "Hi")
        self.assertEqual(out["message"]["thinking"], "t")
        self.assertEqual(out["message"]["tool_calls"], [{"function": {"name": "f", "arguments": {}}}])
        self.assertEqual(out["eval_count"], 1)
        self.assertTrue(out["done"])

    def test_error_form(self):
        st, body = gw.openai_error_to_ollama(400, b'{"error": {"message": "bad", "type": "invalid_request_error"}}')
        self.assertEqual((st, body), (400, {"error": "bad"}))
        st, body = gw.openai_error_to_ollama(502, b"gateway down")
        self.assertEqual(body, {"error": "gateway down"})

    def test_show_and_tags(self):
        entry = {"name": "qwen36-35b", "path": "/x", "family": "qwen36", "format": "colibri", "quant": "int4",
                 "size": 5, "modified_at": "2026-01-01T00:00:00.000000000Z", "digest": "sha256:abc"}
        tag = gw.tags_entry(entry)
        self.assertEqual(tag["name"], "qwen36-35b")
        self.assertEqual(tag["details"]["family"], "qwen3")
        self.assertEqual(tag["details"]["quantization_level"], "int4")
        show = gw.show_response(entry, {"context_tokens": 16384, "capabilities": ["chat"], "input_modalities": ["text", "image"]})
        self.assertEqual(show["model_info"]["qwen3.context_length"], 16384)
        self.assertEqual(show["capabilities"], ["completion", "tools", "vision", "thinking"])


class GatewayAttached(unittest.TestCase):
    """The gateway in front of a real APIServer whose engine is scripted."""

    def setUp(self):
        self.engine = ScriptedEngine(chunks=("Hel", "lo"), prompt_tokens=5)
        self.api = APIServer(("127.0.0.1", 0), self.engine, "test-model", "secret", 16, kv_slots=1)
        t = threading.Thread(target=self.api.serve_forever, args=(0.01,), daemon=True); t.start()
        self.addCleanup(t.join, timeout=2); self.addCleanup(self.api.server_close)
        self.addCleanup(self.api.shutdown); self.addCleanup(self.api.scheduler.close)
        registry = gw.ModelRegistry(None, [], {})
        registry.entries["test-model"] = {"name": "test-model", "path": "", "family": "qwen36", "format": "colibri",
                                          "quant": "int4", "params": None, "size": 0, "modified_at": "2026-01-01T00:00:00.000000000Z",
                                          "digest": "sha256:t", "args": [], "env": {}, "context_length": None, "license": "", "pinned": True}
        manager = gw.EngineManager("coli", 0, api_key="secret", engine_url=f"http://127.0.0.1:{self.api.server_port}",
                                   engine_model="test-model")
        self.gw = gw.Gateway(("127.0.0.1", 0), registry, manager, gw.Puller("coli", None), default_model="test-model")
        g = threading.Thread(target=self.gw.serve_forever, args=(0.01,), daemon=True); g.start()
        self.addCleanup(g.join, timeout=2); self.addCleanup(self.gw.server_close); self.addCleanup(self.gw.shutdown)
        self.base = f"http://127.0.0.1:{self.gw.server_port}"

    def _get(self, path):
        with urlopen(self.base + path, timeout=10) as r:
            return r.status, json.loads(r.read().decode())

    def _post(self, path, body, raw=False):
        req = Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=20) as r:
                data = r.read()
                return r.status, (data if raw else json.loads(data.decode()))
        except HTTPError as e:
            return e.code, json.loads(e.read().decode())

    def _ndjson(self, path, body):
        status, data = self._post(path, body, raw=True)
        return status, [json.loads(l) for l in data.decode().splitlines() if l.strip()]

    def test_version_tags_ps_show(self):
        self.assertEqual(self._get("/api/version")[1]["version"], gw.GATEWAY_VERSION)
        st, tags = self._get("/api/tags")
        self.assertEqual([m["name"] for m in tags["models"]], ["test-model"])
        st, ps = self._get("/api/ps")
        self.assertEqual(ps["models"][0]["name"], "test-model")
        st, show = self._post("/api/show", {"model": "test-model:latest"})
        self.assertEqual(st, 200)
        self.assertIn("completion", show["capabilities"])
        self.assertEqual(show["details"]["family"], "qwen3")
        st, body = self._post("/api/show", {"model": "nope"})
        self.assertEqual(st, 404)
        self.assertIn("not found", body["error"])

    def test_chat_stream_and_plain(self):
        st, lines = self._ndjson("/api/chat", {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200)
        text = "".join(l["message"]["content"] for l in lines if not l["done"])
        self.assertEqual(text, "Hello")
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(lines[-1]["eval_count"], 2)
        self.assertEqual(lines[-1]["prompt_eval_count"], 5)
        st, reply = self._post("/api/chat", {"model": "test-model", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        self.assertEqual(st, 200)
        self.assertEqual(reply["message"]["content"], "Hello")
        self.assertTrue(reply["done"])

    def test_generate(self):
        st, lines = self._ndjson("/api/generate", {"model": "test-model", "prompt": "hi"})
        self.assertEqual("".join(l["response"] for l in lines), "Hello")
        st, reply = self._post("/api/generate", {"model": "test-model", "prompt": "hi", "stream": False})
        self.assertEqual(reply["response"], "Hello")
        st, reply = self._post("/api/generate", {"model": "test-model", "prompt": ""})
        self.assertEqual(reply["done_reason"], "load")

    def test_unsupported_and_proxy(self):
        st, body = self._post("/api/embed", {"model": "test-model", "input": "x"})
        self.assertEqual(st, 501)
        st, body = self._post("/api/create", {"model": "x"})
        self.assertEqual(st, 501)
        req = Request(self.base + "/v1/models", headers={"Authorization": "Bearer secret"})
        with urlopen(req, timeout=10) as r:
            models = json.loads(r.read().decode())
        self.assertEqual(models["data"][0]["id"], "test-model")
        req = Request(self.base + "/health", headers={"Authorization": "Bearer secret"})
        with urlopen(req, timeout=10) as r:
            self.assertEqual(json.loads(r.read().decode())["status"], "ok")

    def test_unknown_model(self):
        st, body = self._post("/api/chat", {"model": "other", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        self.assertEqual(st, 404)


if __name__ == "__main__":
    unittest.main()
