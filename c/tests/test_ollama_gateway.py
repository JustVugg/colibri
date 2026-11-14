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

    def test_think_default_from_the_model_entry(self):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertEqual(gw.apply_think_default(body, "low")["think"], "low")
        self.assertNotIn("think", body)                      # the request's body is left as it was
        self.assertIs(gw.apply_think_default(body, None), body)
        self.assertFalse(gw.apply_think_default(body, False)["think"])
        self.assertTrue(gw.apply_think_default({"messages": [], "think": True}, "low")["think"])   # explicit wins
        self.assertFalse(gw.apply_think_default({"messages": [], "think": False}, "low")["think"])
        with self.assertRaises(gw.GatewayError):
            gw.apply_think_default(body, "xhigh")
        req, _ = gw.ollama_chat_to_openai(gw.apply_think_default(body, "low"), "m")
        self.assertEqual((req["enable_thinking"], req["reasoning_effort"]), (True, "low"))

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

    def test_load_time_is_not_eval_time(self):
        import time
        started = time.time() - 100.0            # the request began 100 s ago; 90 of them were the load
        out = gw.openai_reply_to_ollama({"choices": [{"message": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}}, "m", started, load_seconds=90.0)
        self.assertGreaterEqual(out["total_duration"], 100 * 10**9)
        self.assertEqual(out["load_duration"], 90 * 10**9)
        self.assertLess(out["eval_duration"], 11 * 10**9)
        conv = gw.SseToOllama("m", started=started, load_seconds=90.0)
        lines = conv.feed('data: {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": "stop"}]}')
        lines += conv.finish()
        self.assertLess(lines[-1]["prompt_eval_duration"], 11 * 10**9)
        self.assertEqual(lines[-1]["load_duration"], 90 * 10**9)

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

    def test_openai_routes_resolve_the_model(self):
        """An OpenAI client names the model as Ollama's clients do: aliases and :latest resolve,
        the engine gets the id it serves, an unknown name is OpenAI's 404."""
        st, reply = self._post("/v1/chat/completions", {"model": "test-model:latest", "stream": False,
                                                        "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 200, reply)
        self.assertEqual(reply["model"], "test-model")
        self.assertEqual(reply["choices"][0]["message"]["content"], "Hello")
        st, reply = self._post("/v1/chat/completions", {"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(st, 404)
        self.assertEqual(reply["error"]["code"], "model_not_found")
        st, models = self._get("/v1/models/test-model:latest")
        self.assertEqual(models["id"], "test-model")
        st, body = self._get("/v1/models")
        self.assertEqual([m["id"] for m in body["data"]], ["test-model"])

    def test_think_default_openai(self):
        body = {"messages": []}
        self.assertEqual(gw.apply_think_default_openai(body, "low"), {"messages": [], "enable_thinking": True, "reasoning_effort": "low"})
        self.assertEqual(gw.apply_think_default_openai(body, False), {"messages": [], "enable_thinking": False, "reasoning_effort": "none"})
        self.assertEqual(gw.apply_think_default_openai(body, True), {"messages": [], "enable_thinking": True})
        self.assertIs(gw.apply_think_default_openai(body, None), body)
        explicit = {"messages": [], "reasoning_effort": "none"}
        self.assertIs(gw.apply_think_default_openai(explicit, "low"), explicit)

    def test_unknown_model(self):
        st, body = self._post("/api/chat", {"model": "other", "messages": [{"role": "user", "content": "hi"}], "stream": False})
        self.assertEqual(st, 404)


FAKE_COLI = r"""
import json, os, sys, time
# a stand-in for `coli setup --yes --model ID --no-start --no-browser --dir ROOT`: it writes the
# install-state.json the real setup writes, in the phases the gateway turns into Ollama's status lines
home = os.environ["COLI_SETUP_HOME"]; os.makedirs(home, exist_ok=True)
state = os.path.join(home, "install-state.json")
model = sys.argv[sys.argv.index("--model") + 1]
def put(d):
    with open(state + ".tmp", "w") as f: json.dump(d, f)
    os.replace(state + ".tmp", state)
for done in (0, 500, 1000):
    put({"phase": "download", "model": model, "file": "model-00000.safetensors", "total": 1000, "done": done}); time.sleep(0.5)
put({"phase": "build", "model": model}); time.sleep(0.15)
if model == "broken":
    print("the converter said no"); sys.exit(3)
root = sys.argv[sys.argv.index("--dir") + 1]
os.makedirs(os.path.join(root, model), exist_ok=True)
with open(os.path.join(root, model, "config.json"), "w") as f: json.dump({"model_type": "qwen3_5_moe"}, f)
with open(os.path.join(root, model, "qwen36_meta.json"), "w") as f: json.dump({"ebits": 4, "group_size": 64}, f)
put({"phase": "ready", "model": model})
"""


class _Entry:
    def __init__(self, id):
        self.id = id


class PullWithFakeSetup(unittest.TestCase):
    """/api/pull runs `coli setup` as a child and reads its install-state.json: here a stand-in
    coli writes the phases, so the status lines, the failure and the registry refresh are checked
    without a download."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="coli-gw-pull-")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.coli = os.path.join(self.tmp, "coli")
        with open(self.coli, "w") as f:
            f.write(FAKE_COLI)
        self.models = os.path.join(self.tmp, "models"); os.makedirs(self.models)
        self.home = os.path.join(self.tmp, "home")
        self.old_home = os.environ.get("COLI_SETUP_HOME")
        os.environ["COLI_SETUP_HOME"] = self.home     # the parent's state_path() reads it too
        self.addCleanup(self._restore_home)
        registry = gw.ModelRegistry(self.models, [], {})
        manager = gw.EngineManager(self.coli, 0, engine_url="http://127.0.0.1:1", engine_model="none")
        puller = gw.Puller(self.coli, self.models, setup_home=self.home)
        puller.catalog = lambda: {"tiny": _Entry("tiny"), "broken": _Entry("broken")}
        self.gw = gw.Gateway(("127.0.0.1", 0), registry, manager, puller)
        g = threading.Thread(target=self.gw.serve_forever, args=(0.01,), daemon=True); g.start()
        self.addCleanup(g.join, timeout=2); self.addCleanup(self.gw.server_close); self.addCleanup(self.gw.shutdown)
        self.base = f"http://127.0.0.1:{self.gw.server_port}"

    def _restore_home(self):
        if self.old_home is None:
            os.environ.pop("COLI_SETUP_HOME", None)
        else:
            os.environ["COLI_SETUP_HOME"] = self.old_home

    def _post(self, path, body):
        req = Request(self.base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=30) as r:
                return r.status, r.read().decode()
        except HTTPError as e:
            return e.code, e.read().decode()

    def test_pull_streams_progress_then_lists_the_model(self):
        st, text = self._post("/api/pull", {"model": "tiny"})
        self.assertEqual(st, 200)
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
        self.assertEqual(lines[0], {"status": "pulling manifest"})
        progress = [l for l in lines if l["status"].startswith("pulling model-00000")]
        self.assertTrue(progress, lines)
        self.assertEqual(progress[-1]["total"], 1000)
        self.assertEqual(progress[-1]["completed"], 1000)
        self.assertEqual(lines[-1], {"status": "success"})
        with urlopen(self.base + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read().decode())
        self.assertEqual([m["name"] for m in tags["models"]], ["tiny"])
        self.assertEqual(tags["models"][0]["details"]["quantization_level"], "int4-g64")
        self.assertEqual(tags["models"][0]["details"]["format"], "colibri")
        # pulling a model that is there is a no-op success, as with Ollama
        st, text = self._post("/api/pull", {"model": "tiny", "stream": False})
        self.assertEqual(json.loads(text), {"status": "success"})
        # and /api/delete removes it (it is under the models directory)
        req = Request(self.base + "/api/delete", data=json.dumps({"model": "tiny"}).encode(), method="DELETE",
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=10) as r:
            self.assertEqual(json.loads(r.read().decode()), {"status": "success"})
        self.assertFalse(os.path.exists(os.path.join(self.models, "tiny")))

    def test_pull_failure_and_unknown(self):
        st, text = self._post("/api/pull", {"model": "broken"})
        lines = [json.loads(l) for l in text.splitlines() if l.strip()]
        self.assertIn("error", lines[-1])
        self.assertIn("the converter said no", lines[-1]["error"])
        st, text = self._post("/api/pull", {"model": "nothing-like-it", "stream": False})
        self.assertEqual(st, 404)
        self.assertIn("not in colibri's catalog", json.loads(text)["error"])


class ConfigReload(unittest.TestCase):
    def test_an_edited_gateway_json_is_taken_on_refresh(self):
        import tempfile
        tmp = tempfile.mkdtemp(prefix="coli-gw-cfg-")
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        model = os.path.join(tmp, "m"); os.makedirs(model)
        with open(os.path.join(model, "config.json"), "w") as f:
            json.dump({}, f)
        cfg = os.path.join(tmp, "gateway.json")
        def write(ctx, think):
            with open(cfg, "w") as f:
                json.dump({"models": {"m": {"path": model, "ctx": ctx, "think": think, "aliases": ["mm"]}}}, f)
        write(16384, "low")
        reg = gw.ModelRegistry(None, [], gw.load_gateway_config(cfg), config_path=cfg)
        self.assertEqual((reg.entries["m"]["context_length"], reg.entries["m"]["think"]), (16384, "low"))
        write(65536, False)
        reg.refresh()
        self.assertEqual((reg.entries["m"]["context_length"], reg.entries["m"]["think"]), (65536, False))
        self.assertIs(reg.resolve("mm"), reg.entries["m"])
        with open(cfg, "w") as f:
            f.write("{ broken")
        reg.refresh()                     # a broken file: the last good config stays
        self.assertEqual(reg.entries["m"]["context_length"], 65536)


class ModelFiles(unittest.TestCase):
    """What /api/tags says about a folder: the converter's meta file, a GGUF's type, a sidecar."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp(prefix="coli-gw-files-")
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)

    def _dir(self, name, files):
        d = os.path.join(self.tmp, name); os.makedirs(d)
        for fname, content in files.items():
            if content is None:
                os.makedirs(os.path.join(d, fname))
            else:
                with open(os.path.join(d, fname), "w") as f:
                    json.dump(content, f)
        return d

    def test_labels(self):
        d = self._dir("i4", {"config.json": {"model_type": "qwen3_5_moe"}, "qwen36_meta.json": {"ebits": 4}})
        self.assertEqual((gw._model_format(d), gw._model_quant(d)), ("colibri", "int4"))
        d = self._dir("i4g", {"config.json": {}, "qwen38_meta.json": {"ebits": 4, "group_size": 64}})
        self.assertEqual((gw._model_format(d), gw._model_quant(d)), ("colibri", "int4-g64"))
        d = self._dir("gg", {"config.json": {}, "gguf.json": {"format": "colibri-gguf-v1", "gguf": "/x/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"}})
        self.assertEqual((gw._model_format(d), gw._model_quant(d)), ("gguf", "Q4_K_M"))
        d = self._dir("fp8", {"config.json": {"quantization_config": {"quant_method": "fp8"}}, "experts-int4g64": None})
        self.assertEqual((gw._model_format(d), gw._model_quant(d)), ("safetensors", "FP8+int4-g64"))
        d = self._dir("bf16", {"config.json": {"torch_dtype": "bfloat16"}})
        self.assertEqual((gw._model_format(d), gw._model_quant(d)), ("safetensors", "bfloat16"))


class StopFile(unittest.TestCase):
    def test_coli_stop_request_ends_the_gateway(self):
        import tempfile
        registry = gw.ModelRegistry(None, [], {})
        manager = gw.EngineManager("coli", 0, engine_url="http://127.0.0.1:1", engine_model="none")
        server = gw.Gateway(("127.0.0.1", 0), registry, manager, gw.Puller("coli", None))
        fd, server.stopfile = tempfile.mkstemp(prefix="coli-gw-stop-"); os.close(fd)
        t = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True); t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "the gateway did not stop on the stop file")
        self.assertFalse(os.path.exists(server.stopfile), "the stop file was not taken")
        server.server_close()


if __name__ == "__main__":
    unittest.main()


class KeepAlive(unittest.TestCase):
    """Ollama's keep_alive: durations, and the idle unload between requests."""

    def test_parse(self):
        p = gw.parse_keep_alive
        self.assertEqual(p(None, default=-1.0), -1.0)
        self.assertEqual(p("20m"), 1200.0)
        self.assertEqual(p("1h30m"), 5400.0)
        self.assertEqual(p("90s"), 90.0)
        self.assertEqual(p("300"), 300.0)
        self.assertEqual(p(300), 300.0)
        self.assertEqual(p(0), 0.0)
        self.assertEqual(p("0"), 0.0)
        self.assertEqual(p("0m"), 0.0)
        self.assertEqual(p(-1), -1.0)
        self.assertEqual(p("-1"), -1.0)
        self.assertEqual(p("forever", default=1200.0), 1200.0)
        self.assertEqual(p(True, default=5.0), 5.0)

    def _manager(self, keep_alive):
        m = gw.EngineManager("coli", 0, keep_alive=keep_alive)
        m.current, m.state = "m", "ready"
        m.stopped = []
        m._stop_locked = lambda: (m.stopped.append(m.current), setattr(m, "current", None), setattr(m, "state", "stopped"))
        return m

    def test_idle_unload_after_the_request(self):
        m = self._manager(1200.0)
        entry = {"name": "m"}
        m.begin(); m.end(entry, None, now=1000.0)
        self.assertEqual(m.expires_at, 2200.0)
        self.assertIsNone(m.idle_check(now=2199.0))
        self.assertEqual(m.idle_check(now=2200.0), "m")
        self.assertEqual(m.stopped, ["m"]); self.assertEqual(m.state, "stopped")

    def test_request_keep_alive_wins_then_the_entry_then_the_gateway(self):
        m = self._manager(1200.0)
        m.begin(); m.end({"name": "m", "keep_alive": "5m"}, "1h", now=0.0)
        self.assertEqual(m.expires_at, 3600.0)
        m.begin(); m.end({"name": "m", "keep_alive": "5m"}, None, now=0.0)
        self.assertEqual(m.expires_at, 300.0)
        m.begin(); m.end({"name": "m"}, None, now=0.0)
        self.assertEqual(m.expires_at, 1200.0)
        m.begin(); m.end({"name": "m"}, -1, now=0.0)
        self.assertIsNone(m.expires_at)
        self.assertIsNone(m.idle_check(now=1e9))

    def test_never_under_a_request_and_not_attached(self):
        m = self._manager(10.0)
        m.begin(); m.end({"name": "m"}, None, now=0.0)
        m.begin()                                   # a second request runs
        self.assertIsNone(m.idle_check(now=100.0))  # busy: stays
        m.end({"name": "m"}, None, now=100.0)
        self.assertEqual(m.idle_check(now=110.0), "m")
        a = gw.EngineManager("coli", 0, engine_url="http://127.0.0.1:1", engine_model="m", keep_alive=1.0)
        a.begin(); a.end({"name": "m"}, None, now=0.0)
        self.assertIsNone(a.idle_check(now=1e9))    # attached: never spawned, never stopped

    def test_a_request_without_a_model_leaves_the_timer(self):
        m = self._manager(1200.0)
        m.begin(); m.end({"name": "m"}, None, now=0.0)
        m.begin(); m.end(None, None, now=1000.0)    # /api/show
        self.assertEqual(m.expires_at, 1200.0)

    def test_ps_shows_the_expiry(self):
        out = gw.ps_entry({"name": "m", "path": ""}, 0, expires_at=0.0)
        self.assertEqual(out["expires_at"], "1970-01-01T00:00:00.000000000Z")
        self.assertEqual(gw.ps_entry({"name": "m", "path": ""}, 0)["expires_at"], "0001-01-01T00:00:00Z")
