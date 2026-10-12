#!/usr/bin/env python3
"""The kolibri1 gateway with the REAL kolibri1 binary on the tiny fixture.

`openai_server.py --model <tiny fixture + tokenizer> --engine ./kolibri1` as a subprocess, real
HTTP, the real family lookup, plan and Engine class, and the real SERVE loop of kolibri1.c. This
replaces the stand-in of test_kolibri1_gateway_serve.py (kept: it checks the same gateway paths
against an engine that answers as a function of the prompt).

The tiny model is random, so what it says is noise. Two things make the answers testable:
  * K1_TEST_SERVE_SCRIPT (engine test hook, loud on stderr): the ids the engine samples are
    replaced by the tokenised lines of a file, line k answers engine request k. The forward,
    cache, finite check, stop handling, DATA framing and tokenizer decode are all real; only
    the choice of the next id is scripted. That gives a thinking block, a tool call, a final answer.
  * K1_SERVE_LOG=1: the engine logs, per request, a hash of the prompt ids it received. The test
    renders the expected prompt with the gateway's own renderer, tokenises it with the engine's
    tokenizer (`kolibri1 --tokenize`) and compares the hashes: the bytes the gateway sent are the
    bytes the engine tokenised, for every reasoning effort and for the tool round trip.
Without a script (greedy, temperature 0) the real model's own tokens are compared across requests:
isolation (A, B, A), streaming equals plain, prefix reuse over HTTP.

Needs numpy (fixture generator) and the ./kolibri1 binary; skipped without them.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kolibri1_serve_fixture as FX  # noqa: E402

BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
KEY = "k1-real-key"
EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Wetter für eine Stadt",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
USER = [{"role": "user", "content": "Grüße aus Köln"}]
HOOKS = ("K1_CAP", "K1_CHUNK", "K1_CTX", "K1_TOP_K", "K1_SEED", "SEED", "K1_STOP_IDS", "K1_SERVE_PREFIX",
         "COLI_KV_PREFIX", "K1_TEST_SERVE_SCRIPT", "K1_RELEASE", "K1_LAYERS", "K1_TEST_SERVE_IDS",
         "K1_TEST_SERVE_NAN_REQ", "K1_LOGITS", "K1_CAPTURE", "K1_FEED", "COLI_CONTINUE_ASSISTANT", "COLI_TEMP")


def fnv(ids):
    h = 1469598103934665603
    for i in ids:
        h = ((h ^ i) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def escape(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n")


@unittest.skipUnless(HAVE_NUMPY and BINARY.exists(), "needs numpy and the kolibri1 binary")
class Kolibri1GatewayRealEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="k1-real-gw-")
        cls.root = Path(cls.tmp.name)
        cls.model = FX.make_serve_fixture(cls.root / "k1")
        import openai_server
        cls.gw = openai_server

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- harness ------------------------------------------------------------------------

    def start(self, script=None, extra_env=None):
        name = f"run{len(list(self.root.glob('run*.err')))}"
        err_path = self.root / f"{name}.err"
        env = {k: v for k, v in os.environ.items() if k not in HOOKS}
        env.update({"RAM_GB": "64", "CTX": "4096", "OMP_NUM_THREADS": "2", "K1_SERVE_LOG": "1"})
        if script is not None:
            sp = self.root / f"{name}.script"
            sp.write_text("".join(escape(line) + "\n" for line in script), encoding="utf-8")
            env["K1_TEST_SERVE_SCRIPT"] = str(sp)
        env.update(extra_env or {})
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        err = open(err_path, "wb")
        proc = subprocess.Popen(
            [sys.executable, str(ROOT / "openai_server.py"), "--model", str(self.model), "--engine", str(BINARY),
             "--port", str(port), "--api-key", KEY, "--max-tokens", "3000"],
            env=env, stdout=subprocess.DEVNULL, stderr=err)
        self.addCleanup(self.stop, proc, err)
        base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 90
        while True:
            try:
                with urlopen(Request(base + "/v1/models", headers={"Authorization": "Bearer " + KEY}), timeout=2) as r:
                    json.loads(r.read())
                    break
            except Exception:
                if proc.poll() is not None or time.time() > deadline:
                    raise RuntimeError("gateway did not start:\n" + err_path.read_text(errors="replace")[-3000:])
                time.sleep(0.2)
        return base, err_path

    @staticmethod
    def stop(proc, err):
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        err.close()

    @staticmethod
    def post(base, body, stream=False, timeout=120):
        body = dict(body, model="kolibri-1-colibri", stream=stream)
        request = Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        if not stream:
            return json.loads(raw)
        reasoning, content, calls, finish, usage = [], [], {}, None, None
        for frame in raw.split("\n\n"):
            frame = frame.strip()
            if not frame.startswith("data:") or frame == "data: [DONE]":
                continue
            obj = json.loads(frame[5:])
            usage = obj.get("usage") or usage
            for choice in obj["choices"]:
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                for call in delta.get("tool_calls") or []:
                    calls[call["index"]] = call
                finish = choice.get("finish_reason") or finish
        return {"choices": [{"message": {"reasoning_content": "".join(reasoning), "content": "".join(content),
                                         "tool_calls": [calls[k] for k in sorted(calls)]},
                             "finish_reason": finish}], "usage": usage}

    def engine_log(self, err_path, count, timeout=20):
        """the engine's per-request lines [{id, prompt_tokens, reused, idhash, emitted, status}], once `count` exist"""
        end = time.time() + timeout
        while True:
            rows = []
            for ln in err_path.read_text(errors="replace").splitlines():
                if ln.startswith("[kolibri1] serve req ") and "prompt_tokens=" in ln:
                    head, rest = ln[len("[kolibri1] serve req "):].split(": ", 1)
                    row = dict(kv.split("=", 1) for kv in rest.split())
                    row["id"] = head
                    row["reused_n"] = int(row["reused"].split("(")[0])
                    rows.append(row)
            if len(rows) >= count or time.time() > end:
                return rows
            time.sleep(0.05)

    def expected_hash(self, messages, effort=None, tools=None):
        """render with the gateway's own renderer, tokenise with the engine's tokenizer, hash"""
        from unittest.mock import patch
        with patch("openai_server.ARCH", "kolibri1"):
            prompt = self.gw.render_chat_for_arch(messages, None, effort, tools, None)
        r = subprocess.run([str(BINARY), "--tokenize", str(self.model)], input=prompt.encode("utf-8"),
                           capture_output=True, check=True)
        ids = [int(x) for x in r.stdout.split()]
        return fnv(ids), len(ids)

    # ---- tests --------------------------------------------------------------------------

    def test_each_reasoning_effort_plain_and_streamed(self):
        script, plan = [], []
        for effort in EFFORTS:
            for stream in (False, True):
                if effort == "none":
                    script.append(f"Antwort {effort}: Grüße aus München \U0001f600")
                else:
                    script.append(f"<think>\nPrüfe die Aufgabe, Aufwand {effort}.\n</think>\n\nAntwort {effort}: Grüße aus München \U0001f600")
                plan.append((effort, stream))
        base, err = self.start(script)
        for effort, stream in plan:
            with self.subTest(effort=effort, stream=stream):
                out = self.post(base, {"messages": USER, "reasoning_effort": effort, "temperature": 0}, stream)
                message, choice = out["choices"][0]["message"], out["choices"][0]
                self.assertEqual(message["content"].lstrip("\n"), f"Antwort {effort}: Grüße aus München \U0001f600")
                if effort == "none":
                    self.assertFalse(message.get("reasoning_content"))
                else:
                    self.assertEqual(message["reasoning_content"].strip(), f"Prüfe die Aufgabe, Aufwand {effort}.")
                self.assertEqual(choice["finish_reason"], "stop")
        rows = self.engine_log(err, len(plan))
        self.assertEqual(len(rows), len(plan))
        seen = {}
        for (effort, stream), row in zip(plan, rows):
            want, n = self.expected_hash(USER, effort)
            self.assertEqual(row["idhash"], want, f"{effort}: the engine tokenised different bytes than the gateway rendered")
            self.assertEqual(int(row["prompt_tokens"]), n)
            self.assertEqual(row["status"], "ok")
            seen.setdefault(effort, set()).add(row["idhash"])
        self.assertTrue(all(len(v) == 1 for v in seen.values()))     # plain and streamed send the same prompt
        # the effort reaches the model: the prompts differ (none / low / medium / high share the sentences the template has)
        self.assertGreaterEqual(len({next(iter(v)) for v in seen.values()}), 4)

    def test_tool_call_round_trip_plain_and_streamed(self):
        call_text = ("<think>\nIch brauche das Wetter.\n</think>\n\nIch schaue nach.\n<tool_call>\n"
                     '{"name": "get_weather", "arguments": {"city": "Köln"}}\n</tool_call>')
        final = "<think>\nDas Ergebnis liegt vor.\n</think>\n\nEs sind 12 Grad in Köln."
        base, err = self.start([call_text, final, call_text, final])
        question = [{"role": "user", "content": "Wetter in Köln?"}]
        histories = []
        for stream in (False, True):
            with self.subTest(stream=stream):
                first = self.post(base, {"messages": question, "tools": [WEATHER], "temperature": 0}, stream)
                choice = first["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                call = choice["message"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "get_weather")
                self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Köln"})
                self.assertEqual(choice["message"]["content"].strip(), "Ich schaue nach.")
                history = question + [
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": call["id"], "type": "function", "function": call["function"]}]},
                    {"role": "tool", "tool_call_id": call["id"], "content": "{\"temp\": 12}"}]
                histories.append(history)
                second = self.post(base, {"messages": history, "tools": [WEATHER], "temperature": 0}, stream)
                self.assertEqual(second["choices"][0]["message"]["content"].strip(), "Es sind 12 Grad in Köln.")
                self.assertEqual(second["choices"][0]["finish_reason"], "stop")
        rows = self.engine_log(err, 4)
        self.assertEqual(len(rows), 4)
        h1, _ = self.expected_hash(question, None, [WEATHER])
        self.assertEqual(rows[0]["idhash"], h1)
        self.assertEqual(rows[2]["idhash"], h1)
        # the second prompt carries the tool call and the tool response: the engine got exactly the bytes the renderer
        # makes of the same history (the call id is not rendered, so it does not matter that it is random)
        self.assertEqual(rows[1]["idhash"], self.expected_hash(histories[0], None, [WEATHER])[0])
        self.assertEqual(rows[3]["idhash"], self.expected_hash(histories[1], None, [WEATHER])[0])
        self.assertNotEqual(rows[1]["idhash"], rows[0]["idhash"])
        self.assertTrue(all(r["status"] == "ok" for r in rows))

    def test_client_disconnect_cancels_and_the_engine_serves_on(self):
        long_reply = "<think>\n" + "Gedanken " * 300 + "\n</think>\n\nlange Antwort"
        base, err = self.start([long_reply, "kurz"])
        host, port = base[len("http://"):].split(":")
        conn = http.client.HTTPConnection(host, int(port), timeout=60)
        body = json.dumps({"model": "kolibri-1-colibri", "messages": USER, "stream": True, "temperature": 0,
                           "max_tokens": 3000})
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
        response = conn.getresponse()
        frames = 0
        while frames < 12:                                            # past the role frame: tokens are flowing
            line = response.fp.readline()
            self.assertTrue(line, "the stream ended before the client left")
            frames += line.startswith(b"data:")
        response.close()
        conn.close()                                                  # the client leaves while tokens are flowing
        out = self.post(base, {"messages": USER, "temperature": 0, "reasoning_effort": "none"})
        self.assertEqual(out["choices"][0]["message"]["content"].strip(), "kurz")
        rows = self.engine_log(err, 2)
        self.assertEqual([r["status"] for r in rows], ["cancelled", "ok"])
        self.assertLess(int(rows[0]["emitted"]), 2000)

    def test_concurrent_requests_are_serialised_and_both_answered(self):
        script = ["Antwort eins", "Antwort zwei"]
        base, err = self.start(script)
        results = {}

        def call(name, effort):
            results[name] = self.post(base, {"messages": USER, "reasoning_effort": effort, "temperature": 0})
        threads = [threading.Thread(target=call, args=("a", "none")), threading.Thread(target=call, args=("b", "none"))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(120)
        contents = sorted(r["choices"][0]["message"]["content"].strip() for r in results.values())
        self.assertEqual(contents, sorted(script))                    # each reply is one whole scripted answer
        rows = self.engine_log(err, 2)
        self.assertEqual([r["status"] for r in rows], ["ok", "ok"])

    def test_greedy_isolation_streaming_and_prefix_reuse_over_http(self):
        """no script: the tiny model's own greedy tokens"""
        base, err = self.start(None)
        A = USER
        B = [{"role": "user", "content": "Ganz anderer Text 12345"}]
        body = lambda m: {"messages": m, "temperature": 0, "max_tokens": 24}  # noqa: E731
        a1 = self.post(base, body(A))
        b1 = self.post(base, body(B))
        a2 = self.post(base, body(A))
        a3 = self.post(base, body(A), stream=True)
        text = lambda o: json.dumps([o["choices"][0]["message"].get("reasoning_content"), o["choices"][0]["message"].get("content")],  # noqa: E731
                                    ensure_ascii=False)
        self.assertTrue(text(a1) or a1["choices"][0]["message"].get("reasoning_content"))
        self.assertEqual(text(a1), text(a2))                          # isolation: B in between changed nothing
        self.assertEqual(text(a1), text(a3))                          # streaming equals plain
        self.assertNotEqual(text(a1), text(b1))
        # a second turn: the transcript is resent with the reply, the prompt extends the first one
        turn2 = A + [{"role": "assistant", "content": "Hallo!"}, {"role": "user", "content": "Und weiter?"}]
        self.post(base, body(turn2))
        rows = self.engine_log(err, 5)
        self.assertEqual(len(rows), 5)
        self.assertEqual([r["status"] for r in rows], ["ok"] * 5)
        self.assertEqual(rows[1]["reused_n"], 0)
        self.assertEqual(rows[2]["reused_n"], 0)                      # B -> A: no shared prefix with B's state
        self.assertEqual(rows[2]["idhash"], rows[0]["idhash"])
        self.assertEqual(rows[3]["reused_n"], 0)                      # the same prompt again recomputes
        # the second turn's prompt starts with the whole first prompt (the template renders the past assistant turn
        # as `<|im_start|>assistant\n{content}`): the snapshot at the end of the last prompt is reused
        self.assertEqual(rows[4]["reused_n"], int(rows[3]["prompt_tokens"]))
        self.assertGreater(int(rows[4]["prompt_tokens"]), rows[4]["reused_n"])


if __name__ == "__main__":
    unittest.main()
