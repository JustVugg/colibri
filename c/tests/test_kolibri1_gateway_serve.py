#!/usr/bin/env python3
"""The kolibri1 gateway started the way `coli serve` starts it, on the tiny fixture.

`python openai_server.py --model <tiny kolibri1 fixture> --engine tests/kolibri1_serve_stub.py`
as a real subprocess, real HTTP. Nothing in the gateway is stubbed or patched: model_type
"kolibri1" goes through resolve_model() and family_registry, the cap through cap_for_arch()
and a real build_plan() over the fixture's safetensors, the engine through the real Engine
class and the SERVE protocol. Only the engine process is a stand-in (the stub answers as a
function of the rendered prompt, see its docstring; tests/test_kolibri1_gateway_real_engine.py runs
the same gateway against the real kolibri1 SERVE loop).

Needs numpy to generate the fixture (tools/make_kolibri1_tiny.py); skipped without it.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
STUB = ROOT / "tests" / "kolibri1_serve_stub.py"
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
KEY = "k1-test-key"
WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Wetter für eine Stadt",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}


@unittest.skipUnless(HAVE_NUMPY, "numpy is needed to generate the tiny fixture")
class Kolibri1GatewayServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="k1-gateway-")
        root = Path(cls.tmp.name)
        cls.model = root / "kolibri1_tiny"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.model), "--layout", "release"],
                       check=True, capture_output=True)
        cls.log = root / "stub.jsonl"
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = dict(os.environ, KOLIBRI1_STUB_LOG=str(cls.log), RAM_GB="64", CTX="2048",
                   OMP_NUM_THREADS="2")
        env.pop("COLI_CONTINUE_ASSISTANT", None)
        cls.stderr = open(root / "gateway.err", "wb")
        cls.proc = subprocess.Popen(
            [sys.executable, str(ROOT / "openai_server.py"), "--model", str(cls.model),
             "--engine", str(STUB), "--port", str(port), "--api-key", KEY,
             "--max-tokens", "256"],
            env=env, stdout=subprocess.DEVNULL, stderr=cls.stderr)
        cls.base = f"http://127.0.0.1:{port}"
        deadline = time.time() + 60
        while True:
            try:
                with urlopen(Request(cls.base + "/v1/models",
                                     headers={"Authorization": "Bearer " + KEY}), timeout=2) as r:
                    cls.models = json.loads(r.read())
                    break
            except Exception:
                if cls.proc.poll() is not None or time.time() > deadline:
                    cls.stderr.flush()
                    raise RuntimeError("gateway did not start:\n" +
                                       (root / "gateway.err").read_text(errors="replace")[-3000:])
                time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        cls.stderr.close()
        cls.tmp.cleanup()

    def post(self, body, stream=False):
        body = dict(body, model="kolibri-1-colibri", stream=stream)
        request = Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                          headers={"Authorization": "Bearer " + KEY,
                                   "Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8")
        if not stream:
            return json.loads(raw)
        reasoning, content, calls, finish = [], [], {}, None
        for frame in raw.split("\n\n"):
            frame = frame.strip()
            if not frame.startswith("data:") or frame == "data: [DONE]":
                continue
            for choice in json.loads(frame[5:])["choices"]:
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content") or "")
                content.append(delta.get("content") or "")
                for call in delta.get("tool_calls") or []:
                    calls[call["index"]] = call
                finish = choice.get("finish_reason") or finish
        return {"choices": [{"message": {"reasoning_content": "".join(reasoning),
                                         "content": "".join(content),
                                         "tool_calls": [calls[k] for k in sorted(calls)]},
                             "finish_reason": finish}]}

    def last_logged(self):
        return json.loads(self.log.read_text(encoding="utf-8").splitlines()[-1])

    USER = [{"role": "user", "content": "Grüße"}]

    def test_the_real_registry_serves_the_model(self):
        ids = [entry["id"] for entry in self.models["data"]]
        self.assertEqual(ids, ["kolibri-1-colibri"])
        self.post({"messages": self.USER})
        first = json.loads(self.log.read_text(encoding="utf-8").splitlines()[0])
        # launch path: the engine got the model dir in SNAP, SERVE=1, the plan-derived cap as
        # argv[1] (cap_for_arch -> build_plan over the fixture) and the context from CTX
        self.assertEqual(first["env"]["SNAP"], str(self.model))
        self.assertEqual(first["env"]["SERVE"], "1")
        self.assertGreaterEqual(int(first["argv"][0]), 1)
        self.assertLessEqual(int(first["argv"][0]), 8)           # the tiny fixture has 8 experts

    def test_each_reasoning_effort_plain_and_streamed(self):
        for effort in ("none", "minimal", "low", "medium", "high", "xhigh", "max"):
            for stream in (False, True):
                with self.subTest(effort=effort, stream=stream):
                    out = self.post({"messages": self.USER, "reasoning_effort": effort}, stream)
                    message = out["choices"][0]["message"]
                    sent = self.last_logged()["prompt"]
                    label = {"none": "none", "minimal": "low", "low": "low",
                             "medium": "medium"}.get(effort, "high")
                    self.assertIn(f"Antwort ({label}): Grüße aus München \U0001f600",
                                  message["content"])
                    if effort == "none":
                        self.assertTrue(sent.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
                        self.assertFalse(message.get("reasoning_content"))
                    else:
                        self.assertTrue(sent.endswith("<|im_start|>assistant\n"))
                        self.assertIn(f"Aufwand {label}", message["reasoning_content"])
                    self.assertEqual(out["choices"][0]["finish_reason"], "stop")

    def test_tool_call_round_trip_plain_and_streamed(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                first = self.post({"messages": [{"role": "user", "content": "Wetter in Köln?"}],
                                   "tools": [WEATHER]}, stream)
                choice = first["choices"][0]
                self.assertEqual(choice["finish_reason"], "tool_calls")
                call = choice["message"]["tool_calls"][0]
                self.assertEqual(call["function"]["name"], "get_weather")
                self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "Köln"})
                self.assertIn("<tools>", self.last_logged()["prompt"])
                history = [{"role": "user", "content": "Wetter in Köln?"},
                           {"role": "assistant", "content": None,
                            "tool_calls": [{"id": call["id"], "type": "function",
                                            "function": call["function"]}]},
                           {"role": "tool", "tool_call_id": call["id"], "content": "{\"temp\": 12}"}]
                second = self.post({"messages": history, "tools": [WEATHER]}, stream)
                # the gateway's splitter keeps the "\n\n" the template puts after </think>
                self.assertEqual(second["choices"][0]["message"]["content"].lstrip("\n"),
                                 "Es sind 12 Grad in Köln.")
                self.assertEqual(second["choices"][0]["finish_reason"], "stop")
                sent = self.last_logged()["prompt"]
                self.assertIn("<tool_response>\n{\"temp\": 12}\n</tool_response>", sent)
                self.assertIn('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Köln"}}',
                              sent)

    def test_trailing_assistant_turn_is_an_open_turn(self):
        messages = self.USER + [{"role": "assistant", "content": "Im Wald, da"}]
        for stream in (False, True):
            with self.subTest(stream=stream):
                out = self.post({"messages": messages}, stream)
                sent = self.last_logged()["prompt"]
                self.assertTrue(sent.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\nIm Wald, da"))
                message = out["choices"][0]["message"]
                self.assertFalse(message.get("reasoning_content"))
                self.assertTrue(message["content"].startswith(" und dann weiter"))

    def test_bad_requests_are_400_not_engine_traffic(self):
        self.post({"messages": self.USER})              # the log exists whatever the test order
        before = len(self.log.read_text(encoding="utf-8").splitlines())
        for body in ({"messages": self.USER, "reasoning_effort": "extreme"},
                     {"messages": self.USER + [{"role": "assistant", "content": "x "}]}):
            with self.assertRaises(HTTPError) as caught:
                self.post(body)
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        self.assertEqual(len(self.log.read_text(encoding="utf-8").splitlines()), before)


if __name__ == "__main__":
    unittest.main()
