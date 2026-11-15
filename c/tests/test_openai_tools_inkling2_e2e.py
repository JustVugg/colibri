"""End-to-end Inkling-2 tool calling through the gateway, against a mock engine speaking the
SERVE wire protocol (no checkpoint needed).

The mock answers as Inkling-2-Small does (its reply copied from a run of the real model): a
reasoning part, then the call as a model message of its own under the tool's name,
<|content_invoke_tool_json|>{"name": ..., "args": {...}}. The gateway must turn it into
tool_calls in both response shapes and in Anthropic's, keep the markers out of the streamed
content, and render the call and its result back in the template's form.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "openai_server.py"

MOCK_ENGINE = r'''#!/usr/bin/env python3
import sys, os
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

CALL = ("<|message_model|><|content_thinking|>I should look it up.<|end_message|>"
        "<|message_model|>get_weather<|content_invoke_tool_json|>"
        '{"name":"get_weather","args":{"city":"Rome"}}<|end_message|>')

def reply(rid, text, chunks=1):
    data = text.encode("utf-8")
    n = max(1, len(data) // chunks)
    for i in range(0, len(data), n):
        part = data[i:i+n]
        out.write(("DATA %s %d\n" % (rid, len(part))).encode() + part + b"\n"); out.flush()
    out.write(("DONE %s STAT %d 1.0 50.0 10.0 42 0\n" % (rid, len(text.split()))).encode())
    out.flush()

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen = f[1], int(f[3])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    with open(os.environ["MOCK_LOG"], "a") as log:
        log.write(prompt + "\n\x00\n")
    if "<|message_tool|>" in prompt:
        reply(rid, "<|message_model|><|content_text|>25 degrees and sunny in Rome.<|end_message|>")
    elif "weather in Rome" in prompt:
        reply(rid, CALL, chunks=13)   # markers split across chunks
    else:
        reply(rid, "Hello from the mock Inkling-2 engine.<|end_message|>")
'''

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


@unittest.skipUnless(os.name == "posix", "the mock engine is a shebang script the gateway execs directly")
class ToolCallingInkling2E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(json.dumps({
            "model_type": "inkling2", "eos_token_id": 200006,
            "text_config": {"hidden_size": 128, "num_hidden_layers": 2}}), encoding="utf-8")
        mock = Path(cls.tmp.name) / "mock_engine.py"
        mock.write_text(MOCK_ENGINE)
        mock.chmod(0o755)
        cls.mock_log = Path(cls.tmp.name) / "prompts.log"
        cls.mock_log.touch()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, MOCK_LOG=str(cls.mock_log))
        env.pop("COLI_API_KEY", None)
        cls.server = subprocess.Popen(
            [sys.executable, str(SERVER), "--model", cls.tmp.name, "--engine", str(mock),
             "--arch", "inkling", "--port", str(cls.port), "--cap", "8"],
            env=env, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.base + "/models", timeout=2):
                    pass
                return
            except OSError:
                if cls.server.poll() is not None:
                    raise RuntimeError("gateway exited during startup")
                time.sleep(0.1)
        raise RuntimeError("gateway did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.tmp.cleanup()

    def post(self, body, path="/chat/completions"):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def model_id(self):
        with urllib.request.urlopen(self.base + "/models", timeout=5) as resp:
            return json.loads(resp.read())["data"][0]["id"]

    def ask(self, **extra):
        return {"model": self.model_id(), "messages": [{"role": "user", "content": "weather in Rome?"}],
                "tools": TOOLS, "temperature": 0, "max_tokens": 128, **extra}

    def test_tool_call_non_stream(self):
        choice = self.post(self.ask())["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        message = choice["message"]
        self.assertFalse(message.get("content"))
        self.assertEqual(message.get("reasoning_content"), "I should look it up.")
        calls = message.get("tool_calls") or []
        self.assertEqual([(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls],
                         [("get_weather", {"city": "Rome"})])

    def test_tool_result_round_trip(self):
        msgs = [{"role": "user", "content": "weather in Rome?"}]
        calls = self.post(self.ask())["choices"][0]["message"]["tool_calls"]
        msgs += [{"role": "assistant", "content": None, "tool_calls": calls},
                 {"role": "tool", "tool_call_id": calls[0]["id"], "content": "25 C, sunny"}]
        out = self.post(self.ask(messages=msgs))
        self.assertIn("25 degrees", out["choices"][0]["message"]["content"] or "")
        prompts = self.mock_log.read_text()
        self.assertIn("<|message_system|>tool_declare<|content_xml|>[{", prompts)
        self.assertIn('<|message_model|>get_weather<|content_invoke_tool_json|>'
                      '{"name":"get_weather","args":{"city":"Rome"}}<|end_message|><|content_model_end_sampling|>'
                      '<|message_tool|>get_weather<|content_text|>25 C, sunny<|end_message|>', prompts)

    def test_tool_call_streamed_markers_suppressed(self):
        req = urllib.request.Request(self.base + "/chat/completions", json.dumps(self.ask(stream=True)).encode(),
                                     {"Content-Type": "application/json"})
        calls, content, reasoning, finish = [], "", "", None
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:])["choices"]:
                    delta = choice.get("delta", {})
                    calls += delta.get("tool_calls") or []
                    content += delta.get("content") or ""
                    reasoning += delta.get("reasoning_content") or ""
                    finish = choice.get("finish_reason") or finish
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome"})
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(reasoning, "I should look it up.")
        for leak in ("<|", "", "", '"args"'):
            self.assertNotIn(leak, content)

    def test_anthropic_tool_use(self):
        out = self.post({"model": self.model_id(), "max_tokens": 128,
                         "messages": [{"role": "user", "content": "weather in Rome?"}],
                         "tools": [{"name": "get_weather", "description": "Current weather for a city",
                                    "input_schema": TOOLS[0]["function"]["parameters"]}]}, "/messages")
        uses = [b for b in out["content"] if b.get("type") == "tool_use"]
        self.assertEqual([(b["name"], b["input"]) for b in uses], [("get_weather", {"city": "Rome"})])
        self.assertEqual(out["stop_reason"], "tool_use")

    def test_no_tools_plain_text(self):
        out = self.post({"model": self.model_id(), "messages": [{"role": "user", "content": "hello"}],
                         "temperature": 0, "max_tokens": 64})
        self.assertEqual(out["choices"][0]["message"]["content"], "Hello from the mock Inkling-2 engine.")


if __name__ == "__main__":
    unittest.main()
