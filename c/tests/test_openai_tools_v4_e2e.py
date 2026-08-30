"""End-to-end DeepSeek V4 tool-calling test for the OpenAI gateway (SPEC-56a).

Mirrors test_openai_tools_e2e.py (#401) but exercises the deepseek_v4 arch: tool
declaration rendering in the checkpoint's native DSML encoding (v4_dsml.py,
vendored from the DeepSeek-V4-Flash reference), DSML marker suppression in
streamed deltas, tool_calls in both response shapes, and the
<tool_result> round trip. Runs against a mock engine speaking the SERVE wire
protocol; no real checkpoint needed.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

SERVER = Path(__file__).resolve().parent.parent / "openai_server.py"

# Mock engine: replies are keyed on the prompt so one process covers every case.
# Prompts received are appended to MOCK_LOG for assertions on the rendering.
MOCK_ENGINE = r'''#!/usr/bin/env python3
import sys, os
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

DSML = ("\n\n<\uff5cDSML\uff5ctool_calls>\n"
        "<\uff5cDSML\uff5cinvoke name=\"get_weather\">\n"
        "<\uff5cDSML\uff5cparameter name=\"location\" string=\"true\">Rome</\uff5cDSML\uff5cparameter>\n"
        "<\uff5cDSML\uff5cparameter name=\"unit\" string=\"true\">celsius</\uff5cDSML\uff5cparameter>\n"
        "</\uff5cDSML\uff5cinvoke>\n"
        "</\uff5cDSML\uff5ctool_calls>")

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
    if "<tool_result>" in prompt:
        reply(rid, "25 degrees and sunny in Rome.")
    elif "weather in Rome" in prompt:
        reply(rid, DSML, chunks=12)   # split across chunks: suppression must hold
    else:
        reply(rid, "Hello from the mock V4 engine.")
'''

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
                   "required": ["location"]}}}]


@unittest.skipUnless(os.name == "posix",
                     "the mock engine is a shebang script the gateway execs directly")
class ToolCallingV4E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8")
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
            [sys.executable, str(SERVER), "--model", cls.tmp.name,
             "--engine", str(mock), "--arch", "deepseek_v4", "--port", str(cls.port)],
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
                import time
                time.sleep(0.1)
        raise RuntimeError("gateway did not come up")

    @classmethod
    def tearDownClass(cls):
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.tmp.cleanup()

    def post(self, body, path="/chat/completions"):
        req = urllib.request.Request(
            self.base + path, json.dumps(body).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def model_id(self):
        with urllib.request.urlopen(self.base + "/models", timeout=5) as resp:
            return json.loads(resp.read())["data"][0]["id"]

    def test_tool_call_non_stream(self):
        out = self.post({"model": self.model_id(),
                         "messages": [{"role": "user", "content": "weather in Rome?"}],
                         "tools": TOOLS, "temperature": 0, "max_tokens": 128})
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"].get("tool_calls") or []
        self.assertEqual(len(calls), 1)
        fn = calls[0]["function"]
        self.assertEqual(fn["name"], "get_weather")
        self.assertEqual(json.loads(fn["arguments"]), {"location": "Rome", "unit": "celsius"})

    def test_tool_result_round_trip(self):
        mid = self.model_id()
        msgs = [{"role": "user", "content": "weather in Rome?"}]
        out = self.post({"model": mid, "messages": msgs, "tools": TOOLS,
                         "temperature": 0, "max_tokens": 128})
        calls = out["choices"][0]["message"]["tool_calls"]
        msgs.append({"role": "assistant", "content": None, "tool_calls": calls})
        msgs.append({"role": "tool", "tool_call_id": calls[0]["id"],
                     "content": json.dumps({"temp_c": 25})})
        out2 = self.post({"model": mid, "messages": msgs, "tools": TOOLS,
                          "temperature": 0, "max_tokens": 128})
        self.assertIn("25", out2["choices"][0]["message"]["content"] or "")
        prompts = self.mock_log.read_text()
        self.assertIn("## Tools", prompts)                       # DSML declaration
        self.assertIn("<\uff5cDSML\uff5ctool_calls>", prompts) # native block syntax
        self.assertIn("<tool_result>", prompts)                  # result merged into user turn

    def test_tool_call_streamed_markers_suppressed(self):
        req = urllib.request.Request(
            self.base + "/chat/completions",
            json.dumps({"model": self.model_id(),
                        "messages": [{"role": "user", "content": "weather in Rome?"}],
                        "tools": TOOLS, "temperature": 0, "max_tokens": 128,
                        "stream": True}).encode(),
            {"Content-Type": "application/json"})
        got_calls, leak = False, False
        with urllib.request.urlopen(req, timeout=30) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[6:])["choices"]:
                    delta = choice.get("delta", {})
                    if delta.get("tool_calls"):
                        got_calls = True
                    elif "DSML" in (delta.get("content") or ""):
                        leak = True
        self.assertTrue(got_calls, "no tool_calls deltas streamed")
        self.assertFalse(leak, "DSML markers leaked into streamed content")

    def test_no_tools_plain_text(self):
        out = self.post({"model": self.model_id(),
                         "messages": [{"role": "user", "content": "hello"}],
                         "temperature": 0, "max_tokens": 64})
        self.assertEqual(out["choices"][0]["message"]["content"],
                         "Hello from the mock V4 engine.")


if __name__ == "__main__":
    unittest.main()
