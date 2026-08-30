"""End-to-end tool-calling test for the Kimi K3 gateway path (#1143).

Same harness shape as test_openai_tools_e2e.py: openai_server.py runs as a real
subprocess against a mock engine speaking the SERVE wire protocol, exercised
over real HTTP. What is K3-specific and pinned here: the K3CHAT1 tool records
in the rendered payload (declaration, B/F/V call history, O results), the
literal XTML markers the engine re-emits for a generated call, their
suppression in streamed deltas across chunk boundaries, and tool_calls in both
response shapes.
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

MOCK_ENGINE = r'''#!/usr/bin/env python3
import sys, os
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

def reply(rid, text, chunks=1):
    data = text.encode("utf-8")
    n = max(1, len(data) // chunks)
    for i in range(0, len(data), n):
        part = data[i:i+n]
        out.write(("DATA %s %d\n" % (rid, len(part))).encode() + part + b"\n"); out.flush()
    out.write(("DONE %s STAT %d 1.0 50.0 10.0 42 0\n" % (rid, len(text.split()))).encode())
    out.flush()

CALL = ('<|open|>tools<|sep|>'
        '<|open|>call tool="get_weather" index="1"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Rome<|close|>argument<|sep|>'
        '<|close|>call<|sep|>'
        '<|close|>tools<|sep|>')

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen = f[1], int(f[3])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    with open(os.environ["MOCK_LOG"], "a") as log:
        log.write(prompt + "\n\x00\n")
    if "O 1 11 5\nget_weathersunny" in prompt:
        reply(rid, "25 degrees and sunny in Rome.")
    elif "weather in Rome" in prompt:
        reply(rid, CALL)
    elif "weather in Milan" in prompt:
        # the markers straddle many tiny DATA chunks: streamed suppression must hold
        reply(rid, "Checking. " + CALL.replace("Rome", "Milan"), chunks=25)
    else:
        reply(rid, "Hello from the mock K3 engine.")
'''

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


@unittest.skipUnless(os.name == "posix",
                     "the mock engine is a shebang script the gateway execs directly; "
                     "Windows CreateProcess cannot run it. The gateway logic under test "
                     "is platform-independent and covered by the POSIX CI jobs.")
class K3ToolCallingE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(
            json.dumps({"model_type": "kimi_k3"}), encoding="utf-8")
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
             "--engine", str(mock), "--port", str(cls.port)],
            env=env, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.base + "/models", timeout=2) as resp:
                    cls.model_id = json.loads(resp.read())["data"][0]["id"]
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

    def post(self, body, stream=False):
        req = urllib.request.Request(
            self.base + "/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            if stream:
                return resp.read().decode()
            return json.loads(resp.read().decode())

    def prompts(self):
        return [p for p in self.mock_log.read_text().split("\n\x00\n") if p]

    def test_declaration_reaches_the_wire_and_call_parses(self):
        out = self.post({"model": self.model_id, "tools": TOOLS,
                         "messages": [{"role": "user",
                                       "content": "What is the weather in Rome?"}]})
        msg = out["choices"][0]["message"]
        self.assertEqual(out["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(len(msg["tool_calls"]), 1)
        call = msg["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(json.loads(call["arguments"]), {"city": "Rome"})
        # the rendered K3CHAT1 payload carried the tool declaration record
        last = self.prompts()[-1]
        self.assertIn("Y 12 ", last)
        self.assertIn("tool-declare# Tools", last)
        self.assertNotIn("<|open|>", last)   # gateway ships records, never raw XTML

    def test_round_trip_renders_call_history_and_result(self):
        out = self.post({"model": self.model_id, "tools": TOOLS, "messages": [
            {"role": "user", "content": "What is the weather in Rome?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]})
        msg = out["choices"][0]["message"]
        self.assertIn("sunny", msg["content"])
        self.assertFalse(msg.get("tool_calls"))
        last = self.prompts()[-1]
        self.assertIn("B 0 0 0 1\n", last)
        self.assertIn("F 11 1\nget_weather", last)
        self.assertIn("V 4 6 4\ncitystringRome", last)
        self.assertIn("O 1 11 5\nget_weathersunny", last)

    def test_streamed_markers_are_suppressed_across_chunks(self):
        raw = self.post({"model": self.model_id, "tools": TOOLS, "stream": True,
                         "messages": [{"role": "user",
                                       "content": "What is the weather in Milan?"}]},
                        stream=True)
        self.assertNotIn("<|open|>", raw)
        self.assertNotIn("<|sep|>", raw)
        deltas = [json.loads(l[len("data: "):]) for l in raw.splitlines()
                  if l.startswith("data: ") and l != "data: [DONE]"]
        text = "".join(d["choices"][0]["delta"].get("content") or "" for d in deltas)
        self.assertEqual(text.strip(), "Checking.")
        calls = [d for d in deltas if d["choices"][0]["delta"].get("tool_calls")]
        self.assertTrue(calls, "no tool_calls delta in the stream")
        fn = calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]
        self.assertEqual(fn["name"], "get_weather")

    def test_tool_choice_none_omits_declaration(self):
        self.post({"model": self.model_id, "tools": TOOLS, "tool_choice": "none",
                   "messages": [{"role": "user", "content": "Hello"}]})
        last = self.prompts()[-1]
        self.assertNotIn("tool-declare", last)


if __name__ == "__main__":
    unittest.main()
