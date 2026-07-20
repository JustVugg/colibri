"""End-to-end tool-calling test for the OpenAI gateway (#401).

Unlike the unit tests in test_openai_server.py (which call parse_tool_calls /
render_chat directly), this suite runs openai_server.py as a real subprocess
against a mock engine that speaks the actual SERVE wire protocol
(READY / SUBMIT / DATA / DONE), then talks to it over real HTTP. It pins down
the full path a coding client exercises: tool declaration rendering, marker
suppression in streamed deltas (across chunk boundaries), tool_calls in both
response shapes, and the <|observation|><tool_response> round trip.
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
MODEL_ID = "glm-5.2-colibri"

# Mock engine: replies are keyed on the prompt so one process covers every case.
# Prompts received are appended to MOCK_LOG for assertions on the rendering.
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

while True:
    line = inp.readline()
    if not line: break
    f = line.decode().strip().split()
    if not f or f[0] != "SUBMIT": continue
    rid, plen = f[1], int(f[3])
    prompt = inp.read(plen).decode("utf-8", "replace"); inp.read(1)
    with open(os.environ["MOCK_LOG"], "a") as log:
        log.write(prompt + "\n\x00\n")
    if "<tool_response>" in prompt:
        reply(rid, "25 degrees and sunny in Rome.")
    elif "weather in Rome" in prompt:
        reply(rid, "<tool_call>get_weather<arg_key>city</arg_key>"
                   "<arg_value>Rome</arg_value></tool_call>")
    elif "weather in Milan" in prompt:
        # split across many tiny DATA chunks: streamed marker suppression must
        # hold even when a marker straddles a chunk boundary
        reply(rid, "Checking. <tool_call>get_weather<arg_key>city</arg_key>"
                   "<arg_value>Milan</arg_value></tool_call>", chunks=20)
    else:
        reply(rid, "Hello from the mock engine.")
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
class ToolCallingE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        mock = Path(cls.tmp.name) / "mock_engine.py"
        mock.write_text(MOCK_ENGINE)
        mock.chmod(0o755)
        cls.mock_log = Path(cls.tmp.name) / "prompts.log"
        cls.mock_log.touch()
        with socket.socket() as probe:          # free port, then hand it to the server
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, MOCK_LOG=str(cls.mock_log))
        cls.server = subprocess.Popen(
            [sys.executable, str(SERVER), "--model", cls.tmp.name,
             "--engine", str(mock), "--port", str(cls.port)],
            env=env, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}/v1"
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.base + "/models", timeout=2)
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
        resp = urllib.request.urlopen(req, timeout=30)
        if not stream:
            return json.loads(resp.read())
        events = []
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[6:]))
        return events

    def test_tool_call_non_stream(self):
        r = self.post({"model": MODEL_ID, "tools": TOOLS,
                       "messages": [{"role": "user",
                                     "content": "What is the weather in Rome?"}]})
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome"})
        self.assertNotIn("<tool_call>", choice["message"].get("content") or "")

    def test_tool_call_streamed_markers_suppressed(self):
        events = self.post({"model": MODEL_ID, "tools": TOOLS, "stream": True,
                            "messages": [{"role": "user",
                                          "content": "What is the weather in Milan?"}]},
                           stream=True)
        deltas = [e["choices"][0]["delta"] for e in events if e["choices"]]
        text = "".join(d.get("content") or "" for d in deltas)
        self.assertNotIn("<tool_call>", text)
        self.assertNotIn("<arg_key>", text)
        calls = [d["tool_calls"] for d in deltas if d.get("tool_calls")]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0][0]["function"]["arguments"]),
                         {"city": "Milan"})
        finish = [e["choices"][0]["finish_reason"] for e in events
                  if e["choices"] and e["choices"][0].get("finish_reason")]
        self.assertEqual(finish, ["tool_calls"])

    def test_tool_result_round_trip(self):
        r = self.post({"model": MODEL_ID, "tools": TOOLS, "messages": [
            {"role": "user", "content": "What is the weather in Rome?"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_x", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": "{\"city\": \"Rome\"}"}}]},
            {"role": "tool", "tool_call_id": "call_x",
             "content": "25 degrees, sunny"}]})
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertFalse(choice["message"].get("tool_calls"))
        self.assertIn("25 degrees", choice["message"]["content"])
        rendered = self.mock_log.read_text().split("\x00")[-2]
        self.assertIn("<|observation|><tool_response>25 degrees, sunny</tool_response>",
                      rendered)
        self.assertIn("# Tools", rendered)
        self.assertIn('"get_weather"', rendered)

    def test_no_tools_plain_text(self):
        r = self.post({"model": MODEL_ID,
                       "messages": [{"role": "user", "content": "Hi!"}]})
        choice = r["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertIn("mock engine", choice["message"]["content"])


if __name__ == "__main__":
    unittest.main()
