"""End-to-end DeepSeek V4.1 tool-calling test for the OpenAI gateway.

Mirrors test_openai_tools_v4_e2e.py against the deepseek_v41 arch: tool
declaration in the checkpoint's native DSML encoding (v41_dsml.py, vendored
from the DeepSeek-V4.1-Flash reference), DSML marker suppression in streamed
deltas, tool_calls in both response shapes, and the <tool_result> round trip.
Runs against a mock engine speaking the SERVE wire protocol; no real
checkpoint needed.

The second class needs no server at all: it renders the checkpoint's own
encoding test cases and compares them byte for byte with the vendor's expected
prompts, which is the only check that catches a tag name drifting by one
character.
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

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "openai_server.py"
sys.path.insert(0, str(HERE.parent))

ASSISTANT = "<｜Assistant｜>"

# Mock engine: replies are keyed on the prompt so one process covers every case.
# Prompts received are appended to MOCK_LOG for assertions on the rendering.
# Note the space inside every tag name: that is V4.1's format, not a typo.
MOCK_ENGINE = r'''#!/usr/bin/env python3
import sys, os
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

DSML = ("\n\n<｜DSML｜ calls>\n"
        "<｜DSML｜ invoke name=\"get_weather\">\n"
        "<｜DSML｜ parameter name=\"location\" string=\"true\">Rome</｜DSML｜ parameter>\n"
        "<｜DSML｜ parameter name=\"unit\" string=\"true\">celsius</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>")

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
        reply(rid, "Hello from the mock V4.1 engine.")
'''

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"}, "unit": {"type": "string"}},
                   "required": ["location"]}}}]


@unittest.skipUnless(os.name == "posix",
                     "the mock engine is a shebang script the gateway execs directly")
class ToolCallingV41E2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v41"}), encoding="utf-8")
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
             "--engine", str(mock), "--arch", "deepseek_v41", "--port", str(cls.port)],
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
        self.assertIn("## Tools", prompts)                         # DSML declaration
        self.assertIn("<｜DSML｜ calls>", prompts)         # native block syntax
        self.assertNotIn("<｜DSML｜calls>", prompts)       # V4 spelling, without the space
        self.assertIn("<tool_result>", prompts)                    # result merged into user turn

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
                         "Hello from the mock V4.1 engine.")


class Dsv41VendorEncodingCases(unittest.TestCase):
    """The checkpoint's own encoding test cases, rendered by the gateway.

    fixtures/dsv41_encoding_caseN.json and .txt are copies of
    encoding/tests/test_input_N.json and test_output_N.txt from the
    DeepSeek-V4.1-Flash reference: a conversation and the exact prompt its
    encoder produces for it. The gateway renders a GENERATION prompt, so where
    a case ends on an assistant turn the comparison drops that turn and
    expects the assistant header plus the thinking cue the reference itself
    appends after the preceding user turn.

    Two of the five reference cases are not modelled by the gateway and are
    not copied here: case 3 and case 4 use the `latest_reminder` role and the
    internal classification `task` tokens, and case 5 carries multimodal
    content blocks that the gateway expands into image placeholders upstream
    of the renderer (expand_dsv41_images), not inside it.
    """

    @staticmethod
    def _case(number):
        payload = json.loads((HERE / f"fixtures/dsv41_encoding_case{number}.json").read_text())
        if isinstance(payload, list):
            payload = {"messages": payload}
        vendor = (HERE / f"fixtures/dsv41_encoding_case{number}.txt").read_text()
        return payload, vendor

    def _assert_case(self, number, reasoning_effort=None):
        import openai_server

        payload, vendor = self._case(number)
        messages = payload["messages"]
        thinking = payload.get("thinking_mode") == "thinking"
        expected = vendor
        if messages[-1].get("role") == "assistant":
            messages = messages[:-1]
            head = vendor.rfind(ASSISTANT) + len(ASSISTANT)
            expected = vendor[:head] + ("<think>" if thinking else "</think>")
        rendered = openai_server.render_chat_dsv41(
            messages, enable_thinking=thinking, reasoning_effort=reasoning_effort,
            tools=payload.get("tools"))
        self.assertEqual(rendered, expected)

    def test_case1_tools_and_tool_result_in_thinking_mode(self):
        # two tools declared, an assistant turn with a DSML call, a tool result
        self._assert_case(1)

    def test_case2_plain_multi_turn_chat(self):
        self._assert_case(2)

    def test_tag_names_carry_the_leading_space(self):
        import v41_dsml

        self.assertEqual(v41_dsml.TOOL_CALLS_OPEN, "<｜DSML｜ calls>")
        self.assertEqual(v41_dsml.TOOL_CALL_PREFIX, "<｜DSML｜ invoke")
        block = v41_dsml.render_tool_calls(
            [{"type": "function",
              "function": {"name": "f", "arguments": json.dumps({"a": "x", "b": 2})}}])
        self.assertIn('<｜DSML｜ parameter name="a" string="true">x', block)
        self.assertIn('<｜DSML｜ parameter name="b" string="false">2', block)
        content, calls = v41_dsml.parse_completion_text("said it" + block)
        self.assertEqual(content, "said it")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"a": "x", "b": 2})

    def test_truncated_block_never_reaches_the_client(self):
        import openai_server

        openai_server.ARCH = "deepseek_v41"
        try:
            content, calls = openai_server.parse_dsv41_tool_calls(
                "here you go\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\"")
        finally:
            openai_server.ARCH = "glm"
        self.assertEqual(calls, [])
        self.assertEqual(content, "here you go")


if __name__ == "__main__":
    unittest.main()
