"""Prompt-injected tool translation for families with no native tool syntax.

OLMoE and Qwen3.6 have no tool tokens in their chat templates, so the gateway
answers 400 for `tools[]` and for a `role: "tool"` turn rather than invent a
format. `COLI_TOOL_FALLBACK=1` opts into a translation that writes the
declaration, the prior assistant calls and the tool results as ordinary turns,
in the GLM wire format `parse_tool_calls()` already reads back (#1378).

These tests pin both halves of that switch: the default still refuses, and with
the flag set a full two-turn agent loop completes through
`/v1/chat/completions`. Runs against a mock engine speaking the SERVE wire
protocol, so no checkpoint is needed.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVER = HERE.parent / "openai_server.py"
sys.path.insert(0, str(HERE.parent))

# The call the injected preamble asks for, in the format parse_tool_calls reads.
CALL = ("<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>Rome</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>")

MOCK_ENGINE = r'''#!/usr/bin/env python3
import sys, os
out, inp = sys.stdout.buffer, sys.stdin.buffer
out.write(b"\x01\x01READY\x01\x01\n" + b"STAT 0 0 0 0 0\n"); out.flush()

CALL = ("<tool_call>get_weather"
        "<arg_key>location</arg_key><arg_value>Rome</arg_value>"
        "<arg_key>unit</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>")

def reply(rid, text):
    data = text.encode("utf-8")
    out.write(("DATA %s %d\n" % (rid, len(data))).encode() + data + b"\n"); out.flush()
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
        reply(rid, CALL)
    else:
        reply(rid, "Hello from the mock engine.")
'''

TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Current weather for a city",
    "parameters": {"type": "object",
                   "properties": {"location": {"type": "string"},
                                  "unit": {"type": "string"}},
                   "required": ["location"]}}}]


@unittest.skipUnless(os.name == "posix",
                     "the mock engine is a shebang script the gateway execs directly; "
                     "Windows CreateProcess cannot run it. The gateway logic under test "
                     "is platform-independent and covered by the POSIX CI jobs.")
class _FallbackBase(unittest.TestCase):
    """Boots the gateway for one arch, with COLI_TOOL_FALLBACK under test."""

    arch = None
    # config.json model_type for the family (family_registry.model_types), which
    # is not always the --arch id: qwen36's checkpoints say "qwen3_5_moe".
    model_type = None
    fallback = "1"

    @classmethod
    def setUpClass(cls):
        if cls.arch is None:
            raise unittest.SkipTest("base class")
        cls.tmp = tempfile.TemporaryDirectory()
        (Path(cls.tmp.name) / "config.json").write_text(
            json.dumps({"model_type": cls.model_type or cls.arch}), encoding="utf-8")
        mock = Path(cls.tmp.name) / "mock_engine.py"
        mock.write_text(MOCK_ENGINE)
        mock.chmod(0o755)
        cls.mock_log = Path(cls.tmp.name) / "prompts.log"
        cls.mock_log.touch()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            cls.port = probe.getsockname()[1]
        env = dict(os.environ, MOCK_LOG=str(cls.mock_log),
                   COLI_TOOL_FALLBACK=cls.fallback)
        env.pop("COLI_API_KEY", None)
        cls.server = subprocess.Popen(
            [sys.executable, str(SERVER), "--model", cls.tmp.name,
             "--engine", str(mock), "--arch", cls.arch, "--port", str(cls.port)],
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
        if getattr(cls, "server", None) is None:
            return
        cls.server.terminate()
        cls.server.wait(timeout=5)
        cls.tmp.cleanup()

    def post(self, body):
        req = urllib.request.Request(
            self.base + "/chat/completions", json.dumps(body).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def model_id(self):
        with urllib.request.urlopen(self.base + "/models", timeout=5) as resp:
            return json.loads(resp.read())["data"][0]["id"]


class _TwoTurnLoop(_FallbackBase):
    """The two-turn loop the issue asks for, with the flag on."""

    def test_tool_call_parsed_back(self):
        out = self.post({"model": self.model_id(),
                         "messages": [{"role": "user", "content": "weather in Rome?"}],
                         "tools": TOOLS, "temperature": 0, "max_tokens": 128})
        choice = out["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        calls = choice["message"].get("tool_calls") or []
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"location": "Rome", "unit": "celsius"})

    def test_second_turn_completes(self):
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
        self.assertIn("<tools>", prompts)          # declaration was injected
        self.assertIn("<tool_call>get_weather", prompts)   # prior call replayed
        self.assertIn("<tool_response>", prompts)  # result reached the prompt


class OlmoeToolFallbackE2E(_TwoTurnLoop):
    arch = "olmoe"


class Qwen36ToolFallbackE2E(_TwoTurnLoop):
    arch = "qwen36"
    model_type = "qwen3_5_moe"


class _RefusedByDefault(_FallbackBase):
    """With the flag unset the gateway must still refuse, not half-support."""

    fallback = "0"

    def _expect_400(self, body):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.post(body)
        self.assertEqual(caught.exception.code, 400)
        return json.loads(caught.exception.read())

    def test_tools_refused(self):
        err = self._expect_400({"model": self.model_id(),
                                "messages": [{"role": "user", "content": "hi"}],
                                "tools": TOOLS, "max_tokens": 16})
        self.assertIn("COLI_TOOL_FALLBACK", json.dumps(err))

    def test_tool_role_refused(self):
        self._expect_400({"model": self.model_id(),
                          "messages": [{"role": "user", "content": "hi"},
                                       {"role": "tool", "content": "18C"}],
                          "max_tokens": 16})


class OlmoeToolRefusedByDefault(_RefusedByDefault):
    arch = "olmoe"


class Qwen36ToolRefusedByDefault(_RefusedByDefault):
    arch = "qwen36"
    model_type = "qwen3_5_moe"


# The shared base classes must not run as tests themselves.
del _FallbackBase, _TwoTurnLoop, _RefusedByDefault


if __name__ == "__main__":
    unittest.main()
