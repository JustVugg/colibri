"""COLI_TOOL_TRACE: one JSON line per chat request with tools, and nothing else changes.

Runs an in-process gateway (default glm family) on a scripted engine; no engine binary.
"""
import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from openai_server import APIServer

TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}}}}}]
CALL = ["deciding to call", "</think>",
        "<tool_call>get_weather<arg_key>city</arg_key>", "<arg_value>Paris</arg_value></tool_call>"]
CUT = ["thinking", "</think>", "Leggo.", "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Par"]


class _Engine:
    def __init__(self, chunks, length_limited=False):
        self.chunks, self.length_limited = chunks, length_limited

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        if on_accept is not None:
            on_accept({"prompt_tokens": 7})
        for chunk in self.chunks:
            on_text(chunk)
        return {"prompt_tokens": 7, "completion_tokens": len(self.chunks),
                "length_limited": self.length_limited}


class ToolTraceTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.trace = Path(self.dir.name) / "trace.jsonl"
        quiet = contextlib.redirect_stderr(io.StringIO())   # the parsers' [api] lines
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)

    def _post(self, chunks, stream, tools=TOOLS, length_limited=False):
        server = APIServer(("127.0.0.1", 0), _Engine(chunks, length_limited), "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        body = {"model": "test-model", "stream": stream, "enable_thinking": True,
                "messages": [{"role": "user", "content": "weather?"}]}
        if tools:
            body["tools"] = tools
        req = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                      data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            return response.headers["x-request-id"], response.read().decode()

    def _lines(self):
        return [json.loads(line) for line in self.trace.read_text(encoding="utf-8").splitlines()]

    def test_off_writes_nothing(self):
        with patch.dict(os.environ):
            os.environ.pop("COLI_TOOL_TRACE", None)
            self._post(CALL, True)
            self._post(CALL, False)
        self.assertFalse(self.trace.exists())

    def test_replies_identical_on_and_off(self):
        def strip(raw, stream):
            # ids, created and the request id are per request; everything else must match
            if not stream:
                reply = json.loads(raw)
                reply.pop("id"), reply.pop("created")
                for call in reply["choices"][0]["message"].get("tool_calls", []):
                    call.pop("id")
                return reply
            events = []
            for line in raw.splitlines():
                if line.startswith("data: {"):
                    event = json.loads(line[6:])
                    event.pop("id"), event.pop("created")
                    for choice in event["choices"]:
                        for call in (choice.get("delta") or {}).get("tool_calls", []):
                            call.pop("id")
                    events.append(event)
            return events
        for stream in (True, False):
            for chunks in (CALL, CUT):
                with patch.dict(os.environ):
                    os.environ.pop("COLI_TOOL_TRACE", None)
                    off = strip(self._post(chunks, stream)[1], stream)
                with patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
                    on = strip(self._post(chunks, stream)[1], stream)
                self.assertEqual(off, on, (stream, chunks))

    def test_on_streaming_call(self):
        with patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
            request_id, _raw = self._post(CALL, True)
        [line] = self._lines()
        self.assertEqual(line["request_id"], request_id)
        self.assertEqual((line["stream"], line["finish"], line["n_calls"], line["names"]),
                         (True, "tool_calls", 1, ["get_weather"]))
        self.assertEqual((line["openers"], line["parse_error"], line["length_limited"]), (1, None, False))
        self.assertEqual(line["reasoning_tail"], "deciding to call")
        self.assertIn("<arg_value>Paris</arg_value>", line["answer"])

    def test_on_lost_call_is_named(self):
        with patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
            self._post(CUT, False, length_limited=True)
            self._post(CUT, True, length_limited=True)
        for line in self._lines():
            self.assertEqual((line["n_calls"], line["finish"], line["length_limited"]),
                             (0, "length", True))
            self.assertEqual(line["parse_error"], "1 call opener(s), 0 call(s) parsed")
            self.assertEqual(line["reasoning_tail"], "thinking")
            self.assertTrue(line["answer"].startswith("Leggo.<tool_call>"))

    def test_on_v41_block_the_client_never_sees(self):
        # One newline before the block: the strict parse finds no block and the
        # stream cuts at the marker, so the client gets neither a call nor markup.
        d = "｜DSML｜"
        bad = (f'Leggo.\n<{d} calls>\n<{d} invoke name="get_weather">\n'
               f'<{d} parameter name="city" string="true">Paris</{d} parameter>\n</{d} invoke>\n</{d} calls>')
        with patch.multiple("openai_server", ARCH="deepseek_v41", CHAT_FLAVOR=None), \
                patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
            _request_id, raw = self._post(["pensiero", "</think>", bad], True)
        self.assertNotIn("DSML", raw)
        self.assertNotIn('tool_calls":[', raw)
        [line] = self._lines()
        self.assertEqual((line["family"], line["openers"], line["n_calls"], line["finish"]),
                         ("deepseek_v41", 1, 0, "stop"))
        self.assertEqual(line["parse_error"], "1 call opener(s), 0 call(s) parsed")

    def test_on_no_tools_no_line(self):
        with patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
            self._post(["</think>", "hello"], True, tools=None)
            self._post(["</think>", "hello"], False, tools=None)
        self.assertFalse(self.trace.exists())

    def test_clips_answer_and_reasoning(self):
        chunks = ["r" * 3000, "</think>", "a" * 20000]
        with patch.dict(os.environ, {"COLI_TOOL_TRACE": str(self.trace)}):
            self._post(chunks, True)
        [line] = self._lines()
        self.assertEqual((len(line["answer"]), line["answer_chars"]), (16384, 20000))
        self.assertEqual((len(line["reasoning_tail"]), line["reasoning_chars"]), (2048, 3000))


if __name__ == "__main__":
    unittest.main()
