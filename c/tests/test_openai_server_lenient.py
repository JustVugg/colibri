"""GLM argument recovery, replayed through the real chat HTTP routes.

The whole / 1,3,7-byte / every two-piece split corpus method comes from
strata's serve/test_tool_corpus.py at d7113273 (see THIRD_PARTY_NOTICES.md).
The fixtures and GLM envelopes here are written for Colibri.
"""

import codecs
import contextlib
import io
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

import openai_server
from tests.test_openai_server import ScriptedEngine, _post_chat, _spawn_test_server


TOOLS = [{"type": "function", "function": {"name": "record", "parameters": {
    "type": "object", "properties": {"items": {"type": "array"},
                                      "options": {"type": "object"},
                                      "text": {"type": "string"}}}}}]


def call(key, value, name="record"):
    return ("<tool_call>" + name + "<arg_key>" + key + "</arg_key><arg_value>" +
            value + "</arg_value></tool_call>")


# (declared parameter, model text, literal client value). No donor specimens copied.
REPAIRED = (
    ("items", "[1, 2,]", [1, 2]),
    ("items", "['a']", ["a"]),
    ("items", "[1, 2", [1, 2]),
    ("items", "[4, 9,]", [4, 9]),
    ("items", "[4,,9]", [4, 9]),
    ("items", "[,,4,,,9,,]", [4, 9]),
    ("items", "['red', 'blue']", ["red", "blue"]),
    ("items", "[4, 9", [4, 9]),
    ("items", "[4,", [4]),
    ("items", "[", []),
    ("items", "[, ,]", []),
    ("items", '[{"code": 8,},]', [{"code": 8}]),
    ("items", "[[4,], [9,,]]", [[4], [9]]),
    ("items", '[{"code": [4,9', [{"code": [4, 9]}]),
    ("items", "[true, false, null,]", [True, False, None]),
    ("items", "[-2.5, 3e2,]", [-2.5, 300.0]),
    ("items", "['café', '雪',]", ["café", "雪"]),
    ("items", r"['it\'s', 'a\nb', '\u263a',]", ["it's", "a\nb", "☺"]),
    ("options", '{"limit": 8,}', {"limit": 8}),
    ("options", '{"limit": 8,, "dry": false}', {"limit": 8, "dry": False}),
    ("options", "{'limit': 8, 'dry': false}", {"limit": 8, "dry": False}),
    ("options", '{"limit": 8', {"limit": 8}),
    ("options", '{"limit": 8,', {"limit": 8}),
    ("options", "{", {}),
    ("options", "{,,}", {}),
    ("options", '{"mode": "fast", "unused", "dry": true}', {"mode": "fast", "dry": True}),
    ("options", '{"mode": "fast", "unused"}', {"mode": "fast"}),
    ("options", "{'tags': ['red',,], 'nested': {'ok': true,}}",
     {"tags": ["red"], "nested": {"ok": True}}),
    ("options", '{"tags": [4, 9', {"tags": [4, 9]}),
)

UNCHANGED = (
    ("items", "[4, 9]", [4, 9]),
    ("options", '{"b": 2, "a": 1}', {"b": 2, "a": 1}),
    ("items", '["café", "雪"]', ["café", "雪"]),
    ("text", "[4,9,]", "[4,9,]"),
    ("text", "{'limit':8}", "{'limit':8}"),
    ("extra", "[4,9,]", "[4,9,]"),
    ("items", '["unfinished', '["unfinished'),
    ("options", '{"limit":', '{"limit":'),
    ("options", '{"limit" 8}', '{"limit" 8}'),
    ("items", "[unknown,]", "[unknown,]"),
    ("items", "[4 9]", "[4 9]"),
    ("items", "[4,}", "[4,}"),
    ("options", '{"limit": 8]', '{"limit": 8]'),
    ("items", "[4,] trailing", "[4,] trailing"),
    ("items", "[4,]]", "[4,]]"),
    ("items", "{'limit':8,}", "{'limit':8,}"),
    ("options", "[4,]", "[4,]"),
    ("items", r"['bad\q',]", r"['bad\q',]"),
    ("items", "[True,]", "[True,]"),
    ("items", "[NaN,]", "[NaN,]"),
    ("items", "[1e999,]", "[1e999,]"),
)


def partitions(text):
    raw = text.encode("utf-8")
    yield "whole", (raw,)
    for size in (1, 3, 7):
        yield f"pieces-{size}", tuple(raw[i:i + size] for i in range(0, len(raw), size))
    for at in range(1, len(raw)):
        yield f"cut-{at}", (raw[:at], raw[at:])


def decoded(pieces):
    # Engine decodes DATA bytes incrementally before handing text to the HTTP path.
    decoder = codecs.getincrementaldecoder("utf-8")()
    return tuple(decoder.decode(piece) for piece in pieces) + (decoder.decode(b"", final=True),)


class LenientToolArgumentsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(openai_server, "_TOOL_LENIENT", True))
        self.enterContext(patch.object(openai_server, "ARCH", "glm"))
        self.log = self.enterContext(contextlib.redirect_stderr(io.StringIO()))

    def test_declared_array_with_trailing_comma(self):
        _content, calls = openai_server.parse_tool_calls(call("items", "[4, 9,]"), TOOLS)
        self.assertEqual(calls[0]["function"]["arguments"], '{"items": [4, 9]}')

    def test_corpus_whole_pieces_and_every_byte_split_streaming_on_and_off(self):
        engine = ScriptedEngine()
        base = _spawn_test_server(self, engine)
        replays = 0
        for key, raw, want in REPAIRED + UNCHANGED:
            for label, pieces in partitions(call(key, raw)):
                engine.chunks = decoded(pieces)
                for stream in (False, True):
                    with self.subTest(key=key, raw=raw, partition=label, stream=stream):
                        with _post_chat(base, {"model": "test-model",
                                              "messages": [{"role": "user", "content": "record"}],
                                              "tools": TOOLS, "stream": stream}) as response:
                            body = response.read().decode()
                        if stream:
                            events = [json.loads(line[6:]) for line in body.splitlines()
                                      if line.startswith("data: {")]
                            choices = [choice for event in events for choice in event["choices"]]
                            calls = [tc for choice in choices
                                     for tc in choice["delta"].get("tool_calls", [])]
                            self.assertEqual(choices[-1]["finish_reason"], "tool_calls")
                            self.assertFalse(any(choice["delta"].get("content") for choice in choices))
                        else:
                            choice = json.loads(body)["choices"][0]
                            self.assertEqual(choice["finish_reason"], "tool_calls")
                            calls = choice["message"]["tool_calls"]
                        self.assertEqual(len(calls), 1)
                        self.assertEqual(calls[0]["function"]["name"], "record")
                        self.assertEqual(calls[0]["function"]["arguments"],
                                         json.dumps({key: want}, ensure_ascii=False))
                        replays += 1
        print(f"lenient corpus: {len(REPAIRED) + len(UNCHANGED)} specimens, {replays} HTTP replays")

    def test_off_switch_restores_text_and_valid_arguments_exactly(self):
        with patch.object(openai_server, "_TOOL_LENIENT", False):
            for key, raw, _want in REPAIRED:
                with self.subTest(key=key, raw=raw):
                    _content, calls = openai_server.parse_tool_calls(call(key, raw), TOOLS)
                    self.assertEqual(calls[0]["function"]["arguments"],
                                     json.dumps({key: raw}, ensure_ascii=False))
        for enabled in (False, True):
            with patch.object(openai_server, "_TOOL_LENIENT", enabled):
                _content, calls = openai_server.parse_tool_calls(
                    call("options", ' {"b":2,"a":1} '), TOOLS)
                self.assertEqual(calls[0]["function"]["arguments"], '{"options": {"b": 2, "a": 1}}')

    def test_undeclared_tool_is_not_repaired_and_strict_mode_still_refuses_it(self):
        # The permissive parser already forwards complete unknown calls on this base.
        _content, calls = openai_server.parse_tool_calls(call("items", "[4,]", "unknown"), TOOLS)
        self.assertEqual(calls[0]["function"]["arguments"], '{"items": "[4,]"}')
        self.assertEqual(openai_server.parse_tool_calls("<tool_call>unknown", TOOLS)[1], [])
        with self.assertRaises(openai_server.APIError) as caught:
            openai_server.parse_glm_tool_calls_strict(call("items", "[4,]", "unknown"), TOOLS)
        self.assertEqual(caught.exception.code, "invalid_model_tool_call")

    def test_strict_mode_never_repairs_declared_arguments(self):
        for key, raw, _want in REPAIRED:
            with self.subTest(key=key, raw=raw):
                with self.assertRaises(openai_server.APIError) as caught:
                    openai_server.parse_glm_tool_calls_strict(call(key, raw), TOOLS)
                self.assertEqual(caught.exception.code, "invalid_model_tool_call")

    def test_reader_limits_only_apply_after_strict_json_fails(self):
        cases = (("[" + " " * ((1 << 20) - 1), True),
                 ("[" + " " * (1 << 20), False),
                 ("[" * 64 + "0" + "]" * 63, True),
                 ("[" * 65 + "0" + "]" * 64, False),
                 ("[" * 65 + "0" + "]" * 65, True))
        for raw, is_array in cases:
            with self.subTest(length=len(raw), is_array=is_array):
                _content, calls = openai_server.parse_tool_calls(call("items", raw), TOOLS)
                value = json.loads(calls[0]["function"]["arguments"])["items"]
                if is_array:
                    self.assertIsInstance(value, list)
                else:
                    self.assertEqual(value, raw)

    def test_other_declared_types_never_enable_container_recovery(self):
        for declared in (None, "string", "integer", "number", "boolean", "null"):
            with self.subTest(declared=declared):
                self.assertEqual(openai_server._coerce_arg("[4,]", declared, lenient=True), "[4,]")
        # The existing nullable-schema normalizer still selects the declared array.
        tools = [{"type": "function", "function": {"name": "record", "parameters": {
            "type": "object", "properties": {"items": {"type": ["array", "null"]}}}}}]
        _content, calls = openai_server.parse_tool_calls(call("items", "[4,]"), tools)
        self.assertEqual(calls[0]["function"]["arguments"], '{"items": [4]}')

    def test_environment_switch_controls_recovery_at_startup(self):
        script = ("import openai_server as s; print(s.parse_tool_calls(" +
                  repr(call("items", "[4,]")) + ", " + repr(TOOLS) +
                  ")[1][0]['function']['arguments'])")
        for setting in (None, "0", "1"):
            env = os.environ.copy()
            env.pop("COLI_TOOL_LENIENT", None)
            if setting is not None:
                env["COLI_TOOL_LENIENT"] = setting
            with self.subTest(setting=setting):
                result = subprocess.run([sys.executable, "-c", script], env=env,
                                        cwd=Path(openai_server.__file__).parent,
                                        capture_output=True, text=True, check=True)
                self.assertEqual(result.stdout.strip(),
                                 '{"items": [4]}' if setting == "1" else '{"items": "[4,]"}')

    def test_metrics_count_arguments_and_log_once_per_repaired_call(self):
        engine = ScriptedEngine()
        base = _spawn_test_server(self, engine)

        def count():
            with urlopen(Request(base + "/metrics", headers={"Authorization": "Bearer secret"}),
                         timeout=5) as response:
                text = response.read().decode()
            self.assertIn("# TYPE colibri_tool_args_repaired_total counter\n", text)
            return int(next(line.split()[1] for line in text.splitlines()
                            if line.startswith("colibri_tool_args_repaired_total ")))

        before = count()
        engine.chunks = (call("items", "[4,]").replace("</tool_call>", "") +
                         '<arg_key>options</arg_key><arg_value>{"limit":8,}</arg_value></tool_call>',)
        for stream in (False, True):
            self.log.truncate(0)
            self.log.seek(0)
            with _post_chat(base, {"model": "test-model",
                                  "messages": [{"role": "user", "content": "record"}],
                                  "tools": TOOLS, "stream": stream}) as response:
                response.read()
            self.assertEqual(count() - before, 2)
            before += 2
            repair_lines = [line for line in self.log.getvalue().splitlines()
                            if "tool args repaired" in line]
            self.assertEqual(len(repair_lines), 1)
            for name in ("record", "items", "options"):
                self.assertIn(name, repair_lines[0])
        with patch.object(openai_server, "_TOOL_LENIENT", False):
            openai_server.parse_tool_calls(engine.chunks[0], TOOLS)
        openai_server.parse_tool_calls(call("items", "[4]"), TOOLS)
        openai_server.parse_tool_calls(call("text", "[4,]"), TOOLS)
        openai_server.parse_tool_calls(call("items", "[unknown,]"), TOOLS)
        self.assertEqual(count(), before)


if __name__ == "__main__":
    unittest.main()
