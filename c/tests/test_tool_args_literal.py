"""A declared array or object argument written as a Python literal is read as that container.

Measured on Flint (GLM's call form): asked to run a query with the parameters 7, 'x,y', null
and true, it wrote `[7, 'x,y', null, true]` for the declared array, in both rounds. The GLM
form passed that on as a string, the Qwen form (Qwen3.x, LFM3, MiMo) dropped such a call, Muse
and Kimi K3 kept the text, and the DeepSeek V4/V4.1 decoders put it into `arguments` as it
was, so the client received arguments that were not JSON.
"""
import json
import threading
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

import openai_server as gw
import v41_dsml
import v4_dsml
from tests.test_openai_server import _ChunkEngine

READ = [  # (text, declared, the value)
    ("[7, 'x,y', null, true]", "array", [7, "x,y", None, True]),
    ("[7, 'x,y', None, True]", "array", [7, "x,y", None, True]),
    ("[1, 2,]", "array", [1, 2]),
    ("['Mike\\'s \"special\" coffee', 'pane']", "array", ["Mike's \"special\" coffee", "pane"]),
    ("{'theme': 'dark', 'notifications': False, 'font_size': 14}", "object",
     {"theme": "dark", "notifications": False, "font_size": 14}),
    ("[{'x': 0, 'y': 1.5}, {'x': 1, 'y': -2.25},]", "array", [{"x": 0, "y": 1.5}, {"x": 1, "y": -2.25}]),
    ("  {'a': [], 'b': {}, 'c': 'é'}\n", "object", {"a": [], "b": {}, "c": "é"}),
]
NOT_READ = [  # (text, declared): the parser's previous answer stands
    ("(1, 2)", "array"), ("'a', 'b'", "array"), ("{1, 2}", "array"), ("{1: 'a'}", "object"),
    ("[1e999]", "array"), ("[Infinity]", "array"), ("[NaN]", "array"), ("[b'x']", "array"),
    ("[1+2j]", "array"), ("[x]", "array"), ("__import__('os').getcwd()", "array"),
    ("[().__class__]", "array"), ("[1, 2", "array"), ("{'a': 1", "object"), ("not-json", "array"),
    ("[1]", "object"), ("{'a': 1}", "array"), ("['a' + 'b']", "array"),
    ("[" * 300 + "]" * 300, "array"), ("[" + "1, " * 100000 + "]", "array"),
]


def _is_json(raw):
    try:
        json.loads(raw)
        return True
    except ValueError:
        return False


# what each form answered before for a text that is not JSON (JSON itself, NaN included, is
# each form's own business and unchanged)
TEXT = [(raw, declared) for raw, declared in NOT_READ[:12] if not _is_json(raw)]


def tools(declared):
    return [{"type": "function", "function": {"name": "f", "parameters": {
        "type": "object", "properties": {"q": {"type": "string"}, "v": {"type": declared}},
        "required": ["q", "v"]}}}]


def glm_reply(raw):
    return "<tool_call>f<arg_key>q</arg_key><arg_value>x</arg_value><arg_key>v</arg_key>" \
           f"<arg_value>{raw}</arg_value></tool_call>"


def qwen_reply(raw):
    return f"<tool_call>\n<function=f>\n<parameter=q>\nx\n</parameter>\n<parameter=v>\n{raw}\n" \
           "</parameter>\n</function>\n</tool_call>"


def mimo_reply(raw):
    return f"<tool_call>\n<function=f>\n<parameter=q>x</parameter>\n<parameter=v>{raw}</parameter>\n" \
           "</function>\n</tool_call>"


def muse_reply(raw):
    return ('<atem:function_calls><atem:invoke name="f"><atem:parameter name="q">x</atem:parameter>'
            f'<atem:parameter name="v">{raw}</atem:parameter></atem:invoke></atem:function_calls>')


def k3_reply(raw, typ):
    return ('<|open|>tools<|sep|><|open|>call tool="f" index="0"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>x<|close|>argument<|sep|>'
            f'<|open|>argument key="v" type="{typ}"<|sep|>{raw}<|close|>argument<|sep|>'
            '<|close|>call<|sep|><|close|>tools<|sep|>')


def dsml_reply(mod, raw):
    """The DSML block a reply carries, its "v" parameter (string="false") written as `raw`."""
    block = mod.render_tool_calls([{"function": {"name": "f", "arguments": {"q": "x", "v": 98765}}}])
    assert block.count(">98765<") == 1
    return "ok" + block.replace(">98765<", f">{raw}<")


def _no_constant(token):
    raise ValueError("not JSON: " + token)


def args_of(calls):
    return [json.loads(c["function"]["arguments"], parse_constant=_no_constant) for c in calls]


class LiteralReaderTest(unittest.TestCase):
    def test_python_and_json_literals_become_the_declared_container(self):
        for raw, declared, value in READ:
            with self.subTest(raw=raw):
                self.assertEqual(gw._literal_container(raw, (declared,)), value)
                self.assertEqual(gw._literal_container(raw, ["null", declared]), value)

    def test_everything_else_is_not_read(self):
        for raw, declared in NOT_READ:
            with self.subTest(raw=raw[:40]):
                self.assertIsNone(gw._literal_container(raw, (declared,)))
        for raw, _declared, _value in READ:   # only a declared array or object is read
            for declared in ("string", "integer", "number", "boolean", None):
                self.assertIsNone(gw._literal_container(raw, (declared,)))

    def test_the_text_is_never_run(self):
        with patch("builtins.__import__", side_effect=AssertionError("imported")):
            self.assertIsNone(gw._literal_container("__import__('os')", ("array",)))
            self.assertIsNone(gw._literal_container("[open('x')]", ("array",)))


class EveryFormTest(unittest.TestCase):
    """Each call form reads the literal the same way, and keeps its old answer for the rest."""

    def test_glm_and_flint_form(self):
        for raw, declared, value in READ:
            with self.subTest(raw=raw):
                _, calls = gw.parse_tool_calls(glm_reply(raw), tools(declared))
                self.assertEqual(args_of(calls), [{"q": "x", "v": value}])
        for raw, declared in TEXT:
            _, calls = gw.parse_tool_calls(glm_reply(raw), tools(declared))
            self.assertEqual(args_of(calls), [{"q": "x", "v": raw}])   # the text, as before

    def test_glm_strict_mode_reads_json_only(self):
        raw, declared, _ = READ[0]
        with self.assertRaises(gw.APIError):
            gw.parse_glm_tool_calls_strict(glm_reply(raw), tools(declared))
        _, calls = gw.parse_glm_tool_calls_strict(glm_reply('[7, "x,y"]'), tools("array"))
        self.assertEqual(args_of(calls), [{"q": "x", "v": [7, "x,y"]}])

    def test_qwen_form_and_mimo(self):
        for parse, reply in ((gw.parse_qwen_tool_calls, qwen_reply), (gw.parse_mimo_tool_calls, mimo_reply)):
            for raw, declared, value in READ:
                with self.subTest(parse=parse.__name__, raw=raw):
                    _, calls = parse(reply(raw), tools(declared))
                    self.assertEqual(args_of(calls), [{"q": "x", "v": value}])
            for raw, declared in TEXT:
                with self.subTest(parse=parse.__name__, raw=raw):
                    self.assertEqual(parse(reply(raw), tools(declared))[1], [])   # dropped, as before

    def test_muse(self):
        for raw, declared, value in READ:
            with self.subTest(raw=raw):
                _, calls = gw.parse_muse_tool_calls(muse_reply(raw), tools(declared))
                self.assertEqual(args_of(calls), [{"q": "x", "v": value}])
        _, calls = gw.parse_muse_tool_calls(muse_reply("[Infinity]"), tools("array"))
        self.assertEqual(args_of(calls), [{"q": "x", "v": "[Infinity]"}])

    def test_kimi_k3_by_the_written_type(self):
        for raw, declared, value in READ:
            with self.subTest(raw=raw):
                _, calls = gw.parse_k3_tool_calls(k3_reply(raw, declared), tools(declared))
                self.assertEqual(args_of(calls), [{"q": "x", "v": value}])
        _, calls = gw.parse_k3_tool_calls(k3_reply("[1, 2,]", "number"), tools("array"))
        self.assertEqual(args_of(calls), [{"q": "x", "v": "[1, 2,]"}])

    def test_deepseek_v4_and_v41_arguments_are_json(self):
        cases = [(raw, value) for raw, _d, value in READ] + [('[7, "x"]', [7, "x"]), ("3", 3)]
        for mod, parse in ((v4_dsml, gw.parse_dsv4_tool_calls), (v41_dsml, gw.parse_dsv41_tool_calls)):
            for raw, value in cases:
                with self.subTest(mod=mod.__name__, raw=raw):
                    content, calls = parse(dsml_reply(mod, raw))
                    self.assertEqual(content, "ok")
                    self.assertEqual(args_of(calls), [{"q": "x", "v": value}])
            # arguments that read as nothing stay as the decoder wrote them
            _, calls = parse(dsml_reply(mod, "(1, 2)"))
            self.assertEqual(calls[0]["function"]["arguments"], '{"q": "x", "v": (1, 2)}')


class LiteralHTTPTest(unittest.TestCase):
    """Flint's reply through /v1/chat/completions, whole and streamed."""

    def setUp(self):
        self.engine = _ChunkEngine([])
        self.server = gw.APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)

    def calls(self, reply, declared, stream):
        self.engine.chunks = list(reply)
        body = {"model": "test-model", "messages": [{"role": "user", "content": "run it"}],
                "max_tokens": 1024, "stream": stream, "tools": tools(declared), "enable_thinking": False}
        request = Request(f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=10) as response:
            raw = response.read().decode()
        if not stream:
            return args_of(json.loads(raw)["choices"][0]["message"].get("tool_calls") or [])
        parts = {}
        for line in raw.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for choice in json.loads(line[6:])["choices"]:
                    for c in (choice.get("delta") or {}).get("tool_calls") or []:
                        parts[c["index"]] = parts.get(c["index"], "") + (c.get("function") or {}).get("arguments", "")
        return [json.loads(parts[i]) for i in sorted(parts)]

    def test_flint_reply_and_the_qwen_form(self):
        raw, declared, value = READ[0]
        for arch, reply in (("glm", glm_reply(raw)), ("qwen36", qwen_reply(raw))):
            with patch.multiple("openai_server", ARCH=arch, CHAT_FLAVOR=None):
                for stream in (False, True):
                    with self.subTest(arch=arch, stream=stream):
                        self.assertEqual(self.calls(reply, declared, stream), [{"q": "x", "v": value}])


if __name__ == "__main__":
    unittest.main()
