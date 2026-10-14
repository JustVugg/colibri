#!/usr/bin/env python3
"""The gateway's Qwen3-Coder renderer against the release's own template.

The gateway renders prompts by hand instead of running jinja per request, and a hand copy
can drift from the original while the model keeps answering. Here the real template
(tests/fixtures/qwen3_coder_chat_template.jinja, Apache-2.0, from
Qwen/Qwen3-Coder-30B-A3B-Instruct, sha256 5a38bfa0...) is rendered with jinja2 the way
transformers does and compared byte for byte with render_chat_qwen3_coder.

One documented difference: an OpenAI client sends tool-call arguments as a JSON string,
which the template cannot iterate; the gateway reads it as the object it is. The reference
side of each case gets the object.
"""
import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import openai_server  # noqa: E402

TEMPLATE = HERE / "fixtures" / "qwen3_coder_chat_template.jinja"
TEMPLATE_SHA256 = "5a38bfa05833266240066aedc497decc9b00cc0d3e3b8cceea98cf530196ab06"

try:
    import jinja2
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "  Current weather of a city.  ",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "the city"},
        "days": {"type": "integer", "minimum": 1},
        "units": {"type": "array", "items": {"type": "string"}, "enum": [["C"], ["F"]]}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "strict": True, "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}}}}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"}]},
    "history": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools, no system": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Rome?"}]},
    "tools and a system": {"tools": [WEATHER], "messages": [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Weather in Rome?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Rome and Milan for 3 days?"},
        {"role": "assistant", "content": "Checking both.", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Rome", "days": 3,
                                                     "units": ["C", "mm"]}}},
            {"id": "b", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Milan", "days": 3}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "a tool result last": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "Find colibri"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "search", "arguments": {"query": "colibri"}}}]},
        {"role": "tool", "content": "a pure-C engine"}]},
}


def reference_render(case, add_generation_prompt=True):
    """The template, rendered the way transformers' apply_chat_template does."""
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)
    env.filters["tojson"] = tojson
    template = env.from_string(TEMPLATE.read_text(encoding="utf-8"))
    kwargs = {"messages": case["messages"], "add_generation_prompt": add_generation_prompt}
    if case.get("tools"):
        kwargs["tools"] = case["tools"]
    return template.render(**kwargs)


def as_client_sends(messages):
    """The same history with tool-call arguments as JSON strings, the OpenAI shape."""
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
    return out


@unittest.skipUnless(jinja2, "jinja2 is not installed")
class Qwen3CoderTemplateTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "qwen3_coder"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    def test_template_fixture_is_the_release(self):
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), TEMPLATE_SHA256)

    def test_byte_identical(self):
        for name, case in CASES.items():
            with self.subTest(name):
                want = reference_render(case)
                got = openai_server.render_chat_qwen3_coder(
                    as_client_sends(case["messages"]), tools=case.get("tools"))
                self.assertEqual(got, want)

    def test_dispatch_never_thinks(self):
        for thinking in (False, True):
            prompt = openai_server.render_chat_for_arch(
                [{"role": "user", "content": "ciao"}], enable_thinking=thinking)
            self.assertEqual(prompt, "<|im_start|>user\nciao<|im_end|>\n<|im_start|>assistant\n")

    def test_open_turn_is_the_past_turn_without_its_terminator(self):
        case = {"messages": [{"role": "user", "content": "Write a loop"},
                             {"role": "assistant", "content": "for i in"}]}
        full = reference_render(case, add_generation_prompt=False)
        self.assertTrue(full.endswith("<|im_end|>\n"))
        got = openai_server.render_chat_qwen3_coder(case["messages"],
                                                    add_generation_prompt=False)
        self.assertEqual(got, full[:-len("<|im_end|>\n")])

    def test_tool_choice_none_offers_no_tools(self):
        got = openai_server.render_chat_qwen3_coder([{"role": "user", "content": "x"}],
                                                    tools=[WEATHER], tool_choice="none")
        self.assertNotIn("<tools>", got)

    def test_bad_arguments_are_a_400(self):
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": "{not json"}}]}]
        with self.assertRaises(openai_server.APIError) as caught:
            openai_server.render_chat_qwen3_coder(messages)
        self.assertEqual(caught.exception.status, 400)

    def test_its_calls_parse_with_types_from_the_schema(self):
        reply = ("Let me check.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n"
                 "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>")
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER])
        self.assertEqual(text.strip(), "Let me check.")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome", "days": 3})

    def test_a_python_style_boolean_is_a_boolean(self):
        # seen on the real model: the template prints past bools as True/False
        tool = {"type": "function", "function": {"name": "run_tests", "parameters": {
            "type": "object", "properties": {"verbose": {"type": "boolean"},
                                             "workers": {"type": "integer"}}}}}
        reply = ("<tool_call>\n<function=run_tests>\n<parameter=verbose>\nTrue\n</parameter>\n"
                 "<parameter=workers>\n4\n</parameter>\n</function>\n</tool_call>")
        _text, calls = openai_server.parse_arch_tool_calls(reply, [tool])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"verbose": True, "workers": 4})


class Qwen3CoderFlavorTest(unittest.TestCase):

    def test_the_template_names_the_flavor(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "chat_template.jinja").write_bytes(TEMPLATE.read_bytes())
            self.assertEqual(openai_server.detect_chat_flavor("qwen36", tmp), "qwen3_coder")
            self.assertIsNone(openai_server.detect_chat_flavor("mimo", tmp))


if __name__ == "__main__":
    unittest.main()
