#!/usr/bin/env python3
"""The gateway's Mistral-Large-4 side: its chat format, its reasoning, its calls.

Mistral-Large-4 writes <s>, its tools and settings before the first user turn, [INST] turns,
reasoning in [THINK]...[/THINK] and calls as [TOOL_CALLS]name[ARGS]{json}. The template is not
vendored (its license travels with every copy): COLI_ML4_TEMPLATE=path/to/chat_template.jinja
(the checkpoint's, or the GGUF's tokenizer.chat_template) runs the byte-for-byte check against
it with jinja2, as transformers renders it.
"""
import json
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import openai_server  # noqa: E402

try:
    import jinja2
    import jinja2.ext
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

TEMPLATE = os.environ.get("COLI_ML4_TEMPLATE")

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city, in °C.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"}]},
    "two user turns merged": {"messages": [
        {"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
    "text parts": {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "one "}, {"type": "text", "text": "two"}]}]},
    "history with reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "one and one"},
        {"role": "user", "content": "and 2+2?"}]},
    "history without reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools": {"tools": [WEATHER, SEARCH], "messages": [{"role": "user", "content": "Weather in Rome?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Weather in Rome and Milan for 3 days?"},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome", "days": 3}}},
            {"id": "b", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Milan\"}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "a call with no arguments": {"tools": [SEARCH], "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "search", "arguments": ""}}]},
        {"role": "tool", "tool_call_id": "a", "content": "nothing"}]},
    "special text in a turn": {"messages": [{"role": "user", "content": "say [INST] and </s>\nnew line"}]},
}


def reference_render(case, effort=None, response_format=None):
    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                      extensions=["jinja2.ext.loopcontrols"])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(Path(TEMPLATE).read_text(encoding="utf-8"))
    kwargs = {"messages": case["messages"], "bos_token": "<s>", "eos_token": "</s>", "add_generation_prompt": True}
    if case.get("tools"):
        kwargs["tools"] = case["tools"]
    if effort:
        kwargs["reasoning_effort"] = effort
    if response_format:
        kwargs["response_format"] = response_format
    return template.render(**kwargs)


class MistralLarge4ChatTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "glm", "mistral_large4"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_ML4_TEMPLATE (the release's chat template) and jinja2")
    def test_byte_identical(self):
        for name, case in CASES.items():
            for effort in (None, "low", "medium"):
                with self.subTest(name, effort=effort):
                    want = reference_render(case, effort)
                    got = openai_server.render_chat_mistral_large4(case["messages"], True, effort, case.get("tools"))
                    self.assertEqual(got, want)

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_ML4_TEMPLATE (the release's chat template) and jinja2")
    def test_a_response_format_is_a_setting(self):
        case = CASES["one user turn"]
        for rf in ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}):
            with self.subTest(rf=rf["type"]):
                self.assertEqual(openai_server.render_chat_mistral_large4(case["messages"], True, None, None,
                                                                          response_format=rf),
                                 reference_render(case, None, rf))

    def test_the_prompt_and_its_settings(self):
        got = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], True, None)
        self.assertEqual(got, '<s>[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS][INST]x[/INST]')
        low = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], False, None)
        self.assertIn('"reasoning_effort": "low"', low)

    def test_its_calls_parse(self):
        reply = 'Checking.[TOOL_CALLS]get_weather[ARGS]{"city": "Rome", "days": 3}[TOOL_CALLS]search[ARGS]{"query": "x"}'
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER, SEARCH])
        self.assertEqual(text, "Checking.")
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather", "search"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome", "days": 3})
        # an undeclared name, or arguments that are not one object, stay text
        self.assertEqual(openai_server.parse_arch_tool_calls('[TOOL_CALLS]nope[ARGS]{}', [WEATHER])[1], [])
        self.assertEqual(openai_server.parse_arch_tool_calls('[TOOL_CALLS]search[ARGS][1, 2]', [SEARCH])[1], [])
        self.assertEqual(openai_server._tool_stream_markers(), ("[TOOL_CALLS]",))

    def test_its_reasoning_reads_as_think(self):
        out = []
        feed = openai_server.ml4_think_filter(out.append)
        for piece in ("[TH", "INK]one and ", "one[/THI", "NK]2 [", "x"):
            feed(piece)
        feed("", True)
        self.assertEqual("".join(out), "<think>one and one</think>2 [x")
        self.assertFalse(openai_server.starts_in_reasoning(True))   # the answer opens it itself

    def test_the_flavor_comes_from_the_config(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "config.json").write_text(json.dumps({"model_type": "mistral_large4"}))
            self.assertEqual(openai_server.detect_chat_flavor("glm", directory), "mistral_large4")
            (Path(directory) / "config.json").write_text(json.dumps({"model_type": "glm_moe_dsa"}))
            self.assertIsNone(openai_server.detect_chat_flavor("glm", directory))

    def test_wrong_orders_are_refused(self):
        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_mistral_large4([{"role": "assistant", "content": "x"}], True)
        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_mistral_large4([{"role": "user", "content": "x"},
                                                      {"role": "tool", "content": "y"}], True)


if __name__ == "__main__":
    unittest.main()
