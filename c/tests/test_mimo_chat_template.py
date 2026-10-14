#!/usr/bin/env python3
"""The gateway's MiMo-V2.6 renderer and tool-call parser against the release's template.

The gateway renders prompts by hand instead of running jinja per request, and a hand copy
can drift from the original without anyone noticing: the model answers anyway. Here the
real template (tests/fixtures/mimo_v26_chat_template.jinja, MIT, from
XiaomiMiMo/MiMo-V2.6-Flash-MOPD @ 2479e2d0, sha256 11ea52e1...) is rendered with jinja2 the
way transformers does and compared byte for byte with render_chat_mimo.

Two documented differences are asserted, not hidden:
  - with thinking on the gateway's cue ends in `<think>`, the first token the model writes
    on every turn of this template; the template leaves it to the model;
  - an OpenAI client sends tool-call arguments as a JSON string, which the template would
    print raw; the gateway reads it as the object it is, which is what the template renders
    when the arguments ARE an object. The reference side of each case gets the object.
"""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import openai_server  # noqa: E402

TEMPLATE = HERE / "fixtures" / "mimo_v26_chat_template.jinja"
TEMPLATE_SHA256 = "11ea52e156de38a458e6b7720ad45915d65b97d4ec979a09f55e3c9bd1b4d059"

try:
    import jinja2
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Il meteo di una citta'",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "units": {"type": "array", "items": {"type": "string"}}},
        "required": ["city"]}}}
SEARCH = {"type": "function", "function": {
    "name": "search", "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}}}}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Sei conciso."},
        {"role": "user", "content": "capitale della Francia?"}]},
    "history with an answer": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "e 2+2?"}]},
    "history with reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "uno piu' uno"},
        {"role": "user", "content": "e 2+2?"}]},
    "text parts": {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "prima "},
                                     {"type": "text", "text": "seconda"}]}]},
    "tools declared": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Usa gli strumenti."},
        {"role": "user", "content": "Che tempo fa a Roma?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Meteo a Roma e Milano per 3 giorni?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Roma", "days": 3,
                                                     "units": ["C", "mm"]}}},
            {"id": "b", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Milano", "days": 3}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sole, 24 gradi"},
        {"role": "tool", "tool_call_id": "b", "content": "pioggia, 17 gradi"},
        {"role": "assistant", "content": "A Roma sole, a Milano pioggia."},
        {"role": "user", "content": "E Napoli?"}]},
    "an image placeholder": {"messages": [
        {"role": "user", "content": "<|vision_start|>" + "<|image_pad|>" * 6
                                    + "<|vision_end|>Cosa vedi?"}]},
}


def reference_render(case, add_generation_prompt=True, enable_thinking=None):
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
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
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
class MimoTemplateTest(unittest.TestCase):

    def setUp(self):
        self.saved_arch = openai_server.ARCH
        openai_server.ARCH = "mimo"

    def tearDown(self):
        openai_server.ARCH = self.saved_arch

    def test_template_fixture_is_the_release(self):
        digest = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
        self.assertEqual(digest, TEMPLATE_SHA256)

    def test_thinking_off_is_byte_identical(self):
        for name, case in CASES.items():
            with self.subTest(name):
                want = reference_render(case, enable_thinking=False)
                got = openai_server.render_chat_mimo(
                    as_client_sends(case["messages"]), enable_thinking=False,
                    tools=case.get("tools"))
                self.assertEqual(got, want)

    def test_thinking_on_adds_only_the_opening_think(self):
        for name, case in CASES.items():
            with self.subTest(name):
                want = reference_render(case)          # enable_thinking undefined: thinking
                got = openai_server.render_chat_mimo(
                    as_client_sends(case["messages"]), enable_thinking=True,
                    tools=case.get("tools"))
                self.assertEqual(got, want + "<think>")

    def test_open_turn_is_the_past_turn_without_its_terminator(self):
        case = {"messages": [
            {"role": "user", "content": "Scrivi una poesia"},
            {"role": "assistant", "content": "Nel mezzo del", "reasoning_content": "Dante"}]}
        full = reference_render(case, add_generation_prompt=False)
        self.assertTrue(full.endswith("<|im_end|>"))
        got = openai_server.render_chat_mimo(case["messages"], add_generation_prompt=False)
        self.assertEqual(got, full[:-len("<|im_end|>")])

    def test_dispatch_and_continuation(self):
        prompt = openai_server.render_chat_for_arch(
            [{"role": "user", "content": "ciao"}], enable_thinking=False)
        self.assertEqual(prompt, "<|im_start|>user\nciao<|im_end|><|im_start|>assistant\n"
                                 "<think></think>")
        self.assertIn("mimo", openai_server.CONTINUATION_FAMILIES)

    def test_tool_choice_none_offers_no_tools(self):
        got = openai_server.render_chat_mimo([{"role": "user", "content": "x"}],
                                             tools=[WEATHER], tool_choice="none")
        self.assertNotIn("<tools>", got)

    def test_bad_arguments_are_a_400(self):
        messages = [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": "{not json"}}]}]
        with self.assertRaises(openai_server.APIError) as caught:
            openai_server.render_chat_mimo(messages)
        self.assertEqual(caught.exception.status, 400)


class MimoToolParseTest(unittest.TestCase):

    def test_inline_calls_with_typed_arguments(self):
        reply = ("Controllo.<tool_call><function=get_weather><parameter=city>Roma</parameter>"
                 "<parameter=days>3</parameter><parameter=units>[\"C\", \"mm\"]</parameter>"
                 "</function></tool_call><tool_call><function=search><parameter=query>42"
                 "</parameter></function></tool_call>")
        text, calls = openai_server.parse_mimo_tool_calls(reply, [WEATHER, SEARCH])
        self.assertEqual(text, "Controllo.")
        self.assertEqual([c["function"]["name"] for c in calls], ["get_weather", "search"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Roma", "days": 3, "units": ["C", "mm"]})
        # a string parameter stays a string even when it looks like a number
        self.assertEqual(json.loads(calls[1]["function"]["arguments"]), {"query": "42"})

    def test_values_on_their_own_lines(self):
        reply = ("<tool_call>\n<function=get_weather>\n<parameter=city>\nRoma\n</parameter>\n"
                 "</function>\n</tool_call>")
        _text, calls = openai_server.parse_mimo_tool_calls(reply, [WEATHER])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Roma"})

    def test_json_body_is_the_arguments(self):
        reply = '<tool_call><function=get_weather>{"city": "Roma"}</function></tool_call>'
        _text, calls = openai_server.parse_mimo_tool_calls(reply, [WEATHER])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Roma"})

    def test_round_trip_through_the_renderer(self):
        reply = ("<tool_call><function=get_weather><parameter=city>Roma</parameter>"
                 "<parameter=days>3</parameter></function></tool_call>")
        _text, calls = openai_server.parse_mimo_tool_calls(reply, [WEATHER])
        rendered = openai_server.render_chat_mimo(
            [{"role": "user", "content": "x"},
             {"role": "assistant", "content": "", "tool_calls": calls}],
            add_generation_prompt=False)
        self.assertTrue(rendered.endswith("<think></think>" + reply))

    def test_dispatch_in_parse_arch_tool_calls(self):
        saved = openai_server.ARCH
        openai_server.ARCH = "mimo"
        try:
            _text, calls = openai_server.parse_arch_tool_calls(
                "<tool_call><function=search><parameter=query>x</parameter></function>"
                "</tool_call>", [SEARCH])
        finally:
            openai_server.ARCH = saved
        self.assertEqual(calls[0]["function"]["name"], "search")


if __name__ == "__main__":
    unittest.main()
