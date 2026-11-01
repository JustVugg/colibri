#!/usr/bin/env python3
"""The gateway's Flint renderer against the release's own template.

The real template (tests/fixtures/flint_chat_template.jinja, Apache-2.0, from
RecursiveSI/flint, sha256 7940f4ff...) is rendered with jinja2 the way transformers does
and compared byte for byte with render_chat_flint. Tool-call arguments go to the gateway as
the JSON string an OpenAI client sends and to the template as the object it iterates.

Flint has no thinking switch: a turn starts as text and the model writes <think> when it
decides to reason, so the gateway's split opens reasoning at that marker.
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

TEMPLATE = HERE / "fixtures" / "flint_chat_template.jinja"
TEMPLATE_SHA256 = "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466"

try:
    import jinja2
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city, in °C.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "units": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"}]},
    "role tokens in text": {"messages": [
        {"role": "user", "content": "say <|assistant|> and <think>x</think> and \x00"}]},
    "developer": {"messages": [
        {"role": "developer", "content": "Answer in Italian."},
        {"role": "user", "content": "Hello"}]},
    "history with reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "one and one"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools, no system": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Rome?"}]},
    "tools after two system turns": {"tools": [WEATHER], "messages": [
        {"role": "system", "content": "Use the tools."},
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "Weather in Rome?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "user", "content": "Weather in Rome and Milan for 3 days?"},
        {"role": "assistant", "content": "", "reasoning_content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Rome", "days": 3, "units": ["C", "mm"]}}},
            {"id": "b", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Milan", "days": 3}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "only a system turn and tools": {"tools": [SEARCH], "messages": [
        {"role": "system", "content": "You search."}]},
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
    out = copy.deepcopy(messages)
    for message in out:
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            if isinstance(fn.get("arguments"), dict):
                fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
    return out


class FlintGatewayTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "flint"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    def test_template_fixture_is_the_release(self):
        # a CRLF checkout (the Windows runner's) holds the same template
        self.assertEqual(hashlib.sha256(TEMPLATE.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), TEMPLATE_SHA256)

    @unittest.skipUnless(jinja2, "jinja2 is not installed")
    def test_byte_identical(self):
        for name, case in CASES.items():
            for open_turn in (True, False):
                with self.subTest(name, add_generation_prompt=open_turn):
                    want = reference_render(case, open_turn)
                    got = openai_server.render_chat_flint(
                        as_client_sends(case["messages"]), tools=case.get("tools"),
                        add_generation_prompt=open_turn)
                    self.assertEqual(got, want)

    def test_the_flavor_comes_from_the_template(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "chat_template.jinja").write_bytes(TEMPLATE.read_bytes())
            self.assertEqual(openai_server.detect_chat_flavor("qwen36", directory), "flint")

    def test_dispatch_has_no_thinking_switch(self):
        for thinking in (False, True):
            prompt = openai_server.render_chat_for_arch(
                [{"role": "user", "content": "ciao"}], enable_thinking=thinking)
            self.assertEqual(prompt, "<|user|>ciao<|assistant|>")

    def test_reasoning_opens_where_the_model_writes_it(self):
        for thinking in (False, True):
            self.assertEqual(openai_server.split_thinking_reply(
                "<think>Rome is the capital</think>Rome.", thinking), ("Rome is the capital", "Rome."))
            self.assertEqual(openai_server.split_thinking_reply("Rome.", thinking), ("", "Rome."))

    def test_reasoning_split_across_stream_chunks(self):
        thinking, text = [], []
        split = openai_server.ThinkingStreamSplit(
            thinking.append, text.append,
            initial_thinking=openai_server.starts_in_reasoning(True))
        for chunk in ("<th", "ink>a", "b</th", "ink>an", "swer"):
            split.feed(chunk)
        split.finish()
        self.assertEqual(("".join(thinking), "".join(text)), ("ab", "answer"))

    def test_its_calls_parse_with_types_from_the_schema(self):
        reply = ("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Rome</arg_value>"
                 "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>")
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER])
        self.assertEqual(text.strip(), "")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"city": "Rome", "days": 3})

    def test_tool_choice_none_offers_no_tools(self):
        got = openai_server.render_chat_flint([{"role": "user", "content": "x"}],
                                              tools=[WEATHER], tool_choice="none")
        self.assertNotIn("<tools>", got)


if __name__ == "__main__":
    unittest.main()
