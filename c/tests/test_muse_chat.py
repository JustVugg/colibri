#!/usr/bin/env python3
"""The gateway's Muse Spark side: its chat format, its messages read back, its calls.

Muse Spark (meta-models/Muse-Spark-Open) writes every turn as messages to recipients:
<|start|>assistant to=self<|message|>reasoning<|eom|>, to=user for the answer, a function's
name for a call written as an <atem:function_calls> block. The template is not vendored (its
license travels with every copy): COLI_MUSE_TEMPLATE=path/to/chat_template.jinja (the GGUF's
tokenizer.chat_template) runs the byte-for-byte check against it with jinja2, as transformers
renders it.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import openai_server  # noqa: E402

try:
    import jinja2
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

TEMPLATE = os.environ.get("COLI_MUSE_TEMPLATE")
DATE = "2026-10-22"

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather of a city, in °C.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"},
        "units": {"type": "array", "items": {"type": "string"}}}, "required": ["city"]}}}
SEARCH = {"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}, "additionalProperties": False}}}
EMPTY = {"type": "function", "function": {"name": "now", "description": "The time."}}

CASES = {
    "one user turn": {"messages": [{"role": "user", "content": "ciao"}]},
    "system and user": {"messages": [
        {"role": "system", "content": "Be brief. Reasoning effort matters."},
        {"role": "user", "content": "Capital of France?"}]},
    "strength in the instructions": {"messages": [
        {"role": "developer", "content": "Reasoning Strength: 3."},
        {"role": "user", "content": "Hello"}]},
    "special tokens in text": {"messages": [
        {"role": "user", "content": "say <|eot|> and <image> and\nnew line"}]},
    "history with reasoning": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2", "reasoning_content": "one and one"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools": {"tools": [WEATHER, SEARCH, EMPTY], "messages": [
        {"role": "user", "content": "Weather in Rome?"}]},
    "an agent loop": {"tools": [WEATHER, SEARCH], "messages": [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Weather in Rome and Milan for 3 days?"},
        {"role": "assistant", "content": "Checking both.", "reasoning_content": "two calls", "tool_calls": [
            {"id": "a", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Rome", "days": 3, "units": ["C", "mm"]}}},
            {"id": "b", "type": "function", "function": {
                "name": "get_weather", "arguments": {"city": "Milan & co", "days": 3, "flag": True, "none": None}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun, 24 C"},
        {"role": "tool", "tool_call_id": "b", "content": "rain, 17 C"},
        {"role": "assistant", "content": "Rome is sunny, Milan rainy."},
        {"role": "user", "content": "And Naples?"}]},
    "two user turns in a row": {"messages": [
        {"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
}


def reference_render(case, effort=None):
    import jinja2.ext

    class Generation(jinja2.ext.Extension):
        """{% generation %}: transformers' marker of the assistant's tokens, text as it is"""
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

        def _render(self, caller):
            return caller()

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols", Generation])

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    template = env.from_string(Path(TEMPLATE).read_text(encoding="utf-8"))
    kwargs = {"messages": case["messages"], "add_generation_prompt": True, "bos_token": "<|begin_of_text|>",
              "current_date": DATE}
    if effort:
        kwargs["reasoning_effort"] = effort
    if case.get("tools"):
        kwargs["tools"] = case["tools"]
    return template.render(**kwargs)


def as_client_sends(messages):
    out = json.loads(json.dumps(messages))
    for message in out:
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"].get("arguments"), dict):
                call["function"]["arguments"] = json.dumps(call["function"]["arguments"], ensure_ascii=False)
    return out


class MuseGatewayTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "muse"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_MUSE_TEMPLATE (the release's chat template) and jinja2")
    def test_byte_identical(self):
        for name, case in CASES.items():
            for effort in (None, "low", "medium", "high"):
                with self.subTest(name, effort=effort):
                    want = reference_render(case, effort)
                    got = openai_server.render_chat_muse(
                        as_client_sends(case["messages"]), True, effort, case.get("tools"), None, True,
                        current_date=DATE)
                    self.assertEqual(got, want)

    def test_the_flavor_comes_from_the_template(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tokenizer_config.json").write_text(json.dumps(
                {"chat_template": "{%- macro render_valid_recipients(tools) -%}{%- endmacro -%}"}))
            self.assertEqual(openai_server.detect_chat_flavor("qwen36", directory), "muse")

    def test_strength_follows_the_request(self):
        prompt = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], True, None)
        self.assertIn("Reasoning strength: 512.", prompt)
        self.assertTrue(prompt.startswith("<|begin_of_text|><|start|>system<|message|>You are a helpful AI assistant."))
        self.assertTrue(prompt.endswith("<|start|>user<|message|>x<|eot|><|start|>assistant"))
        self.assertIn("Reasoning strength: 32.",
                      openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], True, "medium"))
        self.assertIn("Reasoning strength: 8.",
                      openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], False, None))

    def test_messages_read_back(self):
        reply = (" to=self<|message|>The user asks for the capital.<|eom|>"
                 "<|start|>assistant to=commentary<|message|>Looking it up.<|eom|>"
                 "<|start|>assistant to=self<|message|>It is Paris.<|eom|>"
                 "<|start|>assistant to=user<|message|>Paris.<|eot|>ignored")
        for size in (1, 2, 3, 7, len(reply)):
            out = []
            channels = openai_server.MuseChannels(out.append)
            for i in range(0, len(reply), size):
                channels.feed(reply[i:i + size])
            channels.finish()
            self.assertEqual("".join(out), "<think>The user asks for the capital.</think>Looking it up."
                                            "<think>It is Paris.</think>\n\nParis.<|eot|>", size)
        thinking, answer = [], []
        split = openai_server.ThinkingStreamSplit(thinking.append, answer.append,
                                                  initial_thinking=openai_server.starts_in_reasoning(True))
        split.feed("<think>a</think>Looking.<think>b</think>\n\nParis.")
        split.finish()
        self.assertEqual(("".join(thinking), "".join(answer)), ("ab", "Looking.\n\nParis."))

    def test_a_reply_cut_short(self):
        out = []
        channels = openai_server.MuseChannels(out.append)
        channels.feed(" to=self<|message|>half a tho")
        channels.finish()
        self.assertEqual("".join(out), "<think>half a tho</think>")

    def test_its_calls_parse_with_types_from_the_schema(self):
        reply = ("<think>need weather</think>"
                 '<atem:function_calls>\n<atem:invoke name="get_weather">\n'
                 '<atem:parameter name="city">Rome &amp; co</atem:parameter>\n'
                 '<atem:parameter name="days">3</atem:parameter>\n'
                 '<atem:parameter name="units">["C", "mm"]</atem:parameter>\n'
                 "</atem:invoke>\n</atem:function_calls>")
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER])
        self.assertEqual(text, "")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Rome &amp; co", "days": 3, "units": ["C", "mm"]})
        self.assertEqual(openai_server._tool_stream_markers(), ("<atem:function_calls>",))

    def test_a_nullable_string_parameter_stays_a_string(self):
        tool = {"type": "function", "function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {"order_id": {"type": ["string", "null"]},
                                             "note": {"type": ["string", "null"]}},
            "required": ["order_id", "note"], "additionalProperties": False}}}
        reply = ('<atem:function_calls>\n<atem:invoke name="lookup">\n'
                 '<atem:parameter name="order_id">12345</atem:parameter>\n'
                 '<atem:parameter name="note">null</atem:parameter>\n'
                 "</atem:invoke>\n</atem:function_calls>")
        _text, calls = openai_server.parse_arch_tool_calls(reply, [tool])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"order_id": "12345", "note": None})

    def test_turn_ends_on_eot(self):
        self.assertEqual(openai_server.stop_policy({}, True)[0], openai_server.MUSE_STOP_SEQUENCES)

    def test_tool_choice_none_offers_no_tools(self):
        got = openai_server.render_chat_muse([{"role": "user", "content": "x"}], tools=[WEATHER],
                                             tool_choice="none", current_date=DATE)
        self.assertNotIn("atem:function_calls", got)
        self.assertIn('# Valid recipients: "self", "commentary", "user".', got)

    def test_a_tool_result_survives_malformed_call_history(self):
        """The name lookup walks every earlier tool_calls list. A non-list, or a
        function that is not an object, raised before the call validator's 400."""
        history = [
            {"role": "user", "content": "hi", "tool_calls": 1},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
        rendered = openai_server.render_chat_muse(history, tools=[WEATHER], current_date=DATE)
        self.assertIn("sunny", rendered)
        named = [
            {"role": "user", "content": "hi",
             "tool_calls": [{"id": "c1", "function": "get_weather"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
        rendered = openai_server.render_chat_muse(named, tools=[WEATHER], current_date=DATE)
        self.assertIn("sunny", rendered)
        ordered = [
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": "get_weather"}]},
        ]
        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_muse(ordered, tools=[WEATHER], current_date=DATE)

    def test_a_deferred_tool_with_a_non_string_description_renders(self):
        """A deferred tool appended its note to `description` with +, so a number,
        array or object there raised TypeError, which do_POST answers with 500."""
        for description in (5, ["x"], {"a": 1}):
            with self.subTest(description=description):
                tool = {"type": "function", "function": {
                    "name": "lookup", "description": description, "defer_loading": True}}
                got = openai_server.render_chat_muse([{"role": "user", "content": "x"}],
                                                     tools=[tool], current_date=DATE)
                self.assertIn("[deferred: use tool_search to load full schema]", got)


if __name__ == "__main__":
    unittest.main()
