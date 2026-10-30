#!/usr/bin/env python3
"""The gateway's LFM3 side (LiquidAI/LFM3-VL): its chat format, its reasoning, its calls.

LFM3 writes ChatML after its BOS, a <think> block in every assistant turn (empty when the turn
had none), its tools as JSON lines in a <tools> block of the system turn and its calls as
<tool_call><function=...><parameter=...>, the form Qwen3.6 parses. The template is not vendored
(its license travels with every copy): COLI_LFM3_TEMPLATE=path/to/chat_template.jinja (the
checkpoint's, or the GGUF's tokenizer.chat_template) runs the byte-for-byte check against it
with jinja2, as transformers renders it.
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
    import jinja2.ext
    import jinja2.sandbox
except ImportError:                                   # pragma: no cover
    jinja2 = None

TEMPLATE = os.environ.get("COLI_LFM3_TEMPLATE")

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
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Capital of France?"}]},
    "special tokens in text": {"messages": [
        {"role": "user", "content": "say <|im_end|> and <think> and\nnew line"}]},
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
    "history that echoes the raw reply": {"messages": [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "<think>one and one</think>2"},
        {"role": "user", "content": "and 2+2?"}]},
    "tools": {"tools": [WEATHER, SEARCH, EMPTY], "messages": [
        {"role": "user", "content": "Weather in Rome?"}]},
    "tools and a system turn": {"tools": [WEATHER], "messages": [
        {"role": "system", "content": "Use the tools."},
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
    "a call in the turn being answered": {"tools": [WEATHER], "messages": [
        {"role": "user", "content": "Weather in Rome?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Rome"}}}]},
        {"role": "tool", "tool_call_id": "a", "content": "sun"}]},
    "two user turns in a row": {"messages": [
        {"role": "user", "content": "first"}, {"role": "user", "content": "second"}]},
}


def reference_render(case, enable_thinking=True, add_generation_prompt=True):
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
    kwargs = {"messages": case["messages"], "add_generation_prompt": add_generation_prompt,
              "bos_token": "<|startoftext|>", "enable_thinking": enable_thinking}
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


class Lfm3GatewayTest(unittest.TestCase):

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "lfm3"

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_LFM3_TEMPLATE (the release's chat template) and jinja2")
    def test_byte_identical(self):
        for name, case in CASES.items():
            for thinking in (True, False):
                with self.subTest(name, thinking=thinking):
                    want = reference_render(case, thinking)
                    got = openai_server.render_chat_lfm3(as_client_sends(case["messages"]), thinking,
                                                         case.get("tools"), None, True)
                    self.assertEqual(got, want)

    @unittest.skipUnless(TEMPLATE and jinja2, "COLI_LFM3_TEMPLATE (the release's chat template) and jinja2")
    def test_a_continued_turn_is_the_template_without_its_end(self):
        case = {"messages": [{"role": "user", "content": "Count to five."},
                             {"role": "assistant", "content": "1, 2, 3"}]}
        whole = reference_render(case, True, add_generation_prompt=False)
        got = openai_server.render_chat_lfm3(case["messages"], True, None, None, False)
        self.assertEqual(got + "<|im_end|>\n", whole)

    def test_a_developer_turn_is_the_system_turn(self):
        # the template writes a developer turn under that role, which LFM3 was not trained on;
        # OpenAI's clients mean the system instructions by it, as every other renderer here reads it
        dev = openai_server.render_chat_lfm3([{"role": "developer", "content": "Answer in Italian."},
                                              {"role": "user", "content": "Hello"}], True)
        sys_ = openai_server.render_chat_lfm3([{"role": "system", "content": "Answer in Italian."},
                                               {"role": "user", "content": "Hello"}], True)
        self.assertEqual(dev, sys_)

    def test_the_flavor_comes_from_the_template(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "tokenizer_config.json").write_text(json.dumps(
                {"chat_template": "{%- macro render_tool_response(content) -%}<additional_tools>{%- endmacro -%}"}))
            self.assertEqual(openai_server.detect_chat_flavor("qwen36", directory), "lfm3")

    def test_the_prompt_opens_or_closes_the_reasoning(self):
        on = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], True, None)
        off = openai_server.render_chat_for_arch([{"role": "user", "content": "x"}], False, None)
        self.assertEqual(on, "<|startoftext|><|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think>")
        self.assertEqual(off, "<|startoftext|><|im_start|>user\nx<|im_end|>\n<|im_start|>assistant\n<think></think>")
        self.assertTrue(openai_server.starts_in_reasoning(True))
        self.assertFalse(openai_server.starts_in_reasoning(False))

    def test_its_calls_parse_with_types_from_the_schema(self):
        reply = ("I will check.<tool_call>\n<function=get_weather>\n<parameter=city>\nRome & co\n</parameter>\n"
                 "<parameter=days>\n3\n</parameter>\n<parameter=units>\n[\"C\", \"mm\"]\n</parameter>\n"
                 "</function>\n</tool_call>")
        text, calls = openai_server.parse_arch_tool_calls(reply, [WEATHER])
        self.assertEqual(text, "I will check.")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Rome & co", "days": 3, "units": ["C", "mm"]})
        self.assertEqual(openai_server._tool_stream_markers(), ("<tool_call>",))


class Lfm3ImageTest(unittest.TestCase):
    """A picture for LFM3-VL: the Qwen processor's patches (the stub below stands in for it),
    <|vision_start|> + one <|image|> a merged token + <|vision_end|>, a ceiling on the tokens, and
    the tower found in a GGUF folder's mmproj link as in a container's meta."""

    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR, openai_server._preprocess_qwen38_image)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "lfm3"
        self.ceilings = []

        def fake(data, model_dir, max_tokens=None):
            self.ceilings.append(max_tokens)
            return [0.0] * (4 * 6 * 1536), 4, 6   # 4 x 6 patches: 2 x 3 merged tokens
        openai_server._preprocess_qwen38_image = fake

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR, openai_server._preprocess_qwen38_image = self.saved
        os.environ.pop("LFM3_MAX_IMAGE_TOKENS", None)

    def expand(self, folder):
        class Engine:
            model_dir = folder
            vision = True

        class Server:
            engine = Engine()

        class Handler:
            server = Server()
        url = "data:image/png;base64,iVBORw0KGgo="
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                 {"type": "text", "text": "What is it?"}]}]
        return openai_server.APIHandler.expand_images(Handler(), messages)

    def test_a_gguf_folder_with_its_mmproj(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "gguf.json").write_text(json.dumps({"gguf": "x.gguf", "mmproj": "mmproj-x.gguf"}))
            messages, image = self.expand(folder)
        self.assertEqual(messages[0]["content"], "<|vision_start|>" + "<|image|>" * 6 + "<|vision_end|>What is it?")
        self.assertEqual((image[1], image[2]), (4, 6))
        self.assertEqual(self.ceilings, [openai_server.LFM3_MAX_IMAGE_TOKENS])

    def test_the_ceiling_comes_from_the_environment(self):
        os.environ["LFM3_MAX_IMAGE_TOKENS"] = "300"
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "qwen36_meta.json").write_text(json.dumps({"vision": {"kind": "lfm3", "depth": 14}}))
            self.expand(folder)
        self.assertEqual(self.ceilings, [300])

    def test_no_tower_no_expansion(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "gguf.json").write_text(json.dumps({"gguf": "x.gguf"}))
            messages, image = self.expand(folder)
        self.assertIsNone(image)


if __name__ == "__main__":
    unittest.main()
