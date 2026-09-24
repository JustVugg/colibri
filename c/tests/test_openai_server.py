import http.client
import io
import json
import math
import os
import socket
import tempfile
import threading
import sys
import time
import struct
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path

from openai_server import (APIError, APIHandler, APIServer, ClientCancelled,
                           CONTINUATION_FAMILIES,
                           DEFAULT_CHAT_STOP_SEQUENCES, END, GenerationScheduler,
                           READY, Engine, InklingStreamSplit, StopFilter, ThinkingStreamSplit,
                           _engine_error, _image_bytes_from_url, cap_for_arch,
                           conversation_cache_slot, model_arch,
                           generation_options, parse_tool_calls, parse_dsv4_tool_calls,
                           parse_arch_tool_calls, parse_k3_tool_calls, parse_qwen38_tool_calls,
                           read_engine_turn, render_chat, render_chat_for_arch,
                           render_chat_glm53, render_chat_inkling, render_chat_kimi,
                           render_chat_olmoe,
                           render_chat_qwen38, render_chat_v4, render_chat_dsv41,
                           _dsv4_tool_calls, serve,
                           resolve_generation_prompt, split_thinking_reply,
                           starts_in_reasoning,
                           stop_policy, tune_child_env)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.stop_requests = 0

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        if on_accept is not None:                 # simulate the engine's ACCEPT frame (#597)
            on_accept({"prompt_tokens": 7})
        for chunk in ("Hé", "llo"):
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        return {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text, cache_slot,
                                cancelled, grammar, stopped, on_accept)


class TemplateTest(unittest.TestCase):
    def test_renders_text_subset_of_official_template(self):
        prompt = render_chat([
            {"role": "system", "content": "System"},
            {"role": "developer", "content": "Developer"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": " Hello "},
            {"role": "user", "content": "Again"},
        ])
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|system|>System<|system|>Developer<|user|>Hi"
            "<|assistant|><think></think>Hello<|user|>Again"
            "<|assistant|><think></think>",
        )

    def test_rejects_non_text_content(self):
        with self.assertRaisesRegex(APIError, "text message content only"):
            render_chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "x"}}
            ]}])

    def test_renders_thinking_prefix(self):
        self.assertEqual(
            render_chat([{"role": "user", "content": "Hi"}], True, "high"),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>",
        )

    def test_qwen38_defaults_to_xhigh_instruction_and_thinking(self):
        prompt = render_chat_qwen38([{"role": "user", "content": "Hi"}])
        self.assertEqual(
            prompt,
            "<|im_start|>system\n"
            "Reasoning effort is set to xhigh. Please think carefully through the task, "
            "validate key assumptions, consider plausible alternatives, and prioritize "
            "correctness, consistency, and clarity in the final answer."
            "<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n",
        )

    def test_qwen38_reasoning_efforts_and_disabled_thinking(self):
        medium = render_chat_qwen38([{"role": "user", "content": "Hi"}],
                                    reasoning_effort="medium")
        self.assertEqual(medium,
                         "<|im_start|>user\nHi<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n")
        low = render_chat_qwen38([{"role": "user", "content": "Hi"}],
                                  reasoning_effort="low")
        self.assertIn("Reasoning effort is set to low.", low)
        high = render_chat_qwen38([{"role": "user", "content": "Hi"}],
                                   reasoning_effort="high")
        self.assertIn("Reasoning effort is set to xhigh.", high)
        disabled = render_chat_qwen38([{"role": "user", "content": "Hi"}],
                                      enable_thinking=False)
        self.assertEqual(disabled,
                         "<|im_start|>user\nHi<|im_end|>\n"
                         "<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_qwen38_still_rejects_non_text_content(self):
        # Tools are wired up now; images are not. The engine is text-only, so a
        # picture must still be refused rather than silently dropped.
        with self.assertRaisesRegex(APIError, "text message content only"):
            render_chat_qwen38([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "x"}}
            ]}])

    def test_qwen38_renders_and_parses_its_own_tool_format(self):
        tool = {"type": "function", "function": {
            "name": "weather", "description": "w",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"},
                                          "days": {"type": "integer"}}}}}
        prompt = render_chat_qwen38([{"role": "user", "content": "Rome?"}], tools=[tool])
        # The declaration teaches the model the syntax it must emit, so the
        # preamble is transcribed from chat_template.jinja and not paraphrased.
        self.assertIn("# Tools\n\nYou have access to the following functions:\n\n<tools>",
                      prompt)
        self.assertIn("<function=example_function_name>", prompt)
        self.assertIn("</tools>", prompt)

        # A call with no preceding text attaches directly; one with text is
        # separated by a blank line. Getting that wrong changes the prompt.
        with_text = render_chat_qwen38([
            {"role": "user", "content": "Rome?"},
            {"role": "assistant", "content": "Checking.", "tool_calls": [
                {"type": "function", "function": {
                    "name": "weather", "arguments": {"city": "Rome"}}}]},
            {"role": "tool", "content": "clear"},
            {"role": "user", "content": "thanks"},
        ], tools=[tool])
        self.assertIn("Checking.\n\n<tool_call>\n<function=weather>\n"
                      "<parameter=city>\nRome\n</parameter>\n</function>\n</tool_call>",
                      with_text)
        # Consecutive tool results share one user turn.
        self.assertIn("<|im_start|>user\n<tool_response>\nclear\n</tool_response><|im_end|>",
                      with_text)

        text, calls = parse_qwen38_tool_calls(
            "Sure.\n\n<tool_call>\n<function=weather>\n<parameter=city>\nRome\n"
            "</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>",
            [tool])
        self.assertEqual(text, "Sure.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "weather")
        # The template writes a string argument unquoted, so the type comes back
        # from the declared schema: city stays a string, days becomes an int.
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Rome", "days": 3})

    def test_qwen38_tool_choice_none_suppresses_the_declaration(self):
        tool = {"type": "function", "function": {"name": "f", "description": "d"}}
        prompt = render_chat_qwen38([{"role": "user", "content": "Hi"}],
                                    tools=[tool], tool_choice="none")
        self.assertNotIn("<tools>", prompt)

    def test_kimi_payload_preserves_utf8_lengths_and_turns(self):
        prompt = render_chat_kimi([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "你好\nKimi"},
            {"role": "assistant", "content": "你好。"},
            {"role": "user", "content": "Continue"},
        ], enable_thinking=True)
        self.assertEqual(
            prompt,
            "K3CHAT1\n"
            "M system 11\nBe precise."
            "M user 11\n你好\nKimi"
            "A 0 9\n你好。"
            "M user 8\nContinue"
            "G 1\n",
        )

    def test_kimi_renders_tool_declaration_and_choice(self):
        tools = [{"type": "function", "function": {
            "name": "get_weather", "parameters": {"type": "object"}}}]
        body = ("# Tools\nHere are the available tools, described in JSONSchema.\n\n"
                "```json\n" + json.dumps(tools, ensure_ascii=False, separators=(",", ":"),
                                         sort_keys=True) + "\n```")
        prompt = render_chat_kimi([{"role": "user", "content": "Hi"}], tools=tools)
        self.assertEqual(prompt, "K3CHAT1\n"
                         f"Y 12 {len(body.encode('utf-8'))}\ntool-declare{body}"
                         "M user 2\nHi"
                         "G 0\n")
        # tool_choice=none: the tools are not offered at all
        self.assertEqual(render_chat_kimi([{"role": "user", "content": "Hi"}],
                                          tools=tools, tool_choice="none"),
                         "K3CHAT1\nM user 2\nHiG 0\n")
        # tool_choice=required appends the reference's tool-choice system message
        prompt = render_chat_kimi([{"role": "user", "content": "Hi"}],
                                  tools=tools, tool_choice="required")
        self.assertIn("\ntool-choiceThe system is invoked with `tool_choice=required`.", prompt)
        self.assertTrue(prompt.endswith("G 0\n"))

    def test_kimi_renders_tool_calls_and_results(self):
        messages = [
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Rome", "days": 1e2, "metric": true}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        prompt = render_chat_kimi(messages)
        # B: assistant turn carrying the call; V records keep the exact JSON
        # literal for non-strings (1e2 stays 1e2) and decode strings.
        self.assertIn("B 0 0 0 1\n", prompt)
        self.assertIn("F 11 3\nget_weather", prompt)
        self.assertIn("V 4 6 4\ncitystringRome", prompt)
        self.assertIn("V 4 6 3\ndaysnumber1e2", prompt)
        self.assertIn("V 6 7 4\nmetricbooleantrue", prompt)
        # O: the result resolves its name through tool_call_id
        self.assertIn("O 1 11 5\nget_weathersunny", prompt)

    def test_kimi_tool_results_resort_by_call_id(self):
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function",
                 "function": {"name": "first", "arguments": "{}"}},
                {"id": "b", "type": "function",
                 "function": {"name": "second", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "b", "content": "B"},
            {"role": "tool", "tool_call_id": "a", "content": "A"},
        ]
        prompt = render_chat_kimi(messages)
        # Results are re-sorted into tool_calls order, names resolved from ids.
        self.assertIn("O 1 5 1\nfirstA", prompt)
        self.assertIn("O 2 6 1\nsecondB", prompt)
        self.assertLess(prompt.index("O 1 5 1"), prompt.index("O 2 6 1"))

    def test_kimi_json_fallback_for_unparseable_arguments(self):
        prompt = render_chat_kimi([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function", "function": {
                    "name": "fn", "arguments": "not json"}}]}])
        self.assertIn("J 2 8\nfnnot json", prompt)

    def test_glm_renders_a_tool_call_whose_arguments_are_not_an_object(self):
        """`arguments` that parses but is not an object must not kill the request.

        Both GLM renderers already tolerate `arguments` that does not parse at
        all -- the except branch sets {} -- and every sibling renderer (Kimi
        above, Qwen3.8, DeepSeek V4/V4.1) renders the call without arguments
        rather than failing. Only the "parsed, but not an object" case reached
        .items(), raised AttributeError, and came back as HTTP 500 "The colibri
        engine failed to process the request." on a request the engine never
        saw.
        """
        import openai_server as srv
        for arguments in ('[1, 2]', '"text"', '5', [1, 2], 7):
            with self.subTest(arguments=arguments):
                messages = [
                    {"role": "user", "content": "run it"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "x", "type": "function",
                         "function": {"name": "fn", "arguments": arguments}}]},
                    {"role": "tool", "content": "done"},
                ]
                glm = render_chat(list(messages))
                self.assertIn("fn", glm)
                self.assertNotIn("<arg_key>", glm)
                glm53 = srv.render_chat_glm53(list(messages))
                self.assertIn("<tool_call>fn", glm53)
                self.assertNotIn("<arg_key>", glm53)
        # An object still renders its arguments, on both renderers.
        renders = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "fn", "arguments": '{"city": "Rome"}'}}]}]
        self.assertIn("<arg_key>city</arg_key><arg_value>Rome</arg_value>",
                      render_chat(list(renders)))
        self.assertIn("<arg_key>city</arg_key><arg_value>Rome</arg_value>",
                      srv.render_chat_glm53(list(renders)))

    def test_kimi_still_rejects_unknown_roles(self):
        with self.assertRaisesRegex(APIError, "Unsupported role"):
            render_chat_kimi([{"role": "critic", "content": "hm"}])
        with self.assertRaisesRegex(APIError, "resolvable tool name"):
            render_chat_kimi([{"role": "tool", "content": "orphan result"}])

    def test_kimi_parses_generated_tool_calls(self):
        reply = ('Sure.<|open|>tools<|sep|>'
                 '<|open|>call tool="get_weather" index="1"<|sep|>'
                 '<|open|>argument key="city" type="string"<|sep|>Rome<|close|>argument<|sep|>'
                 '<|open|>argument key="days" type="number"<|sep|>1e2<|close|>argument<|sep|>'
                 '<|close|>call<|sep|>'
                 '<|close|>tools<|sep|>')
        text, calls = parse_k3_tool_calls(reply)
        self.assertEqual(text, "Sure.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Rome", "days": 100.0})

    def test_kimi_parses_json_block_and_unclosed_tail(self):
        reply = ('<|open|>tools<|sep|>'
                 '<|open|>call tool="a&amp;b" index="1"<|sep|>'
                 '<|open|>json type="object"<|sep|>{"x":1}<|close|>json<|sep|>'
                 '<|close|>call<|sep|>')       # tools block never closed: budget ran out
        text, calls = parse_k3_tool_calls(reply)
        self.assertEqual(text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "a&b")
        self.assertEqual(calls[0]["function"]["arguments"], '{"x":1}')

    def test_kimi_authoritative_sideband_does_not_promote_data_lookalikes(self):
        lookalike = ('Echo: <|open|>tools<|sep|>'
                     '<|open|>call tool="danger" index="1"<|sep|>'
                     '<|close|>call<|sep|><|close|>tools<|sep|>')
        with patch("openai_server.ARCH", "kimi"):
            content, calls = parse_arch_tool_calls(lookalike, [{"type": "function"}], "")
        self.assertEqual(content, lookalike)
        self.assertEqual(calls, [])

    def test_kimi_preserves_prior_reasoning_channel(self):
        self.assertEqual(
            render_chat_kimi([{"role": "assistant", "reasoning_content": "why",
                               "content": "answer"}], enable_thinking=True),
            "K3CHAT1\nA 3 6\nwhyanswerG 1\n",
        )

    def test_olmoe_renders_native_chat_template(self):
        # Matches allenai/OLMoE-1B-7B-0125-Instruct's tokenizer_config.json
        # chat_template exactly: one leading bos_token, per-role turns closed
        # by a trailing newline, a prior (non-final) assistant turn also closed
        # by eos_token before that newline (bos_token == eos_token ==
        # "|||IP_ADDRESS|||" in this tokenizer), and a trailing
        # "<|assistant|>\n" generation prompt.
        prompt = render_chat_olmoe([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Continue"},
        ])
        self.assertEqual(
            prompt,
            "|||IP_ADDRESS|||<|system|>\nBe terse.\n<|user|>\nHi\n"
            "<|assistant|>\nHello|||IP_ADDRESS|||\n<|user|>\nContinue\n"
            "<|assistant|>\n",
        )

    def test_olmoe_rejects_tools_and_unknown_roles(self):
        with self.assertRaisesRegex(APIError, "Tool use"):
            render_chat_olmoe([{"role": "user", "content": "Hi"}],
                              tools=[{"type": "function"}])
        with self.assertRaisesRegex(APIError, "Unsupported role"):
            render_chat_olmoe([{"role": "tool", "content": "result"}])

    def test_olmoe_rejects_empty_messages(self):
        with self.assertRaisesRegex(APIError, "non-empty array"):
            render_chat_olmoe([])

    def test_validates_generation_limits(self):
        self.assertEqual(generation_options({"max_tokens": 4, "temperature": 0, "top_p": 1}, 8),
                         (4, 0.0, 1.0, None, ()))
        # max_tokens above the server cap is clamped, not rejected (#260): OpenAI
        # clients default to large values; erroring breaks them.
        self.assertEqual(generation_options({"max_tokens": 9, "temperature": 0, "top_p": 1}, 8),
                         (8, 0.0, 1.0, None, ()))
        # non-positive / non-int max_tokens is still a hard error
        with self.assertRaises(APIError):
            generation_options({"max_tokens": 0}, 8)
        with self.assertRaises(APIError):
            generation_options({"temperature": math.nan}, 8)
        with self.assertRaises(APIError):
            generation_options({"top_p": math.inf}, 8)
        self.assertEqual(generation_options({"temperature": None, "top_p": None}, 8),
                         (8, 0.7, 0.9, None, ()))
        # response_format -> grammar plumbing (draft source, never a constraint)
        opts = generation_options({"max_tokens": 4, "response_format": {"type": "json_object"}}, 8)
        self.assertIn("root ::=", opts[3])
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        opts = generation_options({"max_tokens": 4, "response_format":
                                   {"type": "json_schema", "json_schema": {"schema": schema}}}, 8)
        self.assertEqual(json.loads(opts[3]), schema)
        opts = generation_options({"max_tokens": 4, "response_format":
                                   {"type": "gbnf", "grammar": 'root ::= "x"'}}, 8)
        self.assertEqual(opts[3], 'root ::= "x"')
        with self.assertRaises(APIError):
            generation_options({"response_format": {"type": "yaml"}}, 8)
        with self.assertRaises(APIError):
            generation_options({"response_format": {"type": "json_schema", "json_schema": {}}}, 8)
        # a json_schema that is not an object is the client's mistake too: a 400
        # naming the parameter, not an AttributeError the handler turns into a
        # 500 "engine failed" (which OpenAI SDKs retry)
        for json_schema in ('{"schema": {}}', [schema], 5, True):
            with self.subTest(json_schema=json_schema):
                with self.assertRaises(APIError) as caught:
                    generation_options({"response_format": {"type": "json_schema",
                                                            "json_schema": json_schema}}, 8)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.param, "response_format")
        with self.assertRaises(APIError):   # non-dict response_format
            generation_options({"response_format": "json"}, 8)
        with self.assertRaises(APIError):   # empty gbnf
            generation_options({"response_format": {"type": "gbnf", "grammar": "  "}}, 8)
        with self.assertRaises(APIError):   # oversized grammar (> 1 MiB pre-check)
            generation_options({"response_format": {"type": "gbnf", "grammar": "x" * ((1 << 20) + 1)}}, 8)
        # malformed GBNF passes the gateway by design: the ENGINE fail-softs it
        # (draft source only — bad grammar costs the speedup, never the request)
        opts = generation_options({"response_format": {"type": "gbnf", "grammar": "not a grammar ::="}}, 8)
        self.assertEqual(opts[3], "not a grammar ::=")

    def test_tool_choice_function_that_is_not_an_object_is_a_400(self):
        """The same read, six places: generation_options and five renderers.

        `(tool_choice.get("function") or {}).get("name")` raised AttributeError
        on {"type": "function", "function": "search"} -- the name written where
        the object goes -- and that became a 500 "engine failed" instead of the
        400 generation_options already had two lines further down. The renderers
        crashed on the identical expression before validation was even reached.
        """
        import openai_server as srv
        tools = [{"type": "function", "function": {"name": "search"}}]
        messages = [{"role": "user", "content": "hi"}]
        for function in ("search", ["search"], 5, True):
            with self.subTest(function=function):
                choice = {"type": "function", "function": function}
                with self.assertRaises(APIError) as caught:
                    generation_options({"tools": tools, "tool_choice": choice}, 8)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.param, "tool_choice")
                # the renderers must get past the read so that 400 is what runs
                for render in (render_chat, srv.render_chat_glm53, render_chat_kimi,
                               render_chat_v4, srv.render_chat_dsv41):
                    render(list(messages), tools=list(tools), tool_choice=choice)
        # A well-formed choice, and the legacy {"type": "function", "name": ...}
        # spelling, still force the tool.
        for choice in ({"type": "function", "function": {"name": "search"}},
                       {"type": "function", "name": "search"}):
            with self.subTest(choice=choice):
                generation_options({"tools": tools, "tool_choice": choice}, 8)
                self.assertIn("You must call the function `search`",
                              render_chat(list(messages), tools=list(tools),
                                          tool_choice=choice))

    def test_tool_function_that_is_not_an_object_is_a_400(self):
        """The same slip as tool_choice, on tools[] this time.

        generation_options already has 400 "Tool function must be an object".
        The GLM and DeepSeek declaration blocks then did fn.items() on
        {"type": "function", "function": "search"} and raised AttributeError
        before that 400 ran, so do_POST answered 500 "engine failed".
        """
        import openai_server as srv
        messages = [{"role": "user", "content": "hi"}]
        for function in ("search", ["search"], 5, True):
            with self.subTest(function=function):
                tools = [{"type": "function", "function": function}]
                with self.assertRaises(APIError) as caught:
                    generation_options({"tools": tools}, 8)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.param, "tools.0.function")
                for render in (render_chat, srv.render_chat_glm53, render_chat_kimi,
                               render_chat_v4, srv.render_chat_dsv41):
                    try:
                        render(list(messages), tools=list(tools))
                    except APIError as error:
                        self.assertEqual(error.status, 400)
        well = [{"type": "function", "function": {"name": "search"}}]
        generation_options({"tools": well}, 8)
        self.assertIn('"name": "search"', render_chat(list(messages), tools=well))

    def test_coli_temp_is_the_default_for_requests_that_omit_temperature(self):
        with patch.dict("openai_server.os.environ", {"COLI_TEMP": "0.25"}):
            self.assertEqual(generation_options({}, 8)[1], 0.25)
            self.assertEqual(generation_options({"temperature": 0}, 8)[1], 0.0)
        for invalid in ("malformed", "5", "-1", "nan", "1e999"):
            with self.subTest(invalid=invalid):
                with patch.dict("openai_server.os.environ", {"COLI_TEMP": invalid}):
                    self.assertEqual(generation_options({}, 8)[1], 0.7)

    def test_validates_stop_sequences(self):
        self.assertEqual(generation_options({"stop": "END"}, 8)[4], ("END",))
        self.assertEqual(generation_options({"stop": ["ONE", "TWO"]}, 8)[4],
                         ("ONE", "TWO"))
        for value in ("", [], [""], ["1", "2", "3", "4", "5"], 7, ["ok", 7]):
            with self.subTest(value=value), self.assertRaises(APIError):
                generation_options({"stop": value}, 8)

    def test_glm_chat_defaults_role_stops_without_changing_other_policies(self):
        with patch("openai_server.ARCH", "glm"):
            self.assertEqual(stop_policy({}, True), (DEFAULT_CHAT_STOP_SEQUENCES, True))
            self.assertEqual(stop_policy({}, False), ((), False))
            self.assertEqual(stop_policy({"stop": "END"}, True), (("END",), False))
            self.assertEqual(stop_policy({
                "stop": "END", "x_colibri_ignore_leading_stop": True,
            }, True), (("END",), True))
        with patch("openai_server.ARCH", "inkling"):
            self.assertEqual(stop_policy({}, True), ((), False))
            self.assertEqual(stop_policy({"stop": "END"}, True), (("END",), False))
        with self.assertRaises(APIError):
            stop_policy({"x_colibri_ignore_leading_stop": "yes"}, True)


class StopFilterTest(unittest.TestCase):
    def test_explicit_stop_composes_with_inkling_stream_split(self):
        content = []
        reasoning = []
        splitter = InklingStreamSplit(content.append, reasoning.append)
        stop_filter = StopFilter(("END",), splitter.feed)
        for chunk in ("<|content_thinking|>why<|content_text|>answer EN", "Dignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        splitter.close()
        self.assertEqual("".join(reasoning), "why")
        self.assertEqual("".join(content), "answer ")
        self.assertEqual(stop_filter.matched, "END")

    def test_hides_match_split_across_chunks(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        for chunk in ("answer S", "TO", "Pignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        self.assertEqual("".join(output), "answer ")
        self.assertEqual(stop_filter.matched, "STOP")

    def test_flushes_partial_prefix_when_generation_finishes(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        stop_filter.feed("answer ST")
        stop_filter.finish()
        self.assertEqual("".join(output), "answer ST")

    def test_optional_patient_mode_ignores_only_leading_matches(self):
        output = []
        stop_filter = StopFilter(("<|user|>",), output.append, ignore_leading=True)
        for chunk in ("<|us", "er|>answer", "<|user|>ignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        self.assertEqual("".join(output), "answer")
        self.assertEqual(stop_filter.matched, "<|user|>")
        self.assertEqual(stop_filter.leading_matches_ignored, 1)

    def test_patient_mode_preserves_remainder_after_same_chunk_leading_match(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append, ignore_leading=True)
        stop_filter.feed("STOPuseful STOPdiscarded")
        stop_filter.finish()
        self.assertEqual("".join(output), "useful ")
        self.assertEqual(stop_filter.matched, "STOP")

    def test_strict_mode_still_stops_on_a_leading_match(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        stop_filter.feed("STOPignored")
        stop_filter.finish()
        self.assertEqual(output, [])
        self.assertEqual(stop_filter.matched, "STOP")


class ProtocolTest(unittest.TestCase):
    def test_reads_payload_and_extended_status(self):
        stream = io.BytesIO(b"hello" + END + b"STAT 2 3.5 44 1.2 7 1\n")
        chunks = []
        stats = read_engine_turn(stream, END, chunks.append)
        self.assertEqual(b"".join(chunks), b"hello")
        self.assertEqual(stats["prompt_tokens"], 7)
        self.assertTrue(stats["length_limited"])

    def test_rejects_invalid_kv_pool_before_engine_start(self):
        with self.assertRaisesRegex(ValueError, "kv_slots"):
            serve("/missing", kv_slots=0)

    def test_occupied_port_fails_before_engine_start(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            with patch("openai_server.subprocess.Popen") as popen:
                with self.assertRaises(OSError):
                    serve("/missing", port=listener.getsockname()[1])
            popen.assert_not_called()
        finally:
            listener.close()


class GenerationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.server = APIServer(("127.0.0.1", 0), FakeEngine(), "test")
        self.addCleanup(self.server.server_close)

    def test_records_first_output_once_and_preserves_text_and_stats(self):
        output = []
        with patch("openai_server.time.monotonic", side_effect=[10, 10.5, 13]):
            stats = self.server.generate("prompt", 4, 0, 1, output.append)
        self.assertEqual(output, ["Hé", "llo"])
        self.assertEqual(stats["completion_tokens"], 2)
        metrics = self.server.scheduler.prometheus()
        self.assertIn("colibri_scheduler_first_output_seconds_sum 0.5\n", metrics)
        self.assertIn("colibri_scheduler_first_output_seconds_count 1\n", metrics)
        self.assertIn("colibri_scheduler_engine_call_seconds_sum 3.0\n", metrics)
        self.assertIn("colibri_scheduler_engine_call_seconds_count 1\n", metrics)

    def test_empty_output_does_not_count_but_tool_output_does(self):
        text, tools = [], []
        def generate(prompt, maximum, temperature, top_p, on_text, **kwargs):
            on_text("")
            kwargs["on_tool"]("")
            kwargs["on_tool"]("tool payload")
            on_text("tail")
            return {"completion_tokens": 2}
        with patch.object(self.server.engine, "generate", side_effect=generate), \
             patch("openai_server.time.monotonic", side_effect=[10, 12, 15]):
            self.server.generate("prompt", 4, 0, 1, text.append, on_tool=tools.append)
        self.assertEqual(text, ["", "tail"])
        self.assertEqual(tools, ["", "tool payload"])
        metrics = self.server.scheduler.prometheus()
        self.assertIn("colibri_scheduler_first_output_seconds_sum 2.0\n", metrics)
        self.assertIn("colibri_scheduler_first_output_seconds_count 1\n", metrics)

    def test_error_and_cancellation_before_output_do_not_invent_first_output(self):
        for error in (RuntimeError("failed"), ClientCancelled()):
            with patch.object(self.server.engine, "generate", side_effect=error), \
                 patch("openai_server.time.monotonic", side_effect=[10, 14]):
                with self.assertRaises(type(error)):
                    self.server.generate("prompt", 4, 0, 1, lambda text: None)
        metrics = self.server.scheduler.prometheus()
        self.assertIn("colibri_scheduler_first_output_seconds_count 0\n", metrics)
        self.assertIn("colibri_scheduler_engine_call_seconds_sum 8.0\n", metrics)
        self.assertIn("colibri_scheduler_engine_call_seconds_count 2\n", metrics)

    def test_failure_after_output_keeps_both_observations(self):
        def generate(prompt, maximum, temperature, top_p, on_text):
            on_text("partial")
            raise RuntimeError("failed after output")
        with patch.object(self.server.engine, "generate", side_effect=generate), \
             patch("openai_server.time.monotonic", side_effect=[10, 11, 12]):
            with self.assertRaises(RuntimeError):
                self.server.generate("prompt", 4, 0, 1, lambda text: None)
        metrics = self.server.scheduler.prometheus()
        self.assertIn("colibri_scheduler_first_output_seconds_count 1\n", metrics)
        self.assertIn("colibri_scheduler_engine_call_seconds_count 1\n", metrics)


class SchedulerTest(unittest.TestCase):
    def test_engine_failure_is_not_a_completed_request(self):
        scheduler = GenerationScheduler()
        with self.assertRaisesRegex(RuntimeError, "engine failed"):
            with scheduler.admit():
                raise RuntimeError("engine failed")
        stats = scheduler.snapshot()
        self.assertEqual(stats["failed"], 1)
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["active"], 0)
        with scheduler.admit():
            pass
        self.assertEqual(scheduler.snapshot()["completed"], 1)

    def test_prometheus_histograms_measure_admission_and_slot_occupancy(self):
        scheduler = GenerationScheduler()
        with patch("openai_server.time.monotonic", side_effect=[10, 10.25, 10.25, 12.25]):
            with scheduler.admit():
                active = scheduler.prometheus()
                self.assertIn("colibri_scheduler_active 1\n", active)
                self.assertIn("colibri_scheduler_slot_duration_seconds_count 0\n", active)
        metrics = scheduler.prometheus()
        self.assertIn("# TYPE colibri_scheduler_completed_total counter\n", metrics)
        self.assertIn("colibri_scheduler_completed_total 1\n", metrics)
        self.assertIn("colibri_scheduler_queue_wait_seconds_sum 0.25\n", metrics)
        self.assertIn('colibri_scheduler_queue_wait_seconds_bucket{le="0.1"} 0\n', metrics)
        self.assertIn('colibri_scheduler_queue_wait_seconds_bucket{le="0.5"} 1\n', metrics)
        self.assertIn('colibri_scheduler_queue_wait_seconds_bucket{le="+Inf"} 1\n', metrics)
        self.assertIn("colibri_scheduler_slot_duration_seconds_sum 2.0\n", metrics)
        self.assertIn("colibri_scheduler_slot_duration_seconds_count 1\n", metrics)

    def test_failed_and_cancelled_admissions_are_timed(self):
        scheduler = GenerationScheduler()
        for error in (RuntimeError("failed"), ClientCancelled()):
            with self.assertRaises(type(error)):
                with scheduler.admit():
                    raise error
        metrics = scheduler.prometheus()
        self.assertIn("colibri_scheduler_failed_total 1\n", metrics)
        self.assertIn("colibri_scheduler_cancelled_total 1\n", metrics)
        self.assertIn("colibri_scheduler_completed_total 0\n", metrics)
        self.assertIn("colibri_scheduler_slot_duration_seconds_count 2\n", metrics)

    def test_admits_up_to_capacity_without_serializing(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1, capacity=2)
        with scheduler.admit() as first:
            with scheduler.admit() as second:
                self.assertEqual({first[1], second[1]}, {0, 1})
                self.assertEqual(scheduler.snapshot()["active"], 2)

    def test_rejects_when_waiting_queue_is_full(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1)
        with scheduler.admit():
            with self.assertRaises(APIError) as caught:
                with scheduler.admit():
                    pass
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.code, "queue_full")
        self.assertEqual(scheduler.snapshot()["rejected"], 1)

    def test_any_slot_request_uses_capacity_not_reserved_by_older_waiters(self):
        scheduler = GenerationScheduler(capacity=2)
        # State immediately after both slots are released, before the older
        # slot-0 waiter reacquires the condition lock.
        older = (object(), 0)
        scheduler.queue.append(older)
        with patch.object(scheduler.condition, "wait", side_effect=AssertionError("unused free slot")):
            with scheduler.admit() as (_, slot):
                self.assertEqual(slot, 1)
                self.assertEqual(list(scheduler.queue), [older])
        self.assertEqual(scheduler.free_slots, {0, 1})

    def test_older_any_slot_and_same_slot_waiters_keep_priority(self):
        for older_slot, requested in ((None, None), (None, 1), (0, 0)):
            with self.subTest(older_slot=older_slot, requested=requested):
                scheduler = GenerationScheduler(capacity=2)
                older = (object(), older_slot)
                scheduler.queue.append(older)
                with patch.object(scheduler.condition, "wait",
                                  side_effect=lambda _timeout: scheduler.queue.remove(older)) as wait:
                    with scheduler.admit(slot=requested) as (_, slot):
                        self.assertEqual(slot, 0 if requested is None else requested)
                wait.assert_called_once()
                self.assertEqual(scheduler.snapshot()["queued"], 0)

    def test_full_queue_still_admits_unreserved_free_slot(self):
        for requested in (None, 1):
            with self.subTest(requested=requested):
                scheduler = GenerationScheduler(max_queue=1, capacity=2)
                with scheduler.admit(slot=0):
                    older = (object(), 0)
                    scheduler.queue.append(older)
                    try:
                        with scheduler.admit(slot=requested) as (_, slot):
                            self.assertEqual(slot, 1)
                            self.assertEqual(scheduler.snapshot()["active"], 2)
                            self.assertEqual(list(scheduler.queue), [older])
                    finally:
                        scheduler.queue.remove(older)
                self.assertEqual(scheduler.snapshot()["rejected"], 0)
                self.assertEqual(scheduler.snapshot()["completed"], 2)

    def test_full_queue_does_not_bypass_older_slot_reservations(self):
        for older_slot, requested in ((None, None), (None, 1), (0, 0)):
            with self.subTest(older_slot=older_slot, requested=requested):
                scheduler = GenerationScheduler(max_queue=1, capacity=2)
                older = (object(), older_slot)
                scheduler.queue.append(older)
                with self.assertRaises(APIError) as caught:
                    with scheduler.admit(slot=requested):
                        self.fail("bypassed older waiter")
                self.assertEqual(caught.exception.code, "queue_full")
                self.assertEqual(list(scheduler.queue), [older])
                self.assertEqual(scheduler.snapshot()["rejected"], 1)

    def test_zero_queue_rejects_busy_pinned_slot_with_spare_capacity(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=0.01, capacity=2)
        with scheduler.admit(slot=0):
            with self.assertRaises(APIError) as caught:
                with scheduler.admit(slot=0):
                    self.fail("busy pinned slot admitted")
            self.assertEqual(caught.exception.code, "queue_full")
            with scheduler.admit(slot=1) as (_, slot):
                self.assertEqual(slot, 1)
        stats = scheduler.snapshot()
        self.assertEqual((stats["rejected"], stats["timed_out"], stats["queued"]), (1, 0, 0))
        self.assertEqual((stats["admitted"], stats["completed"], stats["active"]), (2, 2, 0))
        self.assertIn("colibri_scheduler_queue_wait_seconds_count 2\n", scheduler.prometheus())

    def test_times_out_and_cancels_queued_requests(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=0.02)
        with scheduler.admit():
            with self.assertRaises(APIError) as timed_out:
                with scheduler.admit():
                    pass
            with self.assertRaises(ClientCancelled):
                with scheduler.admit(lambda: True):
                    pass
        stats = scheduler.snapshot()
        self.assertEqual(timed_out.exception.code, "queue_timeout")
        self.assertEqual(stats["timed_out"], 1)
        self.assertEqual(stats["cancelled"], 1)

    def test_queue_deadline_wins_when_slot_becomes_free(self):
        for released_at in (0.9, 1.0, 1.1):
            with self.subTest(released_at=released_at):
                scheduler = GenerationScheduler(queue_timeout=1)
                now = [0.0]
                with patch("openai_server.time.monotonic", side_effect=lambda: now[0]):
                    holder = scheduler.admit()
                    holder.__enter__()
                    def release(_timeout):
                        now[0] = released_at
                        holder.__exit__(None, None, None)
                    with patch.object(scheduler.condition, "wait", side_effect=release):
                        if released_at < 1:
                            with scheduler.admit():
                                pass
                        else:
                            with self.assertRaises(APIError) as caught:
                                with scheduler.admit():
                                    pass
                            self.assertEqual(caught.exception.code, "queue_timeout")
                    stats = scheduler.snapshot()
                    expected = 2 if released_at < 1 else 1
                    self.assertEqual((stats["admitted"], stats["completed"]), (expected, expected))
                    self.assertEqual((stats["active"], stats["queued"], stats["timed_out"]),
                                     (0, 0, int(released_at >= 1)))
                    self.assertIn(f"colibri_scheduler_slot_duration_seconds_count {expected}\n",
                                  scheduler.prometheus())
                    with scheduler.admit():
                        pass

    def test_cancelled_request_does_not_acquire_a_free_slot(self):
        scheduler = GenerationScheduler()
        with self.assertRaises(ClientCancelled):
            with scheduler.admit(lambda: True):
                self.fail("cancelled request admitted")
        stats = scheduler.snapshot()
        self.assertEqual((stats["active"], stats["queued"], stats["admitted"], stats["cancelled"]),
                         (0, 0, 0, 1))
        self.assertIn("colibri_scheduler_queue_wait_seconds_count 0\n", scheduler.prometheus())
        with scheduler.admit():
            pass

    def test_cancellation_wins_when_a_waiting_slot_becomes_free(self):
        scheduler = GenerationScheduler(queue_timeout=1)
        waiting = threading.Event()
        cancelled = threading.Event()
        outcomes = []
        holder = scheduler.admit()
        holder.__enter__()
        def is_cancelled():
            waiting.set()
            return cancelled.is_set()
        def run():
            try:
                with scheduler.admit(is_cancelled):
                    outcomes.append("admitted")
            except ClientCancelled:
                outcomes.append("cancelled")
        thread = threading.Thread(target=run)
        thread.start()
        observed = waiting.wait(1)
        # Publish cancellation and release capacity under the same lock, so
        # the waiter must observe both on its next scheduling pass.
        with scheduler.condition:
            cancelled.set()
            holder.__exit__(None, None, None)
        thread.join(2)
        self.assertTrue(observed)
        self.assertFalse(thread.is_alive())
        self.assertEqual(outcomes, ["cancelled"])
        stats = scheduler.snapshot()
        self.assertEqual((stats["admitted"], stats["completed"], stats["cancelled"]), (1, 1, 1))
        self.assertEqual((stats["active"], stats["queued"]), (0, 0))
        self.assertIn("colibri_scheduler_slot_duration_seconds_count 1\n", scheduler.prometheus())

    def test_counts_admitted_client_cancellation_without_completion(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1)
        with self.assertRaises(ClientCancelled):
            with scheduler.admit():
                raise ClientCancelled()
        stats = scheduler.snapshot()
        self.assertEqual(stats["active"], 0)
        self.assertEqual(stats["admitted"], 1)
        self.assertEqual(stats["completed"], 0)
        self.assertEqual(stats["cancelled"], 1)

        with scheduler.admit():
            pass
        self.assertEqual(scheduler.snapshot()["completed"], 1)

    def test_admits_waiters_in_fifo_order(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        order = []

        def run(name, block=False):
            with scheduler.admit():
                order.append(name)
                if block:
                    entered.set()
                    release.wait(1)

        first = threading.Thread(target=run, args=("first", True))
        second = threading.Thread(target=run, args=("second",))
        third = threading.Thread(target=run, args=("third",))
        first.start(); entered.wait(1)
        second.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 1: break
            threading.Event().wait(0.005)
        third.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 2: break
            threading.Event().wait(0.005)
        release.set()
        first.join(1); second.join(1); third.join(1)
        self.assertEqual(order, ["first", "second", "third"])
        self.assertEqual(scheduler.snapshot()["completed"], 3)

    def test_close_rejects_waiters(self):
        scheduler = GenerationScheduler(max_queue=1, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def active():
            with scheduler.admit():
                entered.set(); release.wait(1)

        def waiting():
            try:
                with scheduler.admit(): pass
            except APIError as error:
                errors.append(error.code)

        first = threading.Thread(target=active); first.start(); entered.wait(1)
        second = threading.Thread(target=waiting); second.start()
        scheduler.close(); release.set(); first.join(1); second.join(1)
        self.assertEqual(errors, ["scheduler_closed"])


class BlockingStream:
    def __init__(self, initial=b""):
        self.buffer = bytearray(initial)
        self.closed = False
        self.condition = threading.Condition()

    def feed(self, data):
        with self.condition:
            self.buffer.extend(data)
            self.condition.notify_all()

    def read(self, size=1):
        with self.condition:
            while len(self.buffer) < size and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            size = min(size, len(self.buffer))
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def readline(self):
        with self.condition:
            while b"\n" not in self.buffer and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            end = self.buffer.find(b"\n")
            size = len(self.buffer) if end < 0 else end + 1
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class FakeProcess:
    def __init__(self, on_write):
        self.stdout = BlockingStream(READY + b"STAT 0 0 0 0\n")
        self.stdin = self
        self.on_write = on_write
        self.writes = []
        self.returncode = None

    def write(self, data):
        # `_write_all` (production) hands every write a `memoryview` slice,
        # first write included -- the fakes below pattern-match frame bytes
        # (`frame.split()`, equality against a literal), so normalise here
        # rather than asking each one to know about the view.
        data = bytes(data)
        self.writes.append(data)
        self.on_write(self, data)
        return len(data)

    def flush(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.stdout.close()

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminate()


class DispatcherTest(unittest.TestCase):
    def test_inkling_audio_request_and_response_transcript_is_byte_exact(self):
        prompt = "<|message_user|><|content_audio_input|><|audio|><|end_message|>"
        payload = prompt.encode("utf-8")
        audio = bytes(range(16)) * 5
        expected = (f"SUBMIT 1 0 {len(payload)} 4 0.25 0.9 {len(audio)}\n".encode() +
                    payload + audio + b"\n")

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"DATA 1 4\nA\n\xc3\xa9\n"
                b"DONE 1 STAT 1 2.500 50.0 1.25 7 0\n"
            )

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "inkling"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("inkling", "model")
            chunks = []
            stats = engine.generate(prompt, 4, 0.25, 0.9, chunks.append,
                                    audio=audio)
        engine.close()

        self.assertEqual(process.writes, [expected])
        self.assertEqual(chunks, ["A\né"])
        self.assertEqual(stats["prompt_tokens"], 7)

    def test_v4_request_and_response_transcript_is_byte_exact(self):
        prompt = "<｜begin▁of▁sentence｜>System<｜User｜>Hello<｜Assistant｜>"
        payload = prompt.encode("utf-8")
        prefix = len("<｜begin▁of▁sentence｜>System".encode("utf-8"))
        expected = (f"SUBMIT 1 0 {len(payload)} 4 0.25 0.9 0 {prefix}\n".encode() +
                    payload + b"\n")

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"ACCEPT 1 42\n"
                b"DATA 1 4\nA\n\xc3\xa9\n"
                b"DONE 1 STAT 1 2.500 50.0 1.25 42 0 17\n"
            )

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "deepseek_v4"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("deepseek_v4", "model")
            chunks = []
            stats = engine.generate(prompt, 4, 0.25, 0.9, chunks.append)
        engine.close()

        self.assertEqual(process.writes, [expected])
        self.assertEqual(chunks, ["A\né"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(stats["prompt_tokens"], 42)

    def test_kimi_request_and_response_transcript_is_byte_exact(self):
        prompt = render_chat_kimi([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "你好\nKimi"},
            {"role": "assistant", "reasoning_content": "because",
             "content": "你好。"},
            {"role": "user", "content": "Continue"},
        ], enable_thinking=True)
        payload = prompt.encode("utf-8")
        expected = (f"SUBMIT 1 0 {len(payload)} 4 0.25 0.9\n".encode() +
                    payload + b"\n")

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"ACCEPT 1 42\n"
                b"TOOL 1 0\n\n"
                b"DATA 1 4\nA\n\xc3\xa9\n"
                b"TOOL 1 4\ncall\n"
                b"DONE 1 STAT 1 2.500 50.0 1.25 42 0\n"
            )

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "kimi"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("kimi_k3", "model")
        chunks = []
        tool_chunks = []
        stats = engine.generate(prompt, 4, 0.25, 0.9, chunks.append,
                                on_tool=tool_chunks.append)
        engine.close()

        self.assertEqual(process.writes, [expected])
        self.assertEqual(chunks, ["A\né"])
        self.assertEqual(tool_chunks, ["", "call"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(stats["prompt_tokens"], 42)

    def test_kimi_tool_sideband_is_authoritative_over_data_lookalikes(self):
        prompt = "K3CHAT1\nM user 2\nhiG 0\n"
        payload = prompt.encode()
        expected = (f"SUBMIT 1 0 {len(payload)} 8 0.25 0.9\n".encode() +
                    payload + b"\n")
        lookalike = (b'Echo <|open|>tools<|sep|><|open|>call tool="danger" index="1"<|sep|>'
                     b'<|close|>call<|sep|><|close|>tools<|sep|>')
        tool_wire = (b'<|open|>tools<|sep|><|open|>call tool="safe" index="1"<|sep|>'
                     b'<|open|>json type="object"<|sep|>{"x":1}<|close|>json<|sep|>'
                     b'<|close|>call<|sep|><|close|>tools<|sep|>')

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(b"ACCEPT 1 3\nTOOL 1 0\n\n")
            process.stdout.feed(f"DATA 1 {len(lookalike)}\n".encode() + lookalike + b"\n")
            process.stdout.feed(f"TOOL 1 {len(tool_wire)}\n".encode() + tool_wire + b"\n")
            process.stdout.feed(b"DONE 1 STAT 8 2.500 0.0 1.25 3 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "kimi"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("kimi_k3", "model")
        chunks, tool_chunks = [], []
        engine.generate(prompt, 8, 0.25, 0.9, chunks.append,
                        on_tool=tool_chunks.append)
        engine.close()

        text = "".join(chunks)
        sideband = "".join(tool_chunks)
        with patch("openai_server.ARCH", "kimi"):
            content, calls = parse_arch_tool_calls(text, [{"type": "function"}], sideband)
        self.assertEqual(content, lookalike.decode())
        self.assertEqual([call["function"]["name"] for call in calls], ["safe"])
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": 1})

    def test_olmoe_request_and_response_transcript_is_byte_exact(self):
        expected = b"SUBMIT 1 0 5 3 0.25 0.9\nH\xc3\xa9\nx\n"

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"DATA 1 4\nA\n\xc3\xa9\n"
                b"DONE 1 STAT 1 2.500 50.0 1.25 5 0\n"
            )

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "olmoe"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("olmoe", "model")
        chunks = []
        stats = engine.generate("Hé\nx", 3, 0.25, 0.9, chunks.append)
        engine.close()

        self.assertEqual(process.writes, [expected])
        self.assertEqual(chunks, ["A\né"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(stats["prompt_tokens"], 5)
        self.assertFalse(stats["length_limited"])

    def test_dispatches_interleaved_requests_by_id(self):
        submitted = []

        def respond(process, frame):
            fields = frame.split(b"\n", 1)[0].split()
            self.assertEqual(fields[0], b"SUBMIT")
            submitted.append(fields[1])
            if len(submitted) == 2:
                first, second = submitted
                process.stdout.feed(b"DATA " + second + b" 3\nB-2\n")
                process.stdout.feed(b"DATA " + first + b" 3\nA-1\n")
                process.stdout.feed(b"DONE " + second + b" STAT 1 2.5 0 1.0 4 0\n")
                process.stdout.feed(b"DATA " + first + b" 3\nA-2\n")
                process.stdout.feed(b"DONE " + first + b" STAT 2 3.5 0 1.0 5 1\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model", kv_slots=2)
        results = {}

        def generate(name, prompt, slot):
            chunks = []
            stats = engine.generate(prompt, 8, 0.7, 0.9, chunks.append, slot)
            results[name] = ("".join(chunks), stats)

        threads = [threading.Thread(target=generate, args=("a", "alpha", 0)),
                   threading.Thread(target=generate, args=("b", "beta", 1))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        engine.close()

        self.assertEqual(results["a"][0], "A-1A-2")
        self.assertTrue(results["a"][1]["length_limited"])
        self.assertEqual(results["b"][0], "B-2")
        headers = [frame.split(b"\n", 1)[0].split() for frame in process.writes]
        self.assertEqual({int(header[2]) for header in headers}, {0, 1})
        self.assertEqual({header[3] for header in headers}, {b"4", b"5"})

    def test_routes_engine_error_to_request(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ERROR " + request_id + b" slot is busy\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "slot is busy"):
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        engine.close()

    def test_close_wakes_pending_generation_and_is_idempotent(self):
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        errors = []

        def generate():
            try:
                engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=generate)
        thread.start()
        for _ in range(100):
            with engine.pending_lock:
                if engine.pending:
                    break
            threading.Event().wait(0.01)
        engine.close()
        engine.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["colibri engine is shutting down"])
        self.assertFalse(engine.dispatcher.is_alive())
        with engine.pending_lock:
            self.assertFalse(engine.pending)
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            engine.generate("again", 4, 0.7, 0.9, lambda _: None)

    def test_protocol_corruption_fails_request_and_stops_dispatcher(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" -1\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "DATA size"):
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "dispatcher stopped"):
            engine.generate("again", 4, 0.7, 0.9, lambda _: None)
        engine.close()

    def test_decodes_utf8_split_across_data_frames(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 1\n\xc3\n")
            process.stdout.feed(b"DATA " + request_id + b" 1\n\xa9\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        chunks = []
        engine.generate("hello", 4, 0.7, 0.9, chunks.append)
        engine.close()
        self.assertEqual(chunks, ["é"])

    def test_records_profile_snapshots_from_prof_lines(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 2\nok\n")
            process.stdout.feed(b"PROF 2.500 7 12 0.400 0.100 0.900 0.600 0.200 15\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 12 4.8 0 1.0 7 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        engine.generate("hello", 16, 0.7, 0.9, lambda _: None)
        engine.close()
        self.assertEqual(engine.profile_seq, 1)
        self.assertEqual(list(engine.profile), [{
            "wall_s": 2.5, "prompt_tokens": 7, "completion_tokens": 12,
            "expert_disk_s": 0.4, "expert_wait_s": 0.1, "expert_matmul_s": 0.9,
            "attention_s": 0.6, "lm_head_s": 0.2, "forwards": 15,
        }])

    def test_accepts_u7a_echo_and_extended_data_frames(self):
        # U7a forward-compat: the engine's opt-in per-token numeric channel --
        # ECHO frames for echoed prompt positions and DATA frames extended
        # with "<lp> <k> [tid tlp]*k" -- must NOT trip the dispatcher's
        # catch-all (which kills every in-flight request). Text delivery and
        # the DONE stats stay exactly as for legacy frames; the numeric
        # fields are consumed by the server feature half (U7b).
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + request_id + b" 3\n")
            process.stdout.feed(b"ECHO " + request_id + b" 2 0 nan 0\nHi\n")
            process.stdout.feed(
                b"ECHO " + request_id +
                b" 6 1 -0.105361 2 7 -0.105361 9 -2.302585\n world\n")
            process.stdout.feed(
                b"ECHO " + request_id +
                b" 1 2 -1.203973 2 4 -0.803973 6 -1.203973\n!\n")
            process.stdout.feed(
                b"DATA " + request_id +
                b" 2 -0.223144 2 3 -0.223144 8 -1.723144\nok\n")
            process.stdout.feed(
                b"DONE " + request_id + b" STAT 1 2.5 0 1.0 3 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        chunks = []
        stats = engine.generate("Hi world!", 4, 0.0, 1.0, chunks.append)
        self.assertEqual(chunks, ["ok"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(stats["prompt_tokens"], 3)
        self.assertIsNone(engine.dispatcher_error)
        engine.close()

    def test_unknown_frame_still_stops_dispatcher(self):
        # The catch-all that makes an unrecognized frame a hard failure is
        # load-bearing for the U7a compatibility asymmetry (a new engine's
        # frame reaching an OLD server kills the dispatcher -- the reason the
        # engine half ships first and stays opt-in). Accepting ECHO/extended
        # DATA must not have widened acceptance beyond those frames.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"LOGPROB " + request_id + b" 0.5\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine response: LOGPROB"):
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        engine.close()

    def test_cancels_generation_after_consumer_disconnects(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"CANCEL":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        # Il consumatore se ne va DOPO aver ricevuto qualcosa: e' lo scenario che
        # il nome promette, e va costruito invece che sperato. Con cancelled che
        # tornava True da subito, la corsa era fra il ramo idle -- che cancella
        # prima di ogni dato -- e l'arrivo del frame: su Windows vinceva il primo
        # e l'asserzione su output falliva senza che nulla fosse rotto (#1328).
        # Qui il flag si alza dentro il sink, e nel ramo "data" decode() consegna
        # il token PRIMA che cancelled() venga interrogato: l'ordine e' garantito
        # dal codice, non dallo scheduler.
        #
        # Il tetto sui poll non e' decorativo. Senza, un guasto che impedisce la
        # consegna del token lascerebbe il flag basso per sempre: niente cancel,
        # niente eccezione, e il test si APPENDE invece di fallire -- misurato
        # rompendo decode() apposta. Il ramo idle interroga cancelled() ogni 50 ms,
        # quindi dopo un secondo si cancella comunque e l'asserzione su output
        # fallisce subito, dicendo la cosa giusta. Deterministico quando funziona,
        # rapido a fallire quando no.
        #
        # Il caso "cancella prima del primo frame" resta coperto, in modo
        # deterministico, da test_cancels_generation_before_first_frame (#908):
        # prima i due si sovrapponevano a caso e uno dei due vinceva a sorte.
        disconnected = False
        polls = 0

        def sink(text):
            nonlocal disconnected
            output.append(text)
            disconnected = True

        def consumer_gone():
            nonlocal polls
            polls += 1
            return disconnected or polls > 20

        with self.assertRaises(ClientCancelled):
            engine.generate("hello", 8, 0.7, 0.9, sink, cancelled=consumer_gone)
        engine.close()
        self.assertEqual(output, ["x"])
        self.assertEqual(process.writes[-1].split(), [b"CANCEL", request_id])

    def test_cancels_generation_before_first_frame(self):
        # #908: a client that disconnects while the engine is still prefilling
        # (no DATA frame has arrived) must cancel too. cancelled() used to be
        # polled only in the "data" branch, so the CANCEL never went out and
        # the turn ran to its token limit while this thread stayed blocked.
        # The fake engine emits nothing until it sees CANCEL -- exactly the
        # pre-first-frame regime -- and must still get one.
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
            elif fields[0] == b"CANCEL":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        flag = {"cancelled": False}
        outcome = []

        def generate():
            try:
                engine.generate("hello", 8, 0.7, 0.9, lambda _: None,
                                cancelled=lambda: flag["cancelled"])
            except ClientCancelled:
                outcome.append("cancelled")

        thread = threading.Thread(target=generate)
        thread.start()
        for _ in range(200):
            if any(frame.startswith(b"SUBMIT") for frame in process.writes):
                break
            time.sleep(0.01)
        flag["cancelled"] = True
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        engine.close()
        self.assertEqual(outcome, ["cancelled"])
        self.assertEqual(process.writes[-1].split(), [b"CANCEL", request_id])

    def test_stops_generation_through_successful_done_path(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"STOP":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        stats = engine.generate("hello", 8, 0.7, 0.9, output.append,
                                stopped=lambda: output == ["x"])
        engine.close()
        self.assertEqual(output, ["x"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(process.writes[-1].split(), [b"STOP", request_id])


def _capture_frames(body, path="/v1/completions"):
    """Sends `body` to `path` against a FakeProcess-backed Engine/APIServer
    and returns (status, parsed_response, frames_written_to_the_engine).
    Shared by the test classes below so the harness lives in one place.
    """
    frames = []

    def respond(process, frame):
        frames.append(frame)
        rid = frame.split()[1]
        process.stdout.feed(b"DATA " + rid + b" 5\nHello\n")
        process.stdout.feed(b"DONE " + rid + b" STAT 1 2.5 0 1.0 4 0\n")

    process = FakeProcess(respond)
    with patch("openai_server.subprocess.Popen", return_value=process):
        engine = Engine("glm", "model")
    server = APIServer(("127.0.0.1", 0), engine, "test-model", "secret", 16)
    thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    try:
        data = json.dumps(body).encode()
        headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        request = Request(f"http://127.0.0.1:{server.server_port}{path}",
                          data=data, headers=headers)
        with urlopen(request, timeout=2) as response:
            status = response.status
            parsed = json.load(response)
    finally:
        server.scheduler.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        engine.close()
    return status, parsed, frames


class BaseWireContractTest(unittest.TestCase):
    """Pins the SUBMIT frame written for a request that uses none of the
    optional fields; any change here is a wire-format change and must be
    deliberate.
    """

    def test_no_new_fields_request_emits_the_base_submit_header(self):
        _, _, frames = _capture_frames(
            {"model": "test-model", "prompt": "Complete me", "temperature": 0, "max_tokens": 4})
        self.assertEqual(frames, [b"SUBMIT 1 0 11 4 0 0.9\nComplete me\n"])


class SeedOptionTest(unittest.TestCase):
    """`generation_options()` accepts a `seed` field without raising."""

    def test_seed_is_accepted_by_generation_options(self):
        generation_options({"seed": 1234, "prompt": "hi"}, 16)   # must not raise


class SeedWireFrameTest(unittest.TestCase):
    """`seed` is accepted and ignored. A stub-response equality check alone
    is vacuous here (the scripted `respond` closure inside `_capture_frames`
    always returns the same canned text regardless of any request field) --
    the real proof is that the byte-exact SUBMIT frames the dispatcher
    writes to the engine process (see DispatcherTest above) never carry the
    seed value at all, seeded or not.
    """

    def test_seed_accepted_and_absent_from_submit_frame(self):
        base = {"model": "test-model", "prompt": "Complete me", "temperature": 0, "max_tokens": 4}
        status_plain, body_plain, frames_plain = _capture_frames(base)
        status_seeded, body_seeded, frames_seeded = _capture_frames({**base, "seed": 1234})
        self.assertEqual(status_plain, 200)
        self.assertEqual(status_seeded, 200)
        self.assertEqual(body_seeded["choices"][0], body_plain["choices"][0])
        # Each call uses a freshly-constructed Engine, so both first requests are
        # assigned request id "1" -- the wire frames are directly byte-comparable,
        # no field needs normalizing. Comparing the whole frame list (not just
        # the first frame) closes "reaches no wire frame" literally: if `seed`
        # ever leaked onto any frame, this equality would break.
        self.assertEqual(frames_seeded, frames_plain)

    def test_seed_accepted_and_absent_from_chat_submit_frame(self):
        # Same proof as above, on /v1/chat/completions: generation_options()
        # is shared by both endpoints, but the SUBMIT frame is built from
        # the chat-rendered prompt, so this is not implied by the completions
        # case above -- a divergence between the two call sites would only
        # show up here.
        base = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "temperature": 0, "max_tokens": 4}
        status_plain, body_plain, frames_plain = _capture_frames(base, path="/v1/chat/completions")
        status_seeded, body_seeded, frames_seeded = _capture_frames(
            {**base, "seed": 1234}, path="/v1/chat/completions")
        self.assertEqual(status_plain, 200)
        self.assertEqual(status_seeded, 200)
        self.assertEqual(body_seeded["choices"][0], body_plain["choices"][0])
        self.assertEqual(frames_seeded, frames_plain)


class CapSentinelShimTest(unittest.TestCase):
    # #379 cap-sentinel shim, arch-keyed (#386 r2, F3): an absent cap is
    # "platform-auto" only for the glm engine (colibri.c coli_resolve_cap);
    # inkling reads cap <= 0 as "fit the expert LRU to all available RAM"
    # (inkling.c), so leaking the sentinel to a non-glm arch silently changes
    # its memory behavior. The key is the MODEL's config.json model_type, not
    # the engine binary's file name -- COLI_ENGINE users package the glm
    # engine as glm52/colibri-1.2/glm-metal, and basename keying disabled the
    # platform default for exactly them (and an inkling binary someone names
    # `glm` would get the leak back). Engine() is the one funnel every launch
    # passes through, so the translation is pinned at the argv it emits, over
    # the full matrix: arch x arbitrary executable name x cap absent/explicit.
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _model(self, model_type):
        model = Path(self.tmp.name) / f"model-{model_type}"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text(json.dumps({"model_type": model_type}))
        return str(model)

    def _spawn_argv(self, executable, model, **kwargs):
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process) as popen:
            engine = Engine(executable, model, **kwargs)
            engine.close()
        return popen.call_args[0][0]

    def test_matrix_arch_times_engine_name_times_cap(self):
        # COLI_ENGINE axis: the executable name carries no information; only
        # the model arch and the explicitness of cap may matter.
        executables = ("colibri", "glm", "glm52", "colibri-1.2", "glm-metal",
                       "/opt/custom/inkling", "kimi_k3.exe")
        cases = (  # (model_type, cap kwargs, expected argv cap)
            ("glm_moe_dsa", {}, "0"),          # absent -> glm sentinel
            ("inkling", {}, "8"),              # absent -> legacy 8
            ("kimi_k3", {}, "8"),              # absent -> legacy 8
            ("glm_moe_dsa", {"cap": 5}, "5"),  # explicit -> verbatim
            ("inkling", {"cap": 5}, "5"),
            ("kimi_k3", {"cap": 5}, "5"),
            ("glm_moe_dsa", {"cap": 0}, "0"),  # explicit 0 -> verbatim
            ("inkling", {"cap": 0}, "0"),      # upstream RAM-auto, by request
            ("kimi_k3", {"cap": 0}, "0"),
        )
        for model_type, kwargs, want in cases:
            model = self._model(model_type)
            for executable in executables:
                self.assertEqual(
                    self._spawn_argv(executable, model, **kwargs),
                    [executable, want],
                    f"arch={model_type} exe={executable} kwargs={kwargs}")

    def test_synthetic_engine_without_config_uses_explicit_arch(self):
        self.assertEqual(self._spawn_argv("engine", "/nonexistent/model"),
                         ["engine", "0"])
        model = Path(self.tmp.name) / "model-broken"
        model.mkdir()
        (model / "config.json").write_text("{not json")
        with self.assertRaisesRegex(ValueError, "invalid config.json"):
            self._spawn_argv("engine", str(model))

    def test_cap_for_arch_is_the_single_translation_point(self):
        # 0 is the "you decide" sentinel, and it goes to the engines that
        # actually do decide: glm resolves it platform-aware, olmoe sizes its
        # expert cache from the RAM budget once the dense weights are resident
        # (#1443). The others still get the legacy eight slots per layer.
        self.assertEqual(cap_for_arch("glm", None), 0)
        self.assertEqual(cap_for_arch("olmoe", None), 0)
        self.assertEqual(cap_for_arch("inkling", None), 8)
        self.assertEqual(cap_for_arch("kimi", None), 8)
        self.assertEqual(cap_for_arch("glm", 3), 3)
        self.assertEqual(cap_for_arch("inkling", 3), 3)
        self.assertEqual(cap_for_arch("inkling", 0), 0)   # explicit 0 is explicit
        self.assertEqual(cap_for_arch("glm", 0), 0)

    def test_profile_cap_is_below_explicit_cap_and_above_implicit_default(self):
        profiled = {"COLI_PROFILE_CAP": "24"}
        self.assertEqual(cap_for_arch("inkling", None, profiled), 24)
        self.assertEqual(cap_for_arch("inkling", 7, profiled), 7)
        self.assertEqual(cap_for_arch("inkling", None,
                                      {"COLI_PROFILE_CAP": "invalid"}), 8)
        planned = {"COLI_PLAN_CAP": "11"}
        self.assertEqual(cap_for_arch("qwen38", None, planned), 11)
        self.assertEqual(cap_for_arch(
            "qwen38", None, {"COLI_PLAN_CAP": "11", "COLI_PROFILE_CAP": "7"}), 7)
        self.assertEqual(cap_for_arch(
            "qwen38", 5, {"COLI_PLAN_CAP": "11", "COLI_PROFILE_CAP": "7"}), 5)

    def test_engine_consumes_profile_cap_without_leaking_private_env(self):
        process = FakeProcess(lambda _process, _frame: None)
        model = self._model("inkling")
        with patch("openai_server.subprocess.Popen", return_value=process) as popen:
            engine = Engine("custom-engine", model,
                            env={"COLI_PROFILE_CAP": "24", "KEEP": "yes"})
            engine.close()
        command = popen.call_args[0][0]
        child_env = popen.call_args[1]["env"]
        self.assertEqual(command, ["custom-engine", "24"])
        self.assertNotIn("COLI_PROFILE_CAP", child_env)
        self.assertEqual(child_env["KEEP"], "yes")

    def test_engine_consumes_planned_cap_without_leaking_private_env(self):
        process = FakeProcess(lambda _process, _frame: None)
        model = self._model("qwen4_exp_text")
        with patch("openai_server.subprocess.Popen", return_value=process) as popen:
            engine = Engine("qwen38", model,
                            env={"COLI_PLAN_CAP": "13", "KEEP": "yes"})
            engine.close()
        self.assertEqual(popen.call_args[0][0], ["qwen38", "13"])
        child_env = popen.call_args[1]["env"]
        self.assertNotIn("COLI_PLAN_CAP", child_env)
        self.assertEqual(child_env["KEEP"], "yes")

    def test_v41_ram_plan_reaches_engine_argv(self):
        model = self._model("deepseek_v41")
        for settings, expected_ram in (({"RAM_GB": "120", "CTX": "8192"}, 120),
                                       ({"CTX": "8192"}, 0),
                                       ({"RAM_GB": "auto", "CTX": "8192"}, 0)):
            with self.subTest(settings=settings):
                process = FakeProcess(lambda _process, _frame: None)
                with patch("resource_plan.build_plan", return_value={
                        "tiers": {"ram": {"cache_slots_per_layer": 96}}}) as planner, \
                        patch("openai_server.subprocess.Popen", return_value=process) as popen:
                    engine = Engine("deepseek_v41", model, env=settings)
                    engine.close()
                planner.assert_called_once_with(model, ram_gb=expected_ram,
                                                context=8192, gpu_indices=[])
                self.assertEqual(popen.call_args[0][0], ["deepseek_v41", "96"])

    def test_v41_explicit_and_calibrated_caps_bypass_planning(self):
        for cap, env, expected in ((7, {}, 7), (0, {}, 0),
                                   (None, {"COLI_PROFILE_CAP": "12"}, 12),
                                   (None, {"COLI_PLAN_CAP": "24"}, 24)):
            with self.subTest(cap=cap, env=env), patch("resource_plan.build_plan") as planner:
                self.assertEqual(cap_for_arch("deepseek_v41", cap, env, model="model"),
                                 expected)
                planner.assert_not_called()

    def test_v41_insufficient_ram_refuses_before_spawning(self):
        model = self._model("deepseek_v41")
        with patch("resource_plan.build_plan", return_value={
                "tiers": {"ram": {"cache_slots_per_layer": 0}}}), \
                patch("openai_server.subprocess.Popen") as popen:
            with self.assertRaisesRegex(ValueError, "one expert slot"):
                Engine("deepseek_v41", model, env={"RAM_GB": "8"})
            popen.assert_not_called()

    def test_model_arch_reads_model_type(self):
        self.assertEqual(model_arch(self._model("glm_moe_dsa")), "glm")
        self.assertEqual(model_arch(self._model("inkling")), "inkling")
        self.assertEqual(model_arch(self._model("kimi_k3")), "kimi")
        self.assertEqual(model_arch(self._model("deepseek_v4")), "deepseek_v4")
        self.assertEqual(model_arch(self._model("olmoe")), "olmoe")
        self.assertEqual(model_arch(self._model("qwen4_exp")), "qwen38")
        self.assertEqual(model_arch(self._model("qwen4_exp_text")), "qwen38")
        with self.assertRaisesRegex(ValueError, "cannot read config.json"):
            model_arch("/nonexistent")

    def test_direct_v4_server_gets_bounded_dspark_defaults(self):
        env = {"V4_MTP_CONF": "0.7"}
        with patch("resource_plan.physical_cpu_count",
                   side_effect=AssertionError("V4 server sized the team")), \
             patch("openai_server.sys.platform", "linux"):
            tune_child_env(env, "deepseek_v4")
        self.assertNotIn("OMP_NUM_THREADS", env)
        self.assertEqual(env["OMP_PROC_BIND"], "close")
        self.assertEqual(env["V4_DRAFT"], "0")
        self.assertEqual(env["V4_MTP"], "0")
        self.assertEqual(env["V4_MTP_DRAFT"], "3")
        self.assertEqual(env["V4_MTP_GB"], "0.45")
        self.assertEqual(env["V4_MTP_CONF"], "0.7")  # explicit override wins
        self.assertEqual(env["V4_MTP_GPU"], "0")     # GPU drafting opt-in, off by default

    def test_direct_v4_server_preserves_explicit_omp_threads(self):
        env = {"OMP_NUM_THREADS": "3"}
        tune_child_env(env, "deepseek_v4")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")

    def test_direct_v4_server_honours_omp_kill_switch(self):
        env = {"COLI_NO_OMP_TUNE": "1"}
        tune_child_env(env, "deepseek_v4")
        for key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
                    "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES"):
            self.assertNotIn(key, env)


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0),cls.engine,"test-model","secret",16,kv_slots=2)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, body=None, key="secret"):
        headers = {"Authorization": f"Bearer {key}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=2)

    def test_lists_models_and_checks_auth(self):
        with self.request("/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "test-model")
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/models", key="wrong")
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 401)

    def test_metrics_counts_http_engine_failure_without_success(self):
        before = self.server.scheduler.snapshot()
        with patch.object(self.engine, "generate", side_effect=RuntimeError("injected failure")):
            with self.assertRaises(HTTPError) as caught:
                self.request("/v1/chat/completions", {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "hello"}], "max_tokens": 1})
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 500)
        after = self.server.scheduler.snapshot()
        self.assertEqual(after["failed"], before["failed"] + 1)
        self.assertEqual(after["completed"], before["completed"])
        with self.request("/metrics") as response:
            text = response.read().decode()
        self.assertIn(f'colibri_scheduler_failed_total {after["failed"]}\n', text)
        self.assertIn("colibri_scheduler_active 0\n", text)

    def test_metrics_exposes_prometheus_text_with_auth(self):
        with self.request("/metrics") as response:
            self.assertEqual(response.headers["Content-Type"],
                             "text/plain; version=0.0.4; charset=utf-8")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            text = response.read().decode()
        self.assertIn("colibri_scheduler_capacity 2\n", text)
        self.assertIn("# TYPE colibri_scheduler_queue_wait_seconds histogram\n", text)
        for key in ("wrong", ""):
            with self.assertRaises(HTTPError) as caught:
                self.request("/metrics", key=key)
            self.addCleanup(caught.exception.close)
            self.assertEqual(caught.exception.code, 401)
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/metrics", timeout=2)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 401)

    def test_health_reports_scheduler_and_kv_slots(self):
        with self.request("/health") as response:
            health = json.load(response)
            scheduler = health["scheduler"]
        self.assertEqual(scheduler["max_queue"], 8)
        self.assertIn("queued", scheduler)
        self.assertEqual(health["kv_slots"], 2)

    def test_health_reports_the_continuation_switch(self):
        """The web UI shows Continue only when this is true: with the switch off a
        trailing assistant turn is answered fresh, which a Continue button would
        present as a resumption. Unauthed probes keep the bare liveness shape."""
        for value, expected in (("1", True), ("0", False)):
            with patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": value}), \
                 self.request("/health") as response:
                self.assertIs(json.load(response)["continue_assistant"], expected)
        with urlopen(self.base + "/health", timeout=2) as response:
            self.assertNotIn("continue_assistant", json.load(response))

    def test_profile_requires_auth(self):
        """/profile is served before require_auth(), so it needs its own gate.

        This test previously asserted the opposite -- that the telemetry was
        readable without a key. That was not a client requirement: the web
        dashboard sends `Authorization: Bearer` to /profile exactly as it does
        to /health and /experts (web/src/lib/api.ts). The turns carry prompt and
        completion token counts and per-phase timings for the last 120 requests,
        which describes what the operator is running and how much of it, so an
        anonymous caller now gets the same empty shape those two endpoints give.
        """
        turn = {"wall_s": 2.5, "prompt_tokens": 7, "completion_tokens": 12,
                "expert_disk_s": 0.4, "expert_wait_s": 0.1, "expert_matmul_s": 0.9,
                "attention_s": 0.6, "lm_head_s": 0.2, "forwards": 15}
        self.engine.profile = [turn]
        self.engine.profile_seq = 1
        try:
            with urlopen(self.base + "/profile", timeout=2) as response:
                self.assertEqual(json.load(response), {"seq": 0, "turns": []},
                                 "unauthenticated caller received telemetry")
            with self.request("/profile") as response:
                self.assertEqual(json.load(response), {"seq": 1, "turns": [turn]},
                                 "authenticated caller lost access")
        finally:
            del self.engine.profile, self.engine.profile_seq

    def test_browser_preflight(self):
        request = Request(self.base + "/v1/chat/completions", method="OPTIONS", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])

    def test_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4, "cache_slot": 1,
        }) as response:
            body = json.load(response)
            queue_wait = response.headers.get("x-colibri-queue-wait-ms")
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(body["usage"], {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})
        self.assertIsNotNone(queue_wait)
        self.assertIn("<|user|>Hi<|assistant|><think></think>", self.engine.calls[-1][0])
        self.assertEqual(self.engine.calls[-1][4], 1)

    def test_kimi_chat_completion_uses_multiturn_wire_payload(self):
        with patch("openai_server.ARCH", "kimi"):
            with self.request("/v1/chat/completions", {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好。"},
                    {"role": "user", "content": "Continue"},
                ],
                "enable_thinking": False,
            }) as response:
                body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(
            self.engine.calls[-1][0],
            "K3CHAT1\n"
            "M user 6\n你好"
            "M assistant 9\n你好。"
            "M user 8\nContinue"
            "G 0\n",
        )

    def test_chat_completion_stops_across_engine_chunks(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stop": "éll",
        }) as response:
            body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "H")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.stop_requests, before + 1)

    def test_patient_stop_extension_ignores_a_leading_match(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stop": "H", "x_colibri_ignore_leading_stop": True,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "éllo")
        self.assertEqual(self.engine.stop_requests, before)

    def test_patient_stop_extension_requires_a_boolean(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stop": "H", "x_colibri_ignore_leading_stop": "yes",
            })
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_rejects_invalid_cache_slot(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "cache_slot": 2,
            })
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_olmoe_never_files_the_answer_as_reasoning(self):
        # #984: OLMoE has no thinking mode, so its engine never emits </think>.
        # With thinking left on, the reasoning splitter kept the whole answer in
        # reasoning_content and streamed an empty `content` -- content dropped.
        # Forcing thinking off for olmoe must return the answer as content
        # whether or not the client asked for reasoning, streaming or not.
        for streaming in (False, True):
            with self.subTest(stream=streaming), \
                 patch("openai_server.ARCH", "olmoe"):
                with self.request("/v1/chat/completions", {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "reasoning_effort": "high",   # a client asking to think
                    "stream": streaming,
                }) as response:
                    raw = response.read().decode()
                if streaming:
                    payloads = [json.loads(line[6:]) for line in raw.splitlines()
                                if line.startswith("data: ") and line != "data: [DONE]"]
                    content = "".join((c.get("delta") or {}).get("content", "")
                                      for p in payloads for c in p["choices"])
                    reasoning = "".join((c.get("delta") or {}).get("reasoning_content", "")
                                        for p in payloads for c in p["choices"])
                else:
                    msg = json.loads(raw)["choices"][0]["message"]
                    content = msg.get("content") or ""
                    reasoning = msg.get("reasoning_content") or ""
                self.assertIn("Hé", content, "the answer must arrive as content")
                self.assertEqual(reasoning, "", "olmoe must not produce reasoning")

    def test_streaming_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            stream = response.read().decode()
        self.assertIn('\"delta\":{\"role\":\"assistant\",\"content\":\"\"}', stream)
        self.assertIn('\"object\":\"chat.completion.chunk\"', stream)
        self.assertIn('\"content\":\"Hé\"', stream)
        self.assertIn('\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}', stream)
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))

    def test_streaming_stop_never_exposes_partial_sequence(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "stop": "éll",
        }) as response:
            raw = response.read().decode()
        payloads = [json.loads(line[6:]) for line in raw.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join((choice.get("delta") or {}).get("content", "")
                          for payload in payloads for choice in payload["choices"])
        self.assertEqual(content, "H")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.stop_requests, before + 1)

    def test_legacy_completion(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Complete me", "temperature": 0,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "Héllo")
        self.assertEqual(self.engine.calls[-1][0], "Complete me")

    def test_rejects_empty_legacy_completion(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {"model": "test-model", "prompt": ""})
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.load(caught.exception)["error"]["param"], "prompt")

    def test_rejects_invalid_stream_options(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stream": True, "stream_options": "usage",
            })
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_unpaired_surrogate_is_a_client_error(self):
        """JSON can spell a lone UTF-16 surrogate ("\\ud83d": a client that cut a
        string between the two halves of an emoji). json.loads accepts it, no
        UTF-8 can carry it, and Engine.generate's first step, prompt.encode(),
        raised: HTTP 500 "The colibri engine failed". Invalid UTF-8 in the raw
        body is already a 400; this is the same invalid text, escaped."""
        real_generate = self.engine.generate

        def encoding_generate(prompt, *args, **kwargs):
            prompt.encode("utf-8")          # what Engine.generate does before anything else
            return real_generate(prompt, *args, **kwargs)

        cut = "emoji \ud83d"
        cases = (("/v1/chat/completions", {"messages": [{"role": "user", "content": cut}]}),
                 ("/v1/completions", {"prompt": cut}),
                 ("/v1/messages", {"max_tokens": 4, "messages": [{"role": "user", "content": cut}]}))
        with patch.object(self.engine, "generate", side_effect=encoding_generate):
            for path, body in cases:
                with self.subTest(path=path):
                    with self.assertRaises(HTTPError) as caught:
                        self.request(path, dict(body, model="test-model"))
                    self.addCleanup(caught.exception.close)
                    self.assertEqual(caught.exception.code, 400)

    def test_tool_choice_with_a_non_object_function_is_a_client_error(self):
        """{"type": "function", "function": "search"} answered HTTP 500.

        generation_options() already has the 400 for a `tool_choice` function
        object with no name, but it -- and the five renderers that read the
        forced tool the same way -- did `(tool_choice.get("function") or {})
        .get("name")`, so a `function` that is not an object raised
        AttributeError first and do_POST answered 500 "The colibri engine failed
        to process the request." for a request the engine never saw. Every arch
        and both OpenAI endpoints, because the crash was on the read.
        """
        tools = [{"type": "function", "function": {"name": "search"}}]
        for arch in ("glm", "glm53", "kimi", "deepseek_v4", "deepseek_v41",
                     "olmoe", "inkling", "qwen36", "qwen38"):
            for function in ("search", ["search"], 5):
                with self.subTest(arch=arch, function=function):
                    with patch("openai_server.ARCH", arch):
                        with self.assertRaises(HTTPError) as caught:
                            self.request("/v1/chat/completions", {
                                "model": "test-model",
                                "messages": [{"role": "user", "content": "hi"}],
                                "tools": tools,
                                "tool_choice": {"type": "function", "function": function}})
                    self.addCleanup(caught.exception.close)
                    self.assertEqual(caught.exception.code, 400)
        # /v1/completions has no renderer in front of generation_options, so it
        # reached the same read directly.
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {
                "model": "test-model", "prompt": "hi", "tools": tools,
                "tool_choice": {"type": "function", "function": "search"}})
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_a_well_formed_forced_tool_choice_still_runs(self):
        """The read above must not change the shape clients actually send."""
        with self.request("/v1/chat/completions", {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "search"}}],
                "tool_choice": {"type": "function",
                                "function": {"name": "search"}}}) as response:
            self.assertEqual(response.status, 200)
        self.assertIn("You must call the function `search`", self.engine.calls[-1][0])
        # The OpenAI-legacy spelling {"type": "function", "name": ...} still
        # forces the tool: _tool_choice_name keeps that fallback.
        with self.request("/v1/chat/completions", {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "search"}}],
                "tool_choice": {"type": "function", "name": "search"}}) as response:
            self.assertEqual(response.status, 200)
        self.assertIn("You must call the function `search`", self.engine.calls[-1][0])

    def test_tool_with_a_non_object_function_is_a_client_error(self):
        """{"type": "function", "function": "search"} on tools[] answered HTTP 500.

        generation_options already has the 400, but /v1/chat/completions renders
        first. GLM, GLM-5.3, DeepSeek V4 and V4.1 did fn.items() on a string
        and the AttributeError became do_POST's 500 "The colibri engine failed
        to process the request." for a request the engine never saw.
        """
        for arch in ("glm", "glm53", "kimi", "deepseek_v4", "deepseek_v41",
                     "olmoe", "inkling", "qwen36", "qwen38"):
            for function in ("search", ["search"], 5):
                with self.subTest(arch=arch, function=function):
                    with patch("openai_server.ARCH", arch):
                        with self.assertRaises(HTTPError) as caught:
                            self.request("/v1/chat/completions", {
                                "model": "test-model",
                                "messages": [{"role": "user", "content": "hi"}],
                                "tools": [{"type": "function",
                                           "function": function}]})
                    self.addCleanup(caught.exception.close)
                    self.assertEqual(caught.exception.code, 400)
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {
                "model": "test-model", "prompt": "hi",
                "tools": [{"type": "function", "function": "search"}]})
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)

    def test_a_well_formed_tool_function_still_runs(self):
        """The read above must not change the shape clients actually send."""
        with self.request("/v1/chat/completions", {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function",
                           "function": {"name": "search"}}]}) as response:
            self.assertEqual(response.status, 200)
        self.assertIn('"name": "search"', self.engine.calls[-1][0])

    def test_tool_call_arguments_that_are_not_an_object_do_not_fail_the_request(self):
        """A replayed tool call with `arguments: "[1, 2]"` answered HTTP 500.

        render_chat reached `(args or {}).items()` with a parsed list, and the
        AttributeError became do_POST's catch-all 500 "The colibri engine failed
        to process the request." -- which OpenAI SDKs retry, against an engine
        that was never asked anything.
        """
        body = {"model": "test-model", "messages": [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "x", "type": "function",
                 "function": {"name": "fn", "arguments": "[1, 2]"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "done"},
            {"role": "user", "content": "and now?"},
        ]}
        with self.request("/v1/chat/completions", body) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(json.load(response)["object"], "chat.completion")


class ClientHangupTest(unittest.TestCase):
    """A client that disconnects mid-response must not print a traceback.

    `coli chat` polls /health on a 2 s timeout while the model loads and drops
    each connection the moment it has an answer; Ctrl-C during a stream closes
    the socket by design -- the banner tells the user to do exactly that. Both
    reach the handler as BrokenPipeError, and socketserver logs an unhandled
    exception per occurrence, so a normal DeepSeek V4 start buried the loading
    spinner under stack traces and every cancelled answer looked like a crash.
    """

    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                None, 16, kv_slots=1)
        self.errors = []
        # socketserver routes an escaped exception here; the base class prints
        # it to stderr. Recording instead of printing is what lets the test
        # assert on it rather than on captured output.
        self.server.handle_error = lambda request, address: self.errors.append(
            sys.exc_info()[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)

    def _hang_up_after_request(self, path):
        """Send a request, then close without reading the response."""
        sock = socket.create_connection(("127.0.0.1", self.server.server_port), 2)
        sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     f"Connection: close\r\n\r\n".encode())
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))   # RST rather than a clean FIN
        sock.close()
        time.sleep(0.3)

    def test_hangup_on_health_is_not_an_error(self):
        for _ in range(5):
            self._hang_up_after_request("/health")
        self.assertEqual(self.errors, [],
                         "client disconnect surfaced as a server error")

    def test_the_server_still_answers_afterwards(self):
        """The real damage would be a handler thread lost to the exception."""
        self._hang_up_after_request("/health")
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health",
                     timeout=2) as response:
            self.assertEqual(json.load(response)["status"], "ok")

    def _abort(self, *args):
        """Windows' spelling of the same disconnect, raised where it really lands.

        In #854's log the traceback runs do_GET -> send_json -> end_headers ->
        flush_headers -> wfile.write -> sendall, so raising from end_headers
        reproduces the exact shape on any platform.
        """
        raise ConnectionAbortedError(
            10053, "An established connection was aborted by the software in your "
                   "host machine")

    def test_windows_aborted_connection_is_not_an_error(self):
        """ConnectionAbortedError is a SIBLING of BrokenPipeError and
        ConnectionResetError under ConnectionError, not a subclass of either --
        so catching the pair caught the POSIX spellings and let the Windows one
        escape. #854 is pages of WinError 10053 tracebacks from a healthy start.
        """
        self.assertFalse(
            issubclass(ConnectionAbortedError, (BrokenPipeError, ConnectionResetError)),
            "the old except clause would have covered this; the test proves nothing")
        with patch.object(APIHandler, "end_headers", self._abort):
            try:
                with urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=2):
                    pass
            except Exception:
                pass                      # the client sees a broken response; that is fine
            time.sleep(0.3)
        self.assertEqual(self.errors, [],
                         "WinError 10053 surfaced as a server error (#854)")

    def test_the_server_survives_an_aborted_connection(self):
        """Same as the hangup case: the damage is a lost handler, not the log."""
        with patch.object(APIHandler, "end_headers", self._abort):
            try:
                with urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=2):
                    pass
            except Exception:
                pass
            time.sleep(0.3)
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health",
                     timeout=2) as response:
            self.assertEqual(json.load(response)["status"], "ok")


class StaticServingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        dist = root / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("dashboard", encoding="utf-8")
        sibling = root / "dist-private"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("private", encoding="utf-8")
        self.web_dist = patch.object(APIHandler, "WEB_DIST", dist)
        self.web_dist.start()
        self.server = APIServer(("127.0.0.1", 0), FakeEngine(), "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.web_dist.stop()
        self.tmp.cleanup()

    def test_static_root_stays_inside_dist_directory(self):
        with urlopen(self.base + "/", timeout=2) as response:
            self.assertEqual(response.read(), b"dashboard")
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/%2e%2e/dist-private/secret.txt", timeout=2)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 404)


class SchedulerHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = BlockingEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                max_tokens=16, max_queue=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.engine.release.set()
        self.server.scheduler.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self):
        body = json.dumps({"model": "test-model", "messages": [
            {"role": "user", "content": "Hi"}]}).encode()
        return urlopen(Request(self.url, data=body, headers={"Content-Type": "application/json"}), timeout=2)

    def test_queue_full_returns_429_before_generation(self):
        first_errors = []

        def first_request():
            try:
                with self.request() as response: response.read()
            except Exception as error:
                first_errors.append(error)

        first = threading.Thread(target=first_request); first.start()
        self.assertTrue(self.engine.entered.wait(1))
        with self.assertRaises(HTTPError) as caught:
            self.request()
        self.addCleanup(caught.exception.close)
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "1")
        self.assertEqual(error["code"], "queue_full")
        self.engine.release.set(); first.join(2)
        self.assertEqual(first_errors, [])



ORDER_TOOL = [{"type": "function", "function": {
    "name": "lookup_order",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"},
        "qty": {"type": "integer"},
        "express": {"type": "boolean"},
    }, "required": ["order_id"]}}}]


class ToolArgumentTypeTest(unittest.TestCase):
    """The model emits every argument as text. Without the schema, a string-typed value that
    happens to look numeric is json.loads()'d into an int and the tool gets the wrong type."""

    def _args(self, reply, tools=ORDER_TOOL):
        _, calls = parse_tool_calls(reply, tools)
        self.assertEqual(len(calls), 1)
        return json.loads(calls[0]["function"]["arguments"])

    def test_string_parameter_holding_digits_stays_a_string(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>order_id</arg_key><arg_value>12345</arg_value></tool_call>")
        self.assertEqual(args["order_id"], "12345")
        self.assertIsInstance(args["order_id"], str)

    def test_declared_numeric_and_boolean_parameters_are_decoded(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>"
                          "<arg_key>qty</arg_key><arg_value>2</arg_value>"
                          "<arg_key>express</arg_key><arg_value>true</arg_value></tool_call>")
        self.assertEqual(args, {"order_id": "A-1", "qty": 2, "express": True})
        self.assertIsInstance(args["qty"], int)
        self.assertIs(args["express"], True)

    def test_unknown_parameter_keeps_permissive_decoding(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>extra</arg_key><arg_value>7</arg_value></tool_call>")
        self.assertEqual(args["extra"], 7)


class DeepSeekV4ToolCallTest(unittest.TestCase):
    """DeepSeek V4 (#916): DSML tool blocks. Schemas render into the first system/developer
    message, assistant tool_calls render as <｜DSML｜invoke> blocks, tool results merge into
    user turns as <tool_result>, and model output parses back into OpenAI tool_calls."""

    DSML = "｜DSML｜"
    WEATHER = [{"type": "function", "function": {
        "name": "get_weather", "description": "current weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"]}}}]
    CALL = [{"id": "call_1", "type": "function", "function": {
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris", "days": 3})}}]

    def test_tools_declared_on_first_system_message(self):
        prompt = render_chat_v4([{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "Weather?"}], tools=self.WEATHER)
        self.assertLess(prompt.index("Be brief."), prompt.index("## Tools"))
        self.assertIn(f'"{self.WEATHER[0]["function"]["name"]}"', prompt)
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>"))

    def test_tools_prepended_when_no_system_message(self):
        prompt = render_chat_v4([{"role": "user", "content": "hi"}], tools=self.WEATHER)
        # The official encoder renders tools on an empty system message: bos + "\n\n" + tools.
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>\n\n## Tools"))

    def test_tool_choice_none_suppresses_declaration(self):
        prompt = render_chat_v4([{"role": "user", "content": "hi"}],
                                tools=self.WEATHER, tool_choice="none")
        self.assertNotIn("## Tools", prompt)

    def test_developer_message_is_wrapped_in_user_token(self):
        # encoding_dsv4.py wraps developer content in <｜User｜>; system stays bare.
        prompt = render_chat_v4([{"role": "developer", "content": "Be terse."},
                                 {"role": "user", "content": "hi"}], tools=self.WEATHER)
        self.assertIn("<｜User｜>Be terse.", prompt)
        self.assertNotIn("<｜User｜>## Tools", prompt)

    def test_thinking_mode_prepends_effort_prompt(self):
        # encoding_dsv4.py prepends the level prompt after BOS in thinking mode.
        prompt = render_chat_v4([{"role": "user", "content": "hi"}], enable_thinking=True,
                                reasoning_effort="high")
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: High."))
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))
        # V4-native level names work too; low adds nothing.
        self.assertTrue(render_chat_v4([{"role": "user", "content": "hi"}],
                                       enable_thinking=True, reasoning_effort="max").startswith(
            "<｜begin▁of▁sentence｜>Reasoning Effort: Maximum."))
        self.assertFalse(render_chat_v4([{"role": "user", "content": "hi"}],
                                        enable_thinking=True, reasoning_effort="low").startswith(
            "<｜begin▁of▁sentence｜>Reasoning Effort:"))

    def test_assistant_tool_calls_render_as_dsml(self):
        prompt = render_chat_v4([{"role": "user", "content": "Paris?"},
                                 {"role": "assistant", "content": None,
                                  "tool_calls": self.CALL}])
        self.assertIn(f'<{self.DSML}invoke name="get_weather">', prompt)
        self.assertIn(f'<{self.DSML}parameter name="city" string="true">Paris</{self.DSML}parameter>',
                      prompt)
        self.assertIn(f'<{self.DSML}parameter name="days" string="false">3</{self.DSML}parameter>',
                      prompt)
        self.assertTrue(prompt.endswith("</think>"))

    def test_tool_results_merge_into_one_user_turn(self):
        prompt = render_chat_v4([{"role": "user", "content": "Paris?"},
                                 {"role": "assistant", "content": None, "tool_calls": self.CALL},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "21c"},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                                 {"role": "user", "content": "And London?"}])
        self.assertEqual(prompt.count("<｜User｜>"), 2)
        self.assertIn("<tool_result>21c</tool_result>", prompt)
        self.assertIn("<tool_result>sunny</tool_result>", prompt)
        self.assertIn("And London?", prompt)

    def test_parse_dsml_reply_to_openai_tool_calls(self):
        raw = (f"Here you go.\n\n<{self.DSML}tool_calls>\n"
               f"<{self.DSML}invoke name=\"get_weather\">\n"
               f"<{self.DSML}parameter name=\"city\" string=\"true\">Paris</{self.DSML}parameter>\n"
               f"<{self.DSML}parameter name=\"days\" string=\"false\">3</{self.DSML}parameter>\n"
               f"</{self.DSML}invoke>\n</{self.DSML}tool_calls><｜end▁of▁sentence｜>")
        content, calls = parse_dsv4_tool_calls(raw)
        self.assertEqual(content, "Here you go.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Paris", "days": 3})

    def test_parse_truncated_block_leaks_no_markers(self):
        content, calls = parse_dsv4_tool_calls("Almost.\n\n<｜DSML｜tool_calls><｜DSML｜invoke na")
        self.assertEqual(calls, [])
        self.assertEqual(content, "Almost.")
        self.assertNotIn("DSML", content)


class EngineErrorFrameTest(unittest.TestCase):
    """#401: an over-long prompt used to be silently truncated to the first CTX-2 tokens, so the
    model answered from a mutilated prompt and the client got HTTP 200 with junk. The engine now
    refuses, and the refusal has to reach the client as a 400 it can act on -- not a 500."""

    def test_context_exceeded_becomes_a_400_the_client_can_act_on(self):
        err = _engine_error(["CONTEXT_EXCEEDED", "8321", "4094"], "CONTEXT_EXCEEDED 8321 4094")
        self.assertIsInstance(err, APIError)
        self.assertEqual(err.status, 400)
        self.assertEqual(err.code, "context_length_exceeded")
        self.assertEqual(err.param, "messages")
        self.assertIn("4094", err.message)
        self.assertIn("8321", err.message)

    def test_other_engine_errors_stay_runtime_errors(self):
        for frame in (["SLOT_BUSY"], ["BAD_REQUEST"], []):
            err = _engine_error(frame, " ".join(frame) or "engine request failed")
            self.assertIsInstance(err, RuntimeError)
            self.assertNotIsInstance(err, APIError)

    def test_malformed_context_frame_does_not_crash_the_dispatcher(self):
        err = _engine_error(["CONTEXT_EXCEEDED"], "CONTEXT_EXCEEDED")
        self.assertIsInstance(err, APIError)
        self.assertEqual(err.status, 400)
class UnclosedToolCallTest(unittest.TestCase):
    """#401: the model opens <tool_call>, emits a well-formed call, then stops without the
    closing tag (budget ran out, or quantization mangled it). The strict regex needs both tags,
    so the client used to get zero tool_calls -- a total failure from a recoverable output."""

    NO_ARG_TOOL = ORDER_TOOL + [{"type": "function",
                                 "function": {"name": "list_orders", "parameters": {}}}]

    def _calls(self, reply, tools=ORDER_TOOL):
        return parse_tool_calls(reply, tools)

    def test_unclosed_box_is_recovered(self):
        content, calls = self._calls("<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"order_id": "A-1"})
        self.assertEqual(content, "")

    def test_mangled_closing_tag_is_recovered(self):
        _, calls = self._calls("<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>A-1</arg_value></tool_cal")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"order_id": "A-1"})

    def test_leading_prose_is_kept_as_content(self):
        content, calls = self._calls("Let me check.\n<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Let me check.")

    def test_closed_call_followed_by_an_unclosed_one(self):
        _, calls = self._calls("<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>A-1</arg_value></tool_call>"
                               "<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>B-2</arg_value>")
        self.assertEqual([json.loads(c["function"]["arguments"])["order_id"] for c in calls],
                         ["A-1", "B-2"])

    def test_bare_declared_name_recovers_a_zero_argument_call(self):
        _, calls = self._calls("<tool_call>list_orders", self.NO_ARG_TOOL)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_orders")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_prose_mentioning_the_marker_does_not_fabricate_a_call(self):
        content, calls = self._calls("To call a tool, write <tool_call> and then the name.")
        self.assertEqual(calls, [])
        self.assertIn("<tool_call>", content)

    def test_undeclared_name_without_arguments_is_not_recovered(self):
        _, calls = self._calls("<tool_call>drop_all_tables")
        self.assertEqual(calls, [])

    def test_well_formed_output_is_untouched(self):
        content, calls = self._calls("Done.<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>"
                                     "</tool_call>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Done.")


class ToolChoiceTest(unittest.TestCase):
    def test_none_does_not_offer_the_tools(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="none")
        self.assertNotIn("<tools>", prompt)

    def test_auto_offers_the_tools(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="auto")
        self.assertIn("<tools>", prompt)

    def test_required_instructs_the_model_to_call_one(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="required")
        self.assertIn("<tools>", prompt)
        self.assertIn("must call one of the functions", prompt)

    def test_named_function_restricts_to_that_function(self):
        tools = ORDER_TOOL + [{"type": "function", "function": {"name": "other", "parameters": {}}}]
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=tools,
                             tool_choice={"type": "function", "function": {"name": "lookup_order"}})
        self.assertIn("must call the function `lookup_order`", prompt)
        self.assertNotIn('"other"', prompt)

    def test_rejects_unknown_string_and_unknown_function(self):
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tools": ORDER_TOOL, "tool_choice": "maybe"}, 128)
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tools": ORDER_TOOL,
                                "tool_choice": {"type": "function",
                                                "function": {"name": "nope"}}}, 128)

    def test_rejects_tool_choice_without_tools(self):
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tool_choice": "required"}, 128)


class TrailingAssistantTurnTest(unittest.TestCase):
    """A trailing `assistant` message is a turn to CONTINUE, not one already finished.

    The gateway used to fold it into a completed turn and append a fresh generation cue,
    so the model wrote a second assistant turn and the client's opening was dropped. These
    pin what replaced that: the switch that turns continuation on, what the prompt looks
    like when it is on, and what is refused rather than silently reshaped.

    The switch is COLI_CONTINUE_ASSISTANT and not a request field on purpose -- a body
    extension would only be reachable by hand-written JSON, and the clients that want this
    send a message list and nothing else.
    """

    OPEN_TURN = [{"role": "user", "content": "capitale della Francia?"},
               {"role": "assistant", "content": "La capitale e'"}]

    @staticmethod
    def on():
        return patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": "1"})

    @staticmethod
    def off():
        return patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": "0"})

    def test_switch_on_leaves_the_turn_open(self):
        with self.on(), patch("openai_server.ARCH", "glm53"):
            self.assertFalse(resolve_generation_prompt(self.OPEN_TURN, {}))
            prompt = render_chat_glm53(self.OPEN_TURN, enable_thinking=True,
                                       add_generation_prompt=False)
        # ends INSIDE the turn, on the client's opening -- no new cue after it
        self.assertTrue(prompt.endswith("La capitale e'"), prompt[-60:])
        self.assertFalse(prompt.endswith("<|assistant|><think>"))
        self.assertEqual(prompt.count("<|assistant|>"), 1)

    def test_default_is_on(self):
        """Continuation is the default now: unset (the ordinary deployment) continues a
        trailing assistant turn. Only COLI_CONTINUE_ASSISTANT=0 turns it off."""
        with patch.dict(os.environ, {}, clear=False), patch("openai_server.ARCH", "glm53"):
            os.environ.pop("COLI_CONTINUE_ASSISTANT", None)
            self.assertFalse(resolve_generation_prompt(self.OPEN_TURN, {}))

    def test_switch_off_restores_old_behaviour(self):
        """COLI_CONTINUE_ASSISTANT=0 is the off-switch: a deployment that sets it behaves
        exactly as the gateway did before continuation existed -- append a fresh cue."""
        with self.off(), patch("openai_server.ARCH", "glm53"):
            self.assertTrue(resolve_generation_prompt(self.OPEN_TURN, {}))
            self.assertEqual(render_chat_glm53(self.OPEN_TURN, enable_thinking=True),
                             render_chat_glm53(self.OPEN_TURN, enable_thinking=True,
                                               add_generation_prompt=True))
        self.assertTrue(render_chat_glm53(self.OPEN_TURN,
                                          enable_thinking=True).endswith("<|assistant|><think>"))

    def test_no_trailing_assistant_turn_is_untouched_either_way(self):
        messages = [{"role": "user", "content": "a"}]
        for switch in (self.on(), self.off()):
            with switch, patch("openai_server.ARCH", "glm53"):
                self.assertTrue(resolve_generation_prompt(messages, {}))

    def test_rejects_trailing_whitespace(self):
        """The template strips it, so the model would resume from different bytes than the
        ones sent -- the same reason Anthropic's own prefill validator refuses it."""
        messages = [{"role": "user", "content": "a"},
                    {"role": "assistant", "content": "La capitale e' "}]
        with self.on(), patch("openai_server.ARCH", "glm53"):
            with self.assertRaises(APIError):
                resolve_generation_prompt(messages, {})

    def test_rejects_an_empty_continuation(self):
        """An empty one ends the prompt on a closed, empty <think></think>: the
        out-of-distribution position #1327 removed."""
        for empty in ("", "   ", None):
            messages = [{"role": "user", "content": "a"},
                        {"role": "assistant", "content": empty}]
            with self.on(), patch("openai_server.ARCH", "glm53"):
                with self.assertRaises(APIError):
                    resolve_generation_prompt(messages, {})

    def test_rejects_tools_and_tool_calls(self):
        with self.on(), patch("openai_server.ARCH", "glm53"):
            with self.assertRaises(APIError):
                resolve_generation_prompt(self.OPEN_TURN, {"tools": ORDER_TOOL})
            calls = [{"role": "user", "content": "a"},
                     {"role": "assistant", "content": "x", "tool_calls": [
                         {"type": "function", "function": {"name": "f", "arguments": "{}"}}]}]
            with self.assertRaises(APIError):
                resolve_generation_prompt(calls, {})

    def test_unimplemented_family_passes_through(self):
        """Continuation is on by default, so a family whose renderer has no open-turn shape
        yet must render as before -- append the cue -- not reject a request nobody opted into.
        Every shipped family is now in CONTINUATION_FAMILIES (Kimi K3 too, via its C `C`
        record), so the backstop is exercised with a hypothetical future arch: it must pass
        through, not error, the day a new renderer lands before its open-turn shape does."""
        self.assertNotIn("future_family", CONTINUATION_FAMILIES)
        with self.on(), patch("openai_server.ARCH", "future_family"):
            self.assertTrue(resolve_generation_prompt(self.OPEN_TURN, {}))

    def test_continuation_open_turn_deepseek_v4(self):
        """deepseek_v4 has no authoritative vendored jinja template to diff against:
        render_chat_v4 is pinned to the official encoding_dsv4.py, and the community
        reap-150b template on the Hub diverges on the reasoning-block convention (a bare
        </think> for a direct answer vs <think></think>). So this is the expected-string
        pin the maintainer allows for such families -- the open turn is pinned to a literal
        here, with the past-turn-minus-EOS invariant kept as an added check, in both
        thinking modes."""
        EOS = "<｜end▁of▁sentence｜>"
        ASSISTANT = "<｜Assistant｜>"
        # The literal open turn, written out so the test does not lean on another renderer
        # call for its only expected value -- a bug that corrupts render_chat_v4 and the
        # continuation path identically would slip past the comparison below but not this.
        # Identical in both thinking modes: the open turn is the shape of the PAST turn,
        # which carries no generation cue for enable_thinking to steer.
        EXPECTED_OPEN = ("<｜begin▁of▁sentence｜><｜User｜>capitale della Francia?"
                         "<｜Assistant｜></think>La capitale e'")
        for enable_thinking in (True, False):
            cue = ASSISTANT + ("<think>" if enable_thinking else "</think>")
            with patch("openai_server.ARCH", "deepseek_v4"):
                normal = render_chat_v4(self.OPEN_TURN, enable_thinking=enable_thinking)
                cont = render_chat_for_arch(self.OPEN_TURN, enable_thinking=enable_thinking,
                                            add_generation_prompt=False)
            self.assertEqual(cont, EXPECTED_OPEN)
            # and the invariant tying it to the normal render: past turn without its EOS + cue
            self.assertEqual(normal, cont + EOS + cue)
            self.assertTrue(cont.endswith("La capitale e'"), cont[-40:])
            self.assertFalse(cont.endswith(EOS))

    def test_continuation_open_turn_deepseek_v41(self):
        """deepseek_v41, like deepseek_v4, is pinned to encoding.py rather than a diffable
        vendored jinja template, so it gets the same expected-string pin: the open turn is a
        literal here, with the past-turn-minus-EOS invariant kept as an added check, in both
        thinking modes.

        Unlike v4, the v41 open turn is NOT identical across the modes: thinking-on carries the
        <｜System｜> Reasoning Effort line and closes the continued turn's (empty) reasoning as
        <think></think>; thinking-off has neither. The empty <think></think> is the SAFE
        position #1327 is about, not the out-of-distribution one -- it is followed by the
        client's content ('La capitale e''), never left dangling at the end of the prompt,
        which resolve_generation_prompt guarantees by refusing an empty continuation."""
        EOS = "<｜end▁of▁sentence｜>"
        ASSISTANT = "<｜Assistant｜>"
        EFFORT = ("<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the "
                  "more thorough the reasoning)\n\n")
        # The literal open turns, written out so the test does not lean on another renderer
        # call for its only expected value (see the deepseek_v4 test), one per thinking mode.
        EXPECTED_OPEN = {
            True: ("<｜begin▁of▁sentence｜>" + EFFORT + "<｜User｜>capitale della Francia?"
                   "<｜Assistant｜><think></think>La capitale e'"),
            False: ("<｜begin▁of▁sentence｜><｜User｜>capitale della Francia?"
                    "<｜Assistant｜></think>La capitale e'"),
        }
        for enable_thinking in (True, False):
            cue = ASSISTANT + ("<think>" if enable_thinking else "</think>")
            with patch("openai_server.ARCH", "deepseek_v41"):
                normal = render_chat_dsv41(self.OPEN_TURN, enable_thinking=enable_thinking)
                cont = render_chat_for_arch(self.OPEN_TURN, enable_thinking=enable_thinking,
                                            add_generation_prompt=False)
            self.assertEqual(cont, EXPECTED_OPEN[enable_thinking])
            # and the invariant tying it to the normal render: past turn without its EOS + cue
            self.assertEqual(normal, cont + EOS + cue)
            self.assertTrue(cont.endswith("La capitale e'"), cont[-40:])
            self.assertFalse(cont.endswith(EOS))

    def test_continuation_open_turn_inkling(self):
        """inkling uses its own markers, and render_chat_inkling deliberately deviates from
        the template's generation cue (it prefills <|content_text|> in the thinking-off case
        to force content mode, and defaults thinking off), so it is pinned with an
        expected-string test: the open turn is pinned to a literal here, with the
        past-turn-minus-terminators invariant kept as an added check, in both thinking modes."""
        END = "<|end_message|><|content_model_end_sampling|>"
        for enable_thinking in (True, False):
            # eff is 0.9 with thinking on, 0.0 off; the off cue prefills the content channel
            eff = "0.9" if enable_thinking else "0"
            cue = "<|message_model|>" + ("" if enable_thinking else "<|content_text|>")
            # The literal open turn, written out so the test does not lean on another renderer
            # call for its only expected value (see the deepseek_v4 test). The two modes differ
            # only in the system effort line; both end on the prefilled content channel.
            expected = ("<|message_system|><|content_text|>Thinking effort level: " + eff +
                        "<|end_message|><|message_user|><|content_text|>capitale della Francia?"
                        "<|end_message|><|message_model|><|content_text|>La capitale e'")
            with patch("openai_server.ARCH", "inkling"):
                normal = render_chat_inkling(self.OPEN_TURN, enable_thinking=enable_thinking)
                cont = render_chat_for_arch(self.OPEN_TURN, enable_thinking=enable_thinking,
                                            add_generation_prompt=False)
            self.assertEqual(cont, expected)
            # and the invariant tying it to the normal render: past turn without terminators + cue
            self.assertEqual(normal, cont + END + cue)
            self.assertTrue(cont.endswith("La capitale e'"), cont[-40:])
            self.assertFalse(cont.endswith("<|end_message|>"))

    def test_only_the_final_assistant_turn_is_opened(self):
        """Across every continuation family: only the TRAILING assistant turn is opened, and
        each earlier turn renders exactly as it does in a completed conversation. The
        single-turn fixtures elsewhere cannot see this -- their assistant turn is trivially
        last -- but the terminator is dropped by a per-family `index == last` check, written
        out by hand in each renderer; a family that lost that check would open every assistant
        turn, and nothing else in the suite would notice.

        Family-agnostic on purpose, because the open-turn SHAPE is not uniform: qwen36 injects
        an empty <think></think>, GLM carries no per-turn terminator at all, ChatML drops an
        <|im_end|>. What IS uniform is that the completed render (a fresh generation cue
        appended) and the open render share their entire prefix up to the final turn -- so the
        SECOND user turn must survive into their common prefix. If the first assistant turn
        lost its terminator in the open render, that prefix would break right after it, before
        this text. Checked in both thinking modes; the set drives the loop so a newly added
        family is covered the day it joins.

        Kimi K3 is excluded: render_chat_for_arch returns its engine-side K3CHAT1 wire, not a
        string prompt, so this string-level invariant doesn't apply -- its open turn is pinned
        at the token level in tests/test_k3_chat_tools.c against the tiny tokenizer instead."""
        multi = [{"role": "user", "content": "1+1?"},
                 {"role": "assistant", "content": "2"},
                 {"role": "user", "content": "capitale della Francia?"},
                 {"role": "assistant", "content": "La capitale e'"}]
        for arch in sorted(CONTINUATION_FAMILIES):
            if arch == "kimi":
                continue
            for enable_thinking in (True, False):
                where = (arch, enable_thinking)
                with patch("openai_server.ARCH", arch):
                    completed = render_chat_for_arch(multi, enable_thinking=enable_thinking)
                    opened = render_chat_for_arch(multi, enable_thinking=enable_thinking,
                                                  add_generation_prompt=False)
                common = os.path.commonprefix([opened, completed])
                # the prior assistant turn (and its terminator) rendered identically: the turn
                # AFTER it survives into the shared prefix
                self.assertIn("capitale della Francia?", common, where)
                # and only the last turn is open -- the prompt ends on the client's opening
                self.assertTrue(opened.endswith("La capitale e'"), (where, opened[-40:]))

    def test_splitter_starts_in_content_mode_on_a_continued_turn(self):
        """Measured on glm53 int4, CPU: content '' with reasoning_chars 10 and 109,
        clean stop, and a byte-correct open turn on the wire. The model was fine; the
        splitter was primed from enable_thinking alone, so it waited for a </think> the
        prompt had already passed and filed the whole answer as reasoning.

        starts_in_reasoning's own docstring is about exactly this invariant -- a continued
        turn is the third state it did not model. Nothing else in the suite covers it:
        every other thinking test runs against a prompt with the generation cue appended,
        where enable_thinking really does say where the block was left.

        Pinned to glm53 because that is the only family the switch serves. Left on the
        module default (ARCH = "glm") this exercised the continuation path on a family
        resolve_generation_prompt refuses, and passed for the wrong reason.

        The family rule itself -- whether a NEW turn starts inside the block -- belongs to
        #1278 and is pinned by its own test; asserted here it would only duplicate it. What
        this test owns is the axis crossing it: a continued turn opens no block, whatever
        the family rule says."""
        with patch("openai_server.ARCH", "glm53"):
            self.assertTrue(starts_in_reasoning(True))                   # cue: block left open
            self.assertFalse(starts_in_reasoning(True, add_generation_prompt=False))
            self.assertFalse(starts_in_reasoning(False, add_generation_prompt=False))

            # what the engine actually returns after "...The capital of France is": no
            # markers, because the turn's <think></think> is already behind it in the prompt
            emitted = " Paris."
            reasoning, answer = split_thinking_reply(emitted, enable_thinking=True,
                                                     add_generation_prompt=False)
            self.assertEqual(answer, emitted)
            self.assertEqual(reasoning, "")
            # and the bug it replaces, so this test fails if the priming is ever reverted
            reasoning, answer = split_thinking_reply(emitted, enable_thinking=True,
                                                     add_generation_prompt=True)
            self.assertEqual(answer, "")
            self.assertEqual(reasoning, emitted)

    def test_rejects_a_continuation_with_nothing_to_continue_from(self):
        lone = [{"role": "assistant", "content": "La capitale e'"}]
        with self.on(), patch("openai_server.ARCH", "glm53"):
            with self.assertRaises(APIError):
                resolve_generation_prompt(lone, {})


class TrailingAssistantEndToEndTest(unittest.TestCase):
    """End-to-end, through the real HTTP handler and a fake engine (no model weights): a
    request whose last message is an `assistant` turn must reach the engine as a CONTINUATION
    prompt -- ending on the client's own opening, no generation cue appended -- and the text
    the engine generates must come back as the message `content`.

    The TrailingAssistantTurnTest cases pin the prompt string in isolation; none of them prove
    the wiring from an HTTP request through resolve_generation_prompt to the engine and back,
    which a serve() refactor could silently drop. /v1/messages is a translation layer onto the
    same engine path (not a second one), so /v1/chat/completions covers both endpoints. The
    generated text arriving as content (not reasoning_content) also shows the continued turn
    primes the splitter into content mode over the wire -- the third state #1327 fixed."""

    OPEN = {"model": "test-model",
            "messages": [{"role": "user", "content": "capitale della Francia?"},
                         {"role": "assistant", "content": "La capitale e'"}]}

    def _server(self, engine):
        server = APIServer(("127.0.0.1", 0), engine, "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def test_continuation_reaches_the_engine_and_returns_as_content(self):
        engine = FakeEngine()
        with patch("openai_server.ARCH", "glm53"), \
             patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": "1"}):
            server = self._server(engine)
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
            self.addCleanup(conn.close)
            conn.request("POST", "/v1/chat/completions", body=json.dumps(self.OPEN),
                         headers={"Content-Type": "application/json"})
            response = conn.getresponse()
            status, payload = response.status, response.read()
        self.assertEqual(status, 200, payload)
        # the engine saw the continuation prompt: it ends on the client's opening, no cue
        self.assertEqual(len(engine.calls), 1)
        prompt = engine.calls[0][0]
        self.assertTrue(prompt.endswith("La capitale e'"), prompt[-60:])
        self.assertFalse(prompt.endswith("<|assistant|><think>"))
        # and the generated text comes back as content, not misfiled as reasoning
        message = json.loads(payload)["choices"][0]["message"]
        self.assertEqual(message["content"], "Héllo")
        self.assertFalse(message.get("reasoning_content"))


class AllowedHostsTest(unittest.TestCase):
    """#597: the DNS-rebinding guard must accept operator-trusted reverse-proxy
    Host values, while still rejecting everything else by default."""

    def _make_server(self, allowed_hosts=()):
        server = APIServer(("127.0.0.1", 0), FakeEngine(), "test-model",
                           allowed_hosts=allowed_hosts)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def _get_models(self, port, host_header):
        # http.client lets us set an arbitrary Host header (urlopen forces its own).
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            conn.putrequest("GET", "/v1/models", skip_host=True)
            conn.putheader("Host", host_header)
            conn.endheaders()
            return conn.getresponse().status
        finally:
            conn.close()

    def test_allowlist_wiring_normalises_and_filters(self):
        server = self._make_server(allowed_hosts=("  Proxy.Example.TS.net ", "", "   "))
        self.assertEqual(server.allowed_hosts, ("proxy.example.ts.net",))

    def test_trusted_reverse_proxy_host_is_accepted(self):
        server = self._make_server(allowed_hosts=("proxy.example.ts.net",))
        port = server.server_port
        # trusted name (with a port suffix, case-insensitive) passes the guard
        self.assertEqual(self._get_models(port, "Proxy.Example.TS.net:8000"), 200)
        # loopback still works, unaffected by the allowlist
        self.assertEqual(self._get_models(port, "localhost"), 200)

    def test_untrusted_host_is_rejected_by_default(self):
        server = self._make_server()               # no allowlist: loopback/bind only
        self.assertEqual(self._get_models(server.server_port, "evil.example.com"), 403)

    def test_untrusted_host_still_rejected_with_allowlist(self):
        server = self._make_server(allowed_hosts=("proxy.example.ts.net",))
        self.assertEqual(self._get_models(server.server_port, "evil.example.com"), 403)

    def test_wildcard_accepts_any_host(self):
        # #990: a Docker/LAN bind reached by an unpredictable IP. The wildcard is
        # an explicit operator opt-out, so ANY host passes -- but loopback and a
        # real name still work, i.e. it widens rather than replaces.
        server = self._make_server(allowed_hosts=("*",))
        port = server.server_port
        self.assertEqual(self._get_models(port, "10.20.30.40:36873"), 200)
        self.assertEqual(self._get_models(port, "colibri.example.com"), 200)
        self.assertEqual(self._get_models(port, "localhost"), 200)

    def test_rejection_message_names_the_host_and_the_fix(self):
        server = self._make_server()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            conn.putrequest("GET", "/v1/models", skip_host=True)
            conn.putheader("Host", "myserver.lan:36873")
            conn.endheaders()
            resp = conn.getresponse()
            body = resp.read().decode()
        finally:
            conn.close()
        self.assertEqual(resp.status, 403)
        self.assertIn("myserver.lan", body)          # the offending host, so the user can copy it
        self.assertIn("--allowed-host", body)        # and how to fix it


class ThinkingSplitUnitTest(unittest.TestCase):
    """#597 item 4: the GLM reasoning splitter, incl. mkelcb's cross-chunk cases."""

    def test_single_chunk(self):
        self.assertEqual(split_thinking_reply("abc</think>def"), ("abc", "def"))

    def test_close_tag_split_across_chunks(self):
        thinking, answer = [], []
        s = ThinkingStreamSplit(thinking.append, answer.append)
        s.feed("abc</thi"); s.feed("nk>def"); s.finish()
        self.assertEqual(("".join(thinking), "".join(answer)), ("abc", "def"))

    def test_open_tag_split_and_stray_open_marker(self):
        thinking, answer = [], []
        s = ThinkingStreamSplit(thinking.append, answer.append)
        s.feed("abc<thi"); s.feed("nk>def</think>ghi"); s.finish()
        self.assertEqual(("".join(thinking), "".join(answer)), ("abcdef", "ghi"))

    def test_thinking_disabled_is_all_answer(self):
        # initial_thinking=False: a pure answer with no markers must not be filed as reasoning
        self.assertEqual(split_thinking_reply("plain answer", enable_thinking=False),
                         ("", "plain answer"))

    def test_glm53_starts_in_reasoning_even_with_thinking_off(self):
        """#1278: render_chat_glm53 opens <think> unconditionally (the template
        has no switch; "off" only lowers the effort), so the reply always starts
        inside the block. With the splitter started in text mode the reasoning
        streamed as `content`, glued in front of the answer. The family, not the
        client flag, decides where the output starts."""
        import openai_server as srv
        with patch("openai_server.ARCH", "glm53"):
            self.assertTrue(srv.starts_in_reasoning(False))
            self.assertEqual(split_thinking_reply("why</think>answer", enable_thinking=False),
                             ("why", "answer"))
        with patch("openai_server.ARCH", "glm"):
            self.assertFalse(srv.starts_in_reasoning(False),
                             "GLM-5.2 closes the block in the prompt when thinking is off")

    def test_glm53_bare_tool_call_turn_matches_what_the_model_wrote(self):
        """#1576: a replayed assistant turn has to be the tokens the model made.

        The official template writes "\\n<tool_call>" and GLM-5.3 does not: on a
        turn that is nothing but a tool call it writes "</think><tool_call>".
        Rendering the newline anyway put one extra token into the replayed
        prefix, and the reuse gate in glm53.c is all-or-nothing, so the entire
        cached prefix went and the turn re-prefilled from scratch. The reporter
        measured twenty minutes of it on a 3k-token agent history.

        The turn that also carries text keeps its newline, and that is not an
        oversight: there the model's own trailing newline is stripped and put
        back, the tokens line up, and it is the case that works today.
        """
        import openai_server as srv
        bare = srv.render_chat_glm53([
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function",
                             "function": {"name": "bash",
                                          "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "content": "a.txt"},
        ])
        self.assertIn("</think><tool_call>bash", bare,
                      "a bare tool call must follow </think> with no newline")
        self.assertNotIn("</think>\n<tool_call>", bare)

        with_text = srv.render_chat_glm53([
            {"role": "user", "content": "list the files"},
            {"role": "assistant", "content": "Let me look.",
             "tool_calls": [{"type": "function",
                             "function": {"name": "bash",
                                          "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "content": "a.txt"},
        ])
        self.assertIn("Let me look.\n<tool_call>bash", with_text,
                      "a turn with text keeps the separator it already had")

    def test_missing_close_tag_surfaces_reasoning(self):
        self.assertEqual(split_thinking_reply("thought with no end"),
                         ("thought with no end", ""))


class _ChunkEngine(FakeEngine):
    """Engine that emits a caller-supplied chunk sequence, to exercise the streaming
    reasoning splitter across arbitrary chunk boundaries."""
    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        if on_accept is not None:                 # simulate the engine's ACCEPT frame (#597)
            on_accept({"prompt_tokens": 7})
        for chunk in self.chunks:
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        return {"prompt_tokens": 7, "completion_tokens": len(self.chunks), "length_limited": False}


class GlmReasoningStreamTest(unittest.TestCase):
    """#597 item 4 end-to-end: GLM reasoning streams as reasoning_content, the answer as
    content, no <think>/</think> leaks, cross-chunk-safe, and reasoning never contaminates
    the tool-call buffer."""

    def _server(self, chunks):
        server = APIServer(("127.0.0.1", 0), _ChunkEngine(chunks), "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return f"http://127.0.0.1:{server.server_port}"

    def _post(self, base, body):
        req = Request(base + "/v1/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            return response.read().decode()

    def _deltas(self, raw):
        reasoning, content, tool_calls = [], [], []
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:])["choices"]:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
        return "".join(reasoning), "".join(content), tool_calls

    def test_streaming_splits_reasoning_from_answer(self):
        base = self._server(["I think ", "42", "</think>", "The answer ", "is 42"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "2+2?"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "I think 42")
        self.assertEqual(content, "The answer is 42")
        self.assertNotIn("<think>", raw)
        self.assertNotIn("</think>", raw)

    def test_streaming_close_tag_split_across_chunks(self):
        base = self._server(["reason</thi", "nk>ans", "wer"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "x"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "reason")
        self.assertEqual(content, "answer")
        self.assertNotIn("think>", raw)

    def test_streaming_thinking_off_is_all_content(self):
        base = self._server(["Just ", "the answer"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": False,
                                "messages": [{"role": "user", "content": "x"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Just the answer")

    def test_streaming_reasoning_stays_out_of_tool_call(self):
        base = self._server(["deciding to call", "</think>",
                             "<tool_call>get_weather<arg_key>city</arg_key>"
                             "<arg_value>Paris</arg_value></tool_call>"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "weather?"}],
                                "tools": [{"type": "function", "function": {
                                    "name": "get_weather", "parameters": {"type": "object",
                                    "properties": {"city": {"type": "string"}}}}}]})
        reasoning, content, tool_calls = self._deltas(raw)
        self.assertEqual(reasoning, "deciding to call")
        self.assertTrue(tool_calls, "expected a parsed tool call")
        args = tool_calls[0]["function"]["arguments"]
        self.assertIn("Paris", args)
        self.assertNotIn("deciding", args)     # reasoning must not leak into the tool arguments
        self.assertNotIn("deciding", content)  # nor into the visible answer

    def test_nonstreaming_splits_reasoning(self):
        base = self._server(["mulling ", "it over", "</think>", "final ", "answer"])
        req = Request(base + "/v1/chat/completions",
                      data=json.dumps({"model": "test-model", "enable_thinking": True,
                        "messages": [{"role": "user", "content": "x"}]}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        message = body["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "mulling it over")
        self.assertEqual(message["content"], "final answer")


class AcceptFrameTest(unittest.TestCase):
    """#597 item 6: the engine's ACCEPT frame gates the HTTP commit, and its invariants."""

    def _engine(self, respond):
        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            return Engine("glm", "model")

    def test_accept_fires_before_any_data(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + rid + b" 42\n")
            process.stdout.feed(b"DATA " + rid + b" 3\nHi!\n")
            process.stdout.feed(b"DONE " + rid + b" STAT 1 2.5 0 1.0 4 0\n")
        engine = self._engine(respond)
        seen = []
        engine.generate("hi", 8, 0.7, 0.9, lambda t: seen.append(("text", t)),
                        on_accept=lambda info: seen.append(("accept", info)))
        engine.close()
        self.assertEqual(seen[0], ("accept", {"prompt_tokens": 42}))
        self.assertEqual("".join(t for k, t in seen if k == "text"), "Hi!")

    def test_error_before_accept_never_commits(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ERROR " + rid + b" CONTEXT_EXCEEDED 5000 4094\n")
        engine = self._engine(respond)
        accepts = []
        with self.assertRaises(APIError) as caught:
            engine.generate("hi", 8, 0.7, 0.9, lambda _: None,
                            on_accept=lambda info: accepts.append(info))
        engine.close()
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.code, "context_length_exceeded")
        self.assertEqual(accepts, [])          # nothing committed -> HTTP layer can send a clean 400

    def test_data_before_accept_still_commits_for_old_engine(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"DATA " + rid + b" 3\nHey\n")
            process.stdout.feed(b"DONE " + rid + b" STAT 1 2.5 0 1.0 4 0\n")
        engine = self._engine(respond)
        accepts, chunks = [], []
        engine.generate("hi", 8, 0.7, 0.9, chunks.append,
                        on_accept=lambda info: accepts.append(info))
        engine.close()
        self.assertEqual("".join(chunks), "Hey")
        self.assertEqual(len(accepts), 1)      # first DATA implies acceptance (no ACCEPT frame)
        self.assertIsNone(accepts[0]["prompt_tokens"])

    def test_duplicate_accept_is_a_protocol_error(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + rid + b" 10\n")
            process.stdout.feed(b"ACCEPT " + rid + b" 10\n")
        engine = self._engine(respond)
        with self.assertRaisesRegex(RuntimeError, "duplicate ACCEPT"):
            engine.generate("hi", 8, 0.7, 0.9, lambda _: None, on_accept=lambda _: None)
        engine.close()


class _ContextExceededEngine(FakeEngine):
    """Engine that rejects the prompt before ACCEPT — on_accept is never called."""
    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        raise APIError(400, "This model's maximum context length is 4094 tokens.",
                       "messages", "context_length_exceeded")


class StreamingContextRejectTest(unittest.TestCase):
    """#597 item 6: an oversized prompt on a *streaming* request must return a clean HTTP 400,
    not a committed 200 SSE stream that only later discovers the overflow."""

    def setUp(self):
        self.server = APIServer(("127.0.0.1", 0), _ContextExceededEngine(), "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_streaming_context_exceeded_is_clean_400(self):
        req = Request(self.base + "/v1/chat/completions",
                      data=json.dumps({"model": "test-model", "stream": True,
                        "messages": [{"role": "user", "content": "x" * 100}]}).encode(),
                      headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=3)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)          # a real 400, not a 200 stream
        body = json.load(caught.exception)
        self.assertEqual(body["error"]["code"], "context_length_exceeded")
        self.assertEqual(body["error"]["param"], "messages")


class _ExplodingEngine(FakeEngine):
    """ACCEPTs the prompt (committing the streaming 200), then dies mid-generation."""
    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        if on_accept is not None:
            on_accept({"prompt_tokens": 7})
        on_text("partial")
        raise RuntimeError("engine died mid-stream")


class KeepAliveFramingTest(unittest.TestCase):
    """#597 item 3: HTTP/1.1 persistence must not desynchronise.

    The report was `Bad request syntax ('{...json body...}POST /v1/chat/completions HTTP/1.1')`
    -- a previous body being parsed as the next request line. Two independent causes: an early
    rejection returning before the body is read, and a streaming 200 that neither announced
    close-framing nor stopped offering the socket for reuse when generation failed."""

    CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "x"}]}

    def _server(self, engine=None, **kw):
        server = APIServer(("127.0.0.1", 0), engine or FakeEngine(), "test-model", **kw)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def _conn(self, server):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        self.addCleanup(conn.close)
        return conn

    def _post(self, conn, body=None, headers=None, path="/v1/chat/completions"):
        payload = json.dumps(self.CHAT if body is None else body)
        head = {"Content-Type": "application/json"}
        head.update(headers or {})
        conn.request("POST", path, body=payload, headers=head)
        response = conn.getresponse()
        return response.status, response.read()

    def _raw(self, server, request_bytes, read=4096):
        """Byte-level exchange, for assertions about framing that a client library hides."""
        sock = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        self.addCleanup(sock.close)
        sock.sendall(request_bytes)
        chunks = []
        try:
            while True:
                chunk = sock.recv(read)
                if not chunk:
                    break                      # server closed: the SSE message boundary
                chunks.append(chunk)
        except socket.timeout:
            chunks.append(b"<STILL-OPEN>")     # no EOF: the connection was left reusable
        return b"".join(chunks).decode("utf-8", "replace")

    def _request_bytes(self, body, host="127.0.0.1"):
        payload = json.dumps(body)
        return (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                f"\r\n{payload}").encode()

    # --- an early rejection must not leave its body in the socket -------------------------

    def test_rejected_host_does_not_desync_the_next_request(self):
        server = self._server()
        conn = self._conn(server)
        status, _ = self._post(conn, headers={"Host": "evil.example.com"})
        self.assertEqual(status, 403)
        status, payload = self._post(conn)                  # same connection, valid request
        self.assertEqual(status, 200, "the 403's unread body desynchronised the connection")
        self.assertEqual(json.loads(payload)["object"], "chat.completion")

    def test_rejected_auth_does_not_desync_the_next_request(self):
        server = self._server(api_key="secret")
        conn = self._conn(server)
        status, _ = self._post(conn)                        # no Authorization header
        self.assertEqual(status, 401)
        status, payload = self._post(conn, headers={"Authorization": "Bearer secret"})
        self.assertEqual(status, 200, "the 401's unread body desynchronised the connection")
        self.assertEqual(json.loads(payload)["object"], "chat.completion")

    def test_unknown_model_does_not_desync_the_next_request(self):
        server = self._server()
        conn = self._conn(server)
        status, _ = self._post(conn, body=dict(self.CHAT, model="nope"))
        self.assertEqual(status, 404)
        status, _ = self._post(conn)
        self.assertEqual(status, 200)

    def test_each_body_is_consumed_exactly_once_across_reused_requests(self):
        """The engine sees one prompt per request, with no body bytes bleeding between them."""
        engine = FakeEngine()
        server = self._server(engine)
        conn = self._conn(server)
        for index in range(4):
            body = {"model": "test-model",
                    "messages": [{"role": "user", "content": f"question-{index}"}]}
            status, _ = self._post(conn, body=body)
            self.assertEqual(status, 200)
        self.assertEqual(len(engine.calls), 4)
        for index, call in enumerate(engine.calls):
            self.assertIn(f"question-{index}", call[0])
            self.assertNotIn("question-", call[0].split(f"question-{index}")[1],
                             "a later body leaked into an earlier prompt")

    def test_interleaved_rejections_and_successes_stay_in_sync(self):
        server = self._server(api_key="secret")
        conn = self._conn(server)
        good = {"Authorization": "Bearer secret"}
        for _ in range(3):
            self.assertEqual(self._post(conn)[0], 401)
            self.assertEqual(self._post(conn, headers={"Host": "evil.example.com", **good})[0], 403)
            self.assertEqual(self._post(conn, headers=good)[0], 200)

    # --- bodies we refuse to swallow must close rather than desynchronise -----------------

    def test_oversized_content_length_closes_instead_of_desyncing(self):
        server = self._server()
        raw = self._raw(server, (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 b"Content-Type: application/json\r\n"
                                 b"Content-Length: 99999999\r\n\r\n{}"))
        self.assertIn(" 400 ", raw.splitlines()[0])
        self.assertNotIn("<STILL-OPEN>", raw,
                         "an over-limit body must close the connection, not keep it alive")

    def test_unparseable_content_length_closes_instead_of_desyncing(self):
        server = self._server()
        raw = self._raw(server, (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 b"Content-Type: application/json\r\n"
                                 b"Content-Length: abc\r\n\r\n{}"))
        self.assertIn("400", raw.splitlines()[0])
        self.assertNotIn("<STILL-OPEN>", raw)

    # --- a streaming 200 is close-framed, and says so ------------------------------------

    def test_streaming_response_announces_close_framing(self):
        server = self._server()
        raw = self._raw(server, self._request_bytes(dict(self.CHAT, stream=True)))
        headers = raw.split("\r\n\r\n", 1)[0].lower()
        self.assertIn("content-type: text/event-stream", headers)
        self.assertIn("connection: close", headers,
                      "SSE has no Content-Length, so the close IS the boundary and must be declared")
        self.assertIn("data: [DONE]", raw)
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_anthropic_stream_announces_close_framing(self):
        server = self._server()
        payload = json.dumps({"model": "test-model", "stream": True, "max_tokens": 16,
                              "messages": [{"role": "user", "content": "x"}]})
        raw = self._raw(server, (f"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 f"Content-Type: application/json\r\n"
                                 f"Content-Length: {len(payload)}\r\n\r\n{payload}").encode())
        self.assertIn("connection: close", raw.split("\r\n\r\n", 1)[0].lower())
        self.assertIn("event: message_stop", raw)
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_stream_exit_stops_keepalive_before_releasing_slot(self):
        class CancelledEngine(_ExplodingEngine):
            def generate(self, *args, **kwargs):
                try:
                    return super().generate(*args, **kwargs)
                except RuntimeError:
                    raise ClientCancelled()

        original_thread = threading.Thread
        for path in ("/v1/chat/completions", "/v1/messages"):
            for engine_type, outcome in ((_ExplodingEngine, "failed"),
                                         (CancelledEngine, "cancelled")):
                with self.subTest(path=path, outcome=outcome):
                    pumps = []
                    def thread_factory(*args, **kwargs):
                        thread = original_thread(*args, **kwargs)
                        target = kwargs.get("target")
                        if getattr(target, "__name__", "") in ("_keepalive", "keepalive"):
                            stop = next(cell.cell_contents for cell in target.__closure__
                                        if isinstance(cell.cell_contents, threading.Event))
                            pumps.append((thread, stop))
                        return thread
                    server = self._server(engine_type())
                    try:
                        with patch.object(threading, "Thread", side_effect=thread_factory):
                            status, _ = self._post(self._conn(server),
                                dict(self.CHAT, stream=True, max_tokens=16), path=path)
                        self.assertEqual(status, 200)
                        self.assertEqual(len(pumps), 1)
                        self.assertFalse(pumps[0][0].is_alive(), "keepalive survived stream exit")
                        stats = server.scheduler.snapshot()
                        self.assertEqual(stats[outcome], 1)
                        self.assertEqual((stats["active"], stats["completed"]), (0, 0))
                    finally:
                        for thread, stop in pumps:
                            stop.set()
                            thread.join(2)

    def test_engine_failure_after_commit_does_not_splice_a_second_response(self):
        """Once the 200 is out, a 500 status line would land inside the event stream."""
        server = self._server(_ExplodingEngine())
        raw = self._raw(server, self._request_bytes(dict(self.CHAT, stream=True)))
        self.assertEqual(raw.count("HTTP/1."), 1,
                         "a second HTTP response was spliced into the committed SSE stream")
        self.assertIn("partial", raw)            # the events sent before the failure survive
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_non_streaming_response_still_reuses_the_connection(self):
        """The fix must not turn every response into a close: plain JSON stays persistent."""
        server = self._server()
        conn = self._conn(server)
        self.assertEqual(self._post(conn)[0], 200)
        self.assertEqual(self._post(conn)[0], 200)
        self.assertIsNotNone(conn.sock, "the JSON path should keep the connection open")


class ConversationCacheSlotTest(unittest.TestCase):
    """#634 Defect 1: a conversation must map to one stable KV slot across its turns."""

    def _conv(self, *user_and_assistant_turns, system="you are a helpful assistant"):
        messages = [{"role": "system", "content": system}]
        for i, text in enumerate(user_and_assistant_turns):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
        return messages

    def test_single_slot_is_always_zero(self):
        self.assertEqual(conversation_cache_slot(self._conv("hi"), 1), 0)

    def test_empty_or_bad_input_is_zero(self):
        self.assertEqual(conversation_cache_slot(None, 8), 0)
        self.assertEqual(conversation_cache_slot([], 8), 0)

    def test_slot_is_in_range(self):
        for kv in (2, 3, 8, 16):
            slot = conversation_cache_slot(self._conv("solve x"), kv)
            self.assertTrue(0 <= slot < kv, f"slot {slot} out of range for kv={kv}")

    def test_stable_across_turns_of_one_conversation(self):
        # The engine caches the prefix; every turn of the same conversation must return
        # the same slot so it lands on its warm KV instead of re-prefilling.
        first_turn = self._conv("1+1=")
        second_turn = self._conv("1+1=", "2", "and 2+2?")
        third_turn = self._conv("1+1=", "2", "and 2+2?", "4", "thanks")
        base = conversation_cache_slot(first_turn, 8)
        self.assertEqual(conversation_cache_slot(second_turn, 8), base)
        self.assertEqual(conversation_cache_slot(third_turn, 8), base)

    def test_distinct_conversations_can_differ(self):
        # Not a guarantee for any single pair (hashing collides sometimes), but across a
        # spread of openings we must see more than one slot used, i.e. not everything on 0.
        slots = {conversation_cache_slot(self._conv(f"task number {i}"), 8) for i in range(40)}
        self.assertGreater(len(slots), 1)

    def test_deterministic(self):
        conv = self._conv("same question", "same answer", "again")
        self.assertEqual(conversation_cache_slot(conv, 8), conversation_cache_slot(conv, 8))


class ConnectionLimitTest(unittest.TestCase):
    """Bounds on the accept loop, which nothing bounded before.

    ThreadingHTTPServer spawns a thread per connection with no ceiling, and
    `timeout` is per socket operation, so it restarts on every byte: a client
    dripping one byte kept a thread and a slot forever. Threads at 8 MiB of
    stack each made that a memory-exhaustion DoS, reachable before any Host
    check or auth.
    """

    def setUp(self):
        self.engine = FakeEngine()
        APIServer.MAX_CONNECTIONS = 6
        APIServer.MAX_CONNECTIONS_PER_IP = 3
        APIHandler.READ_DEADLINE = 2
        self.addCleanup(setattr, APIServer, "MAX_CONNECTIONS", 64)
        self.addCleanup(setattr, APIServer, "MAX_CONNECTIONS_PER_IP", 8)
        self.addCleanup(setattr, APIHandler, "READ_DEADLINE", 30)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "m", None, 16, kv_slots=1)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)
        self.port = self.server.server_port
        self.held = []
        self.addCleanup(self._drop_all)

    def _drop_all(self):
        for sock in self.held:
            try:
                sock.close()
            except OSError:
                pass

    def _dribble(self, count):
        """Open connections that send a partial request line and never finish."""
        for _ in range(count):
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), 2)
                sock.settimeout(2)
                sock.sendall(b"GET /health HTTP/1.1\r\n")
                self.held.append(sock)
            except OSError:
                pass
        time.sleep(0.4)

    def test_slowloris_cannot_grow_threads_without_bound(self):
        self._dribble(40)
        self.assertLessEqual(self.server._conn_live, APIServer.MAX_CONNECTIONS)

    def test_one_address_cannot_take_every_slot(self):
        """A global cap alone only turns exhaustion into starvation."""
        self._dribble(40)
        self.assertLessEqual(self.server._conn_live,
                             APIServer.MAX_CONNECTIONS_PER_IP)
        self.assertLess(self.server._conn_live, APIServer.MAX_CONNECTIONS,
                        "one source filled the server cap")

    def test_dripping_connections_are_reclaimed(self):
        """The cumulative deadline, which the per-operation timeout is not."""
        self._dribble(10)
        self.assertGreater(self.server._conn_live, 0)
        time.sleep(APIHandler.READ_DEADLINE + 1.5)
        self.assertEqual(self.server._conn_live, 0,
                         "dripping connections were never reclaimed")


class ReasoningEffortTest(unittest.TestCase):
    """render_chat mapped every level except "high" onto Max (#809).

    The endpoint accepts none/minimal/low/medium/high/xhigh. Four of the five
    that enable thinking rendered Max, so a client asking for `minimal` got
    more reasoning than one asking for `high` -- and on a single machine
    unrequested reasoning spends the token budget before the answer starts.
    """

    MESSAGES = [{"role": "user", "content": "hi"}]

    def effort(self, level):
        import re
        text = render_chat(self.MESSAGES, enable_thinking=True,
                           reasoning_effort=level)
        found = re.search(r"Reasoning Effort: (\w+)", text)
        return found.group(1) if found else None

    def test_levels_are_distinct_and_ordered(self):
        rendered = [self.effort(l) for l in
                    ("minimal", "low", "medium", "high", "xhigh")]
        rank = {"Low": 0, "Medium": 1, "High": 2, "Max": 3}
        scores = [rank[r] for r in rendered]
        self.assertEqual(scores, sorted(scores), rendered)
        self.assertLess(scores[0], scores[-1],
                        "minimal and xhigh render the same effort")

    def test_minimal_is_not_max(self):
        """The reported symptom, pinned on its own."""
        self.assertNotEqual(self.effort("minimal"), "Max")
        self.assertNotEqual(self.effort("low"), "Max")
        self.assertNotEqual(self.effort("medium"), "Max")

    def test_thinking_off_emits_no_effort_line(self):
        text = render_chat(self.MESSAGES, enable_thinking=False,
                           reasoning_effort="xhigh")
        self.assertNotIn("Reasoning Effort", text)


class ImageUrlPathGuard(unittest.TestCase):
    """image_url.url naming a local file is read with the server's rights, so
    it is denied unless the operator sets COLI_IMAGE_ROOT, and then only
    inside it (GHSA follow-up to #1354, whose guards left every absolute path
    readable by default). data: URIs are the client's way in. Errors stay
    generic so a reply never confirms a path or its permissions."""

    def setUp(self):
        self._saved = os.environ.pop("COLI_IMAGE_ROOT", None)

    def tearDown(self):
        os.environ.pop("COLI_IMAGE_ROOT", None)
        if self._saved is not None:
            os.environ["COLI_IMAGE_ROOT"] = self._saved

    def test_data_uri_is_the_default_way_in(self):
        import base64
        payload = base64.b64encode(b"\x89PNG\r\n").decode()
        self.assertEqual(_image_bytes_from_url("data:image/png;base64," + payload), b"\x89PNG\r\n")

    def test_local_paths_are_denied_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            img = Path(root) / "pic.png"
            img.write_bytes(b"\x89PNG\r\n")
            for url in (str(img), "file://" + str(img), "/etc/passwd", "file:///etc/passwd", "relative.png"):
                with self.assertRaises(APIError) as caught:
                    _image_bytes_from_url(url)
                self.assertEqual(caught.exception.status, 400)
                self.assertIn("COLI_IMAGE_ROOT", str(caught.exception))
                self.assertNotIn("passwd", str(caught.exception))

    def test_image_root_allows_inside_and_refuses_outside(self):
        with tempfile.TemporaryDirectory() as root, \
                tempfile.NamedTemporaryFile(suffix=".png") as outside:
            os.environ["COLI_IMAGE_ROOT"] = root
            inside = Path(root) / "ok.png"
            inside.write_bytes(b"ok")
            self.assertEqual(_image_bytes_from_url(str(inside)), b"ok")
            self.assertEqual(_image_bytes_from_url("file://" + str(inside)), b"ok")
            for url in (outside.name, "/etc/passwd", str(Path(root) / ".." / Path(outside.name).name)):
                with self.assertRaises(APIError) as caught:
                    _image_bytes_from_url(url)
                self.assertNotIn(Path(outside.name).name, str(caught.exception))

    def test_a_symlink_escaping_the_root_is_refused(self):
        if not hasattr(os, "symlink"):
            self.skipTest("no symlinks here")
        with tempfile.TemporaryDirectory() as root, \
                tempfile.NamedTemporaryFile(suffix=".png") as outside:
            os.environ["COLI_IMAGE_ROOT"] = root
            link = Path(root) / "link.png"
            try:
                os.symlink(outside.name, link)
            except OSError:
                self.skipTest("cannot create symlinks here")
            with self.assertRaises(APIError):
                _image_bytes_from_url(str(link))

    def test_a_missing_or_file_root_denies_everything(self):
        with tempfile.TemporaryDirectory() as root:
            img = Path(root) / "pic.png"
            img.write_bytes(b"x")
            os.environ["COLI_IMAGE_ROOT"] = str(Path(root) / "nowhere")
            with self.assertRaises(APIError):
                _image_bytes_from_url(str(img))
            os.environ["COLI_IMAGE_ROOT"] = str(img)
            with self.assertRaises(APIError):
                _image_bytes_from_url(str(img))

    def test_error_does_not_leak_the_path(self):
        os.environ["COLI_IMAGE_ROOT"] = tempfile.gettempdir()
        with self.assertRaises(APIError) as caught:
            _image_bytes_from_url("/no/such/secret-name.png")
        self.assertNotIn("secret-name", str(caught.exception))


class ContextExceededMessageTest(unittest.TestCase):
    """#1376: the engine writes `CONTEXT_EXCEEDED prompt_tokens=N requested=M
    capacity=C`. The message took fields[2] ("requested=M", the completion
    budget) as the limit and printed the raw key=value token, so a user with a
    9000-token prompt on an 8192 ceiling read "maximum context length is
    requested=4 tokens". The number that mattered, capacity, appeared nowhere."""

    def test_limit_is_capacity_and_used_is_prompt_tokens(self):
        from openai_server import _engine_error
        err = _engine_error(["CONTEXT_EXCEEDED", "prompt_tokens=9000", "requested=4",
                             "capacity=8192"], "ignored")
        text = str(err)
        self.assertIn("8192", text)
        self.assertIn("9000", text)
        self.assertNotIn("requested=", text)
        self.assertNotIn("prompt_tokens=", text)
        self.assertNotIn("capacity=", text)

    def test_the_positional_spelling_of_colibri_and_deepseek_still_reads(self):
        from openai_server import _engine_error
        text = str(_engine_error(["CONTEXT_EXCEEDED", "8321", "4094"], "ignored"))
        self.assertIn("4094", text)
        self.assertIn("8321", text)

    def test_missing_fields_do_not_crash_the_message(self):
        from openai_server import _engine_error
        text = str(_engine_error(["CONTEXT_EXCEEDED"], "ignored"))
        self.assertIn("the context", text)


class _ShortWritingStream:
    """A fake stdin that hands back at most `chunk` bytes per write() call,
    forcing `_write_all` to loop -- the production pipe does this on a
    signal landing mid-write or a full pipe buffer on a large IMAGE frame."""

    def __init__(self, chunk=3):
        self.chunk = chunk
        self.received = bytearray()

    def write(self, data):
        piece = bytes(data)[:self.chunk]
        self.received.extend(piece)
        return len(piece)


class _ScriptedStream:
    """A fake stdin whose write() answers each entry in `script` in turn --
    an int number of bytes actually taken, or `None` for "took nothing" --
    then takes everything it is offered once the script runs out."""

    def __init__(self, script):
        self.script = list(script)
        self.received = bytearray()
        self.calls = []

    def write(self, data):
        data = bytes(data)
        self.calls.append(data)
        answer = self.script.pop(0) if self.script else len(data)
        if answer is None:
            return None
        taken = data[:answer]
        self.received.extend(taken)
        return len(taken)


class _CountingLock:
    """A `threading.Lock`-alike that counts `with` acquisitions -- used to
    pin that IMAGE and SUBMIT share exactly one `write_lock` acquisition."""

    def __init__(self):
        self._lock = threading.Lock()
        self.acquisitions = 0

    def __enter__(self):
        self.acquisitions += 1
        return self._lock.__enter__()

    def __exit__(self, *exc_info):
        return self._lock.__exit__(*exc_info)


class WriteAllTest(unittest.TestCase):
    """`_write_all` (extracted from the inline server->engine stdin writes):
    loop on a short write until the whole frame is sent, and fail closed --
    never spin -- on the two shapes that are not progress.

    `_write_all` is imported locally in each test method here, not at module
    scope: it does not exist on base, and a module-level import of it would
    make this whole test file fail to collect when overlaid on base product
    code (the procedure the base-pin tests below rely on)."""

    def test_short_writes_reassemble_to_the_full_frame(self):
        from openai_server import _write_all
        stream = _ShortWritingStream(chunk=3)
        frame = b"SUBMIT 1 0 5 3 0.25 0.9\nHello\n"
        _write_all(stream, frame, "SUBMIT")
        self.assertEqual(bytes(stream.received), frame)

    def test_a_short_first_write_still_receives_the_correct_remainder(self):
        from openai_server import _write_all
        # The first call must not be special-cased to the unsliced buffer:
        # every call, including the first, offers a memoryview starting at
        # the bytes not yet sent.
        stream = _ScriptedStream([2])
        _write_all(stream, b"CANCEL 42\n", "CANCEL")
        self.assertEqual(bytes(stream.received), b"CANCEL 42\n")
        self.assertEqual(len(stream.calls), 2)   # the short first write forced a second

    def test_none_return_fails_closed_instead_of_spinning(self):
        from openai_server import _write_all
        stream = _ScriptedStream([None])
        with self.assertRaisesRegex(RuntimeError, "failed to write SUBMIT to the engine"):
            _write_all(stream, b"SUBMIT 1 0 1 1 1 1\nx\n", "SUBMIT")

    def test_zero_return_fails_closed_instead_of_spinning(self):
        from openai_server import _write_all
        stream = _ScriptedStream([0])
        with self.assertRaisesRegex(RuntimeError, "failed to write STOP to the engine"):
            _write_all(stream, b"STOP 1\n", "STOP")


class WriteFailureHTTPTest(unittest.TestCase):
    """A checked engine-stdin write that fails must reach the client as the
    named 500 engine_error while the response is still uncommitted, and must
    never splice anything into a stream that has already committed -- it
    just ends (#597 item 3's `_fail`, exercised here by a genuine broken
    pipe on the engine's stdin, through the real `Engine`, rather than by a
    fake that raises `RuntimeError` directly -- a real `BrokenPipeError` is
    a `ConnectionError`, and it is exactly that unwrapped subclass that a
    correct checked write must keep away from do_POST's client-hangup
    handler)."""

    def _serve(self, process):
        with patch("openai_server.ARCH", "glm"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        server = APIServer(("127.0.0.1", 0), engine, "test-model", None, 16, kv_slots=1)
        # A short poll_interval means server.shutdown() (addCleanup, below)
        # returns almost immediately instead of paying up to the default
        # 0.5s poll -- these tests do no waiting of their own, so that 0.5s
        # would be pure teardown overhead, not a wait under test.
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        self.addCleanup(engine.close)
        return server

    def _raw_post(self, server, path, body):
        """POST over a plain socket and read until the peer closes, so the
        literal status line(s) on the wire are visible -- `urlopen` strips
        the status line into `response.status` before handing back `.read()`,
        so asserting on `.read()` alone cannot tell "one status line" from
        "a second one spliced into the body"."""
        payload = json.dumps(body).encode()
        request = (f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                  f"Connection: close\r\n\r\n").encode() + payload
        sock = socket.create_connection(("127.0.0.1", server.server_port), 2)
        sock.sendall(request)
        sock.settimeout(2)
        chunks = []
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except socket.timeout:
            pass
        sock.close()
        return b"".join(chunks)

    def test_dead_engine_submit_is_a_named_500_engine_error_not_silence(self):
        def respond(process, frame):
            if frame.split()[0] == b"SUBMIT":
                raise BrokenPipeError("broken pipe")

        server = self._serve(FakeProcess(respond))
        body = json.dumps({"model": "test-model", "prompt": "hi"}).encode()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/completions",
                          data=body, headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 500)
        payload = json.load(caught.exception)
        self.assertEqual(payload["error"]["code"], "engine_error")

    def test_dead_engine_submit_is_a_named_500_on_the_chat_streaming_path(self):
        # Same failure, but through /v1/chat/completions with stream: true --
        # the tool-sideband/ThinkingStreamSplit wrapping chat streaming builds
        # around engine.generate() must not swallow or reshape a RuntimeError
        # that fires before anything is committed (nothing here ever reaches
        # that wrapping: the SUBMIT write fails before the first ACCEPT/DATA).
        def respond(process, frame):
            if frame.split()[0] == b"SUBMIT":
                raise BrokenPipeError("broken pipe")

        server = self._serve(FakeProcess(respond))
        body = json.dumps({"model": "test-model", "stream": True,
                           "messages": [{"role": "user", "content": "hi"}]}).encode()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                          data=body, headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 500)
        payload = json.load(caught.exception)
        self.assertEqual(payload["error"]["code"], "engine_error")

    def test_uncommitted_stop_write_failure_is_a_named_500_engine_error(self):
        # The dead-SUBMIT tests above cover the write _write_frame never gets
        # a chance to make (the connection is refused before the first
        # request even lands). This is the case _write_frame's own docstring
        # is actually about: a CANCEL/STOP write that fails while the
        # response is still uncommitted -- non-streaming, so nothing commits
        # until the whole answer is assembled, and the STOP write happens
        # well before that.
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 3\nhi \n")
                process.stdout.feed(b"DATA " + request_id + b" 4\nSTOP\n")
            elif fields[0] == b"STOP":
                raise BrokenPipeError("broken pipe")

        server = self._serve(FakeProcess(respond))
        body = json.dumps({"model": "test-model", "prompt": "hi",
                           "stop": "STOP"}).encode()
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/completions",
                          data=body, headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 500)
        payload = json.load(caught.exception)
        self.assertEqual(payload["error"]["code"], "engine_error")

    def test_write_failure_reaching_the_committed_stream_ends_it_cleanly(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                # "hi " clears StopFilter's hold and reaches the client; the
                # stop sequence itself never does -- feeding them as two
                # frames pins the STOP write to the SECOND one, after the
                # stream has already committed on the first.
                process.stdout.feed(b"DATA " + request_id + b" 3\nhi \n")
                process.stdout.feed(b"DATA " + request_id + b" 4\nSTOP\n")
            elif fields[0] == b"STOP":
                raise BrokenPipeError("broken pipe")

        server = self._serve(FakeProcess(respond))
        raw = self._raw_post(server, "/v1/completions",
                             {"model": "test-model", "prompt": "hi", "stream": True,
                              "stop": "STOP"})
        # Exactly one status line on the whole wire -- the original 200. A
        # response that spliced a second status line (or any framed error)
        # into the already-committed SSE body would show up here as 2.
        self.assertEqual(raw.count(b"HTTP/1.1"), 1)
        self.assertIn(b"200", raw.split(b"\r\n", 1)[0])
        # The committed token reached the client...
        self.assertIn(b'"hi "', raw)
        # ...and nothing else did: no error object, no terminal [DONE]
        # (generate() never returned normally).
        self.assertNotIn(b"engine_error", raw)
        self.assertNotIn(b"data: [DONE]", raw)


class PendingMapCleanupTest(unittest.TestCase):
    """The pending-map entry is dropped on every failed engine write --
    SUBMIT, IMAGE, CANCEL or STOP -- including one forced by a checked
    CANCEL/STOP write that fails closed with no `OSError` (a `None`/zero
    return). A write that never reached the engine gets no DONE/ERROR back,
    so nothing else would ever clear the slot; without this it sits behind
    for `close()`/`_fail_pending` to find stale.

    This is narrower than "every exit": a raise from inside a decode
    callback (`on_text`/`on_accept`/`on_tool`/`on_echo`), or the duplicate-
    ACCEPT guard, still leaves the entry behind, as on `dev`; that is not
    changed here."""

    def test_a_failed_cancel_write_does_not_leave_the_pending_entry_behind(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"CANCEL":
                raise BrokenPipeError("broken pipe")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "failed to write CANCEL to the engine"):
            engine.generate("hello", 8, 0.7, 0.9, lambda _: None, cancelled=lambda: True)
        with engine.pending_lock:
            self.assertEqual(engine.pending, {})
        engine.close()

    def test_a_failed_stop_write_does_not_leave_the_pending_entry_behind(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"STOP":
                raise BrokenPipeError("broken pipe")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        with self.assertRaisesRegex(RuntimeError, "failed to write STOP to the engine"):
            engine.generate("hello", 8, 0.7, 0.9, output.append,
                            stopped=lambda: output == ["x"])
        with engine.pending_lock:
            self.assertEqual(engine.pending, {})
        engine.close()

    def test_a_cancel_write_that_takes_no_bytes_still_drops_the_pending_entry(self):
        # Not an OSError: the pipe accepts the call and answers 0, the same
        # "took nothing" shape _write_all treats as a fail-closed write. If
        # _write_frame's cleanup only ran for OSError, this entry would
        # survive -- the pop has to be keyed on failure of the write, not on
        # which failure shape it took.
        class ZeroOnCancelProcess(FakeProcess):
            def write(self, data):
                text = bytes(data)
                fields = text.split()
                if fields[:1] == [b"CANCEL"]:
                    self.writes.append(text)
                    # Base ignores write()'s return value entirely, so a bare
                    # `return 0` here would leave base's generate() waiting
                    # forever for an ERROR/DONE it never actually asked for
                    # (the CANCEL it believes it sent never reached the
                    # engine) -- a hang, not a failure, on the base-overlay
                    # procedure. Feeding the terminal frame here means base
                    # fails in seconds instead; at head the RuntimeError from
                    # the zero-byte write fires first, so this frame arrives
                    # after the pending entry is already gone and is a no-op.
                    self.stdout.feed(b"ERROR " + fields[1] + b" CANCELLED\n")
                    return 0
                return super().write(data)

        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")

        process = ZeroOnCancelProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "failed to write CANCEL to the engine"):
            engine.generate("hello", 8, 0.7, 0.9, lambda _: None, cancelled=lambda: True)
        with engine.pending_lock:
            self.assertEqual(engine.pending, {})
        engine.close()

    def test_a_failed_image_write_names_the_image_frame_and_drops_the_pending_entry(self):
        # generate()'s outer handler wraps both the IMAGE and the SUBMIT
        # write with one except OSError -- an OSError on the IMAGE write
        # specifically must still name "IMAGE", not fall through to the
        # handler's own default "SUBMIT" label, or one frame kind would
        # report under two different names depending on failure shape
        # (_write_all's own None/zero path already says "IMAGE"; an OSError
        # is the likelier real failure and must match).
        class BrokenPipeOnImageProcess(FakeProcess):
            def write(self, data):
                if bytes(data).split()[:1] == [b"IMAGE"]:
                    raise BrokenPipeError("broken pipe")
                return super().write(data)

        process = BrokenPipeOnImageProcess(lambda _process, _frame: None)
        with patch("openai_server.ARCH", "glm53"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm53", "model")

        class FakePatches:
            def tobytes(self):
                return bytes(range(8))

        with self.assertRaisesRegex(RuntimeError, "failed to write IMAGE to the engine"):
            engine.generate("hello", 8, 0.7, 0.9, lambda _: None, image=(FakePatches(), 2, 2))
        with engine.pending_lock:
            self.assertEqual(engine.pending, {})
        engine.close()


class PlainRequestFrameOrderTest(unittest.TestCase):
    """A request using none of the checked-write machinery's new surface
    (no image, no grammar, no logprobs, no pin) must still put byte-identical
    SUBMIT/STOP and SUBMIT/CANCEL frames on the wire, in the same order, as
    the base module -- literals captured by running the base module's own
    Engine.generate() against this same FakeProcess harness."""

    def test_stop_flow_matches_base(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"STOP":
                process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "glm"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        engine.generate("hello", 8, 0.7, 0.9, output.append,
                        stopped=lambda: output == ["x"])
        engine.close()
        self.assertEqual(process.writes, [b"SUBMIT 1 0 5 8 0.7 0.9\nhello\n", b"STOP 1\n"])

    def test_cancel_flow_matches_base(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"CANCEL":
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "glm"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        disconnected = False

        def sink(text):
            nonlocal disconnected
            disconnected = True

        with self.assertRaises(ClientCancelled):
            engine.generate("hello", 8, 0.7, 0.9, sink, cancelled=lambda: disconnected)
        engine.close()
        self.assertEqual(process.writes, [b"SUBMIT 1 0 5 8 0.7 0.9\nhello\n", b"CANCEL 1\n"])

    def test_image_and_submit_frames_match_base_under_one_lock_acquisition(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
                process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.ARCH", "glm53"), \
             patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm53", "model")
        counting_lock = _CountingLock()
        engine.write_lock = counting_lock

        blob = bytes(range(12))

        class FakePatches:
            def tobytes(self):
                return blob

        output = []
        engine.generate("hello", 8, 0.7, 0.9, output.append, image=(FakePatches(), 2, 2))
        engine.close()
        self.assertEqual(process.writes, [
            b"IMAGE 1 12 2 2\n" + blob + b"\n",
            b"SUBMIT 1 0 5 8 0.7 0.9\nhello\n",
        ])
        # IMAGE must reach the wire before SUBMIT, and both under the SAME
        # lock acquisition -- another request's IMAGE could otherwise land
        # between this one's IMAGE and its SUBMIT.
        self.assertEqual(counting_lock.acquisitions, 1)


if __name__ == "__main__":
    unittest.main()
