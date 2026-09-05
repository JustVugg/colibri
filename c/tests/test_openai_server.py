import http.client
import io
import json
import math
import os
import queue
import socket
import subprocess
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
                           DEFAULT_CHAT_STOP_SEQUENCES, END, GenerationScheduler,
                           LOGPROBS_TOP_K_CAP, READY, Engine, InklingStreamSplit, StopFilter,
                           ThinkingStreamSplit,
                           _engine_error, cap_for_arch, conversation_cache_slot, model_arch,
                           generation_options, logprobs_options, parse_tool_calls,
                           parse_dsv4_tool_calls,
                           parse_arch_tool_calls, parse_k3_tool_calls, parse_qwen38_tool_calls,
                           read_engine_turn, render_chat, render_chat_kimi, render_chat_olmoe,
                           render_chat_qwen38, render_chat_v4, _dsv4_tool_calls, serve,
                           split_thinking_reply,
                           stop_policy, tune_child_env,
                           _chat_logprobs_content, _completions_logprobs_object,
                           _json_float, _order_echo_records, _own_token_label)


class FakeEngine:
    # The U7b capability gate: glm-only in production (Engine.__init__ sets it
    # from `arch == "glm"`). Tests run under the module's default ARCH="glm"
    # unless a test patches it, so True is the representative default for this
    # double; capability-gate tests override it explicitly (see NonGlmEngine
    # below) rather than relying on this.
    supports_logprobs_echo = True

    def __init__(self):
        self.calls = []
        self.stop_requests = 0

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, logprobs=0,
                 echo=False):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        self.last_logprobs = logprobs
        self.last_echo = echo
        if on_accept is not None:                 # simulate the engine's ACCEPT frame (#597)
            on_accept({"prompt_tokens": 7})
        for chunk in ("Hé", "llo"):
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        stats = {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}
        if logprobs:
            stats["logprobs"] = self.logprobs_channel(logprobs)
        return stats

    def logprobs_channel(self, engine_k):
        """Canned U7a logprob records for HTTP-level response-shape tests --
        shaped exactly like Engine.generate()'s real return value: "prompt"
        is a list of (pos, bytes, record), "generated" a list of (bytes,
        record), record = {"lp": float, "topk": [(tid, tlp), ...]}. Position
        0 carries the engine's own "nothing to condition on" sentinel (nan,
        empty table) -- the real wire behavior mux_prefill_echo always sends.
        The tail values are NON-dyadic (e.g. -0.3, -2.7) so a `%.6f`-shaped
        fixture is distinguishable from a `%.17g`-shaped one in tests. "H"=72
        "\\xc3\\xa9"="é" pretend token ids; the generated tokens' own lp is
        bit-identical to one of their own topk entries (mux's logprob_tail
        invariant), which is what the bit-identity property test and
        _own_token_label depend on."""
        k = min(engine_k, 2)
        prompt = [
            (0, b"H", {"lp": float("nan"), "topk": []}),
            (1, b"\xc3\xa9", {"lp": -0.3, "topk": [(72, -0.3), (100, -1.7)][:k]}),
        ]
        generated = [
            (b"H", {"lp": -0.2, "topk": [(72, -0.2), (200, -2.4)][:k]}),
            (b"\xc3\xa9", {"lp": -0.4, "topk": [(101, -0.4), (300, -3.6)][:k]}),
        ]
        return {"prompt": prompt, "generated": generated}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, logprobs=0,
                 echo=False):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text, cache_slot,
                                cancelled, grammar, stopped, on_accept, logprobs, echo)


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

    def test_seed_no_longer_rejected_by_generation_options(self):
        # generation_options() used to 400 on any `seed`; it is now a silent
        # accept-and-discard (documented no-op).
        generation_options({"seed": 1234, "prompt": "hi"}, 16)   # must not raise

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


class LogprobsOptionsTest(unittest.TestCase):
    """logprobs_options(): pure validation/translation, no HTTP or engine.

    Covers the range checks and the zero/false/null semantics, at the unit
    level -- fast, and independent of any fixture.
    """

    def test_completions_valid_integer_logprobs(self):
        self.assertEqual(logprobs_options({"logprobs": 3}, False, True), (3, False, 3))
        self.assertEqual(logprobs_options({"logprobs": 3, "echo": True}, False, True),
                         (3, True, 3))
        self.assertEqual(logprobs_options({"logprobs": 1}, False, True), (1, False, 1))
        self.assertEqual(logprobs_options({"logprobs": LOGPROBS_TOP_K_CAP}, False, True),
                         (LOGPROBS_TOP_K_CAP, False, LOGPROBS_TOP_K_CAP))

    def test_completions_zero_false_null_mean_no_logprobs(self):
        # The zero semantics are explicit -- `0`, `false`, and `null` all
        # mean "no logprobs" on completions, never a truthiness accident
        # and never an engine channel floored on at k=1.
        for off in ({"logprobs": 0}, {"logprobs": False}, {"logprobs": None}, {}):
            with self.subTest(off=off):
                self.assertEqual(logprobs_options(off, False, True), (0, False, 0))
        self.assertEqual(logprobs_options({"logprobs": 0, "echo": True}, False, True),
                         (0, True, 0))

    def test_completions_true_is_a_named_400(self):
        # The legacy completions field is an integer COUNT; a boolean
        # `true` carries no count and is a named 400, not a guess at k.
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": True}, False, True)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.param, "logprobs")
        self.assertEqual(caught.exception.code, "invalid_value")

    def test_break_it_battery_non_integer_negative_huge(self):
        for bad in (1.5, "5", -1, LOGPROBS_TOP_K_CAP + 1):
            with self.subTest(bad=bad):
                with self.assertRaises(APIError) as caught:
                    logprobs_options({"logprobs": bad}, False, True)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.param, "logprobs")
                self.assertEqual(caught.exception.code, "invalid_value")

    def test_completions_echo_requires_a_boolean(self):
        with self.assertRaises(APIError) as caught:
            logprobs_options({"echo": "yes"}, False, True)
        self.assertEqual(caught.exception.param, "echo")

    def test_chat_logprobs_requires_a_boolean(self):
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": 1}, True, True)
        self.assertEqual(caught.exception.param, "logprobs")
        self.assertEqual(caught.exception.code, "invalid_value")

    def test_chat_false_and_null_mean_no_logprobs(self):
        # Chat's boolean gate treats `null` like `false` -- explicitly.
        for off in ({"logprobs": False}, {"logprobs": None}, {}):
            with self.subTest(off=off):
                self.assertEqual(logprobs_options(off, True, True), (0, False, 0))

    def test_chat_echo_is_always_rejected(self):
        # Chat has no echo concept at all -- a named 400, not a silent
        # ignore, whether or not logprobs was also requested.
        with self.assertRaises(APIError) as caught:
            logprobs_options({"echo": True}, True, True)
        self.assertEqual(caught.exception.param, "echo")
        with self.assertRaises(APIError) as caught:
            logprobs_options({"echo": True, "logprobs": True}, True, True)
        self.assertEqual(caught.exception.param, "echo")

    def test_chat_top_logprobs_default_and_cap(self):
        self.assertEqual(logprobs_options({"logprobs": True}, True, True), (1, False, 0))
        self.assertEqual(
            logprobs_options({"logprobs": True, "top_logprobs": 5}, True, True), (5, False, 5))
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": True, "top_logprobs": LOGPROBS_TOP_K_CAP + 1},
                             True, True)
        self.assertEqual(caught.exception.param, "top_logprobs")

    def test_capability_gate_rejects_unsupported_engine(self):
        # The server never emits logprobs= to an engine that does not
        # implement the numeric per-token channel -- a named 400, not a
        # silent downgrade to "no logprobs".
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": 1}, False, False)
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.code, "unsupported_parameter")
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": True}, True, False)
        self.assertEqual(caught.exception.status, 400)
        # Absent/zero logprobs never reach the capability check at all.
        self.assertEqual(logprobs_options({}, False, False), (0, False, 0))
        self.assertEqual(logprobs_options({"logprobs": 0}, False, False), (0, False, 0))

    def test_range_error_precedes_capability_error(self):
        # The named 400 above the cap fires even on a non-supporting
        # engine -- the range check is about the public API surface, not
        # the engine.
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": LOGPROBS_TOP_K_CAP + 1}, False, False)
        self.assertEqual(caught.exception.code, "invalid_value")

    def test_cap_pinned_to_engine_topk_maximum(self):
        # Mirrors the engine's per-request top-k maximum, the
        # COLI_SUBMIT_TOPK_MAX constant defined in c/decode_batch.h:19. A
        # literal check (not the LOGPROBS_TOP_K_CAP symbol) so a change to
        # either side is caught, instead of the test drifting in lockstep
        # with the constant it exists to pin.
        self.assertEqual(LOGPROBS_TOP_K_CAP, 32)

    def test_cap_boundary_literals_both_endpoints(self):
        # Literal 32/33, not LOGPROBS_TOP_K_CAP +/- 1: a boundary check
        # that stays meaningful even if the cap constant itself drifts.
        self.assertEqual(logprobs_options({"logprobs": 32}, False, True), (32, False, 32))
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": 33}, False, True)
        self.assertEqual(caught.exception.param, "logprobs")
        self.assertEqual(caught.exception.code, "invalid_value")
        self.assertEqual(
            logprobs_options({"logprobs": True, "top_logprobs": 32}, True, True),
            (32, False, 32))
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": True, "top_logprobs": 33}, True, True)
        self.assertEqual(caught.exception.param, "top_logprobs")
        self.assertEqual(caught.exception.code, "invalid_value")

    def test_chat_top_logprobs_validated_even_when_logprobs_is_off(self):
        # Pre-existing defect found by review: with `logprobs`
        # false/absent, `top_logprobs` used to be ignored outright --
        # 999 and "x" both passed silently. It is now TYPE- and
        # RANGE-checked regardless, with the same named 400s as when
        # `logprobs` is true; a genuinely valid value stays a documented
        # no-op.
        for bad in ("x", -1, 999):
            with self.subTest(bad=bad):
                with self.assertRaises(APIError) as caught:
                    logprobs_options({"top_logprobs": bad}, True, True)
                self.assertEqual(caught.exception.status, 400)
                self.assertEqual(caught.exception.param, "top_logprobs")
                self.assertEqual(caught.exception.code, "invalid_value")
        self.assertEqual(logprobs_options({"top_logprobs": 5}, True, True), (0, False, 0))
        self.assertEqual(logprobs_options({}, True, True), (0, False, 0))

    def test_echo_non_bool_rejected_same_way_both_endpoints(self):
        # Completions already checked echo's type with isinstance(bool);
        # chat's refusal used truthiness. Both endpoints now validate
        # echo's TYPE first, with the shared "must be a boolean" 400 --
        # chat's "not supported for chat completions" refusal now only
        # ever fires for an actual boolean True.
        for bad in (1, "true"):
            with self.subTest(endpoint="completions", bad=bad):
                with self.assertRaises(APIError) as caught:
                    logprobs_options({"echo": bad}, False, True)
                self.assertEqual(caught.exception.param, "echo")
                self.assertEqual(caught.exception.code, "invalid_value")
            with self.subTest(endpoint="chat", bad=bad):
                with self.assertRaises(APIError) as caught:
                    logprobs_options({"echo": bad}, True, True)
                self.assertEqual(caught.exception.param, "echo")
                self.assertEqual(caught.exception.code, "invalid_value")

    def test_range_error_message_states_the_cap_value(self):
        with self.assertRaises(APIError) as caught:
            logprobs_options({"logprobs": 999}, False, True)
        self.assertIn("32", caught.exception.message)

    def test_chat_top_logprobs_null_normalizes_to_absent(self):
        # `top_logprobs: null` means the same thing as the field being
        # absent -- exactly like `logprobs: null` -- on both sides of the
        # `logprobs` gate. It never itself reaches the type/range check.
        self.assertEqual(
            logprobs_options({"logprobs": False, "top_logprobs": None}, True, True),
            (0, False, 0))
        self.assertEqual(
            logprobs_options({"logprobs": True, "top_logprobs": None}, True, True),
            (1, False, 0))

    def test_completions_ignores_top_logprobs_entirely(self):
        # `top_logprobs` is a chat-only field in the OpenAI request shape;
        # completions never reads it, so a nonsense value there changes
        # nothing -- the result is exactly the completions tuple for
        # `logprobs: 2` alone.
        self.assertEqual(
            logprobs_options({"logprobs": 2, "top_logprobs": 999}, False, True),
            (2, False, 2))


class ResponseAssemblyTest(unittest.TestCase):
    """Pure response-shape builders: _json_float, _order_echo_records,
    _own_token_label, _completions_logprobs_object, _chat_logprobs_content.
    No HTTP, no engine -- fixtures hand-build the (pos, bytes, record) and
    (bytes, record) tuples the wire dispatcher would otherwise produce."""

    def test_json_float_maps_non_finite_to_none(self):
        # Non-finite becomes JSON null on the wire out, never a clamped
        # number and never a raw float (json.dumps would emit the
        # invalid-JSON NaN/Infinity literal for that).
        self.assertIsNone(_json_float(float("nan")))
        self.assertIsNone(_json_float(float("inf")))
        self.assertIsNone(_json_float(float("-inf")))
        self.assertEqual(_json_float(-0.5), -0.5)
        self.assertEqual(json.dumps({"x": _json_float(float("nan"))}), '{"x": null}')

    def test_own_token_label_matches_by_value_not_rank_or_id(self):
        # The chosen token need not sit first in the top-k table (the
        # table is unsorted on the wire) -- the label must be found by
        # exact float match against the position's own logprob, not by
        # assuming table position 0 and not by any id-to-text mapping.
        entry = {"text": "café", "raw_lp": -0.8}
        topk = [(999, -0.05), (42, -0.8)]
        self.assertEqual(_own_token_label(entry, topk, 0), "<token_id:999>")
        self.assertEqual(_own_token_label(entry, topk, 1), "café")

    def test_own_token_label_tie_break_deterministic_first_match(self):
        # The engine prints logprobs to 6 decimal digits, so two distinct
        # candidates can share the exact printed value the chosen token
        # also carries -- a genuine tie the record's own shape (lp + a
        # topk table of raw ids, no chosen-token id) cannot resolve by
        # identity. Documented behavior: the FIRST table entry (in wire
        # order) whose value matches is labeled as the chosen token; a
        # later entry that also matches is labeled by its raw id like any
        # other unidentified candidate -- deterministic, not a claim that
        # the first entry is provably the true chosen one.
        entry = {"text": "cat", "raw_lp": -0.223144}
        topk = [(1, -0.223144), (2, -0.223144)]
        self.assertEqual(_own_token_label(entry, topk, 0), "cat")
        self.assertEqual(_own_token_label(entry, topk, 1), "<token_id:2>")

    def test_completions_logprobs_object_first_prompt_token_is_null(self):
        prompt = [(0, b"The", {"lp": float("nan"), "topk": []}),
                  (1, b" cat", {"lp": -0.3, "topk": [(7, -0.3), (8, -1.1)]})]
        obj = _completions_logprobs_object(prompt, [], display_k=2)
        self.assertEqual(obj["tokens"], ["The", " cat"])
        self.assertIsNone(obj["token_logprobs"][0])
        self.assertEqual(obj["token_logprobs"][1], -0.3)
        self.assertEqual(obj["top_logprobs"][0], {})

    def test_text_offset_reconstruction_and_monotonic(self):
        # "".join(tokens) reproduces the prompt text, and text_offset
        # matches the exact per-position character counts -- a literal
        # expected list, not a self-referential sortedness/start-at-0
        # check that every possible offset sequence satisfies by
        # construction (offsets are a running sum of non-negative token
        # lengths, so both of those would hold even for a wrong sequence).
        prompt_text = "Hé said éé"
        pieces = ["H", "é", " said ", "é", "é"]
        prompt = [(i, p.encode("utf-8"), {"lp": float("nan") if i == 0 else -0.1, "topk": []})
                  for i, p in enumerate(pieces)]
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual("".join(obj["tokens"]), prompt_text)
        self.assertEqual(obj["text_offset"], [0, 1, 2, 8, 9])

    def test_text_offset_base_kwarg_shifts_all_offsets(self):
        # An `echo=false` completions request still needs `text_offset`
        # counted from the end of the prompt under OpenAI's own
        # completions/echo convention -- callers that omit the echoed
        # prompt can pass the prompt's character length here so offsets
        # index the full text instead of just the generated tail. Default
        # (0) preserves the behavior exercised by every other test in
        # this class; deciding whether/when to pass a nonzero base is a
        # caller-side concern, handled outside this helper.
        generated = [(b"cat", {"lp": -0.05, "topk": []}), (b" dog", {"lp": -0.1, "topk": []})]
        obj = _completions_logprobs_object([], generated, display_k=0, text_offset_base=100)
        self.assertEqual(obj["text_offset"], [100, 103])

    def test_trailing_incomplete_multibyte_sequence_flushed_in_completions_path(self):
        # The same stateful-decoder trailing flush _chat_logprobs_content
        # relies on lives in the shared _logprob_positions helper -- prove
        # it from the completions/echo side too, not only via chat, so a
        # regression in the shared helper that happens to leave the chat
        # test green cannot slip through.
        prompt = [(0, b"H", {"lp": float("nan"), "topk": []}),
                  (1, b"\xc3", {"lp": -0.1, "topk": []})]    # first byte of 'é', never completed
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual(obj["tokens"][0], "H")
        self.assertIn("\ufffd", obj["tokens"][1])

    def test_bit_identity_when_chosen_token_is_argmax(self):
        # token_logprobs[i] equals top_logprobs[i][tokens[i]] exactly when
        # token i is itself the argmax of its own table -- checked against
        # the fixture's own argmax logprob literal (-0.05) on both sides,
        # not by comparing two fields of the same call's output to each
        # other (which a shared, consistently-wrong source would pass).
        generated = [(b"cat", {"lp": -0.05, "topk": [(7, -0.05), (8, -3.0)]})]
        obj = _completions_logprobs_object([], generated, display_k=2)
        self.assertEqual(obj["token_logprobs"][0], -0.05)
        self.assertEqual(obj["top_logprobs"][0]["cat"], -0.05)

    def test_display_k_truncates_alternatives_independent_of_engine_k(self):
        generated = [(b"cat", {"lp": -0.05, "topk": [(7, -0.05), (8, -3.0), (9, -4.0)]})]
        obj = _completions_logprobs_object([], generated, display_k=1)
        self.assertEqual(len(obj["top_logprobs"][0]), 1)
        obj0 = _completions_logprobs_object([], generated, display_k=0)
        self.assertEqual(obj0["top_logprobs"][0], {})
        self.assertEqual(obj0["token_logprobs"][0], -0.05)   # still reported

    def test_chat_content_shape_and_no_echo_field(self):
        generated = [(b"cat", {"lp": -0.05, "topk": [(7, -0.05), (8, -3.0)]})]
        content = _chat_logprobs_content(generated, display_k=2)
        self.assertEqual(len(content), 1)
        entry = content[0]
        self.assertEqual(set(entry), {"token", "logprob", "bytes", "top_logprobs"})
        self.assertEqual(entry["token"], "cat")
        self.assertEqual(entry["logprob"], -0.05)
        self.assertEqual(entry["bytes"], [99, 97, 116])
        self.assertEqual(len(entry["top_logprobs"]), 2)
        for alt in entry["top_logprobs"]:
            self.assertEqual(set(alt), {"token", "logprob", "bytes"})
        own = [a for a in entry["top_logprobs"] if a["token"] == "cat"][0]
        self.assertEqual(own["logprob"], -0.05)
        self.assertEqual(own["bytes"], [99, 97, 116])
        other = [a for a in entry["top_logprobs"] if a["token"] != "cat"][0]
        self.assertIsNone(other["bytes"])
        self.assertNotIn("echo", json.dumps(content))

    def test_chat_content_nan_serializes_as_null(self):
        generated = [(b"x", {"lp": float("-inf"), "topk": [(1, float("-inf"))]})]
        content = _chat_logprobs_content(generated, display_k=1)
        self.assertIsNone(content[0]["logprob"])
        self.assertIsNone(content[0]["top_logprobs"][0]["logprob"])

    # Adversarial top-k order: the chosen token's own candidate sits
    # second in the table and is not even the highest logprob present (a
    # sampled-path shape) -- no code here may assume table position 0.

    def test_token_logprobs_sourced_from_chosen_record_not_table_position(self):
        generated = [(b"cat", {"lp": -0.8, "topk": [(999, -0.05), (7, -0.8)]})]
        obj = _completions_logprobs_object([], generated, display_k=2)
        self.assertEqual(obj["token_logprobs"][0], -0.8)
        self.assertEqual(obj["top_logprobs"][0]["cat"], -0.8)
        self.assertNotEqual(obj["token_logprobs"][0], -0.05,
                            "token_logprobs must not be sourced from topk[0]")

    def test_chat_content_logprob_sourced_from_chosen_record_not_table_position(self):
        generated = [(b"cat", {"lp": -0.8, "topk": [(999, -0.05), (7, -0.8)]})]
        content = _chat_logprobs_content(generated, display_k=2)
        self.assertEqual(content[0]["logprob"], -0.8)
        own = [a for a in content[0]["top_logprobs"] if a["token"] == "cat"][0]
        self.assertEqual(own["logprob"], -0.8)

    # Numeric text_offset: exact values, not just monotonic/starts-at-0,
    # over a fixture with a multi-byte character ("é": one Python
    # character, two UTF-8 bytes) so a byte-vs-character-count confusion
    # would fail.

    def test_text_offset_exact_values_with_multibyte_character(self):
        prompt = [(0, b"H", {"lp": float("nan"), "topk": []}),
                  (1, b"\xc3\xa9", {"lp": -0.1, "topk": []}),   # "é", complete on its own
                  (2, b" cat", {"lp": -0.2, "topk": []})]
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual(obj["tokens"], ["H", "é", " cat"])
        self.assertEqual(obj["text_offset"], [0, 1, 2])

    # Multi-byte character split across two adjacent frames -- must
    # reconstruct via the stateful incremental decoder instead of
    # mangling into two replacement-character halves.

    def test_multibyte_character_split_across_adjacent_tokens_reconstructs(self):
        prompt_text = "café"
        prompt = [(0, b"c", {"lp": float("nan"), "topk": []}),
                  (1, b"a", {"lp": -0.1, "topk": []}),
                  (2, b"f", {"lp": -0.1, "topk": []}),
                  (3, b"\xc3", {"lp": -0.1, "topk": []}),    # first byte of 'é'
                  (4, b"\xa9", {"lp": -0.1, "topk": []})]    # second byte of 'é'
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual("".join(obj["tokens"]), prompt_text)
        self.assertEqual(obj["tokens"], ["c", "a", "f", "", "é"])

    def test_multibyte_character_split_across_generated_tokens_reconstructs(self):
        generated = [(b"\xc3", {"lp": -0.1, "topk": []}), (b"\xa9", {"lp": -0.1, "topk": []})]
        content = _chat_logprobs_content(generated, display_k=0)
        self.assertEqual("".join(c["token"] for c in content), "é")
        self.assertEqual([c["token"] for c in content], ["", "é"])
        # `bytes` always stays the frame's own raw payload, independent
        # of what text (if any) it resolved to.
        self.assertEqual(content[0]["bytes"], [0xC3])
        self.assertEqual(content[1]["bytes"], [0xA9])

    def test_trailing_incomplete_multibyte_sequence_still_flushed(self):
        # A dangling partial sequence at the very end of the stream (no
        # more data ever completes it) must still surface via the final
        # flush, not silently vanish.
        generated = [(b"cat", {"lp": -0.1, "topk": []}), (b"\xc3", {"lp": -0.1, "topk": []})]
        content = _chat_logprobs_content(generated, display_k=0)
        self.assertEqual(content[0]["token"], "cat")
        self.assertIn("�", content[1]["token"])

    def test_chat_content_tail_flush_consistent_with_own_alternative_label(self):
        # The trailing-flush text must reach the chosen token's OWN
        # top_logprobs entry, not just the outer `token` field -- decoding
        # is done in one pass over all positions (tail included) before
        # any content entry is built, so the two can never disagree about
        # what the last position's text actually is.
        generated = [(b"\xc3", {"lp": -0.1, "topk": [(1, -0.1)]})]   # never completed
        content = _chat_logprobs_content(generated, display_k=1)
        self.assertIn("�", content[0]["token"])
        self.assertEqual(content[0]["top_logprobs"][0]["token"], content[0]["token"])

    def test_completions_top_logprobs_tie_break_deterministic_first_match(self):
        # Two candidates print the identical 6-decimal logprob the chosen
        # token also carries. The FIRST exact match in wire order is
        # labeled as the chosen token; the second gets its own distinct
        # id-labeled entry rather than being silently merged into the
        # first (a dict keyed only by label would otherwise collapse two
        # genuinely different candidates into one entry).
        generated = [(b"cat", {"lp": -0.223144, "topk": [(1, -0.223144), (2, -0.223144)]})]
        obj = _completions_logprobs_object([], generated, display_k=2)
        self.assertEqual(set(obj["top_logprobs"][0]), {"cat", "<token_id:2>"})

    # ECHO's wire `pos` field, not arrival order, decides placement.

    def test_echo_positions_reassembled_by_wire_pos_not_arrival_order(self):
        prompt = [(1, b"b", {"lp": -1.0, "topk": []}),      # delivered out of order
                  (0, b"a", {"lp": float("nan"), "topk": []})]
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual(obj["tokens"], ["a", "b"])
        self.assertIsNone(obj["token_logprobs"][0])
        self.assertEqual(obj["token_logprobs"][1], -1.0)

    def test_order_echo_records_repeated_token_prompt_not_fooled_by_join_check(self):
        # A join-only check ("abab") can pass even when positions are
        # corrupted, because repeated tokens make the wrong order look
        # identical to the right one under "".join(). Position-indexed
        # assembly must get the per-position values right regardless.
        prompt = [(0, b"a", {"lp": float("nan"), "topk": []}),
                  (2, b"a", {"lp": -2.0, "topk": []}),
                  (1, b"b", {"lp": -1.0, "topk": []}),
                  (3, b"b", {"lp": -3.0, "topk": []})]
        obj = _completions_logprobs_object(prompt, [], display_k=0)
        self.assertEqual(obj["tokens"], ["a", "b", "a", "b"])
        self.assertEqual("".join(obj["tokens"]), "abab")
        self.assertEqual(obj["token_logprobs"], [None, -1.0, -2.0, -3.0])

    def test_duplicate_echo_position_raises_named_error(self):
        prompt = [(0, b"a", {"lp": float("nan"), "topk": []}),
                  (0, b"a2", {"lp": -1.0, "topk": []})]
        with self.assertRaisesRegex(RuntimeError, "invalid engine ECHO position"):
            _order_echo_records(prompt)
        with self.assertRaises(RuntimeError):
            _completions_logprobs_object(prompt, [], display_k=0)

    def test_out_of_range_echo_position_raises_named_error(self):
        prompt = [(0, b"a", {"lp": float("nan"), "topk": []}),
                  (5, b"b", {"lp": -1.0, "topk": []})]
        with self.assertRaisesRegex(RuntimeError, "invalid engine ECHO position"):
            _order_echo_records(prompt)

    def test_negative_echo_position_raises_named_error(self):
        prompt = [(-1, b"a", {"lp": float("nan"), "topk": []})]
        with self.assertRaises(RuntimeError):
            _order_echo_records(prompt)


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


class SchedulerTest(unittest.TestCase):
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
        # catch-all (which kills every in-flight request), and must not
        # change the legacy response shape: text delivery and the DONE
        # stats stay exactly as for legacy frames. The numeric records
        # themselves are collected internally; response assembly reading
        # them back out is separate, later work.
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

    def test_legacy_data_frame_dispatches_bare_bytes_no_record(self):
        # A DATA frame WITHOUT a tail dispatches exactly as on the
        # predecessor dispatcher -- the same event tuple, same bytes, no
        # record object allocated. Guards against a record being allocated
        # unconditionally on the (far more common) non-opted-in path.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(b"DATA " + request_id.encode() + b" 2\nok\n")
        kind, value = events.get(timeout=1)
        self.assertEqual(kind, "data")
        self.assertIs(type(value), bytes)
        self.assertEqual(value, b"ok")
        engine.close()

    def test_echo_position_zero_nan_tail_parses_as_float_nan(self):
        # Position 0 of an echoed prompt carries no preceding token to
        # condition on, so the engine's tail there is always "nan 0" --
        # this must parse to float("nan"), not 0.0.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(b"ECHO " + request_id.encode() + b" 2 0 nan 0\nHi\n")
        kind, (pos, data, record) = events.get(timeout=1)
        self.assertEqual((kind, pos, data), ("echo", 0, b"Hi"))
        self.assertTrue(math.isnan(record["lp"]))
        self.assertEqual(record["topk"], [])
        engine.close()

    def test_nan_and_inf_logprob_tail_values_parse_cleanly(self):
        # A degenerate logit row (all -inf after grammar masking, say)
        # produces a well-formed record, not a parse failure.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(
            b"DATA " + request_id.encode() + b" 1 inf 2 5 inf 6 -inf\nx\n")
        kind, (data, record) = events.get(timeout=1)
        self.assertEqual((kind, data), ("data", b"x"))
        self.assertEqual(record["lp"], float("inf"))
        self.assertEqual(record["topk"], [(5, float("inf")), (6, float("-inf"))])
        engine.close()

    def test_high_precision_and_negative_infinity_tail_values_parse(self):
        # The parser must not assume any particular float rendering -- a
        # 17-significant-digit value (as a higher-precision engine build
        # might emit, versus the shipped build's %.6f) and a bare "-inf"
        # both parse through the same float() call as any other value.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(
            b"DATA " + request_id.encode() +
            b" 1 -0.10536051565782628 2 3 -1.7987654321098765 8 -inf\nx\n")
        kind, (data, record) = events.get(timeout=1)
        self.assertEqual((kind, data), ("data", b"x"))
        self.assertEqual(record["lp"], -0.10536051565782628)
        self.assertEqual(record["topk"], [(3, -1.7987654321098765), (8, float("-inf"))])
        engine.close()

    def test_top_k_stays_in_wire_order_never_sorted(self):
        # The table is unsorted on the wire; a pair earlier in the wire
        # order but numerically/id-smaller later must stay first -- catches
        # an accidental sort by id or by log-probability.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(
            b"DATA " + request_id.encode() + b" 1 -1.0 2 9 -1.0 2 -0.1\nx\n")
        kind, (data, record) = events.get(timeout=1)
        self.assertEqual((kind, data), ("data", b"x"))
        self.assertEqual(record["topk"], [(9, -1.0), (2, -0.1)])
        engine.close()

    def test_short_logprob_tail_raises_rather_than_silently_truncating(self):
        # A tail whose k claims more pairs than are actually present on the
        # wire must fail loudly, never silently truncate to the pairs that
        # happen to be there.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 1 -0.5 2 3 -0.5\nx\n")
            # A DONE trails the malformed frame so a dispatcher that fails to
            # catch the mismatch completes normally instead of hanging --
            # keeping this a clean assertion failure, not a stuck test.
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_long_logprob_tail_raises_rather_than_ignoring_trailing_fields(self):
        # The reverse of the short-tail case: a tail with MORE fields than
        # its own k accounts for must also fail loudly, not silently
        # ignore the trailing garbage.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(
                b"DATA " + request_id + b" 1 -0.5 1 3 -0.5 9 -9.0\nx\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_non_numeric_tail_fields_raise_named_error(self):
        # A non-numeric lp, token id, or per-candidate log-probability must
        # be a named engine-protocol error, not an uncaught ValueError.
        cases = (
            b"DATA {id} 1 nope 1 3 -0.5\nx\n",     # lp
            b"DATA {id} 1 -0.5 1 abc -0.5\nx\n",   # tid
            b"DATA {id} 1 -0.5 1 3 nope\nx\n",     # tlp (not "nan"/"inf")
        )
        for malformed in cases:
            with self.subTest(frame=malformed):
                def respond(process, frame, malformed=malformed):
                    request_id = frame.split()[1]
                    process.stdout.feed(malformed.replace(b"{id}", request_id))
                    process.stdout.feed(
                        b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

                process = FakeProcess(respond)
                with patch("openai_server.subprocess.Popen", return_value=process):
                    engine = Engine("glm", "model")
                with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail"):
                    engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
                engine.close()

    def test_negative_or_oversized_k_raises_named_error(self):
        # k selects how many candidate pairs follow; a negative k or one
        # past the engine's own top-32 cap is malformed, not a huge/empty
        # read.
        for k in (-1, LOGPROBS_TOP_K_CAP + 1):
            with self.subTest(k=k):
                def respond(process, frame, k=k):
                    request_id = frame.split()[1]
                    process.stdout.feed(
                        b"DATA " + request_id + f" 1 -0.5 {k}\n".encode() + b"x\n")
                    process.stdout.feed(
                        b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

                process = FakeProcess(respond)
                with patch("openai_server.subprocess.Popen", return_value=process):
                    engine = Engine("glm", "model")
                with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail"):
                    engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
                engine.close()

    def test_negative_token_id_raises_named_error(self):
        # A candidate's token id is a vocabulary index -- never negative.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(
                b"DATA " + request_id + b" 1 -0.5 1 -3 -0.5\nx\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_echo_frame_bad_terminator_is_a_named_error(self):
        # The byte after an ECHO frame's payload must be the LF terminator;
        # anything else is a named protocol error, matching DATA/TOOL. The
        # stream is closed right after the bad terminator so a check that
        # is missing or weakened surfaces "colibri engine exited
        # unexpectedly" from the next (never-arriving) frame instead of
        # hanging forever.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ECHO " + request_id + b" 2 0 nan 0\nHiX")
            process.stdout.close()

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine ECHO terminator"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_data_size_bound_checked_before_any_payload_read(self):
        # The 65536-byte size bound must be enforced BEFORE any payload
        # byte is read. The fake stream here is closed right after a
        # too-large claimed size and a couple of payload bytes -- a bound
        # check that fires first raises immediately; one that is missing or
        # weakened would instead try to read 70000 bytes from a stream that
        # only ever offers 2 and then closes, surfacing "truncated engine
        # DATA payload" (a different, misleading error) rather than
        # hanging forever waiting for bytes that will never arrive.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 70000\nxy")
            process.stdout.close()

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine DATA size"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_echo_frame_errors_are_named_echo_not_data(self):
        # A malformed ECHO frame's error message must say ECHO, not a
        # copy-pasted DATA -- a wrong frame name in a dispatcher-killing
        # error is actively misleading to whoever reads it. The stream is
        # closed right after the oversized header so a missing/weakened
        # bound check surfaces a truncation error instead of hanging.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ECHO " + request_id + b" 99999 0 nan 0\n")
            process.stdout.close()

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine ECHO size"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_dispatcher_drops_echo_frames_with_no_pending_request(self):
        # An ECHO frame for an id with no pending entry (already DONE, or
        # never admitted) stays droppable, exactly like DATA/ACCEPT already
        # do -- it must not raise or wedge the dispatcher for the NEXT
        # request on the same connection.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DONE " + request_id + b" STAT 0 1 0 1 3 0\n")
            # A stray ECHO for an id that is no longer pending (this one
            # just finished) must be read and dropped, not raise.
            process.stdout.feed(b"ECHO " + request_id + b" 1 0 nan 0\nx\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        # The dispatcher must still be alive and able to serve a second
        # request -- proof the stray frame didn't wedge it.
        engine.generate("hello2", 4, 0.0, 1.0, lambda _: None)
        self.assertIsNone(engine.dispatcher_error)
        engine.close()

    def test_supports_logprobs_echo_flag_set_once_by_arch(self):
        # The capability flag is set once at launch from the arch id,
        # glm only.
        process_glm = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process_glm):
            engine = Engine("glm", "model")
        self.assertTrue(engine.supports_logprobs_echo)
        engine.close()

        process_other = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process_other), \
             patch("openai_server.ARCH", "inkling"):
            engine = Engine("inkling", "model")
        self.assertFalse(engine.supports_logprobs_echo)
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


class SeedWireFrameTest(unittest.TestCase):
    """C1: `seed` is accepted and ignored. A stub-response equality check alone
    is vacuous here (FakeEngine always returns the same canned text regardless
    of any request field) -- the real proof is that the byte-exact SUBMIT frame
    the dispatcher writes to the engine process (see DispatcherTest above) never
    carries the seed value at all, seeded or not.
    """

    def _completion(self, body):
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
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            data = json.dumps(body).encode()
            headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
            request = Request(f"http://127.0.0.1:{server.server_port}/v1/completions",
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
        return status, parsed, frames[0]

    def test_seed_accepted_and_absent_from_submit_frame(self):
        base = {"model": "test-model", "prompt": "Complete me", "temperature": 0, "max_tokens": 4}
        status_plain, body_plain, frame_plain = self._completion(base)
        status_seeded, body_seeded, frame_seeded = self._completion({**base, "seed": 1234})
        self.assertEqual(status_plain, 200)
        self.assertEqual(status_seeded, 200)
        self.assertEqual(body_seeded["choices"][0], body_plain["choices"][0])
        # Each call uses a freshly-constructed Engine, so both first requests are
        # assigned request id "1" -- the wire frames are directly byte-comparable,
        # no field needs normalizing. If `seed` ever leaked onto the SUBMIT
        # header or into an extension field, this equality would break.
        self.assertEqual(frame_seeded, frame_plain)


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
        self.assertEqual(cap_for_arch("glm", None), 0)
        self.assertEqual(cap_for_arch("inkling", None), 8)
        self.assertEqual(cap_for_arch("kimi", None), 8)
        self.assertEqual(cap_for_arch("olmoe", None), 8)
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

    def test_health_reports_scheduler_and_kv_slots(self):
        with self.request("/health") as response:
            health = json.load(response)
            scheduler = health["scheduler"]
        self.assertEqual(scheduler["max_queue"], 8)
        self.assertIn("queued", scheduler)
        self.assertEqual(health["kv_slots"], 2)

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

    def _streaming_commit_probe(self):
        """Real Engine + FakeProcess + a real APIServer/socket, wired so a CANCEL frame
        written to the (fake) engine is observable deterministically (no sleep-and-hope):
        `cancel_seen` fires the instant the CANCEL write happens, synchronously in the
        request-handling thread; `pending_empty()` bounded-polls for the dispatcher
        thread's async pop of the request out of `engine.pending`, which is genuinely
        asynchronous relative to the CANCEL write."""
        cancel_seen = threading.Event()

        def respond(process, frame):
            parts = frame.split()
            if parts[0] == b"SUBMIT":
                process.stdout.feed(b"ACCEPT " + parts[1] + b" 7\n")
            elif parts[0] == b"CANCEL":
                process.stdout.feed(b"ERROR " + parts[1] + b" CANCELLED\n")
                cancel_seen.set()

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        server = APIServer(("127.0.0.1", 0), engine, "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        self.addCleanup(engine.close)

        def pending_empty(timeout=3.0):
            deadline = time.time() + timeout
            while time.time() < deadline:
                if "1" not in engine.pending:
                    return True
                time.sleep(0.01)
            return "1" not in engine.pending

        return process, engine, server, cancel_seen, pending_empty

    def _post_streaming_request(self, server):
        payload = json.dumps({"model": "test-model", "stream": True, "max_tokens": 16,
                              "messages": [{"role": "user", "content": "Hi"}]})
        request = (f"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                  f"Content-Type: application/json\r\n"
                  f"Content-Length: {len(payload)}\r\n\r\n{payload}").encode()
        sock = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        self.addCleanup(sock.close)
        sock.sendall(request)
        return sock

    def test_anthropic_end_headers_failure_at_commit_does_not_orphan_the_pending_request(self):
        """If the client vanishes at the exact moment the engine ACCEPTs, the real header
        flush -- end_headers(), where BaseHTTPRequestHandler actually hits the socket --
        can raise. That must not unwind out of generate()'s dispatch loop with nothing
        sent: the request would sit in the engine's pending map forever, with no CANCEL
        ever going out to free it. Patches end_headers() itself (not send_response), so
        send_response's real _committed/close_connection bookkeeping still runs first,
        exactly as it would for a real dropped socket."""
        process, engine, server, cancel_seen, pending_empty = self._streaming_commit_probe()
        with patch.object(APIHandler, "end_headers",
                          side_effect=BrokenPipeError("client vanished exactly at ACCEPT")):
            self._post_streaming_request(server)
            self.assertTrue(cancel_seen.wait(timeout=3),
                            "the commit failure never triggered a CANCEL")
        self.assertTrue(any(w.startswith(b"CANCEL ") for w in process.writes),
                        "the commit failure never triggered a CANCEL")
        self.assertTrue(pending_empty(),
                        "the request was never removed from the engine's pending map")

    def test_anthropic_first_sse_write_failure_after_commit_does_not_orphan_the_pending_request(self):
        """Companion to the end_headers() variant above: a write failure on the FIRST SSE
        body write (message_start, right after headers commit cleanly) is already caught
        by send_event()'s own try/except -- this is REGRESSION-COVERAGE that pathway keeps
        working, not a new defect. Patches the handler's wfile so only that first write
        after headers raises; the header commit itself goes through for real."""
        process, engine, server, cancel_seen, pending_empty = self._streaming_commit_probe()
        real_end_headers = APIHandler.end_headers
        state = {"headers_done": False}

        def committing_end_headers(handler):
            real_end_headers(handler)
            state["headers_done"] = True
            real_write = handler.wfile.write

            def failing_write(data):
                if state["headers_done"]:
                    state["headers_done"] = False   # only the first post-header write fails
                    raise BrokenPipeError("client vanished on the first SSE write")
                return real_write(data)
            handler.wfile.write = failing_write

        with patch.object(APIHandler, "end_headers", committing_end_headers):
            self._post_streaming_request(server)
            self.assertTrue(cancel_seen.wait(timeout=3),
                            "the first-SSE-write failure never triggered a CANCEL")
        self.assertTrue(any(w.startswith(b"CANCEL ") for w in process.writes),
                        "the first-SSE-write failure never triggered a CANCEL")
        self.assertTrue(pending_empty(),
                        "the request was never removed from the engine's pending map")


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

    def test_anthropic_streaming_context_exceeded_is_clean_400(self):
        # The Anthropic streaming path must defer its 200 the same way the OpenAI path
        # above does: a refusal discovered before the engine accepts the prompt is a
        # real HTTP 400 in the Anthropic error envelope, never a committed 200 followed
        # by a truncated (or empty) SSE body.
        req = Request(self.base + "/v1/messages",
                      data=json.dumps({"model": "test-model", "stream": True,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "x" * 100}]}).encode(),
                      headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=3)
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)          # a real 400, not a 200 stream
        raw = caught.exception.read()
        self.assertNotIn(b"event:", raw, "a pre-accept refusal must not emit any SSE bytes")
        body = json.loads(raw)
        self.assertEqual(body["type"], "error")               # the Anthropic envelope
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertIn("maximum context length", body["error"]["message"])


class AnthropicStreamCommitTest(unittest.TestCase):
    """On a healthy engine the deferred commit is invisible -- the stream's byte-level
    framing (headers, message_start through message_stop order) is unchanged, and the
    200 is provably committed from the engine's ACCEPT, not before generate() is even
    called."""

    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _raw(self, body):
        payload = json.dumps(body)
        request = (f"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                   f"Content-Type: application/json\r\n"
                   f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n"
                   f"{payload}").encode()
        with socket.create_connection(("127.0.0.1", self.server.server_port),
                                      timeout=3) as sock:
            sock.sendall(request)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks).decode("utf-8", "replace")

    def test_success_stream_framing_is_unchanged(self):
        raw = self._raw({"model": "test-model", "stream": True, "max_tokens": 16,
                         "messages": [{"role": "user", "content": "Hi"}]})
        head, body = raw.split("\r\n\r\n", 1)
        self.assertIn("HTTP/1.1 200", head)
        self.assertIn("content-type: text/event-stream", head.lower())
        self.assertIn("connection: close", head.lower())
        # Exact sequence, not mere presence: two chunks ("Hé", "llo") from FakeEngine's
        # default script must produce two content_block_delta events in position, not
        # just an unordered set of the six event names.
        names = [line[len("event: "):] for line in body.splitlines()
                 if line.startswith("event: ")]
        self.assertEqual(names, ["message_start", "content_block_start", "content_block_delta",
                                 "content_block_delta", "content_block_stop", "message_delta",
                                 "message_stop"])
        deltas = "".join(json.loads(line[len("data: "):])["delta"]["text"]
                         for line in body.splitlines()
                         if line.startswith("data: ") and '"text_delta"' in line)
        self.assertEqual(deltas, "Héllo")

    def test_stream_commits_only_after_accept(self):
        # The mechanism itself: the 200 must be sent from the engine's on_accept
        # callback, not unconditionally before generate() is even called.
        committed_at = {}
        commit_flags = []

        def handler_committed():
            return bool(commit_flags)

        class CommitProbeEngine(FakeEngine):
            def generate(self, prompt, maximum, temperature, top_p, on_text,
                         cache_slot=0, cancelled=None, grammar=None, stopped=None,
                         on_accept=None):
                committed_at["before_accept"] = handler_committed()
                if on_accept is not None:
                    on_accept({"prompt_tokens": 7})
                committed_at["after_accept"] = handler_committed()
                on_text("ok")
                return {"prompt_tokens": 7, "completion_tokens": 1, "length_limited": False}

        original = APIHandler.send_response

        def recording_send_response(handler, code, message=None):
            commit_flags.append(code)
            return original(handler, code, message)

        self.server.engine = CommitProbeEngine()
        with patch.object(APIHandler, "send_response", recording_send_response):
            raw = self._raw({"model": "test-model", "stream": True, "max_tokens": 16,
                             "messages": [{"role": "user", "content": "Hi"}]})
        self.assertIn("HTTP/1.1 200", raw)
        self.assertFalse(committed_at["before_accept"],
                         "the Anthropic stream committed its 200 before engine ACCEPT")
        self.assertTrue(committed_at["after_accept"])


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


class NonGlmEngine(FakeEngine):
    """The U7b capability gate's negative case: an engine that predates U7a
    (or isn't glm) never gets the extension fields, never mind what it
    would do with them."""
    supports_logprobs_echo = False


class LogprobsHTTPTest(unittest.TestCase):
    """End-to-end U7b acceptance tests against a real APIServer + APIHandler,
    with FakeEngine standing in for the engine subprocess (its
    logprobs_channel() returns the canned U7a records documented on
    FakeEngine.generate() above).

    On the predecessor head, ANY truthy `logprobs` (an integer, or `True`)
    unconditionally 400s before this server's own validation ever runs, and
    `choices[].logprobs` is otherwise always null. That predecessor check
    happens to also 400 several of this class's "reject this" cases for the
    WRONG reason (a blanket rejection, not this server's specific named
    validation), so only 6 of these 12 methods are regression pins (fail
    outright on the predecessor head): test_chat_logprobs_content_shape,
    test_chat_echo_is_rejected, test_completions_logprobs_true_is_named_400
    (predecessor 400s but with the wrong error code),
    test_bit_identity_end_to_end, test_echo_reconstructs_prompt_text, and
    test_nan_logprob_serializes_as_json_null_over_the_wire. The other 6
    already pass on the predecessor head (a truthy `logprobs`/`echo` request
    there either already 400s for the coincidentally-matching blanket reason,
    or a falsy/absent one already no-ops to `logprobs: null`) and are
    REGRESSION-COVERAGE, pinning that this server's own validation reaches the
    identical observable result: test_completions_logprobs_zero_is_no_logprobs_end_to_end,
    test_break_it_logprobs_out_of_range, test_break_it_echo_without_logprobs_is_a_documented_noop,
    test_break_it_streaming_plus_logprobs_is_named_400,
    test_break_it_logprobs_rejected_for_non_glm_engine, and
    test_golden_fixture_style_plain_request_is_unaffected. [RAN] verified by
    running this class against the predecessor head's server file: 6
    failed, 6 passed.
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0), cls.engine, "test-model", "secret", 16,
                               kv_slots=2)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, body=None):
        headers = {"Authorization": "Bearer secret"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=2)

    def _temp_server(self, engine):
        """A second, throwaway server backed by a different fake engine --
        for the capability-gate negative cases, which must not share
        cls.engine/cls.server with the rest of this class."""
        server = APIServer(("127.0.0.1", 0), engine, "test-model", "secret", 16, kv_slots=1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.scheduler.close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, timeout=2)
        return f"http://127.0.0.1:{server.server_port}"

    # ---- chat logprobs shape ------------------------------------------------

    def test_chat_logprobs_content_shape(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "logprobs": True, "top_logprobs": 2,
        }) as response:
            body = json.load(response)
        content = body["choices"][0]["logprobs"]["content"]
        self.assertGreater(len(content), 0)
        for entry in content:
            self.assertEqual(set(entry), {"token", "logprob", "bytes", "top_logprobs"})
            for alt in entry["top_logprobs"]:
                self.assertEqual(set(alt), {"token", "logprob", "bytes"})
        self.assertNotIn("echo", body["choices"][0])

    def test_chat_echo_is_rejected(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "echo": True,
            })
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.load(caught.exception)["error"]["param"], "echo")

    # ---- end to end -----------------------------------------------------------

    def test_completions_logprobs_zero_is_no_logprobs_end_to_end(self):
        # `logprobs: 0` behaves exactly like an omitted field -- the engine
        # channel stays off and the choice carries logprobs: null.
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "hi", "logprobs": 0,
        }) as response:
            body = json.load(response)
        self.assertIsNone(body["choices"][0]["logprobs"])
        self.assertEqual(self.engine.last_logprobs, 0)

    def test_completions_logprobs_true_is_named_400(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {
                "model": "test-model", "prompt": "hi", "logprobs": True,
            })
        self.assertEqual(caught.exception.code, 400)
        error = json.load(caught.exception)["error"]
        self.assertEqual(error["param"], "logprobs")
        self.assertEqual(error["code"], "invalid_value")

    # ---- bit-identity, end to end -------------------------------------------

    def test_bit_identity_end_to_end(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Hé", "echo": True, "logprobs": 2,
            "max_tokens": 1,
        }) as response:
            body = json.load(response)
        logprobs = body["choices"][0]["logprobs"]
        for i, tok in enumerate(logprobs["tokens"]):
            lp = logprobs["token_logprobs"][i]
            table = logprobs["top_logprobs"][i]
            if lp is not None and tok in table:
                self.assertEqual(table[tok], lp,
                                 f"position {i}: token_logprobs != top_logprobs[tokens[i]]")

    # ---- text_offset / echoed-prompt reconstruction -------------------------

    def test_echo_reconstructs_prompt_text(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Hé", "echo": True, "logprobs": 1,
            "max_tokens": 1,
        }) as response:
            body = json.load(response)
        logprobs = body["choices"][0]["logprobs"]
        # The canned engine always echoes exactly "H", "é" for the prompt
        # positions (FakeEngine.logprobs_channel) -- the first two tokens
        # reconstruct the prompt exactly; anything after that is generated.
        self.assertEqual("".join(logprobs["tokens"][:2]), "Hé")
        offsets = logprobs["text_offset"]
        self.assertEqual(offsets, sorted(offsets))
        self.assertEqual(offsets[0], 0)
        self.assertIsNone(logprobs["token_logprobs"][0])   # first prompt token: null by convention

    # ---- break-it battery ---------------------------------------------------

    def test_break_it_logprobs_out_of_range(self):
        for bad in (-1, 33, 1.5):
            with self.subTest(bad=bad):
                with self.assertRaises(HTTPError) as caught:
                    self.request("/v1/completions", {
                        "model": "test-model", "prompt": "hi", "logprobs": bad,
                    })
                self.assertEqual(caught.exception.code, 400)

    def test_break_it_echo_without_logprobs_is_a_documented_noop(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "hi", "echo": True,
        }) as response:
            body = json.load(response)
        self.assertEqual(response.status, 200)
        self.assertIsNone(body["choices"][0]["logprobs"])

    def test_break_it_streaming_plus_logprobs_is_named_400(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {
                "model": "test-model", "prompt": "hi", "logprobs": 1, "stream": True,
            })
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.load(caught.exception)["error"]["param"], "logprobs")

    def test_break_it_logprobs_rejected_for_non_glm_engine(self):
        base = self._temp_server(NonGlmEngine())
        request = Request(base + "/v1/chat/completions", method="POST",
                          headers={"Authorization": "Bearer secret",
                                   "Content-Type": "application/json"},
                          data=json.dumps({"model": "test-model",
                                           "messages": [{"role": "user", "content": "hi"}],
                                           "logprobs": True}).encode())
        with self.assertRaises(HTTPError) as caught:
            urlopen(request, timeout=2)
        self.assertEqual(caught.exception.code, 400)

    # ---- non-finite serializes as JSON null, end to end ---------------------

    def test_nan_logprob_serializes_as_json_null_over_the_wire(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Hé", "echo": True, "logprobs": 1,
            "max_tokens": 1,
        }) as response:
            raw = response.read()
        self.assertNotIn(b"NaN", raw)
        self.assertNotIn(b"Infinity", raw)
        body = json.loads(raw)
        self.assertIsNone(body["choices"][0]["logprobs"]["token_logprobs"][0])

    # ---- a request that never touches logprobs is unaffected ----------------

    def test_golden_fixture_style_plain_request_is_unaffected(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        }) as response:
            body = json.load(response)
        self.assertIsNone(body["choices"][0]["logprobs"])
        self.assertEqual(self.engine.last_logprobs, 0)


class LogprobsSubmitHeaderRegressionTest(unittest.TestCase):
    """REGRESSION-COVERAGE: a legacy request (no logprobs asked at all) must
    produce a byte-identical SUBMIT header to the predecessor -- the
    extension namespace must never appear unless the client opted in."""

    def test_legacy_request_submit_header_is_byte_identical(self):
        request_id = "1"
        expected = f"SUBMIT {request_id} 0 5 4 0.25 0.9\n".encode() + b"hello\n"

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"DATA " + request_id.encode() + b" 2\nok\n"
                b"DONE " + request_id.encode() + b" STAT 1 2.5 0 1.0 5 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        chunks = []
        engine.generate("hello", 4, 0.25, 0.9, chunks.append)
        engine.close()
        self.assertEqual(process.writes, [expected])

    def test_opted_in_request_submit_header_carries_the_extension(self):
        # The mutation this pins: sending logprobs= on every request (not
        # only opted-in ones) would make this test's own legacy sibling
        # above fail -- the extension field would show up unconditionally.
        request_id = "1"
        expected = (f"SUBMIT {request_id} 0 5 4 0.25 0.9 0 logprobs=2\n".encode() +
                    b"hello\n")

        def respond(process, frame):
            self.assertEqual(frame, expected)
            process.stdout.feed(
                b"ACCEPT " + request_id.encode() + b" 5\n"
                b"ECHO " + request_id.encode() + b" 1 0 nan 0\nh\n"
                b"DATA " + request_id.encode() +
                b" 2 -0.223144 1 3 -0.223144\nok\n"
                b"DONE " + request_id.encode() + b" STAT 1 2.5 0 1.0 5 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        chunks = []
        engine.generate("hello", 4, 0.25, 0.9, chunks.append, logprobs=2)
        engine.close()
        self.assertEqual(process.writes, [expected])


class LogprobsGoldenResponseRegressionTest(unittest.TestCase):
    """REGRESSION-COVERAGE: a golden plain (non-logprobs) request's response
    must be byte-identical to the predecessor's -- this feature must not
    perturb any response field for a request that never asked for
    logprobs."""

    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_golden_plain_completions_response_is_byte_identical(self):
        req = Request(self.base + "/v1/completions",
                      data=json.dumps({"model": "test-model", "prompt": "hi",
                                       "max_tokens": 4, "temperature": 0}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        choice = body["choices"][0]
        self.assertEqual(set(choice), {"index", "text", "logprobs", "finish_reason"})
        self.assertIsNone(choice["logprobs"])
        self.assertEqual(choice["text"], "Héllo")
        self.assertEqual(choice["finish_reason"], "stop")


class LogprobsTailTerminatorAndRangeTest(unittest.TestCase):
    """Two mutation survivors carried from the wire-dispatch work: a
    wrong (not-LF) DATA terminator byte, and a k outside 0..32 presented
    WITH a matching field count -- so the field-count check alone could
    never catch it, only the dedicated range check."""

    def test_data_frame_wrong_terminator_byte_is_a_named_error(self):
        # The byte after a DATA frame's tail-extended payload must be LF;
        # a wrong byte here (not a closed stream) is the same class of
        # protocol error ECHO's terminator check already has a dedicated
        # test for -- DATA's own wrong-byte case had none.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(
                b"DATA " + request_id + b" 2 -0.223144 1 3 -0.223144\nokX")
            process.stdout.close()

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine DATA terminator"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_k_out_of_range_with_matching_field_count_is_a_named_error(self):
        # k = LOGPROBS_TOP_K_CAP + 1, with exactly that many (tid, tlp)
        # pairs actually present -- the field-count check passes cleanly,
        # so only the dedicated `0 <= k <= LOGPROBS_TOP_K_CAP` range check
        # can catch this.
        def respond(process, frame):
            request_id = frame.split()[1]
            k = LOGPROBS_TOP_K_CAP + 1
            pairs = " ".join(f"{i} -0.1" for i in range(k))
            process.stdout.feed(
                b"DATA " + request_id + f" 1 -0.5 {k} {pairs}\n".encode() + b"x\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "invalid engine logprob tail: k=.* out of range"):
            engine.generate("hello", 4, 0.0, 1.0, lambda _: None)
        engine.close()

    def test_g17_precision_tail_value_parses_through_the_same_float_call(self):
        # The shipped engine prints tail numbers as %.6f; a %.17g
        # value (a higher-precision build's own extension, not exactly
        # representable at 6 decimals) must parse through the same
        # float() call, not a format assumption.
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        events = queue.Queue()
        request_id = "1"
        with engine.pending_lock:
            engine.pending[request_id] = events
        process.stdout.feed(
            b"DATA " + request_id.encode() +
            b" 1 -0.30000000000000004 1 3 -0.30000000000000004\nx\n")
        kind, (data, record) = events.get(timeout=1)
        self.assertEqual((kind, data), ("data", b"x"))
        self.assertEqual(record["lp"], -0.30000000000000004)
        self.assertEqual(record["topk"], [(3, -0.30000000000000004)])
        engine.close()


class _DistinctEchoEngine(FakeEngine):
    """Prompt-echo text ("PQ") and generated text ("gen") are chosen so
    they share no character -- unlike LogprobsHTTPTest's shared canned
    fixture, where the generated text happens to start with the same two
    characters as the reconstructed prompt, a coincidence that would make a
    `text.startswith(prompt_text)` check pass even with the prompt never
    actually prepended."""

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, logprobs=0,
                 echo=False):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        self.last_logprobs = logprobs
        self.last_echo = echo
        if on_accept is not None:
            on_accept({"prompt_tokens": 2})
        on_text("gen")
        stats = {"prompt_tokens": 2, "completion_tokens": 1, "length_limited": False}
        if logprobs:
            stats["logprobs"] = {
                "prompt": [
                    (0, b"P", {"lp": float("nan"), "topk": []}),
                    (1, b"Q", {"lp": -0.1, "topk": [(1, -0.1)]}),
                ],
                "generated": [(b"gen", {"lp": -0.2, "topk": [(2, -0.2)]})],
            }
        return stats


class EchoTextPrependTest(unittest.TestCase):
    """`echo: true` must return prompt+completion in `text` itself (the
    OpenAI legacy shape), not the completion alone, with `text_offset`
    indexing that same concatenation."""

    def _server(self):
        engine = _DistinctEchoEngine()
        server = APIServer(("127.0.0.1", 0), engine, "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.scheduler.close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, timeout=2)
        return f"http://127.0.0.1:{server.server_port}"

    def test_echo_true_returns_prompt_plus_completion_in_text(self):
        base = self._server()
        req = Request(base + "/v1/completions",
                      data=json.dumps({"model": "test-model", "prompt": "PQ",
                                       "echo": True, "logprobs": 1,
                                       "max_tokens": 1}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        choice = body["choices"][0]
        logprobs = choice["logprobs"]
        self.assertEqual(choice["text"], "PQgen")
        self.assertEqual(logprobs["tokens"], ["P", "Q", "gen"])
        self.assertEqual(logprobs["text_offset"], [0, 1, 2])
        for offset in logprobs["text_offset"]:
            self.assertLessEqual(offset, len(choice["text"]))

    def test_echo_false_returns_completion_only_in_text(self):
        # The control case: without echo, `text` stays completion-only, as
        # before this fix.
        base = self._server()
        req = Request(base + "/v1/completions",
                      data=json.dumps({"model": "test-model", "prompt": "PQ",
                                       "logprobs": 1, "max_tokens": 1}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        self.assertEqual(body["choices"][0]["text"], "gen")


class _SeamSplitEngine(FakeEngine):
    """A 3-byte UTF-8 character (the Euro sign, "\u20ac") is split 1+2
    across the prompt/completion seam: its leading byte rides the last
    prompt ECHO frame's own bytes, and its two trailing bytes ride the
    first generated DATA frame's own bytes. `on_text` is fed exactly what
    an incremental UTF-8 decoder with no leading-byte context produces for
    those two trailing bytes alone -- two replacement characters -- which
    is what a caller decoding the generated bytes independently of the
    prompt bytes (the seam bug) would see; a decoder that instead sees the
    whole byte stream as one sequence reconstructs the real character."""

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, logprobs=0,
                 echo=False):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        self.last_logprobs = logprobs
        self.last_echo = echo
        if on_accept is not None:
            on_accept({"prompt_tokens": 2})
        on_text("\ufffd\ufffd")
        stats = {"prompt_tokens": 2, "completion_tokens": 1, "length_limited": False}
        if logprobs:
            stats["logprobs"] = {
                "prompt": [
                    (0, b"A", {"lp": float("nan"), "topk": []}),
                    (1, b"\xe2", {"lp": -0.1, "topk": [(1, -0.1)]}),
                ],
                "generated": [(b"\x82\xac", {"lp": -0.2, "topk": [(2, -0.2)]})],
            }
        return stats


class EchoSeamDecodingTest(unittest.TestCase):
    """A UTF-8 codepoint split across the prompt/completion seam must be
    decoded as one character by one decoder spanning both sides, not as
    two independently decoded halves."""

    def test_split_codepoint_at_the_seam_decodes_as_one_character(self):
        engine = _SeamSplitEngine()
        server = APIServer(("127.0.0.1", 0), engine, "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.scheduler.close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, timeout=2)
        base = f"http://127.0.0.1:{server.server_port}"
        req = Request(base + "/v1/completions",
                      data=json.dumps({"model": "test-model", "prompt": "A\u20ac",
                                       "echo": True, "logprobs": 1,
                                       "max_tokens": 1}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        choice = body["choices"][0]
        logprobs = choice["logprobs"]
        self.assertEqual(choice["text"], "A\u20ac")
        self.assertEqual(choice["text"].count("\u20ac"), 1)
        self.assertNotIn("\ufffd", choice["text"])
        offsets = logprobs["text_offset"]
        self.assertEqual(offsets, sorted(offsets), "text_offset must be monotonic")
        for offset in offsets:
            self.assertLessEqual(offset, len(choice["text"]))


class _StopTokenLogprobsEngine(FakeEngine):
    """Emits three generated chunks and a matching logprob record for each;
    a `stop` sequence matching the second chunk exactly withholds it (and
    everything after) from `text` -- its own record, and the record after
    it, must not survive into the response either."""

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None, logprobs=0,
                 echo=False):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        self.last_logprobs = logprobs
        self.last_echo = echo
        if on_accept is not None:
            on_accept({"prompt_tokens": 3})
        for chunk in ("ok ", "STOP", " more"):
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        stats = {"prompt_tokens": 3, "completion_tokens": 3, "length_limited": False}
        if logprobs:
            stats["logprobs"] = {"prompt": [], "generated": [
                (b"ok ", {"lp": -0.1, "topk": [(1, -0.1)]}),
                (b"STOP", {"lp": -0.2, "topk": [(2, -0.2)]}),
                (b" more", {"lp": -0.3, "topk": [(3, -0.3)]}),
            ]}
        return stats


class LogprobsDroppedStopTokenTest(unittest.TestCase):
    """A matched stop sequence withholds its own (and any later) text from
    the response -- the logprobs arrays must not still describe a token the
    client never received."""

    def test_filtered_stop_token_record_is_dropped(self):
        engine = _StopTokenLogprobsEngine()
        server = APIServer(("127.0.0.1", 0), engine, "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.scheduler.close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, timeout=2)
        base = f"http://127.0.0.1:{server.server_port}"
        req = Request(base + "/v1/completions",
                      data=json.dumps({"model": "test-model", "prompt": "hi",
                                       "max_tokens": 8, "stop": ["STOP"],
                                       "logprobs": 1}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        choice = body["choices"][0]
        self.assertEqual(choice["text"], "ok ")
        logprobs = choice["logprobs"]
        # Only the ONE record whose bytes are a prefix of "ok " may survive;
        # the "STOP" record (and the " more" record after it) must not.
        self.assertEqual(logprobs["tokens"], ["ok "])
        self.assertEqual(len(logprobs["token_logprobs"]), 1)
        self.assertEqual(len(logprobs["top_logprobs"]), 1)


class LogprobsOldEngineAcceptTimeoutTest(unittest.TestCase):
    """An engine build that silently rejects the extended SUBMIT header
    must not wedge the caller forever."""

    def test_old_engine_rejection_times_out_with_a_named_503(self):
        # engine.generate() must run on its own thread here: if the
        # accept-deadline check is missing or broken, the call blocks
        # forever on this stub (the "ERROR 0 ..." reply never resolves the
        # real pending request -- see the module docstring above), and a
        # direct call in this test's own thread would hang the whole suite
        # rather than fail this one test. A bounded join turns that failure
        # mode into a normal, fast test failure instead.
        def respond(process, frame):
            process.stdout.feed(b"ERROR 0 BAD_REQUEST\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process), \
             patch("openai_server.LOGPROBS_ACCEPT_TIMEOUT", 0.2):
            engine = Engine("glm", "model")
            outcome = {}

            def run():
                try:
                    engine.generate("hello", 4, 0.0, 1.0, lambda _: None, logprobs=1)
                except Exception as error:               # noqa: BLE001 -- captured, not swallowed
                    outcome["error"] = error

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(),
                             "engine.generate() did not return within the bound -- "
                             "the accept-deadline check did not fire")
        caught = outcome.get("error")
        self.assertIsInstance(caught, APIError, f"wrong exception: {caught!r}")
        self.assertEqual(caught.status, 503)
        self.assertEqual(caught.param, "logprobs")
        self.assertEqual(caught.code, "engine_logprobs_unsupported")
        engine.close()

    def test_legacy_non_opted_in_request_is_unaffected_by_the_timeout(self):
        # The bound only applies when logprobs was requested; a legacy
        # request keeps waiting exactly as before -- proven here by a very
        # short LOGPROBS_ACCEPT_TIMEOUT that would fire immediately if it
        # (wrongly) applied to every request.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 2\nok\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process), \
             patch("openai_server.LOGPROBS_ACCEPT_TIMEOUT", 0.01):
            engine = Engine("glm", "model")
            chunks = []
            stats = engine.generate("hello", 4, 0.0, 1.0, chunks.append)
        self.assertEqual(chunks, ["ok"])
        self.assertEqual(stats["completion_tokens"], 1)
        engine.close()


class LogprobsAcceptTimeoutEnvVarTest(unittest.TestCase):
    """The tests above patch the module attribute `LOGPROBS_ACCEPT_TIMEOUT`
    directly, which proves the accept-deadline logic reacts to that
    attribute but proves nothing about the documented public knob: the
    `COLI_LOGPROBS_ACCEPT_TIMEOUT` environment variable and its 30-second
    default are only read once, at import time. A typo in the env var
    name, or a changed default, would ship silently and green under a
    patch()-only suite. These tests import the module fresh in a real
    subprocess so the env var is read for real, by its real name."""

    SERVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_VAR = "COLI_LOGPROBS_ACCEPT_TIMEOUT"

    def _read_timeout_in_subprocess(self, value=None):
        # Build the child's environment from scratch for this one variable:
        # start from a copy of ours, drop the real name unconditionally,
        # then set it back only if a value was requested -- so neither
        # branch is at the mercy of whatever happens to be in this
        # process's own environment.
        env = dict(os.environ)
        env.pop(self.ENV_VAR, None)
        if value is not None:
            env[self.ENV_VAR] = value
        result = subprocess.run(
            [sys.executable, "-c",
             "import openai_server; print(openai_server.LOGPROBS_ACCEPT_TIMEOUT)"],
            cwd=self.SERVER_DIR, env=env,
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        return float(result.stdout.strip())

    def test_env_var_by_its_real_name_overrides_the_default(self):
        self.assertEqual(self._read_timeout_in_subprocess("5"), 5.0)

    def test_unset_env_var_defaults_to_30(self):
        self.assertEqual(self._read_timeout_in_subprocess(), 30.0)


class LogprobsEchoBufferingTest(unittest.TestCase):
    """Prompt-echo records must not be retained when the caller never asked
    to see them, even though the engine still sends every ECHO frame (there
    is no wire bit for "logprobs but no echo")."""

    def test_prompt_records_stay_empty_without_echo(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + request_id + b" 2\n")
            process.stdout.feed(b"ECHO " + request_id + b" 1 0 nan 0\nh\n")
            process.stdout.feed(
                b"ECHO " + request_id + b" 1 1 -0.1 1 3 -0.1\ni\n")
            process.stdout.feed(
                b"DATA " + request_id + b" 2 -0.223144 1 3 -0.223144\nok\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        stats = engine.generate("hi", 4, 0.0, 1.0, lambda _: None, logprobs=1, echo=False)
        self.assertEqual(stats["logprobs"]["prompt"], [])
        self.assertEqual(len(stats["logprobs"]["generated"]), 1)
        engine.close()

    def test_prompt_records_are_kept_with_echo(self):
        # The control case: the same wire traffic, but the caller DID ask
        # to see the echo table -- the records must still be retained.
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + request_id + b" 2\n")
            process.stdout.feed(b"ECHO " + request_id + b" 1 0 nan 0\nh\n")
            process.stdout.feed(
                b"ECHO " + request_id + b" 1 1 -0.1 1 3 -0.1\ni\n")
            process.stdout.feed(
                b"DATA " + request_id + b" 2 -0.223144 1 3 -0.223144\nok\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        stats = engine.generate("hi", 4, 0.0, 1.0, lambda _: None, logprobs=1, echo=True)
        self.assertEqual(len(stats["logprobs"]["prompt"]), 2)
        engine.close()


if __name__ == "__main__":
    unittest.main()
