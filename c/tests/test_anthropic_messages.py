"""Anthropic Messages API (#343): /v1/messages as a translation layer over the same
generation path the OpenAI endpoint uses.

The point of these tests is the *contract a real Anthropic client depends on* — Claude Code
is the reference client from the issue — not merely that the handler returns 200:
  - request translation: system prompt, content blocks, tool_use/tool_result round trips;
  - response shape: content blocks, stop_reason, usage with Anthropic's own field names;
  - the SSE event sequence, in order, with named events (a client that keys off
    `event:` names breaks on a data-only stream even if the JSON is right);
  - `x-api-key` auth, which is how Anthropic clients authenticate — Bearer keeps working;
  - the Anthropic error envelope, which is not the OpenAI one.
"""
import json
import os
import re
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import (APIServer, anthropic_to_openai, anthropic_tools, APIError,
                           render_chat, render_chat_glm53, render_chat_inkling,
                           render_chat_kimi, render_chat_v4)


class FakeEngine:
    """Emits whatever `script` says, so a test can drive tool syntax through the parser."""
    def __init__(self, script=("Hé", "llo"), length_limited=False):
        self.script = script
        self.length_limited = length_limited
        self.prompts = []
        self.images = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None,
                 on_tool=None, image=None):
        self.prompts.append(prompt)
        self.images.append(image)
        self.emitted = 0
        for chunk in self.script:
            on_text(chunk)
            self.emitted += 1
            if stopped is not None and stopped():
                break                        # the gateway hit a stop sequence
        return {"prompt_tokens": 11, "completion_tokens": 3,
                "length_limited": self.length_limited}


class TranslationTest(unittest.TestCase):
    def test_system_and_blocks_become_openai_messages(self):
        messages = anthropic_to_openai({
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        })
        self.assertEqual(messages, [{"role": "system", "content": "Be brief."},
                                    {"role": "user", "content": "Hi"}])

    def test_tool_use_and_result_round_trip(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Rome"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"},
                {"type": "text", "text": "and tomorrow?"}]},
        ]})
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(messages[1]["tool_calls"][0]["function"]["arguments"]),
                         {"city": "Rome"})
        # the tool result must precede the user's new question, or the model reads the
        # answer as arriving after a question it has not been asked yet
        self.assertEqual([m["role"] for m in messages[2:]], ["tool", "user"])
        self.assertEqual(messages[2]["content"], "18C")
        self.assertEqual(messages[3]["content"], "and tomorrow?")

    def test_tool_result_only_message_adds_no_empty_user_turn(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "done"}]},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])

    def test_prior_assistant_thinking_block_is_preserved(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "reasoning",
                 "signature": "opaque-upstream-signature"},
                {"type": "text", "text": "answer"}]},
            {"role": "user", "content": "follow-up"},
        ]})
        self.assertEqual(messages[1]["reasoning_content"], "reasoning")
        self.assertIn("<|assistant|><think>reasoning</think>answer",
                      render_chat(messages))

    def test_tools_and_choice_translate(self):
        tools, choice = anthropic_tools({
            "tools": [{"name": "f", "description": "d",
                       "input_schema": {"type": "object", "properties": {"x": {"type": "string"}}}}],
            "tool_choice": {"type": "tool", "name": "f"}})
        self.assertEqual(tools[0]["type"], "function")
        self.assertEqual(tools[0]["function"]["parameters"]["properties"], {"x": {"type": "string"}})
        self.assertEqual(choice, {"type": "function", "function": {"name": "f"}})
        self.assertEqual(anthropic_tools({"tool_choice": {"type": "any"}})[1], "required")
        self.assertEqual(anthropic_tools({"tool_choice": {"type": "auto"}})[1], "auto")

    def test_system_role_rejection_triggers_claude_code_fallback(self):
        with self.assertRaises(APIError) as caught:
            anthropic_to_openai({"messages": [{"role": "system", "content": "no"}]})
        error = caught.exception
        self.assertEqual(error.status, 400)
        # Claude Code 2.1.212 retries without its model-gated mid-conversation
        # system turn only when the upstream rejection matches this contract.
        self.assertIn("not supported", error.message)
        self.assertRegex(error.message, re.compile(r"role .{0,2}system", re.IGNORECASE))

    # ---- image blocks -------------------------------------------------------------------
    # An Anthropic client sends a picture as {"type": "image", "source": {"type": "base64",
    # "media_type": ..., "data": ...}}; the OpenAI endpoint already takes the same picture
    # as an image_url data: URI, and expand_*_images() turns that into the engine's
    # placeholders. The translation must produce exactly that part, in place, so the two
    # endpoints cannot disagree about where a picture sits in the conversation.
    PNG = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                       "data": "iVBORw0KGgo="}}
    PNG_PART = {"type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}

    def test_image_block_becomes_an_image_url_part_in_place(self):
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                         self.PNG,
                                         {"type": "text", "text": "Be brief."}]}]})
        self.assertEqual(messages, [{"role": "user", "content": [
            {"type": "text", "text": "What is this?"}, self.PNG_PART,
            {"type": "text", "text": "Be brief."}]}])

    def test_text_only_content_still_collapses_to_a_string(self):
        # Every existing renderer and test relies on this shape; a picture-free
        # message must not start arriving as a parts list.
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "a"},
                                         {"type": "text", "text": "b"}]}]})
        self.assertEqual(messages, [{"role": "user", "content": "ab"}])

    def test_tool_result_image_rides_the_following_user_turn(self):
        # Claude Code's Read tool on a .png returns the picture INSIDE the tool_result.
        # The tool role carries text only, so the picture is carried onto the user
        # turn that follows the results -- the model still sees it right after the
        # tool answered, which is where the client put it.
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "look at it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"p": "x.png"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1",
                 "content": [{"type": "text", "text": "(image)"}, self.PNG]},
                {"type": "text", "text": "and?"}]},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(messages[2]["content"], "(image)")
        self.assertEqual(messages[3]["content"],
                         [self.PNG_PART, {"type": "text", "text": "and?"}])

    def test_tool_result_image_alone_still_makes_a_user_turn(self):
        # ...even when the tool_result is the whole message (no user text): dropping
        # the picture here is exactly the silent loss this exists to prevent.
        messages = anthropic_to_openai({"messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [self.PNG]}]},
        ]})
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool", "user"])
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(messages[3]["content"], [self.PNG_PART])

    def test_image_in_an_assistant_message_is_refused(self):
        for content in ([self.PNG],
                        [{"type": "tool_result", "tool_use_id": "t1", "content": [self.PNG]}]):
            with self.assertRaises(APIError) as caught:
                anthropic_to_openai({"messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": content}]})
            self.assertEqual(caught.exception.status, 400)
            self.assertIn("user messages", caught.exception.message)
            self.assertEqual(caught.exception.param, "messages.1.content.0")

    def test_image_source_is_validated(self):
        for source in ({"type": "base64", "media_type": "image/png"},          # no data
                       {"type": "base64", "data": "abc"},                       # no media_type
                       {"type": "file", "file_id": "f1"},                       # not supported
                       "not-an-object"):
            with self.assertRaises(APIError) as caught:
                anthropic_to_openai({"messages": [{"role": "user", "content": [
                    {"type": "image", "source": source}]}]})
            self.assertEqual(caught.exception.status, 400)
            self.assertTrue(caught.exception.param.startswith("messages.0.content.0.source"),
                            caught.exception.param)

    def test_url_source_is_handed_to_the_same_refusal_as_the_openai_path(self):
        # The OpenAI endpoint refuses to fetch a remote URL by policy; the same URL
        # arriving as an Anthropic `url` source must reach that same refusal, not a
        # different one and not a fetch.
        messages = anthropic_to_openai({"messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}]}]})
        self.assertEqual(messages[0]["content"],
                         [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}])


class MessagesHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model", "secret", 64,
                                kv_slots=2)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def post(self, body, headers=None, path="/v1/messages"):
        head = {"Content-Type": "application/json", "x-api-key": "secret"}
        head.update(headers or {})
        return urlopen(Request(self.base + path, data=json.dumps(body).encode(), headers=head),
                       timeout=3)

    def base_body(self, **extra):
        body = {"model": "test-model", "max_tokens": 32,
                "messages": [{"role": "user", "content": "Hi"}]}
        body.update(extra)
        return body

    def test_message_response_shape(self):
        with self.post(self.base_body()) as response:
            payload = json.load(response)
        self.assertEqual(payload["type"], "message")
        self.assertEqual(payload["role"], "assistant")
        self.assertEqual(payload["content"], [{"type": "text", "text": "Héllo"}])
        self.assertEqual(payload["stop_reason"], "end_turn")
        self.assertIsNone(payload["stop_sequence"])
        # Anthropic's usage field names, not OpenAI's prompt_tokens/completion_tokens
        self.assertEqual(payload["usage"], {"input_tokens": 11, "output_tokens": 3})
        self.assertTrue(payload["id"].startswith("msg_"))

    def test_trailing_assistant_turn_follows_shared_continuation_switch(self):
        """Off preserves the existing cue; on continues the turn on both endpoints."""
        messages = [{"role": "user", "content": "The capital of France is?"},
                    {"role": "assistant", "content": "The capital is"}]
        with patch("openai_server.ARCH", "glm53"):
            with patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": "0"}):
                with self.post(self.base_body(messages=messages)) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(self.engine.prompts[-1], render_chat_glm53(messages))
                self.assertTrue(self.engine.prompts[-1].endswith("<|assistant|><think>"))

            with patch.dict(os.environ, {"COLI_CONTINUE_ASSISTANT": "1"}):
                with self.post(self.base_body(messages=messages)) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(self.engine.prompts[-1],
                                 render_chat_glm53(messages, add_generation_prompt=False))
                self.assertTrue(self.engine.prompts[-1].endswith("The capital is"))

    def test_each_architecture_receives_its_native_chat_prompt(self):
        messages = [{"role": "user", "content": "Hi"}]
        renderers = {
            "glm": render_chat,
            "inkling": render_chat_inkling,
            "kimi": render_chat_kimi,
            "deepseek_v4": render_chat_v4,
        }
        for arch, renderer in renderers.items():
            with self.subTest(arch=arch), patch("openai_server.ARCH", arch):
                self.post(self.base_body()).close()
                self.assertEqual(self.engine.prompts[-1], renderer(messages))

    def test_non_glm_architectures_reject_tools_before_generation(self):
        # kimi left this list in #1143: K3 tool calling is wired up now.
        body = self.base_body(tools=[{"name": "f", "input_schema": {"type": "object"}}])
        for arch in ("inkling",):
            with self.subTest(arch=arch), patch("openai_server.ARCH", arch):
                before = len(self.engine.prompts)
                with self.assertRaises(HTTPError) as caught:
                    self.post(body)
                self.addCleanup(caught.exception.close)
                self.assertEqual(caught.exception.code, 400)
                self.assertIn("tool", json.load(caught.exception)["error"]["message"].lower())
                self.assertEqual(len(self.engine.prompts), before)

    def test_kimi_renders_tools_as_k3chat1_records(self):
        body = self.base_body(tools=[{"name": "f", "input_schema": {
            "type": "object", "properties": {"x": {"type": "string"}}}}])
        with patch("openai_server.ARCH", "kimi"):
            with self.post(body) as response:
                self.assertEqual(response.status, 200)
        prompt = self.engine.prompts[-1]
        self.assertIn("K3CHAT1", prompt)
        self.assertIn("tool-declare# Tools", prompt)
        self.assertIn('"name":"f"', prompt)
        self.assertNotIn("<|open|>", prompt)   # records, never raw XTML

    def test_deepseek_v4_renders_tools_as_dsml_block(self):
        body = self.base_body(tools=[{"name": "f", "input_schema": {
            "type": "object", "properties": {"x": {"type": "string"}}}}])
        with patch("openai_server.ARCH", "deepseek_v4"):
            with self.post(body) as response:
                self.assertEqual(response.status, 200)
        prompt = self.engine.prompts[-1]
        self.assertIn("## Tools", prompt)
        self.assertIn('"name": "f"', prompt)
        self.assertIn("｜DSML｜", prompt)

    def test_x_api_key_and_bearer_both_authenticate(self):
        with self.post(self.base_body(), {"x-api-key": "secret"}) as response:
            self.assertEqual(response.status, 200)
        with self.post(self.base_body(), {"x-api-key": None and "" or "",
                                          "Authorization": "Bearer secret"}) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(HTTPError) as caught:
            self.post(self.base_body(), {"x-api-key": "wrong"})
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 401)

    def test_error_envelope_is_anthropic_shaped(self):
        with self.assertRaises(HTTPError) as caught:
            self.post({"model": "test-model", "messages": [{"role": "user", "content": "x"}]})
        self.addCleanup(caught.exception.close)
        payload = json.load(caught.exception)
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("max_tokens", payload["error"]["message"])
        self.assertNotIn("param", payload)          # OpenAI's envelope must not leak here

    def test_max_tokens_maps_to_stop_reason(self):
        self.engine.length_limited = True
        with self.post(self.base_body()) as response:
            self.assertEqual(json.load(response)["stop_reason"], "max_tokens")

    def test_stream_emits_named_events_in_order(self):
        with self.post(self.base_body(stream=True)) as response:
            raw = response.read().decode()
        names = [line[len("event: "):] for line in raw.splitlines() if line.startswith("event: ")]
        self.assertEqual(names, ["message_start", "content_block_start", "content_block_delta",
                                 "content_block_delta", "content_block_stop", "message_delta",
                                 "message_stop"])
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        text = "".join(p["delta"]["text"] for p in payloads
                       if p["type"] == "content_block_delta")
        self.assertEqual(text, "Héllo")
        self.assertEqual(payloads[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(payloads[-2]["usage"]["output_tokens"], 3)

    def test_role_marker_ends_the_turn(self):
        """/v1/messages must apply the same implicit GLM stop sequences as
        /v1/chat/completions. In serve mode the engine arms only <|endoftext|>
        (#401/#549), so <|user|> arrives as ordinary text; without the filter it is
        detokenized into the reply and generation runs on to max_tokens."""
        script = ("ready", "<|user|>", "the user asks about cake")
        self.engine.script = script
        with self.post(self.base_body()) as response:
            payload = json.load(response)
        self.assertEqual(payload["content"], [{"type": "text", "text": "ready"}])
        self.assertEqual(payload["stop_reason"], "end_turn")
        self.assertEqual(self.engine.emitted, 2)   # tail cancelled, not merely hidden

    def test_role_marker_ends_a_streamed_turn(self):
        self.engine.script = ("ready", "<|user|>", "the user asks about cake")
        with self.post(self.base_body(stream=True)) as response:
            raw = response.read().decode()
        self.assertNotIn("<|user|>", raw)
        deltas = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                  if line.startswith("data: ")]
        text = "".join(d["delta"]["text"] for d in deltas
                       if d["type"] == "content_block_delta" and "text" in d["delta"])
        self.assertEqual(text, "ready")
        self.assertEqual(self.engine.emitted, 2)

    def test_marker_split_across_chunks_is_still_caught(self):
        self.engine.script = ("read", "y<|", "user|>", "cake")
        with self.post(self.base_body()) as response:
            payload = json.load(response)
        self.assertEqual(payload["content"], [{"type": "text", "text": "ready"}])

    def test_tool_call_becomes_tool_use_block(self):
        self.engine.script = ("Sure. <tool_call>get_weather<arg_key>city</arg_key>"
                              "<arg_value>Rome</arg_value></tool_call>",)
        body = self.base_body(tools=[{"name": "get_weather", "description": "w",
                                      "input_schema": {"type": "object",
                                                       "properties": {"city": {"type": "string"}},
                                                       "required": ["city"]}}])
        with self.post(body) as response:
            payload = json.load(response)
        self.assertEqual(payload["stop_reason"], "tool_use")
        kinds = [block["type"] for block in payload["content"]]
        self.assertEqual(kinds, ["text", "tool_use"])
        call = payload["content"][1]
        self.assertEqual(call["name"], "get_weather")
        self.assertEqual(call["input"], {"city": "Rome"})
        self.assertTrue(call["id"])
        # the raw <tool_call> markup must never surface as visible text
        self.assertNotIn("tool_call", payload["content"][0]["text"])

    def test_streamed_tool_call_uses_input_json_delta(self):
        self.engine.script = ("<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>",)
        body = self.base_body(stream=True, tools=[{"name": "f", "input_schema": {
            "type": "object", "properties": {"x": {"type": "string"}}}}])
        with self.post(body) as response:
            raw = response.read().decode()
        self.assertIn('"type":"input_json_delta"', raw)
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        start = [p for p in payloads if p["type"] == "content_block_start"
                 and p["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(start), 1)
        self.assertEqual(start[0]["content_block"]["name"], "f")
        self.assertEqual(start[0]["index"], 1)      # block 0 is the text block
        self.assertEqual(payloads[-2]["delta"]["stop_reason"], "tool_use")

    def test_thinking_enabled_renders_reasoning_prompt(self):
        self.post(self.base_body(thinking={"type": "enabled"})).close()
        self.assertIn("Reasoning Effort", self.engine.prompts[-1])
        self.assertTrue(self.engine.prompts[-1].endswith("<|assistant|><think>"))

    def test_thinking_enabled_returns_separate_blocks(self):
        self.engine.script = ("reasoning</think>answer",)
        with self.post(self.base_body(thinking={"type": "enabled"})) as response:
            payload = json.load(response)
        self.assertEqual(payload["content"], [
            {"type": "thinking", "thinking": "reasoning", "signature": "colibri-local"},
            {"type": "text", "text": "answer"},
        ])

    def test_inkling_thinking_uses_inkling_content_markers(self):
        self.engine.script = ("<|content_thinking|>reason", "ing<|content_text|>answer",)
        with patch("openai_server.ARCH", "inkling"):
            with self.post(self.base_body(thinking={"type": "enabled"})) as response:
                payload = json.load(response)
        self.assertEqual(payload["content"], [
            {"type": "thinking", "thinking": "reasoning", "signature": "colibri-local"},
            {"type": "text", "text": "answer"},
        ])

    def test_streamed_inkling_thinking_uses_inkling_content_markers(self):
        self.engine.script = ("<|content_think", "ing|>reasoning<|content_", "text|>answer",)
        with patch("openai_server.ARCH", "inkling"):
            with self.post(self.base_body(stream=True, thinking={"type": "enabled"})) as response:
                raw = response.read().decode()
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        deltas = [payload["delta"] for payload in payloads
                  if payload["type"] == "content_block_delta"]
        self.assertEqual("".join(delta.get("thinking", "") for delta in deltas), "reasoning")
        self.assertEqual("".join(delta.get("text", "") for delta in deltas), "answer")
        thinking_stop = next(index for index, payload in enumerate(payloads)
                             if payload["type"] == "content_block_stop" and payload["index"] == 0)
        text_start = next(index for index, payload in enumerate(payloads)
                          if payload["type"] == "content_block_start" and payload["index"] == 1)
        self.assertLess(thinking_stop, text_start)
        self.assertNotIn("content_thinking", raw)
        self.assertNotIn("content_text", raw)

    def test_streamed_thinking_marker_can_split_at_every_boundary(self):
        marker = "</think>"
        for boundary in range(1, len(marker)):
            with self.subTest(boundary=boundary):
                self.engine.script = ("reasoning" + marker[:boundary],
                                      marker[boundary:] + "answer")
                with self.post(self.base_body(stream=True,
                                              thinking={"type": "enabled"})) as response:
                    raw = response.read().decode()
                payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                            if line.startswith("data: ")]
                starts = [p for p in payloads if p["type"] == "content_block_start"]
                self.assertEqual([(p["index"], p["content_block"]["type"]) for p in starts],
                                 [(0, "thinking"), (1, "text")])
                deltas = [p for p in payloads if p["type"] == "content_block_delta"]
                self.assertEqual("".join(p["delta"].get("thinking", "") for p in deltas),
                                 "reasoning")
                self.assertEqual("".join(p["delta"].get("text", "") for p in deltas),
                                 "answer")
                self.assertEqual([p["delta"]["type"] for p in deltas
                                  if p["delta"]["type"] == "signature_delta"],
                                 ["signature_delta"])
                signature = next(i for i, p in enumerate(payloads)
                                 if p.get("delta", {}).get("type") == "signature_delta")
                self.assertEqual(payloads[signature]["delta"]["signature"], "colibri-local")
                thinking_stop = next(i for i, p in enumerate(payloads)
                                     if p["type"] == "content_block_stop" and p["index"] == 0)
                text_start = next(i for i, p in enumerate(payloads)
                                  if p["type"] == "content_block_start" and p["index"] == 1)
                self.assertLess(signature, thinking_stop)
                self.assertLess(thinking_stop, text_start)
                self.assertNotIn(marker, raw)

    def test_thinking_budget_exhaustion_returns_thinking_only(self):
        self.engine.script = ("unfinished reasoning",)
        self.engine.length_limited = True
        with self.post(self.base_body(thinking={"type": "enabled"})) as response:
            payload = json.load(response)
        self.assertEqual(payload["content"], [
            {"type": "thinking", "thinking": "unfinished reasoning",
             "signature": "colibri-local"},
        ])
        self.assertEqual(payload["stop_reason"], "max_tokens")

    def test_streamed_thinking_then_tool_uses_next_block(self):
        self.engine.script = ("reasoning</thi",
                              "nk><tool_call>f<arg_key>x</arg_key>"
                              "<arg_value>1</arg_value></tool_call>")
        body = self.base_body(stream=True, thinking={"type": "enabled"},
                              tools=[{"name": "f", "input_schema": {
                                  "type": "object",
                                  "properties": {"x": {"type": "string"}}}}])
        with self.post(body) as response:
            raw = response.read().decode()
        payloads = [json.loads(line[len("data: "):]) for line in raw.splitlines()
                    if line.startswith("data: ")]
        starts = [p for p in payloads if p["type"] == "content_block_start"]
        self.assertEqual([(p["index"], p["content_block"]["type"]) for p in starts],
                         [(0, "thinking"), (1, "tool_use")])
        self.assertNotIn("</think>", raw)
        self.assertEqual(payloads[-2]["delta"]["stop_reason"], "tool_use")

    def test_unsupported_fields_refuse_loudly(self):
        for field, value in (("stop_sequences", ["STOP"]), ("top_k", 40)):
            with self.assertRaises(HTTPError) as caught:
                self.post(self.base_body(**{field: value}))
            self.addCleanup(caught.exception.close)
            self.assertEqual(caught.exception.code, 400)
            self.assertIn(field, json.load(caught.exception)["error"]["message"])

    # ---- images end to end -----------------------------------------------------------
    # The translation alone is not the contract: the picture has to reach the engine.
    # The OpenAI path expands image parts into the family's placeholders and hands the
    # patches to Engine.generate(image=...) beside the prompt; /v1/messages goes
    # through the same expansion, so the placeholder count and the patches cannot
    # disagree between the two endpoints.
    IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": "iVBORw0KGgo="}}

    def fake_preprocess(self, data, model_dir):
        self.preprocessed.append(data)
        return b"\x00" * 16, 4, 4                 # 2x2 = 4 placeholders after merging

    def test_image_block_reaches_the_glm53_engine(self):
        self.preprocessed = []
        messages = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                                 self.IMAGE]}]
        with patch("openai_server.ARCH", "glm53"), \
             patch("openai_server._preprocess_image", self.fake_preprocess):
            with self.post(self.base_body(messages=messages)) as response:
                self.assertEqual(response.status, 200)
        self.assertEqual(self.preprocessed, [b"\x89PNG\r\n\x1a\n"])
        self.assertEqual(self.engine.images, [(b"\x00" * 16, 4, 4)])
        self.assertIn("<|user|>what is this?<|begin_of_image|>" + "<|image|>" * 4
                      + "<|end_of_image|><|assistant|>", self.engine.prompts[-1])

    def test_tool_result_image_reaches_the_glm53_engine_after_the_observation(self):
        self.preprocessed = []
        messages = [
            {"role": "user", "content": "read it"},
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"p": "x.png"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": [self.IMAGE]}]},
        ]
        with patch("openai_server.ARCH", "glm53"), \
             patch("openai_server._preprocess_image", self.fake_preprocess):
            with self.post(self.base_body(messages=messages)) as response:
                self.assertEqual(response.status, 200)
        self.assertEqual(self.engine.images, [(b"\x00" * 16, 4, 4)])
        prompt = self.engine.prompts[-1]
        self.assertLess(prompt.index("<|observation|>"), prompt.index("<|begin_of_image|>"))
        self.assertIn("<|user|><|begin_of_image|>" + "<|image|>" * 4 + "<|end_of_image|>",
                      prompt)

    def test_two_images_are_refused_like_the_openai_path(self):
        self.preprocessed = []
        messages = [{"role": "user", "content": [self.IMAGE, self.IMAGE]}]
        with patch("openai_server.ARCH", "glm53"), \
             patch("openai_server._preprocess_image", self.fake_preprocess):
            with self.assertRaises(HTTPError) as caught:
                self.post(self.base_body(messages=messages))
        self.addCleanup(caught.exception.close)
        self.assertEqual(caught.exception.code, 400)
        self.assertIn("one image per request", json.load(caught.exception)["error"]["message"])
        self.assertEqual(self.engine.images, [])


if __name__ == "__main__":
    unittest.main()
