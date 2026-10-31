import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import APIServer, QwenToolStream, parse_qwen_tool_calls
from tests.test_openai_server import _ChunkEngine

TOOLS = [{"type": "function", "function": {
    "name": "edit", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "edits": {"type": ["array", "null"]},
        "dry_run": {"type": "boolean"}}, "required": ["path", "edits"]}}}]
CALL = ('<function=edit>\n<parameter=path>\na.py\n</parameter>\n'
        '<parameter=edits>\n[{"line": 1}]\n</parameter>\n'
        '<parameter=dry_run>\nTrue\n</parameter>\n</function>')


class QwenToolValidationTest(unittest.TestCase):
    def test_complete_inner_blocks_recover_with_declared_types(self):
        for reply in (CALL, " \n" + CALL + "\n", CALL + "\n" + CALL,
                      "<tool_call>" + CALL + "</tool_call>"):
            with self.subTest(reply=reply):
                text, calls = parse_qwen_tool_calls(reply, TOOLS)
                self.assertEqual(text, "")
                self.assertEqual(len(calls), 2 if reply.count("<function=") == 2 else 1)
                for call in calls:
                    self.assertEqual(json.loads(call["function"]["arguments"]),
                                     {"path": "a.py", "edits": [{"line": 1}], "dry_run": True})

    def test_untrusted_or_incomplete_arguments_are_not_calls(self):
        for broken in (CALL.replace("edit>", "undeclared>"),
                       CALL.replace('<parameter=edits>\n[{"line": 1}]\n</parameter>\n', ''),
                       CALL.replace('[{"line": 1}]', 'not-json'),
                       CALL.replace('</function>', '')):
            for reply in (broken, "<tool_call>" + broken + "</tool_call>"):
                with self.subTest(reply=reply):
                    self.assertEqual(parse_qwen_tool_calls(reply, TOOLS), (reply.strip(), []))

    def test_prose_examples_and_missing_tool_declarations_are_not_recovered(self):
        for reply in ("Example: " + CALL, "```xml\n" + CALL + "\n```", CALL + " is an example"):
            self.assertEqual(parse_qwen_tool_calls(reply, TOOLS), (reply, []))
        self.assertEqual(parse_qwen_tool_calls(CALL), (CALL, []))

    def test_nullable_and_union_types(self):
        cases = [(["string", "null"], "null", None),
                 (["string", "null"], "None", None),
                 (["string", "null"], "hello", "hello"),
                 (["array", "null"], "None", None),
                 (["array", "null"], "[1]", [1]),
                 (["array", "object"], '{"x":1}', {"x": 1}),
                 ("boolean", "False", False), ("integer", "2", 2),
                 ("number", "1.5", 1.5), ("string", "null", "null")]
        for kind, raw, expected in cases:
            tools = [{"function": {"name": "f", "parameters": {
                "properties": {"x": {"type": kind}}, "required": ["x"]}}}]
            reply = f"<function=f><parameter=x>\n{raw}\n</parameter></function>"
            with self.subTest(kind=kind, raw=raw):
                text, calls = parse_qwen_tool_calls(reply, tools)
                self.assertEqual(text, "")
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"x": expected})

    def test_wrong_types_and_nonfinite_values_stay_text(self):
        for kind, raw in (("array", "123"), ("integer", "true"), ("boolean", "[]"),
                          ("integer", "1.5"), ("number", "NaN"), ("array", "[Infinity]")):
            tools = [{"function": {"name": "f", "parameters": {
                "properties": {"x": {"type": kind}}}}}]
            reply = f"<function=f><parameter=x>\n{raw}\n</parameter></function>"
            with self.subTest(kind=kind, raw=raw):
                self.assertEqual(parse_qwen_tool_calls(reply, tools), (reply, []))

    def test_unparsed_body_and_duplicate_parameters_are_not_recovered(self):
        for reply in (CALL.replace("</function>", "unparsed text</function>"),
                      CALL.replace("</function>", "<parameter=path>\nb.py\n</parameter></function>"),
                      CALL + CALL.replace("edit>", "undeclared>")):
            self.assertEqual(parse_qwen_tool_calls(reply, TOOLS), (reply, []))

    def test_only_accepted_wrapped_calls_are_removed(self):
        accepted = "<tool_call>" + CALL + "</tool_call>"
        rejected = accepted.replace("edit>", "undeclared>")
        text, calls = parse_qwen_tool_calls(accepted + "\n" + rejected, TOOLS)
        self.assertEqual(text, rejected)
        self.assertEqual(len(calls), 1)

    def test_stream_release_and_recovery_at_every_split(self):
        examples = (CALL, "<tool_call>" + CALL + "</tool_call>", "Example: " + CALL,
                    "```xml\n" + CALL + "\n```", CALL + " is an example",
                    CALL.replace("edit>", "undeclared>"))
        for reply in examples:
            expected, calls = parse_qwen_tool_calls(reply, TOOLS)
            for split in range(1, len(reply)):
                with self.subTest(reply=reply, split=split):
                    output = []
                    stream = QwenToolStream(output.append, TOOLS)
                    stream.feed(reply[:split])
                    stream.feed(reply[split:])
                    actual_calls = stream.finish()
                    self.assertEqual("".join(output), expected)
                    self.assertEqual(len(actual_calls), len(calls))


class QwenToolHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = _ChunkEngine([])
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)

    def request(self, endpoint, reply, stream):
        self.engine.chunks = list(reply)  # split every marker across engine frames
        body = {"model": "test-model", "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1024, "stream": stream}
        if endpoint == "/v1/messages":
            body["tools"] = [{"name": "edit", "input_schema": TOOLS[0]["function"]["parameters"]}]
            body["thinking"] = {"type": "disabled"}
        else:
            body["tools"] = TOOLS
            body["enable_thinking"] = False
        request = Request(f"http://127.0.0.1:{self.server.server_port}" + endpoint,
                          data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=5) as response:
            raw = response.read().decode()
        if not stream:
            result = json.loads(raw)
            if endpoint == "/v1/messages":
                blocks = result["content"]
                return ("".join(b["text"] for b in blocks if b["type"] == "text"),
                        sum(b["type"] == "tool_use" for b in blocks))
            message = result["choices"][0]["message"]
            return message.get("content") or "", len(message.get("tool_calls") or [])
        text, calls = [], 0
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if endpoint == "/v1/messages":
                if event.get("type") == "content_block_delta":
                    text.append(event["delta"].get("text", ""))
                if event.get("type") == "content_block_start":
                    calls += event["content_block"]["type"] == "tool_use"
            else:
                for choice in event["choices"]:
                    delta = choice.get("delta") or {}
                    text.append(delta.get("content") or "")
                    calls += len(delta.get("tool_calls") or [])
        return "".join(text), calls

    def test_invalid_required_entry_is_a_400_before_generation(self):
        with patch("openai_server.ARCH", "qwen36"):
            for endpoint in ("/v1/chat/completions", "/v1/messages"):
                for invalid in ({}, [], 3, None):
                    with self.subTest(endpoint=endpoint, invalid=invalid):
                        malformed = json.loads(json.dumps(TOOLS))
                        malformed[0]["function"]["parameters"]["required"] = [invalid]
                        with patch(__name__ + ".TOOLS", malformed), self.assertRaises(HTTPError) as ctx:
                            self.request(endpoint, "plain prose", False)
                        self.assertEqual(ctx.exception.code, 400)
                        ctx.exception.close()

    def test_streaming_and_nonstreaming_agree_on_calls_and_examples(self):
        replies = (CALL, "Example: " + CALL, CALL + " is an example",
                   "```xml\n" + CALL + "\n```", CALL.replace("edit>", "undeclared>"),
                   "<tool_call>" + CALL + "</tool_call>",
                   "<tool_call>" + CALL.replace("edit>", "undeclared>") + "</tool_call>")
        for arch in ("qwen36", "qwen38"):
            with patch.multiple("openai_server", ARCH=arch, CHAT_FLAVOR=None):
                for endpoint in ("/v1/chat/completions", "/v1/messages"):
                    for reply in replies:
                        with self.subTest(arch=arch, endpoint=endpoint, reply=reply):
                            expected, calls = parse_qwen_tool_calls(reply, TOOLS)
                            for stream in (False, True):
                                self.assertEqual(self.request(endpoint, reply, stream),
                                                 (expected, len(calls)))


if __name__ == "__main__":
    unittest.main()
