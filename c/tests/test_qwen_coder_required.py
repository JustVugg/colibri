"""Required choice reaches Qwen3-Coder through both public API shapes."""
import copy
import json
import unittest
from unittest.mock import patch
from urllib.request import Request, urlopen

from openai_server import render_chat_qwen3_coder, TOOL_CHOICE_REQUIRED_INSTRUCTION
from tests.test_openai_server import _ChunkEngine, _spawn_test_server

TOOLS = [{"type": "function", "function": {"name": "inspect", "parameters": {
    "type": "object", "properties": {}}}}]
MESSAGES = [{"role": "user", "content": "Inspect this document"}]


class QwenCoderRequiredChoiceTest(unittest.TestCase):
    def test_required_appends_existing_instruction_to_system_tool_block(self):
        for system in ([], [{"role": "system", "content": "Document assistant"}]):
            with self.subTest(system=system):
                tools = copy.deepcopy(TOOLS)
                automatic = render_chat_qwen3_coder(system + MESSAGES, tools=tools)
                required = render_chat_qwen3_coder(system + MESSAGES, tools=tools,
                                                   tool_choice="required")
                end = required.index("<|im_end|>")
                self.assertTrue(required[:end].endswith(TOOL_CHOICE_REQUIRED_INSTRUCTION))
                self.assertEqual(required.replace(TOOL_CHOICE_REQUIRED_INSTRUCTION, ""), automatic)
                self.assertEqual(tools, TOOLS)

    def test_other_choices_retain_existing_behavior(self):
        automatic = render_chat_qwen3_coder(MESSAGES, tools=TOOLS)
        for choice in (None, "auto", {"type": "function", "function": {"name": "inspect"}}):
            with self.subTest(choice=choice):
                self.assertEqual(render_chat_qwen3_coder(MESSAGES, tools=TOOLS, tool_choice=choice),
                                 automatic)
        disabled = render_chat_qwen3_coder(MESSAGES, tools=TOOLS, tool_choice="none")
        self.assertEqual(disabled, render_chat_qwen3_coder(MESSAGES))

    def test_required_and_any_reach_engine_with_streaming_on_and_off(self):
        engine = _ChunkEngine(["Done"])
        base = _spawn_test_server(self, engine)
        with patch.multiple("openai_server", ARCH="qwen36", CHAT_FLAVOR="qwen3_coder"):
            for endpoint in ("/v1/chat/completions", "/v1/messages"):
                for stream in (False, True):
                    with self.subTest(endpoint=endpoint, stream=stream):
                        body = {"model": "test-model", "max_tokens": 16,
                                "messages": MESSAGES, "stream": stream}
                        if endpoint == "/v1/messages":
                            body.update(tools=[{"name": "inspect", "input_schema":
                                                TOOLS[0]["function"]["parameters"]}],
                                        tool_choice={"type": "any"}, thinking={"type": "disabled"})
                        else:
                            body.update(tools=TOOLS, tool_choice="required", enable_thinking=False)
                        request = Request(base + endpoint, data=json.dumps(body).encode(),
                                          headers={"Authorization": "Bearer secret",
                                                   "Content-Type": "application/json"})
                        with urlopen(request, timeout=5) as response:
                            self.assertEqual(response.status, 200)
                            response.read()
                        prompt = engine.calls[-1][0]
                        self.assertIn(TOOL_CHOICE_REQUIRED_INSTRUCTION, prompt)
                        self.assertLess(prompt.index(TOOL_CHOICE_REQUIRED_INSTRUCTION),
                                        prompt.index("<|im_end|>"))


if __name__ == "__main__":
    unittest.main()
