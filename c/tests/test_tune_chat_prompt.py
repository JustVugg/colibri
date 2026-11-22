"""`coli tune` times the request `coli serve` is sent (#2590).

The qwen36 family's checkpoints ship different chat templates (Qwen3.6, Qwen3-Coder, Flint,
Muse Spark, LFM3, the dense Qwen3), and tune used one fixed form for all of them, with an
open <think>. Qwen3-Coder has no thinking mode: after that <think> it ended its turn on the
first token (measured on Qwen3-Coder-30B-A3B int4, CUDA: 0 tokens, where the gateway's own
rendering of the same message gets an answer), the engine reported no decode speed and the
tune failed at its baseline. tune_chat_prompt renders through the gateway instead, so each
checkpoint gets its own template and the default reasoning a chat request gets.
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
import openai_server  # noqa: E402
from family_registry import FAMILIES  # noqa: E402

CODER_TEMPLATE = C_DIR / "tests" / "fixtures" / "qwen3_coder_chat_template.jinja"


class TuneChatPrompt(unittest.TestCase):
    def setUp(self):
        self.saved = (openai_server.ARCH, openai_server.CHAT_FLAVOR)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def tearDown(self):
        openai_server.ARCH, openai_server.CHAT_FLAVOR = self.saved

    def checkpoint(self, model_type, template=None):
        d = Path(self.tmp.name) / model_type
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")
        if template is not None:
            (d / "chat_template.jinja").write_bytes(template.read_bytes())
        return str(d)

    def test_qwen3_coder_gets_its_template_without_think(self):
        model = self.checkpoint("qwen3_moe", CODER_TEMPLATE)
        prompt = openai_server.tune_chat_prompt(model, "Write a palindrome check.")
        self.assertNotIn("<think>", prompt)
        self.assertTrue(prompt.endswith("Write a palindrome check.<|im_end|>\n<|im_start|>assistant\n"), prompt)
        openai_server.ARCH, openai_server.CHAT_FLAVOR = "qwen36", "qwen3_coder"
        self.assertEqual(prompt, openai_server.render_chat_qwen3_coder(
            [{"role": "user", "content": "Write a palindrome check."}]))

    def test_qwen36_gets_the_closed_block_a_chat_request_gets(self):
        model = self.checkpoint("qwen3_5_moe")
        with mock.patch.dict(os.environ, {"COLI_THINK": "0"}):
            prompt = openai_server.tune_chat_prompt(model, "hello")
        self.assertEqual(prompt, "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def test_coli_think_opens_the_block_for_qwen36_and_not_for_the_coder(self):
        with mock.patch.dict(os.environ, {"COLI_THINK": "1"}):
            self.assertTrue(openai_server.tune_chat_prompt(self.checkpoint("qwen3_5_moe"), "hello")
                            .endswith("<|im_start|>assistant\n<think>\n"))
            self.assertNotIn("<think>", openai_server.tune_chat_prompt(
                self.checkpoint("qwen3_moe", CODER_TEMPLATE), "hello"))

    def test_only_the_qwen36_family_renders_through_the_gateway(self):
        self.assertEqual([f.id for f in FAMILIES if f.tune_renders_chat], ["qwen36"])


if __name__ == "__main__":
    unittest.main()
