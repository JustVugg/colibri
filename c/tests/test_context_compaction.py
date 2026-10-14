"""Context compaction's pieces, and the context a chat server starts with.

The end-to-end path (the gateway, a mock engine, HTTP) is test_context_compaction_e2e.py.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import openai_server as gateway  # noqa: E402
from resource_plan import GB, chat_context  # noqa: E402

# Planning keys of Qwen3.6-35B-A3B: ten attention layers, 40 KiB of KV a token
QWEN36_35B_CONFIG = {
    "model_type": "qwen3_5_moe_text",
    "num_hidden_layers": 40,
    "num_experts": 256,
    "num_experts_per_tok": 8,
    "hidden_size": 2048,
    "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(40)],
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def turn(role, text):
    return {"role": role, "content": text}


class CompactionPiecesTest(unittest.TestCase):
    def test_the_summary_joins_the_system_message(self):
        lead = [turn("system", "Be brief.")]
        merged = gateway.with_summary(lead, "S")
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["content"].startswith("Be brief.\n\n"))
        self.assertTrue(merged[0]["content"].endswith(gateway.COMPACT_NOTE + "S"))
        self.assertEqual(lead[0]["content"], "Be brief.")          # the request is not edited

    def test_without_a_system_message_the_summary_is_one(self):
        merged = gateway.with_summary([], "S")
        self.assertEqual(merged, [turn("system", gateway.COMPACT_NOTE + "S")])
        parts = gateway.with_summary([{"role": "system", "content": [{"type": "text", "text": "A"}]}], "S")
        self.assertEqual(parts[0]["content"][-1]["text"], "\n\n" + gateway.COMPACT_NOTE + "S")

    def test_the_tail_starts_at_a_user_turn_and_keeps_the_new_one(self):
        conversation = [turn("user", "a" * 400), turn("assistant", "b" * 400),
                        turn("user", "c" * 40), turn("assistant", "d" * 40), turn("user", "e")]
        # room for the last three messages: the tail starts at the user turn before them
        self.assertEqual(gateway.compaction_tail_start(conversation, 0, 0.25, 100), 2)
        # room for nothing: the new request alone
        self.assertEqual(gateway.compaction_tail_start(conversation, 0, 0.25, 1), 4)
        # everything before the request already summarized: nothing left to do
        self.assertIsNone(gateway.compaction_tail_start(conversation, 4, 0.25, 1))
        self.assertIsNone(gateway.compaction_tail_start([turn("user", "only")], 0, 0.25, 1))

    def test_a_picture_counts_and_becomes_a_mark(self):
        message = {"role": "user", "content": [{"type": "text", "text": "look"},
                                               {"type": "image_url", "image_url": {"url": "data:"}}]}
        self.assertGreater(gateway.estimate_message_tokens(message, 0.25),
                           gateway.COMPACT_IMAGE_TOKENS)
        self.assertEqual(gateway.without_images(message)["content"][1],
                         {"type": "text", "text": "[image]"})

    def test_an_open_reasoning_block_is_closed(self):
        self.assertEqual(gateway.close_open_thinking("x<think>\n"),
                         ("x<think>\n\n</think>\n\n", True))
        self.assertEqual(gateway.close_open_thinking("x<think>"), ("x<think></think>", True))
        self.assertEqual(gateway.close_open_thinking("x<think></think>"), ("x<think></think>", False))

    def test_memory_finds_the_longest_summarized_prefix(self):
        lead = [turn("system", "s")]
        conversation = [turn("user", str(i)) for i in range(6)]
        keys = gateway.compaction_keys(lead, conversation)
        memory = gateway.CompactionMemory(capacity=2)
        self.assertEqual(memory.find(keys), (0, None))
        memory.remember(keys[2], "two")
        memory.remember(keys[4], "four")
        self.assertEqual(memory.find(keys), (4, "four"))
        # another system message is another conversation
        self.assertEqual(memory.find(gateway.compaction_keys([turn("system", "t")], conversation)),
                         (0, None))
        memory.remember(gateway.compaction_keys(lead, [turn("user", "x")])[1], "x")
        self.assertEqual(memory.find(keys[:3]), (0, None))     # capacity 2: "two" went first

    def test_the_opt_in(self):
        self.assertTrue(gateway.compaction_requested({"context_compaction": "auto"}))
        self.assertFalse(gateway.compaction_requested({"context_compaction": "off"}))
        self.assertFalse(gateway.compaction_requested({}))
        with self.assertRaises(gateway.APIError):
            gateway.compaction_requested({"context_compaction": "always"})

    def test_the_engine_error_carries_its_numbers(self):
        error = gateway._engine_error(
            ["CONTEXT_EXCEEDED", "prompt_tokens=9000", "requested=64", "capacity=8192"], "")
        self.assertEqual(error.context, (9000, 8192))
        self.assertEqual(gateway._engine_error(["CONTEXT_EXCEEDED", "70", "64"], "").context, (70, 64))


class ChatContextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        (Path(self.tmp.name) / "config.json").write_text(json.dumps(QWEN36_35B_CONFIG))
        self.model = self.tmp.name

    def test_the_step_follows_the_memory(self):
        # 40 KiB a token: 16384 is 0.67 GB, 32768 1.34 GB, 65536 2.68 GB (up to a quarter of
        # the memory, what agents need), 131072 5.37 GB and 262144 10.7 GB (a tenth)
        for available, expected in ((2 * GB, None), (4 * GB, 16384), (8 * GB, 32768),
                                    (16 * GB, 65536), (32 * GB, 65536), (64 * GB, 131072),
                                    (512 * GB, 262144)):
            with self.subTest(gb=available / GB):
                context, _why = chat_context(self.model, {}, available=available)
                self.assertEqual(context, expected)

    def test_every_kv_slot_holds_its_own(self):
        context, _ = chat_context(self.model, {}, kv_slots=4, available=32 * GB)
        self.assertEqual(context, 32768)

    def test_ram_gb_is_the_budget_and_a_set_context_stays(self):
        self.assertEqual(chat_context(self.model, {"RAM_GB": "16"}, available=512 * GB)[0], 65536)
        context, why = chat_context(self.model, {"Q36_MAXT": "4096"}, available=512 * GB)
        self.assertIsNone(context)
        self.assertIn("Q36_MAXT", why)

    def test_the_gateway_sets_it_for_the_engine(self):
        from family_registry import resolve_model
        family = resolve_model(self.model).descriptor
        env = {"RAM_GB": "16"}
        self.assertEqual(gateway.chat_context_env(env, family, self.model), 65536)
        self.assertEqual(env["Q36_MAXT"], "65536")
        env = {"Q36_MAXT": "12000"}
        self.assertEqual(gateway.chat_context_env(env, family, self.model), 12000)


if __name__ == "__main__":
    unittest.main()
