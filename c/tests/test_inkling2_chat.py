"""Inkling-2 on the inkling engine: its family, its name, and the chat its template renders.

Inkling-2's config says model_type inkling2 (inkling2_text in a flat snapshot or a GGUF import);
the inkling family serves it and calls it Inkling-2-Small. Its chat_template.jinja is not
Inkling's: the effort level before the first message that is not a system one, reasoning as a
message of its own, names after the role tokens, and no generation prompt (the model opens its
own turn). The expected strings below are what thinkingmachines/Inkling-2-Small's
chat_template.jinja renders for the same messages (checked with jinja2), plus the opened text
block render_chat_inkling2 adds when thinking is off.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gw
from family_registry import default_model_id as registry_default_model_id, display_for, resolve_model

EFFORT0 = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|>"
OPEN = "<|message_model|><|content_text|>"


def render(messages, thinking=False, effort=None, **kw):
    return gw.render_chat_inkling2(messages, thinking, effort, **kw)


class Inkling2ChatTest(unittest.TestCase):
    def test_one_user_message_thinking_off_opens_the_text_block(self):
        self.assertEqual(render([{"role": "user", "content": "Hi"}]),
                         EFFORT0 + "<|message_user|><|content_text|>Hi<|end_message|>" + OPEN)

    def test_thinking_on_leaves_the_turn_to_the_model(self):
        self.assertEqual(
            render([{"role": "user", "content": "Hi"}], thinking=True),
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")
        self.assertEqual(
            render([{"role": "user", "content": "Hi"}], effort="xhigh"),
            "<|message_system|><|content_text|>Thinking effort level: 0.99<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_effort_after_the_leading_system_messages(self):
        self.assertEqual(
            render([{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}], effort="high"),
            "<|message_system|><|content_text|>Be brief.<|end_message|>"
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
            "<|message_user|><|content_text|>Hi<|end_message|>")

    def test_history_reasoning_names_and_parts(self):
        got = render([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b", "reasoning_content": "think"},
            {"role": "assistant", "name": "bob", "content": "", "reasoning_content": "only"},
            {"role": "tool", "name": "lookup", "content": "42"},
            {"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]},
        ], effort="medium")
        self.assertEqual(
            got,
            "<|message_system|><|content_text|>Thinking effort level: 0.7<|end_message|>"
            "<|message_user|><|content_text|>a<|end_message|>"
            "<|message_model|><|content_thinking|>think<|end_message|>"
            "<|message_model|><|content_text|>b<|end_message|><|content_model_end_sampling|>"
            "<|message_model|>bob<|content_thinking|>only<|end_message|><|content_model_end_sampling|>"
            "<|message_tool|>lookup<|content_text|>42<|end_message|>"
            "<|message_user|><|content_text|>one<|end_message|>"
            "<|message_user|><|content_text|>two<|end_message|>")

    def test_continuing_an_assistant_turn_leaves_it_open(self):
        self.assertEqual(
            render([{"role": "user", "content": "a"}, {"role": "assistant", "content": "partial"}],
                   effort="none", add_generation_prompt=False),
            EFFORT0 + "<|message_user|><|content_text|>a<|end_message|>"
            "<|message_model|><|content_text|>partial")

    def test_tools_and_images_are_refused(self):
        with self.assertRaises(gw.APIError):
            render([{"role": "user", "content": "x"}], tools=[{"type": "function", "function": {"name": "f"}}])
        with self.assertRaises(gw.APIError):
            render([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}])

    def test_the_reply_splits_into_reasoning_and_answer(self):
        text, reasoning = gw.split_inkling(
            "<|message_model|><|content_thinking|>let me see<|end_message|>"
            "<|message_model|><|content_text|>Paris.<|end_message|>")
        self.assertEqual((text, reasoning), ("Paris.", "let me see"))


class Inkling2FamilyTest(unittest.TestCase):
    def snapshot(self, config):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        (Path(tmp.name) / "config.json").write_text(json.dumps(config))
        return tmp.name

    def test_checkpoint_and_flat_configs(self):
        nested = self.snapshot({"model_type": "inkling2", "eos_token_id": 200006,
                                "text_config": {"hidden_size": 4608, "num_hidden_layers": 48}})
        flat = self.snapshot({"model_type": "inkling2_text", "hidden_size": 128})
        v1 = self.snapshot({"model_type": "inkling_mm_model", "text_config": {"model_type": "inkling"}})
        for d in (nested, flat):
            r = resolve_model(d)
            self.assertEqual(r.descriptor.id, "inkling")
            self.assertEqual(display_for(r), ("Inkling-2-Small", "594B"))
            self.assertEqual(registry_default_model_id(r), "inkling-2-small-colibri")
            self.assertEqual(gw.detect_chat_flavor("inkling", d), "inkling2")
        r = resolve_model(v1)
        self.assertEqual(display_for(r), ("Inkling", "975B"))
        self.assertEqual(registry_default_model_id(r), "inkling-colibri")
        self.assertIsNone(gw.detect_chat_flavor("inkling", v1))

    def test_the_gateway_renders_inkling2_for_an_inkling2_snapshot(self):
        saved = gw.ARCH, gw.CHAT_FLAVOR
        try:
            # thinking on: Inkling-2 ends at the user's message, Inkling hands the turn over
            gw.ARCH, gw.CHAT_FLAVOR = "inkling", "inkling2"
            got = gw.render_chat_for_arch([{"role": "user", "content": "Hi"}], True, None)
            self.assertTrue(got.endswith("Hi<|end_message|>"), got)
            gw.CHAT_FLAVOR = None
            got = gw.render_chat_for_arch([{"role": "user", "content": "Hi"}], True, None, audio_out=[])
            self.assertTrue(got.endswith("<|message_model|>"), got)
        finally:
            gw.ARCH, gw.CHAT_FLAVOR = saved


if __name__ == "__main__":
    unittest.main()
