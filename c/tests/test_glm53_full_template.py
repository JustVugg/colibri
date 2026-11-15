"""The full GLM-5.3 uses its template contract on the shared glm engine."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gw
from test_openai_server import ScriptedEngine, _spawn_test_server, _post_chat

# Signature from zai-org/GLM-5.3's official chat_template.jinja. The full
# checkpoint shares the GLM-5.2 model_type, so config alone is insufficient.
GLM53_TEMPLATE = """[gMASK]<sop>
{%- set effective_reasoning_effort = reasoning_effort if reasoning_effort is defined and reasoning_effort in ['low', 'high'] else 'max' -%}
"""


class Glm53FullTemplateTest(unittest.TestCase):
    def detect(self, template, name="chat_template.jinja", model_type="glm_moe_dsa"):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps({"model_type": model_type}))
            (root / name).write_text(template)
            return gw.detect_chat_flavor("glm", root)

    def test_full_glm53_is_identified_from_its_template(self):
        self.assertEqual(self.detect(GLM53_TEMPLATE), "glm53_full")
        self.assertEqual(self.detect(json.dumps({"chat_template": GLM53_TEMPLATE}), "tokenizer_config.json"), "glm53_full")

    def test_glm52_and_mistral_are_not_reclassified(self):
        glm52 = "[gMASK]<sop> {% if enable_thinking is not defined or enable_thinking %}"
        self.assertIsNone(self.detect(glm52))
        self.assertEqual(self.detect(GLM53_TEMPLATE, model_type="mistral_large4"), "mistral_large4")
        with tempfile.TemporaryDirectory(prefix="GLM-5.3-") as tmp:
            self.assertIsNone(gw.detect_chat_flavor("glm", tmp))

    def test_http_default_is_max_with_open_thinking_and_separate_answer(self):
        engine = ScriptedEngine(chunks=("work</think>answer",))
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", "glm53_full"), patch.dict(os.environ, {"COLI_THINK": "0"}):
            base = _spawn_test_server(self, engine)
            with _post_chat(base, {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}) as response:
                message = json.load(response)["choices"][0]["message"]
            self.assertEqual(engine.calls[0][0], "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>")
            self.assertEqual(message["content"], "answer")
            self.assertEqual(message["reasoning_content"], "work")

    def test_explicit_no_thinking_uses_minimum_supported_effort(self):
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", "glm53_full"):
            for effort in (None, "none"):
                prompt = gw.render_chat_for_arch([{"role": "user", "content": "hi"}], False, effort)
                self.assertIn("Reasoning Effort: Low", prompt)
                self.assertTrue(prompt.endswith("<|assistant|><think>"))
            self.assertTrue(gw.starts_in_reasoning(False))
            self.assertFalse(gw.starts_in_reasoning(False, False))

    def test_parallel_results_follow_call_ids_with_one_observation_block(self):
        calls = [{"id": "a", "function": {"name": "inspect", "arguments": {"path": "A"}}},
                 {"id": "b", "function": {"name": "inspect", "arguments": {"path": "B"}}}]
        messages = [{"role": "user", "content": "inspect"},
                    {"role": "assistant", "content": "", "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "b", "content": "found B"},
                    {"role": "tool", "tool_call_id": "a", "content": "found A"}]
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", "glm53_full"):
            prompt = gw.render_chat_for_arch(messages, True, "low")
            self.assertLess(prompt.index("<tool_response>found A"), prompt.index("<tool_response>found B"))
            self.assertEqual(prompt.count("<|observation|>"), 1)
            self.assertIn("</think>\n<tool_call>", prompt)
            messages[-1]["tool_call_id"] = "b"  # ambiguous duplicates stay in input order
            prompt = gw.render_chat_for_arch(messages, True, "low")
            self.assertLess(prompt.index("<tool_response>found B"), prompt.index("<tool_response>found A"))

    def test_explicit_efforts_and_continuation_keep_the_contract(self):
        with patch.object(gw, "ARCH", "glm"), patch.object(gw, "CHAT_FLAVOR", "glm53_full"):
            for effort, word in (("minimal", "Low"), ("low", "Low"), ("medium", "High"), ("high", "High"), ("xhigh", "Max")):
                prompt = gw.render_chat_for_arch([{"role": "user", "content": "hi"}], True, effort)
                self.assertIn("Reasoning Effort: " + word, prompt)
            prompt = gw.render_chat_for_arch([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer", "reasoning_content": "work"}],
                True, "low", add_generation_prompt=False)
            self.assertTrue(prompt.endswith("<|assistant|><think>work</think>answer"))


if __name__ == "__main__":
    unittest.main()
