"""Responses named tool selection survives translation to Chat Completions."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("responses_shim", Path(__file__).resolve().parents[2] / "ui" / "responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


class NamedToolChoiceTest(unittest.TestCase):
    def test_named_function_is_forced(self):
        chat, _ = shim.to_chat({"input": "check", "tools": [{"type": "function", "name": "inspect"}],
                                "tool_choice": {"type": "function", "name": "inspect"}})
        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "inspect"}})

    def test_named_custom_tool_uses_its_function_wrapper(self):
        chat, custom = shim.to_chat({"input": "patch", "tools": [{"type": "custom", "name": "apply_patch"}],
                                     "tool_choice": {"type": "custom", "name": "apply_patch"}})
        self.assertIn("apply_patch", custom)
        self.assertEqual(chat["tool_choice"], {"type": "function", "function": {"name": "apply_patch"}})

    def test_existing_string_choices_are_preserved(self):
        for choice in ("auto", "none", "required"):
            with self.subTest(choice=choice):
                chat, _ = shim.to_chat({"input": "hello", "tool_choice": choice})
                self.assertEqual(chat["tool_choice"], choice)

    def test_omitted_choice_stays_omitted(self):
        chat, _ = shim.to_chat({"input": "hello"})
        self.assertNotIn("tool_choice", chat)


if __name__ == "__main__":
    unittest.main()
