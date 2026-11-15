"""Muse tool arguments must be strict JSON, including nested numeric values."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gw


class MuseFiniteArgumentsTest(unittest.TestCase):
    def arguments(self, raw, kind):
        tools = [{"type": "function", "function": {"name": "inspect", "parameters": {
            "type": "object", "properties": {"value": {"type": kind}}}}}]
        reply = ('<atem:function_calls><atem:invoke name="inspect">'
                 '<atem:parameter name="value">' + raw + '</atem:parameter>'
                 '</atem:invoke></atem:function_calls>')
        _, calls = gw.parse_muse_tool_calls(reply, tools)
        def reject(token):
            raise ValueError("non-JSON numeric constant " + token)
        return json.loads(calls[0]["function"]["arguments"], parse_constant=reject)["value"]

    def test_nonfinite_and_overflow_values_use_existing_text_fallback(self):
        for raw in ("NaN", "Infinity", "-Infinity", "1e999"):
            with self.subTest(raw=raw):
                self.assertEqual(self.arguments(raw, "number"), raw)

    def test_nonfinite_values_nested_in_containers_do_not_escape(self):
        for raw, kind in (("[NaN]", "array"), ('{"x":1e999}', "object")):
            with self.subTest(raw=raw):
                self.assertEqual(self.arguments(raw, kind), raw)

    def test_valid_json_keeps_its_types(self):
        for raw, kind, expected in (("2.5", "number", 2.5), ("[1,2]", "array", [1, 2]),
                                    ('{"x":null}', "object", {"x": None}), ("true", "boolean", True)):
            with self.subTest(raw=raw):
                self.assertEqual(self.arguments(raw, kind), expected)

    def test_string_arguments_named_nan_remain_literal_text(self):
        self.assertEqual(self.arguments("NaN", "string"), "NaN")


if __name__ == "__main__":
    unittest.main()
