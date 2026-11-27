"""Inkling-2, Kimi K3 and Mistral Large 4 tool arguments must be strict JSON.

Python's JSON reader accepts NaN, Infinity and overflowing exponents such as 1e999, and its
writer prints them back as NaN and Infinity, which strict clients cannot decode. The Muse,
GLM, Qwen and MiMo parsers already keep such a value out of `arguments` (#2551, #3051).
"""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import openai_server as gw

TOOLS = [{"type": "function", "function": {"name": "inspect", "parameters": {
    "type": "object", "properties": {"value": {"type": "number"}}}}}]
NONFINITE = ("NaN", "Infinity", "-Infinity", "1e999", "[1, NaN]", '{"x": 1e999}')


def _reject(token):
    raise ValueError("non-JSON numeric constant " + token)


def _strict(arguments):
    return json.loads(arguments, parse_constant=_reject)


def _inkling2(raw):
    return gw.parse_inkling2_tool_calls(
        gw.INK2_CALL_OPEN + '{"name": "inspect", "args": {"value": ' + raw + "}}"
        + gw.INK2_CALL_CLOSE, TOOLS)


def _k3(raw, kind="number"):
    return gw.parse_k3_tool_calls(
        gw.K3_TOOLS_OPEN + '<|open|>call tool="inspect" index="0"<|sep|>'
        '<|open|>argument key="value" type="' + kind + '"<|sep|>' + raw
        + "<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>", TOOLS)


def _mistral(raw):
    return gw.parse_mistral_tool_calls(
        gw.ML4_CALLS + "inspect" + gw.ML4_ARGS + '{"value": ' + raw + "}", TOOLS)


class ToolArgsNonfiniteTest(unittest.TestCase):
    def test_inkling2_drops_a_call_whose_arguments_are_not_json(self):
        for raw in NONFINITE:
            with self.subTest(raw=raw):
                _, calls = _inkling2(raw)
                self.assertEqual(calls, [])

    def test_k3_keeps_a_nonfinite_value_as_its_text(self):
        for raw in NONFINITE:
            with self.subTest(raw=raw):
                _, calls = _k3(raw)
                self.assertEqual(_strict(calls[0]["function"]["arguments"]), {"value": raw})

    def test_mistral_leaves_the_block_as_text(self):
        for raw in NONFINITE:
            with self.subTest(raw=raw):
                text, calls = _mistral(raw)
                self.assertEqual(calls, [])
                self.assertIn(gw.ML4_CALLS, text)

    def test_finite_values_keep_their_types(self):
        for raw, value in (("2.5", 2.5), ("-3", -3), ("[1, 2]", [1, 2]), ('{"x": null}', {"x": None})):
            with self.subTest(raw=raw):
                for parse in (_inkling2, _k3, _mistral):
                    _, calls = parse(raw)
                    self.assertEqual(_strict(calls[0]["function"]["arguments"]), {"value": value})


if __name__ == "__main__":
    unittest.main()
