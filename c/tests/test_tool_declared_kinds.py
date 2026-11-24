"""One reader for a tool parameter's declared types, behind every family's tool-call parser.

The schema is the one pydantic / FastMCP write for
    f(sku: str | None, n: int | None, box: Box | None, opts: Box, old: Box = ...)
where the types sit in `anyOf`, `$ref` and `allOf` and no parameter has a `type` of its own.
A string of digits must stay a string and a number, an object must come back as one, in the
GLM form (both parsers), the Qwen form, MiMo's and Muse Spark's.
"""
import json
import unittest

import openai_server as gw

REF = {"$ref": "#/$defs/Box"}
PARAMETERS = {
    "type": "object",
    "$defs": {"Box": {"type": "object", "properties": {"w": {"type": "integer"}}}},
    "properties": {
        "sku": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
        "n": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": None},
        "box": {"anyOf": [REF, {"type": "null"}], "default": None},
        "opts": REF,
        "old": {"allOf": [REF], "default": {}}},
    "required": ["sku"]}
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": PARAMETERS}}]
WRITTEN = [("sku", "12345"), ("n", "5"), ("box", '{"w": 1}'), ("opts", '{"w": 2}'), ("old", "{}")]
EXPECTED = {"sku": "12345", "n": 5, "box": {"w": 1}, "opts": {"w": 2}, "old": {}}


def glm(written):
    return "<tool_call>f" + "".join(
        f"<arg_key>{k}</arg_key><arg_value>{v}</arg_value>" for k, v in written) + "</tool_call>"


def qwen(written):
    return "<tool_call>\n<function=f>\n" + "".join(
        f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in written) + "</function>\n</tool_call>"


def mimo(written):
    return "<tool_call><function=f>" + "".join(
        f"<parameter={k}>{v}</parameter>" for k, v in written) + "</function></tool_call>"


def muse(written):
    return '<atem:function_calls>\n<atem:invoke name="f">\n' + "".join(
        f'<atem:parameter name="{k}">{v}</atem:parameter>\n' for k, v in written) + \
        "</atem:invoke>\n</atem:function_calls>"


FORMS = ((gw.parse_tool_calls, glm), (gw.parse_glm_tool_calls_strict, glm),
         (gw.parse_qwen_tool_calls, qwen), (gw.parse_mimo_tool_calls, mimo),
         (gw.parse_muse_tool_calls, muse))


class DeclaredKindsTest(unittest.TestCase):
    def test_the_reader(self):
        root = {"$defs": {"Box": {"type": "object"}, "Loop": {"$ref": "#/$defs/Loop"},
                          "a/b": {"type": "array"}}}
        for spec, kinds in (
                ({"type": "integer"}, ["integer"]),
                ({"type": ["string", "null"]}, ["string", "null"]),
                ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, ["integer", "null"]),
                ({"oneOf": [{"type": ["array", "object"]}, {"type": "null"}]}, ["array", "object", "null"]),
                ({"anyOf": [{"$ref": "#/$defs/Box"}, {"type": "null"}]}, ["object", "null"]),
                ({"anyOf": [{"type": "integer"}, {}]}, ["integer", None]),   # a branch for any value
                ({"allOf": [{"description": "x"}, {"$ref": "#/$defs/Box"}]}, ["object"]),
                ({"$ref": "#/$defs/Box"}, ["object"]),
                ({"$ref": "#/$defs/a~1b"}, ["array"]),
                ({"$ref": "#/$defs/Missing"}, None), ({"$ref": "#/$defs/Loop"}, None),
                ({"$ref": "https://example.com/schema.json"}, None),
                ({"description": "anything"}, None), ({"anyOf": []}, None), ("integer", None), (None, None)):
            with self.subTest(spec=spec):
                self.assertEqual(gw._declared_kinds(spec, root), kinds)

    def test_every_form_types_the_same_schema_the_same_way(self):
        for parse, write in FORMS:
            with self.subTest(parse=parse.__name__):
                calls = parse(write(WRITTEN), TOOLS)[1]
                self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [EXPECTED])


if __name__ == "__main__":
    unittest.main()
