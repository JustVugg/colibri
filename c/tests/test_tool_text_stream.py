"""ToolTextStream: a streamed answer's text with a tool call's syntax kept out of it.

/v1/chat/completions and /v1/messages each had their own copy of this loop; they share the
class now, so what it does is pinned here once, at every way a reply can be cut into chunks.
"""
import json
import types
import unittest
from unittest.mock import patch

import openai_server as gw

CALL = "<tool_call>f<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {
    "type": "object", "properties": {"x": {"type": "integer"}}}}}]


def run(reply, cuts, tools=TOOLS, seen=False):
    out = []
    stream = gw.ToolTextStream(out.append, tools, types.SimpleNamespace(seen=seen))
    start = 0
    for cut in list(cuts) + [len(reply)]:
        stream.feed(reply[start:cut])
        start = cut
    calls = stream.finish()
    return "".join(out), calls


class ToolTextStreamTest(unittest.TestCase):
    def setUp(self):
        for name, value in (("ARCH", "glm"), ("CHAT_FLAVOR", None)):
            patcher = patch.object(gw, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_text_before_a_call_at_every_cut(self):
        reply = "Let me look. " + CALL + " trailing"
        for a in range(1, len(reply)):
            for b in (a, min(len(reply) - 1, a + 3)):
                with self.subTest(cuts=(a, b)):
                    self.assertEqual(run(reply, sorted({a, b})), ("Let me look. ", None))

    def test_a_reply_with_no_call_is_all_text_at_every_cut(self):
        reply = "A < b, and <tool is not a call: <tool_cal"   # a tail that looks like a marker's start
        for a in range(1, len(reply)):
            with self.subTest(cut=a):
                self.assertEqual(run(reply, [a]), (reply, None))

    def test_nothing_is_held_once_the_side_channel_has_the_calls(self):
        out = []
        stream = gw.ToolTextStream(out.append, TOOLS, types.SimpleNamespace(seen=True))
        stream.feed("text <tool")
        self.assertEqual(out, ["text <tool"])          # at once, no tail held
        self.assertIsNone(stream.finish())

    def test_the_qwen_form_gives_its_calls(self):
        reply = ("<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>")
        with patch.object(gw, "ARCH", "qwen36"), patch.object(gw, "CHAT_FLAVOR", "qwen38"):
            for a in range(1, len(reply)):
                with self.subTest(cut=a):
                    text, calls = run(reply, [a])
                    self.assertEqual(text, "")
                    self.assertEqual([json.loads(c["function"]["arguments"]) for c in calls], [{"x": 1}])


if __name__ == "__main__":
    unittest.main()
