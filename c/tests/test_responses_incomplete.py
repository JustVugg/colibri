"""A truncated Chat Completion must not become a successful Responses completion."""
import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("responses_shim", Path(__file__).resolve().parents[2] / "ui" / "responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


class IncompleteResponseTest(unittest.TestCase):
    def events(self, finish_reason):
        e = shim.Events("model", set())
        e.chunk({"choices": [{"delta": {"content": "partial answer"}, "finish_reason": None}]})
        e.chunk({"choices": [{"delta": {}, "finish_reason": finish_reason}]})
        e.chunk({"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
        events = [json.loads(line[6:]) for line in e.finish().splitlines() if line.startswith("data: ")]
        return e, events[-1]

    def test_length_limit_is_incomplete_in_both_response_shapes(self):
        e, last = self.events("length")
        self.assertEqual(last["type"], "response.incomplete")
        for response in (last["response"], e.response("completed")):
            self.assertEqual(response["status"], "incomplete")
            self.assertEqual(response["incomplete_details"], {"reason": "max_output_tokens"})
            self.assertEqual(response["usage"]["total_tokens"], 5)
            self.assertEqual(response["output"][0]["content"][0]["text"], "partial answer")
            self.assertEqual(response["output"][0]["status"], "incomplete")

    def test_content_filter_reason_is_preserved(self):
        _, last = self.events("content_filter")
        self.assertEqual(last["type"], "response.incomplete")
        self.assertEqual(last["response"]["incomplete_details"]["reason"], "content_filter")

    def test_truncated_tool_arguments_remain_incomplete(self):
        e = shim.Events("model", set())
        e.chunk({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_test", "function": {
            "name": "inspect", "arguments": '{"path":'}}]}, "finish_reason": "length"}]})
        events = [json.loads(line[6:]) for line in e.finish().splitlines() if line.startswith("data: ")]
        item = events[-1]["response"]["output"][0]
        self.assertEqual(item["arguments"], '{"path":')
        self.assertEqual(item["status"], "incomplete")

    def test_normal_and_tool_call_finishes_remain_completed(self):
        for reason in ("stop", "tool_calls"):
            with self.subTest(reason=reason):
                _, last = self.events(reason)
                self.assertEqual(last["type"], "response.completed")
                self.assertEqual(last["response"]["status"], "completed")
                self.assertNotIn("incomplete_details", last["response"])


if __name__ == "__main__":
    unittest.main()
