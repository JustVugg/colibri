"""Safety and meaning of the offline human review page."""
import json
import unittest
from examples.atlas_evidence.core import snapshot, build_capture
from examples.atlas_evidence.render import render_html


class RenderTest(unittest.TestCase):
    def capture(self):
        text = '<script>alert("source")</script> is untrusted source content.'
        source = snapshot("source", text, "<img src=x onerror=alert(1)>")
        candidate = {"id": "claim", "text": '<script>alert("claim")</script>', "source_id": "source",
                     "quote": text, "start": 0, "end": len(text)}
        response = {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"claims": [candidate]})}}]}
        return build_capture([source], {"model": "fixture"}, response)

    def test_model_source_and_title_cannot_insert_markup(self):
        html = render_html(self.capture())
        self.assertNotIn("<script>", html)
        self.assertNotIn("<img", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertIn("Unreviewed", html)
        self.assertIn("Exact passage matched", html)

    def test_drift_is_visible_and_does_not_claim_support(self):
        html = render_html(self.capture(), [snapshot("source", "Changed source")])
        self.assertIn("Review required", html)
        self.assertIn("Changed supplied text", html)
        self.assertIn("Check failed", html)
        self.assertNotIn("Exact passage matched</span>", html)

    def test_missing_source(self):
        html = render_html(self.capture(), [])
        self.assertIn("Missing current source", html)


if __name__ == "__main__":
    unittest.main()
