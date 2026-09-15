"""The V4 MoE step must say why it failed.

moe_token_pipeline and v4_moe_batch_union used to fail in some thirty places
with a bare -1, and the block reported "hybrid batched block failed in MoE"
for all of them (#1464). Every failure site now records a reason through
moe_fail(), and both block tails append it. This pins that shape at source
level: the only `result = -1` left in those two functions are the job markers
(`jobs[i].result = -1`, "not finished yet"), and the block tails read
moe_reason(). A silent exit reintroduced by a later change fails here.
"""
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "deepseek_v4.c"


def function_body(text, signature):
    start = text.index(signature)
    end = text.index("\n}\n", start)
    return text[start:end]


class V4MoeReasonSourceTest(unittest.TestCase):
    def setUp(self):
        self.text = SOURCE.read_text(encoding="utf-8")

    def test_no_silent_failure_in_the_two_moe_functions(self):
        for signature in ("static int moe_token_pipeline(",
                          "static int v4_moe_batch_union("):
            body = function_body(self.text, signature)
            for line in body.splitlines():
                stripped = line.strip()
                if re.search(r"(^|[^.\w])result\s*=\s*-1;|^return -1;", stripped):
                    self.assertRegex(
                        stripped, r"^jobs?\[\w+\]\.result = -1;$|^job\.result = -1;$",
                        f"silent failure in {signature}: {stripped!r}")

    def test_both_functions_clear_the_reason_on_entry(self):
        for signature in ("static int moe_token_pipeline(",
                          "static int v4_moe_batch_union("):
            self.assertIn("moe_reason_clear();", function_body(self.text, signature))

    def test_block_tails_append_the_reason(self):
        self.assertIn('"hybrid batched block failed in %s: %s"', self.text)
        self.assertIn('"block computation failed in MoE: %s"', self.text)
        self.assertGreaterEqual(self.text.count("moe_reason()[0]"), 3)


if __name__ == "__main__":
    unittest.main()
