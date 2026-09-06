"""Drift canary for the absorb-path fmt-refusal message shape.

tests/test_fp8_serve_batch_e2e.py machine-matches the engine's death on the
batched fmt=8 serve path with one regex (its REFUSAL constant, the family
`(qt_addrow|qt_matvec_rows): unsupported fmt=`), in two load-bearing
places: the pass lane's engine-death scan, and the expect-bite mode the
fleet's old-binary half PASSES ONLY THROUGH.  The product strings live in
c/colibri.c (qt_addrow's refusal and qt_matvec_rows'); an innocent reword
of either -- "unsupported" to "unhandled", dropping the function-name
prefix -- would not break any build or unit suite, but it would silently
degrade the bite proof's machine-match into a generic "engine died", and
the e2e test only runs where a real fmt=8 container exists (never in CI).
This canary closes that gap in CI: it fails, by name, the moment the
matcher and the source strings drift apart, in either direction.

What is load-bearing is exactly what this file asserts, no more:
  * each function still has at least one refusal-class stderr message
    (its runtime firing is pinned separately by tests/test_qt_addrow.c's
    fork+waitpid refusal cases, which also require the "refus" word);
  * at least ONE refusal-class message from each function matches the e2e
    REFUSAL regex -- imported from the e2e module, never re-typed, so the
    two files cannot agree by coincidence (existential, not universal: a
    future second, differently-worded guard in the same function is a new
    refusal, not drift, and must not fail this canary);
  * the regex is still SELECTIVE -- it must not match an arbitrary death
    line, or expect-bite would file any crash as the bite.
Everything else about the messages (wording after the matched prefix,
fmt lists, line breaks) is deliberately unpinned: message edits that keep
the matchable shape must stay free.
"""
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # sibling import under any invocation style
from test_fp8_serve_batch_e2e import REFUSAL

COLIBRI_C = HERE.parent / "colibri.c"
FUNCTIONS = ("qt_addrow", "qt_matvec_rows")

# One C string literal (escapes included), and an fprintf-to-stderr whose
# message is one or more adjacent literals (the refusals wrap across lines).
_STRING = r'"(?:[^"\\\n]|\\.)*"'
_FPRINTF = re.compile(r'fprintf\s*\(\s*stderr\s*,\s*(' + _STRING + r'(?:\s*' + _STRING + r')*)')


def stderr_messages(source):
    """Every fprintf(stderr, ...) format string in `source`, with adjacent
    literals concatenated (still escaped -- the matched prefix contains no
    escapes, so matching against the raw literal is faithful to what the
    runtime prints before the %d substitution)."""
    messages = []
    for match in _FPRINTF.finditer(source):
        parts = re.findall(_STRING, match.group(1))
        messages.append("".join(part[1:-1] for part in parts))
    return messages


class RefusalShapeCanaryTest(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        cls.messages = stderr_messages(COLIBRI_C.read_text(encoding="utf-8"))

    def messages_of(self, function):
        return [m for m in self.messages if m.startswith(function + ":")]

    def test_each_function_still_names_a_refusal_the_matcher_greps(self):
        for function in FUNCTIONS:
            named = self.messages_of(function)
            self.assertTrue(
                named,
                f"colibri.c has no stderr message starting '{function}:' -- the "
                f"refusal site was renamed or removed, and the D-I2 bite matcher "
                f"(test_fp8_serve_batch_e2e.REFUSAL) can no longer identify this "
                f"death; update the matcher and the fleet bite lane together")
            refusals = [m for m in named if "refus" in m]
            self.assertTrue(
                refusals,
                f"{function}: no refusal-class stderr message left (the 'refus' "
                f"discipline tests/test_qt_addrow.c pins at runtime) -- if the "
                f"refusal moved, this canary and the e2e matcher must follow it")
            self.assertTrue(
                any(REFUSAL.search(m) for m in refusals),
                f"none of {function}'s refusal messages matches the e2e "
                f"REFUSAL regex {REFUSAL.pattern!r} -- an innocent reword "
                f"silently turns the fleet bite proof's machine-match into a "
                f"generic 'engine died'; keep one matchable refusal per "
                f"function or change test_fp8_serve_batch_e2e.REFUSAL in the "
                f"same commit.  (Existential on purpose: an ADDITIONAL, "
                f"differently-worded guard in this function is new coverage, "
                f"not drift.)\nmessages: {refusals}")

    def test_matcher_stays_selective(self):
        """The other drift direction: a loosened REFUSAL regex would make
        expect-bite accept ANY death as the bite.  Pin that it rejects a
        representative non-refusal death line and the empty string, and
        that it keys on the FUNCTION FAMILY, not the bare `unsupported
        fmt=` words -- a matcher loosened to the words alone would accept
        any future non-absorb `unsupported fmt=` message as the bite."""
        self.assertIsNone(REFUSAL.search("malloc: out of memory allocating 42 GB"))
        self.assertIsNone(REFUSAL.search("colibri engine exited unexpectedly"))
        self.assertIsNone(REFUSAL.search(""))
        self.assertIsNone(REFUSAL.search("qt_resolve_fmt: unsupported fmt=9 refusing"))
        # The REAL in-tree near-miss (not synthetic): layer_cuda_shard_kvb's
        # refusal shares the `unsupported ... fmt=` words but is a different
        # guard family (shard-layout, pinned by tests/test_shard_kvb_refuse.c)
        # -- a loosened matcher that swallowed it would misfile a shard
        # refusal as the absorb bite in a fleet lane.
        self.assertIsNone(REFUSAL.search(
            "layer_cuda_shard_kvb: unsupported kv_b fmt=5 for the head-shard "
            "upload (only fmt 1/2/3/4 match the per-row byte/scale strides "
            "computed here) -- refusing the shard"))


if __name__ == "__main__":
    unittest.main()
