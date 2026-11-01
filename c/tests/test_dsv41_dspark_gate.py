"""DeepSeek V4.1's DSpark verify depth from the head's confidence (spec_draft.h's
spec_gate_pick_p, DeepSeek's confidence-scheduled verification): it changes how many drafts
a round verifies, never a token. The tiny fixture (tools/make_dsv41_tiny.py) carries two
DSpark stages with random weights; greedy decoding must give the same bytes with the head
off, with the gate (the default) and without it (V41_DSPARK_GATE=0), and the gate must have
declined some rounds or verified fewer drafts than the head wrote, which says it ran.
"""
import os
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))
PROMPT = bytes((5 * i + 11) % 190 + 33 for i in range(60))


def run(**env):
    base = {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
            "V41_STATS": "1", "COLI_VULKAN": "0", "V41_DSPARK_MINACC": "0"}
    base.update(env)
    engine = ServeEngine(ENGINE, ["8"], base, reuse=False)
    try:
        text = engine.ask("g", PROMPT, 48)
    finally:
        err = engine.close()
        engine.p.stdout.close(); engine.p.stderr.close()
    return text, err


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists(),
                     "built V4.1 engine and tiny fixture required")
class DsparkGateTest(unittest.TestCase):
    def test_the_gate_changes_no_token(self):
        plain, _ = run(V41_DSPARK="0")
        gated, err = run(V41_DSPARK="1")
        full, err_full = run(V41_DSPARK="1", V41_DSPARK_GATE="0")
        self.assertEqual(gated, plain)
        self.assertEqual(full, plain)
        self.assertIn("[v41] DSpark gate:", err)
        self.assertNotIn("[v41] DSpark gate:", err_full)

    def test_the_gate_verifies_fewer_drafts_than_written(self):
        _, err = run(V41_DSPARK="1")
        _, err_full = run(V41_DSPARK="1", V41_DSPARK_GATE="0")
        prop = [int(x) for x in re.findall(r"DSpark: \d+ of (\d+) drafts accepted this turn", err)]
        prop_full = [int(x) for x in re.findall(r"DSpark: \d+ of (\d+) drafts accepted this turn", err_full)]
        declined = [int(x) for x in re.findall(r"(\d+) rounds took a plain step", err)]
        self.assertTrue(prop_full and prop_full[-1] > 0, err_full[-2000:])
        # the random head is wrong almost always: the gate learns it and verifies less
        self.assertTrue((prop and prop[-1] < prop_full[-1]) or (declined and declined[-1] > 0), err[-2000:])

    def test_the_guard_counts_the_drafts_the_gate_left_unverified(self):
        # the stages read their experts for every draft written; a head the gate keeps declining
        # is a cost with nothing to show, so the acceptance guard must still pause it
        plain, _ = run(V41_DSPARK="0")
        text, err = run(V41_DSPARK="1", V41_DSPARK_MINACC="60")
        self.assertEqual(text, plain)
        self.assertIn("pausing for 64 tokens", err)
        written = [int(x) for x in re.findall(r"drafts accepted this turn \((\d+) written\)", err)]
        self.assertTrue(written and written[-1] >= 10, err[-2000:])


if __name__ == "__main__":
    unittest.main()
