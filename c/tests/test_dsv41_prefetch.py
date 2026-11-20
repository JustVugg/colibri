"""DeepSeek V4.1 reads the experts it guesses for the next layer on a worker thread
(V41_PREFETCH, on by default) while the main thread finishes the layer and runs the next
layer's attention. The guess only decides
which slots get filled early; the routing and the matmuls are the ones that would have run,
so the logits must be the same bits with and without it, and the worker must have read
something for the test to mean anything. DUMP writes the logits rows of every forward.
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))
PROMPT = bytes((7 * i + 3) % 200 + 32 for i in range(40))


def run(prefetch, cap="4", tokens=24, floor=None):
    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "logits.bin")
        env = {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
               "V41_DSPARK": "0", "COLI_VULKAN": "0", "DUMP": dump, "V41_STATS": "1",
               "V41_PREFETCH": "1" if prefetch else "0"}
        if prefetch is None:
            del env["V41_PREFETCH"]
        if floor is not None:
            env["V41_PREFETCH_MIN"] = str(floor)
        engine = ServeEngine(ENGINE, [cap], env, reuse=False)
        try:
            text = engine.ask("t", PROMPT, tokens)
        finally:
            err = engine.close()
            engine.p.stdout.close(); engine.p.stderr.close()
        with open(dump, "rb") as f:
            data = f.read()
    found = re.findall(r"prefetch: (\d+) experts guessed off the caches, (\d+) read ahead, (\d+) of them routed", err)
    return text, data, (tuple(map(int, found[-1])) if found else None), err


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists(),
                     "built V4.1 engine and tiny fixture required")
class PrefetchTest(unittest.TestCase):
    def test_reading_ahead_gives_the_same_bits(self):
        text0, dump0, stats0, err0 = run(False)
        text1, dump1, stats1, err1 = run(True)
        self.assertGreaterEqual(len(dump0), 24 * 4 * 256, err0[-2000:])
        self.assertIsNone(stats0, "V41_PREFETCH=0 must turn the worker off")
        self.assertIsNotNone(stats1, err1[-2000:])
        guessed, read, used = stats1
        self.assertGreater(read, 0, err1[-2000:])
        self.assertLessEqual(used, read)
        self.assertEqual(dump0, dump1)
        self.assertEqual(text0, text1)

    def test_a_rank_that_guesses_badly_stops_being_read(self):
        # on this fixture rank 1 guesses about 73% and rank 2 about 51% over 200 tokens:
        # a 60% floor stops reading rank 2 once it has a record, a 0 floor reads both all along
        text0, dump0, _, _ = run(False, tokens=200)
        text1, dump1, all_ranks, err1 = run(True, tokens=200, floor=0)
        text2, dump2, floored, err2 = run(True, tokens=200, floor=60)
        self.assertEqual(dump0, dump1)
        self.assertEqual(dump0, dump2)
        self.assertEqual(text0, text2)
        self.assertLess(floored[1], all_ranks[1], err2[-1500:])
        self.assertGreater(floored[2] / floored[1], all_ranks[2] / all_ranks[1])

    def test_it_is_on_by_default(self):
        text0, dump0, _, _ = run(False)
        text1, dump1, stats1, err1 = run(None)
        self.assertIsNotNone(stats1, err1[-2000:])
        self.assertGreater(stats1[1], 0)
        self.assertEqual(dump0, dump1)

    def test_a_cache_without_room_reads_nothing_ahead(self):
        text0, dump0, _, _ = run(False, cap="2")
        text1, dump1, stats1, err1 = run(True, cap="2")
        self.assertEqual(dump0, dump1)
        self.assertTrue(stats1 is None or stats1[1] == 0, err1[-2000:])


if __name__ == "__main__":
    unittest.main()
