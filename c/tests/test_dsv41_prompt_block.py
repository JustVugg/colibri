"""DeepSeek V4.1 routes a prompt in blocks of rows (V41_MOE_BLOCK), and the experts a block asks for
are read once for all of its rows. A bigger block reads each expert fewer times on a prompt, so a
cache too small to keep them between blocks (cap 2 of 8 here) misses less; the bits must not move:
a row's matmul folds its groups in the same order whatever else shares the block, and the
contributions are summed in rank order afterwards. DUMP writes the logits rows of every forward.
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
PROMPT = bytes((7 * i + 3) % 200 + 32 for i in range(200))   # one id per byte, no runs


def run(block):
    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "logits.bin")
        env = {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
               "V41_DSPARK": "0", "COLI_VULKAN": "0", "DUMP": dump, "V41_STATS": "1"}
        if block is not None:
            env["V41_MOE_BLOCK"] = str(block)
        engine = ServeEngine(ENGINE, ["2"], env, reuse=False)
        try:
            text = engine.ask("t", PROMPT, 8)
        finally:
            err = engine.close()
            engine.p.stdout.close(); engine.p.stderr.close()
        with open(dump, "rb") as f:
            data = f.read()
    found = re.findall(r"(\d+) misses", err)
    return text, data, (int(found[-1]) if found else -1), err


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists(),
                     "built V4.1 engine and tiny fixture required")
class PromptBlockTest(unittest.TestCase):
    def test_a_bigger_block_reads_fewer_experts_and_gives_the_same_bits(self):
        text4, dump4, miss4, err4 = run(4)
        text16, dump16, miss16, _ = run(16)
        text200, dump200, miss200, err200 = run(200)
        self.assertGreaterEqual(len(dump200), 8 * 4 * 256, err200[-2000:])
        self.assertEqual(dump4, dump200)
        self.assertEqual(dump16, dump200)
        self.assertEqual(text4, text200)
        self.assertGreater(miss200, 0, err200[-2000:])
        self.assertGreater(miss4, miss16, err4[-2000:])
        self.assertGreater(miss16, miss200)

    def test_the_default_block_is_exact_too(self):
        text, dump, _, _ = run(None)
        text1, dump1, _, _ = run(1)
        self.assertEqual(dump, dump1)
        self.assertEqual(text, text1)


if __name__ == "__main__":
    unittest.main()
