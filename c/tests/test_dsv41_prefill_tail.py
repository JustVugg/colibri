"""DeepSeek V4.1's prompt tail (v41_prefill_tail): past the first layer with compress_ratio
1 the layers read the long context from that layer's caches and keep only a window of their
own, so they run on a prompt's last D * (W - 1) + 1 rows and give the same bits.

The tiny fixture (tools/make_dsv41_tiny.py) has that layout: six layers, compress ratios
0 2 2 1 1 0, layer 3 the last cache owner and the candidate source, a window of 8, so the
exact tail is 2 * 7 + 1 = 15 rows. A prompt of 200 rows takes it. DUMP writes the logits
rows of every forward, the prompt's and every decode step's after it; the run with the tail
must write the bytes of the run without it (V41_PREFILL_TAIL=0), and a tail one row shorter
(V41_PREFILL_TAIL=14) must not, which says the tail ran and that 15 is the shortest exact one.

Layer 4 runs its own indexer there. In a copy of the fixture whose layer 4 has none, layer 4
reads the list layer 3 published for every row of the prompt, as V4.1-Flash's layers 21 to 23
read layer 20's: the tail's rows must find their own lists in it, on the window ring's rows.

On the Vulkan chain (DSV41_CHAIN=1, a VK=1 build: the deepseek-chain family runs it on
Lavapipe) the same comparison runs with the chain on and chunks of 16 rows, chunk by chunk
(COLI_VK_V41_LAYERWISE=0): the same bits. A prompt of more than one chunk runs layer by layer
by default, each layer's routed experts for all its rows in one step of the tier, whose
kernels follow the rows an expert gets (from 16 on the device's GEMM): the tail's 15 rows and
every row's 200 then round apart in the last bits (2e-7 of the largest logit on an RTX PRO
4000), the text the same. There the comparison is the text and the logits within 1e-5.
"""
import array
import json
import os
import shutil
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


def run(tail, dspark="0", snap=FIXTURE):
    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "logits.bin")
        env = {"SNAP": str(snap), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
               "V41_DSPARK": dspark, "COLI_VULKAN": "0", "DUMP": dump,
               "V41_DSPARK_GATE": "0"}   # every draft verified: the gate's pick follows measured times
        if tail is not None:
            env["V41_PREFILL_TAIL"] = tail
        engine = ServeEngine(ENGINE, ["8"], env, reuse=False)
        try:
            text = engine.ask("t", PROMPT, 24)
        finally:
            err = engine.close()
            engine.p.stdout.close(); engine.p.stderr.close()
        data = b""
        if os.path.exists(dump):
            with open(dump, "rb") as f:
                data = f.read()
    return text, data, err


def run_chain(tail, dspark="0", snap=FIXTURE, layerwise="0"):
    with tempfile.TemporaryDirectory() as d:
        dump = os.path.join(d, "logits.bin")
        env = {"SNAP": str(snap), "SERVE": "1", "CTX": "256", "OMP_NUM_THREADS": "2",
               "V41_DSPARK": dspark, "COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "DUMP": dump, "V41_DSPARK_GATE": "0",
               "COLI_VK_CHAIN_ROWS": "16",   # chunks of 16 rows: the ones before the tail stop at layer 3
               "COLI_VK_V41_LAYERWISE": layerwise}
        if tail is not None:
            env["V41_PREFILL_TAIL"] = tail
        engine = ServeEngine(ENGINE, ["8"], env, reuse=False)
        try:
            text = engine.ask("t", PROMPT, 24)
        finally:
            err = engine.close()
            engine.p.stdout.close(); engine.p.stderr.close()
        data = b""
        if os.path.exists(dump):
            with open(dump, "rb") as f:
                data = f.read()
    return text, data, err


def reused_list_fixture(d):
    """The fixture with layer 4 an index reader instead of an index source."""
    out = Path(d) / "dsv41_tiny_reader"
    shutil.copytree(FIXTURE, out)
    cfg = json.loads((out / "config.json").read_text())
    text = cfg.get("text_config", cfg)
    for key in ("index_source_layers", "index_source_layer_ids"):
        if key in text:
            text[key] = [1, 3]
    (out / "config.json").write_text(json.dumps(cfg))
    return out


def chain_ops(err):
    """The ops the chain recorded, from its report line at exit."""
    import re
    found = re.findall(r"chain: \d+ forwards, \d+ frames \((\d+) ops", err)
    return int(found[-1]) if found else -1


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists(),
                     "built V4.1 engine and tiny fixture required")
class PrefillTailTest(unittest.TestCase):
    def test_the_exact_tail_gives_every_row_s_bits(self):
        text_all, dump_all, err_all = run("0")
        text, dump, err = run(None)
        self.assertNotIn("[v41] prefill: the layers after", err_all)
        self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
        self.assertGreater(len(dump), 20 * 4 * 256, err[-2000:])   # 24 rows of 256 logits: the prompt and its decode steps
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)

    def test_a_layer_reading_the_source_s_list(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = reused_list_fixture(d)
            text_all, dump_all, _ = run("0", snap=fixture)
            text, dump, err = run(None, snap=fixture)
        self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
        self.assertGreater(len(dump), 20 * 4 * 256, err[-2000:])
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)

    def test_a_row_shorter_is_approximate(self):
        _, dump_all, _ = run("0")
        _, dump, err = run("14")
        self.assertIn("last 14 rows (V41_PREFILL_TAIL", err)
        self.assertEqual(len(dump), len(dump_all))
        self.assertNotEqual(dump, dump_all)

    def test_dspark_seeded_from_the_tail_drafts_the_same(self):
        # the stages' window is within the tail: the same drafts, so the same forwards
        text_all, dump_all, _ = run("0", dspark="1")
        text, dump, _ = run(None, dspark="1")
        self.assertGreater(len(dump), 0)
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)


@unittest.skipUnless(os.environ.get("DSV41_CHAIN") == "1" and ENGINE.exists()
                     and (FIXTURE / "model.safetensors").exists(),
                     "DSV41_CHAIN=1, a VK=1 build and a Vulkan device required")
class PrefillTailChainTest(unittest.TestCase):
    def test_the_exact_tail_on_the_chain(self):
        text_all, dump_all, err_all = run_chain("0")
        text, dump, err = run_chain(None)
        self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
        self.assertNotIn("[v41] prefill: the layers after", err_all)
        self.assertGreater(len(dump), 0, err[-2000:])
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)
        # the chunks before the tail stopped after layer 3: fewer ops than every row's run
        self.assertGreater(chain_ops(err_all), 0)
        self.assertLess(chain_ops(err), chain_ops(err_all))

    def test_a_layer_reading_the_source_s_list_on_the_chain(self):
        with tempfile.TemporaryDirectory() as d:
            fixture = reused_list_fixture(d)
            text_all, dump_all, _ = run_chain("0", snap=fixture)
            text, dump, err = run_chain(None, snap=fixture)
        self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
        self.assertGreater(len(dump), 0, err[-2000:])
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)

    def test_the_tail_on_the_layerwise_chain(self):
        """layer by layer (the default): the text, and the logits within 1e-5 of the largest"""
        for snap_of in (lambda d: FIXTURE, reused_list_fixture):
            with tempfile.TemporaryDirectory() as d:
                snap = snap_of(d)
                text_all, dump_all, _ = run_chain("0", snap=snap, layerwise="1")
                text, dump, err = run_chain(None, snap=snap, layerwise="1")
            with self.subTest(snap=snap.name):
                self.assertIn("[v41] prefill: the layers after layer 3 run on a prompt's last 15 rows", err)
                self.assertEqual(text, text_all)
                a, b = array.array("f", dump_all), array.array("f", dump)
                self.assertGreater(len(a), 0, err[-2000:])
                self.assertEqual(len(a), len(b))
                largest = max(abs(v) for v in a)
                self.assertLessEqual(max(abs(x - y) for x, y in zip(a, b)), 1e-5 * largest)

    def test_the_tail_on_the_chain_with_dspark(self):
        text_all, dump_all, _ = run_chain("0", dspark="1")
        text, dump, _ = run_chain(None, dspark="1")
        self.assertEqual(dump, dump_all)
        self.assertEqual(text, text_all)


if __name__ == "__main__":
    unittest.main()
