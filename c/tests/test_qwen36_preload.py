"""Q36_PRELOAD=1 fills the expert cache from loader threads while the engine answers.

It may change when an expert is read and nothing else: a reply must be the bytes
the same engine gives without it, whether the request arrives while the loaders
run or after them, with a cache that holds every expert or one that must evict,
and with one loader or more loaders than there are experts in a layer.

Drives the serve protocol on the generated tiny Qwen3.6 fixtures
(tools/make_qwen36_tiny.py + tools/convert_qwen36.py): the int8 one, and the
int4 gs64 one under both expert kernels where it was generated.
"""
import json
import os
import re
import sys
import time
import unittest
from pathlib import Path
from tests.prefix_serve_harness import ServeEngine, ensure_byte_tokenizer

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))
FIXTURE64 = Path(os.environ.get("QWEN36_TINY64", HERE / "qwen36_tiny64_c"))
PROMPTS = (b"The capital of France is", b"Routing sends this text to other experts: 0123456789",
           b"a third turn, so that a cache of two has had to evict")
DONE = re.compile(r"preload: (\d+) experts loaded in")


def replies(fixture, cap, env, wait_for_loaders):
    """The three replies (8 tokens each) of one engine, its preload count (None without one)."""
    base = {"SNAP": str(fixture), "SERVE": "1", "COLI_DENSE_I8": "0", "Q36_MAXT": "1024"}
    engine = ServeEngine(ENGINE, [str(cap), "8"], dict(base, **env), log_prefix=False, reuse=False)
    try:
        if wait_for_loaders:
            deadline = time.time() + 120
            while not DONE.search("".join(engine._err)):
                if time.time() > deadline:
                    raise AssertionError("the loaders never finished:\n" + "".join(engine._err)[-2000:])
                time.sleep(0.02)
        out = [engine.ask(str(i), p, max_tok=8) for i, p in enumerate(PROMPTS)]
    finally:
        log = engine.close()
        engine.p.stdout.close()
        engine.p.stderr.close()
    done = DONE.search(log)
    return out, (int(done.group(1)) if done else None), log


HAVE = (FIXTURE / "model-globals.safetensors").exists()
HAVE64 = (FIXTURE64 / "model-globals.safetensors").exists()


@unittest.skipUnless(ENGINE.exists(), "build qwen36 first")
class QwenPreloadTest(unittest.TestCase):
    def check(self, fixture, extra):
        ensure_byte_tokenizer(fixture)
        layers = json.loads((fixture / "config.json").read_text())["num_hidden_layers"]
        for cap in (8, 2):                       # every expert of a layer, and a cache that evicts
            want, none, _ = replies(fixture, cap, extra, False)
            self.assertIsNone(none, "the preload ran without Q36_PRELOAD=1")
            self.assertTrue(all(want), "the fixture produced an empty reply")
            for threads in ("1", "4", "16"):
                for wait in (False, True):       # a request racing the loaders, and one after them
                    with self.subTest(fixture=fixture.name, cap=cap, threads=threads, wait=wait, **extra):
                        env = dict(extra, Q36_PRELOAD="1", Q36_PRELOAD_THREADS=threads)
                        got, loaded, log = replies(fixture, cap, env, wait)
                        self.assertEqual(got, want, "a reply changed with the preload on")
                        self.assertIsNotNone(loaded, "the loaders did not report:\n" + log[-2000:])
                        self.assertNotIn("unknown variable Q36_PRELOAD", log, "coli_env.h does not list it")
                        self.assertGreater(loaded, 0)
                        if wait:                 # nothing was resident before them: every free slot, no eviction
                            self.assertEqual(loaded, layers * cap, "the loaders left free slots or evicted")
                        else:
                            self.assertLessEqual(loaded, layers * cap)

    @unittest.skipUnless(HAVE, "generate the tiny fixture first")
    def test_replies_are_the_ones_without_it(self):
        self.check(FIXTURE, {})

    @unittest.skipUnless(HAVE64, "generate the int4 gs64 tiny fixture first")
    def test_replies_are_the_ones_without_it_int4_both_kernels(self):
        for kernel in ("1", "0"):
            self.check(FIXTURE64, {"QWEN_EXPERT_KERNEL": kernel})

    @unittest.skipUnless(HAVE or HAVE64, "generate a tiny fixture first")
    def test_off_values_do_not_start_it(self):
        fixture = FIXTURE if HAVE else FIXTURE64
        ensure_byte_tokenizer(fixture)
        for value in ("0", ""):
            with self.subTest(value=value):
                _, loaded, log = replies(fixture, 8, {"Q36_PRELOAD": value}, False)
                self.assertIsNone(loaded)
                self.assertNotIn("preload:", log)


if __name__ == "__main__":
    unittest.main()
