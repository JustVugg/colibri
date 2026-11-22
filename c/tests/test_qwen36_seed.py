"""SEED seeds qwen36's sampler.

The sampler (qwen36_sample.h) draws from libc's rand(). Before this test's change the engine
never seeded it, so glibc's default seed of 1 gave every run the same draws, and `SEED`, which
colibri, mimo, inkling and deepseek_v41 read, did nothing here. Now:

  * two engines started with the same SEED, asked the same prompt at temperature 1, generate
    the same bytes;
  * an engine started with another SEED generates something else (a random-init 128-token
    fixture at temperature 1 keeps 24 draws from agreeing by chance);
  * temperature 0 (the default header) is greedy and does not depend on SEED at all.

Recipe (CPU is enough; the fixture is generated, not committed):

    make -C c qwen36 qwen36-tiny-v128-generate
    QWEN36_TINY=./c/qwen36_tiny_v128_c python3 -m unittest c.tests.test_qwen36_seed

Without the fixture or the engine the tests skip, unless QWEN36_TINY_REQUIRED=1.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prefix_serve_harness import ServeEngine, ensure_byte_tokenizer   # noqa: E402

ENGINE = HERE.parent / "qwen36"
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE.parent / "qwen36_tiny_v128_c"))
REQUIRED = os.environ.get("QWEN36_TINY_REQUIRED") == "1"


class SampledEngine(ServeEngine):
    """A serve engine whose requests carry a temperature and a top-p."""

    def ask_sampled(self, rid, prompt, max_tok, temp, top_p=1.0, slot=0):
        header = f"SUBMIT {rid} {slot} {len(prompt)} {max_tok} {temp} {top_p}\n".encode()
        self.p.stdin.write(header + prompt + b"\n")
        self.p.stdin.flush()
        chunks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-request:\n" + "".join(self._err)[-2000:])
            text = line.decode("latin-1").rstrip("\n")
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                chunks.append(self.p.stdout.read(int(text.split()[2])))
                self.p.stdout.readline()
            elif kind in ("DONE", "END"):
                return b"".join(chunks)
            elif kind == "ERROR":
                raise RuntimeError(f"engine error: {text}")


def _engine(seed):
    env = {"SNAP": str(FIXTURE), "SERVE": "1", "SERVE_BATCH": "1", "KV_SLOTS": "1"}
    if seed is not None:
        env["SEED"] = str(seed)
    return SampledEngine(ENGINE, [], env, log_prefix=False)


def _generate(seed, temp, n=24, prompt=b"Once upon a time"):
    e = _engine(seed)
    try:
        return e.ask_sampled("r1", prompt, n, temp)
    finally:
        e.close()


class Seed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = []
        if not ENGINE.exists():
            missing.append(f"{ENGINE} (make -C c qwen36)")
        if not (FIXTURE / "config.json").exists():
            missing.append(f"{FIXTURE} (make -C c qwen36-tiny-v128-generate)")
        if missing:
            if REQUIRED:
                raise AssertionError("missing: " + ", ".join(missing))
            raise unittest.SkipTest("missing: " + ", ".join(missing))
        ensure_byte_tokenizer(FIXTURE, size=128)

    def test_same_seed_same_draws(self):
        a = _generate(7, 1.0)
        b = _generate(7, 1.0)
        self.assertTrue(a, "the engine generated nothing")
        self.assertEqual(a, b)

    def test_another_seed_other_draws(self):
        a = _generate(7, 1.0)
        b = _generate(8, 1.0)
        self.assertNotEqual(a, b)

    def test_greedy_ignores_the_seed(self):
        a = _generate(7, 0)
        b = _generate(8, 0)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
