"""DUMP writes every row a token is sampled from, not the last one.

`DUMP=<path>` on qwen36 kept the last token's logits only. Now every row a token is sampled
from goes to the file as it is sampled, raw float32, vocab floats a row, as colibri.c's
`dump_logits` writes them. Over the serve protocol on the generated 128-token fixture:

  * a greedy request of N tokens leaves N rows, and the argmax of row i is the i-th token
    the engine sent (the byte tokenizer maps token id i to byte i);
  * a second request appends its rows: 2N rows after two;
  * the same request twice gives the same rows (greedy, same bytes, same floats).

Recipe (CPU is enough; the fixture is generated, not committed):

    make -C c qwen36 qwen36-tiny-v128-generate
    QWEN36_TINY=./c/qwen36_tiny_v128_c python3 -m unittest c.tests.test_qwen36_dump
"""
import array
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prefix_serve_harness import ServeEngine, ensure_byte_tokenizer   # noqa: E402

ENGINE = Path(os.environ.get("QWEN36_ENGINE", HERE.parent / "qwen36"))
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE.parent / "qwen36_tiny_v128_c"))
REQUIRED = os.environ.get("QWEN36_TINY_REQUIRED") == "1"
VOCAB = 128


def _rows(path):
    data = array.array("f", Path(path).read_bytes())
    assert len(data) % VOCAB == 0, f"{len(data)} floats is not a whole number of {VOCAB}-float rows"
    return [data[i:i + VOCAB] for i in range(0, len(data), VOCAB)]


def _argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


class Dump(unittest.TestCase):
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
        ensure_byte_tokenizer(FIXTURE, size=VOCAB)

    def _engine(self, dump):
        return ServeEngine(ENGINE, [], {"SNAP": str(FIXTURE), "SERVE": "1", "SERVE_BATCH": "1",
                                        "KV_SLOTS": "1", "DUMP": dump}, log_prefix=False)

    def test_one_row_per_token_and_the_argmax_is_the_token(self):
        with tempfile.TemporaryDirectory() as d:
            dump = os.path.join(d, "logits.f32")
            e = self._engine(dump)
            try:
                out = e.ask("r1", b"Once upon a time", max_tok=16)
            finally:
                e.close()
            rows = _rows(dump)
            self.assertEqual(len(rows), len(out), f"{len(rows)} rows for {len(out)} tokens")
            self.assertEqual(bytes(_argmax(r) for r in rows), out)

    def test_requests_append_and_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            dump = os.path.join(d, "logits.f32")
            e = self._engine(dump)
            try:
                out1 = e.ask("r1", b"Once upon a time", max_tok=12)
                out2 = e.ask("r2", b"The weather today", max_tok=12)
            finally:
                e.close()
            rows = _rows(dump)
            self.assertEqual(len(rows), len(out1) + len(out2))
            self.assertEqual(bytes(_argmax(r) for r in rows[:len(out1)]), out1)
            self.assertEqual(bytes(_argmax(r) for r in rows[len(out1):]), out2)
            again = os.path.join(d, "again.f32")
            e = self._engine(again)
            try:
                self.assertEqual(e.ask("r1", b"Once upon a time", max_tok=12), out1)
            finally:
                e.close()
            self.assertEqual(Path(again).read_bytes(), Path(dump).read_bytes()[:len(out1) * VOCAB * 4])


if __name__ == "__main__":
    unittest.main()
