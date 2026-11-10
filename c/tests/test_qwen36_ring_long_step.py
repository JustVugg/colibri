"""A step of a prompt longer than the sliding layers' spare rows, read after position 0 (qwen36.c's
KV ring): every logit a fresh read's, and the engine alive.

Every sliding layer keeps its K and V in a ring of window + spare rows, and ring_pos, the position each
row holds, is shared by all of them. The first sliding layer writes a step's rows into the ring after
its scores; a step longer than the spare rows overwrites there the rows of the window before it, which
the later sliding layers still hold, unwritten, in their own buffers. Each layer checked the window
against ring_pos, so from the second sliding layer on a long step past position 0 found the window
"no longer held" and the engine exited: a chat's next message longer than 256 tokens on LFM3, Flint
or Muse, and with the prefix store any conversation read again whose last messages run long (the
store reads a prompt in pieces where its messages start).

Flint's tiny fixture (tools/make_flint_tiny.py, three sliding layers of seven) with the least spare
(Q36_KV_RING_SPARE=9: a ring of 14 rows over a window of 5), the byte tokenizer with the special
token that opens a message:

- the next message of 63 bytes after the state the last request left: a step of 63 rows at 25;
- the prefix store reading a prompt in pieces where its messages start, the last of 63 bytes;
- each against a fresh engine reading the same prompt in one step, the store off: the same bytes and
  every logit the same.

The test makes its fixture (a vocabulary of 320, for the opener's id 300) when torch is there.
"""
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine, byte_level_vocab  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
VOCAB = 320
OPEN = b"<|im_start|>"
LONG = b"Read every file, then run the tests and keep each change small."   # 63 bytes


def torch_ok():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def ntok(b):
    """tokens of a prompt: a byte each, the message opener one"""
    return len(b) - b.count(OPEN) * (len(OPEN) - 1)


def logits(log):
    return [l for l in log.splitlines() if l.startswith("[LOGITS]")]


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless(torch_ok(), "torch (tools/make_flint_tiny.py makes the fixture)")
class Qwen36RingLongStepTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="q36ring_"))
        gguf = cls.dir / "fx" / "model.gguf"
        subprocess.run([sys.executable, str(HERE / "tools" / "make_flint_tiny.py"), "--out", str(cls.dir / "fx"),
                        "--vocab", str(VOCAB), "--gguf", str(gguf), "--gguf-experts", "F32"],
                       check=True, stdout=subprocess.DEVNULL)
        cls.snap = cls.dir / "snap"
        cls.snap.mkdir()
        (cls.snap / "gguf.json").write_text(json.dumps({"format": "colibri-gguf-v1", "gguf": str(gguf.resolve()),
                                                         "mode": "link", "arch": "flint", "parts": 1}))
        (cls.snap / "tokenizer.json").write_text(json.dumps({
            "version": "1.0",
            "added_tokens": [{"id": 300, "content": OPEN.decode(), "special": True}],
            "model": {"type": "BPE", "vocab": byte_level_vocab(256), "merges": []},
        }), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def spawn(self, **env):
        base = {"SNAP": str(self.snap), "SERVE": "1", "KV_SLOTS": "1", "COLI_DENSE_I8": "0",
                "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0", "COLI_CUDA": "0", "COLI_CUDA_CHAIN": "0",
                "Q36_KV_RING_SPARE": "9", "Q36_MAXT": "512", "COLI_PREFIX_STORE_BLOCK": "16",
                "COLI_LOOKUP": "0", "COLI_LOGIT_DUMP": "1"}
        base.update(env)
        e = ServeEngine(ENGINE, ["8", "8"], base, log_prefix=True)
        timer = threading.Timer(120, e.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(e.close)
        return e

    def fresh(self, prompt, max_tok):
        e = self.spawn(COLI_PREFIX_STORE="0")
        out = e.ask("fresh", prompt, max_tok=max_tok)
        return out, logits(e.close())

    def test_next_message_longer_than_the_spare_rows(self):
        p1 = OPEN + b"user\nHello there, agent."
        p2 = p1 + LONG
        e = self.spawn(COLI_PREFIX_STORE="0")
        e.ask("a", p1, max_tok=1)
        out = e.ask("b", p2, max_tok=6)
        log = e.close()
        self.assertIn("keep a ring of 14 rows", log)
        self.assertIn(f"[PREFIX] reusing {ntok(p1)} of {ntok(p2)} prompt tokens", log)
        f_out, f_lo = self.fresh(p2, 6)
        self.assertEqual(out, f_out)
        self.assertEqual(logits(log)[1:], f_lo)

    def test_store_reads_a_prompt_in_pieces(self):
        prompt = (OPEN + b"system\nYou are a careful agent." + OPEN + b"user\nHi." + OPEN + b"assistant\nHello."
                  + OPEN + b"user\n" + LONG + OPEN + b"assistant\n")
        e = self.spawn()
        out = e.ask("a", prompt, max_tok=6)
        log = e.close()
        self.assertIn("keep a ring of 14 rows", log)
        self.assertNotIn("no longer holds the window", log)
        f_out, f_lo = self.fresh(prompt, 6)
        self.assertEqual(out, f_out)
        self.assertEqual(logits(log), f_lo)


if __name__ == "__main__":
    unittest.main()
