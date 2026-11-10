"""The prefix store on Qwen3.6 (prefix_store.h, qwen36.c's q36_ps_*): prompts read are kept by content,
and a later request that starts with one reads only what follows -- with the bytes a fresh read gives.

Three of every four layers of the tiny fixture are DeltaNet, whose recurrence cannot be wound back:
without the store a new conversation on the same system prompt reads its whole prompt again. The
scenarios, each against a fresh engine with the store off answering the same request alone:

- a new conversation on the first one's system prompt takes the checkpoint where its message starts;
- the first conversation's next turn, after the other took the slot, takes its turn-end checkpoint;
- two slots: the first conversation's next turn sent to the other slot still reads only what is new;
- a budget too small for anything keeps nothing and changes no byte;
- COLI_PREFIX_STORE=0 takes nothing from the store;
- a read-out (logprobs) reads its whole prompt.

The fixture's byte tokenizer gains the special token that opens a message (<|im_start|>, id 300 of the
fixture's 320), so the engine finds where a prompt's messages start. On the CPU every logit is the fresh
read's; QWEN36_PS_BACKEND=vulkan|cuda runs both engines on that chain, where a prompt cut elsewhere adds
its tiles in another order: the same top five, within 1e-4.
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine, byte_level_vocab  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))

OPEN = b"<|im_start|>"
SYSTEM = OPEN + b"system\n" + b"You are a careful agent. Read every file before you edit it, run the tests, " \
         b"keep commits small and explain each change in plain words. " * 2
Q1, Q2, Q3 = b"List the files to read first.", b"Write a commit message.", b"Now run the tests."


def logits(log):
    return [l for l in log.splitlines() if l.startswith("[LOGITS]")]


def chat(*turns):
    """A prompt: the system prompt, then user / assistant messages, then the reply's opening."""
    out = SYSTEM
    for role, text in turns:
        out += OPEN + role + b"\n" + text
    return out + OPEN + b"assistant\n"


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless((FIXTURE / "model-globals.safetensors").exists(),
                     "tiny qwen36 container is absent (tools/convert_qwen36.py)")
class Qwen36PrefixStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="q36ps_"))
        for f in FIXTURE.iterdir():
            if f.name != "tokenizer.json":
                os.symlink(f.resolve(), cls.dir / f.name)
        # every id past the bytes gets a text of its own, so a reply sent back is the same ids
        vocab, added = byte_level_vocab(256), []
        for tid in range(256, 320):
            text = "<|im_start|>" if tid == 300 else f"<|t{tid}|>"
            vocab[text] = tid
            added.append({"id": tid, "content": text, "special": tid == 300})
        (cls.dir / "tokenizer.json").write_text(json.dumps({
            "version": "1.0", "added_tokens": added,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        }), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def spawn(self, slots=1, **env):
        # QWEN36_PS_BACKEND=vulkan|cuda: both engines of a comparison on that chain (the store's
        # rows go up to the device again after a load), else the CPU
        be = os.environ.get("QWEN36_PS_BACKEND", "")
        dev = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "COLI_VK_TIER_SYNC": "1"} if be == "vulkan" else \
              {"COLI_CUDA": "1", "COLI_CUDA_CHAIN": "1", "COLI_GPUS": "0", "COLI_TRUNK_PROBE": "0", "COLI_DENSE_IDOT": "0"} if be == "cuda" else {}
        base = {"SNAP": str(self.dir), "SERVE": "1", "COLI_DENSE_I8": "0" if be != "cuda" else "1", "KV_SLOTS": str(slots),
                "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0", "COLI_CUDA": "0", "COLI_CUDA_CHAIN": "0",
                "COLI_PREFIX_STORE_BLOCK": "16", "COLI_PARK_MIN": "16", "Q36_MAXT": "2048", "COLI_LOGIT_DUMP": "1"}
        base.update(dev)
        base.update(env)
        e = ServeEngine(ENGINE, ["8", "8"], base, log_prefix=True)
        timer = threading.Timer(120, e.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(e.close)
        return e

    def fresh(self, prompt, max_tok, slots=1):
        """A fresh engine's bytes and every step's top logits for this request alone: the store off on
        the CPU (whose sums do not depend on where a prompt is cut), on a chain an empty store, which
        cuts the prompt where the session did (a chain's tiles add in another order past a cut)."""
        cpu = os.environ.get("QWEN36_PS_BACKEND", "") not in ("vulkan", "cuda")
        e = self.spawn(slots=slots, COLI_PREFIX_STORE="0") if cpu else self.spawn(slots=slots)
        out = e.ask("fresh", prompt, max_tok=max_tok)
        return out, logits(e.close())

    def same(self, log, k, prompt, out, max_tok=6, slots=1):
        """Request k (0-based) of a session gave the bytes and the logits a fresh read of it does."""
        f_out, f_lo = self.fresh(prompt, max_tok, slots)
        lo = logits(log)[k * max_tok:(k + 1) * max_tok]
        self.assertEqual(out, f_out, "bytes differ from a fresh read")
        if os.environ.get("QWEN36_PS_BACKEND", "") not in ("vulkan", "cuda"):
            self.assertEqual(lo, f_lo, "logits differ from a fresh read")
            return
        # a chain adds a prompt's tiles in another order where it is cut elsewhere: the same top
        # five, each logit within 1e-4 (CHAIN_SERVE_TOL's bound, tests/vulkan_chain_serve.py)
        self.assertEqual(len(lo), len(f_lo))
        for a, b in zip(lo, f_lo):
            pa = [t.split(":") for t in a.split()[1:]]
            pb = [t.split(":") for t in b.split()[1:]]
            self.assertEqual([x[0] for x in pa], [x[0] for x in pb], f"top five differ: {a} / {b}")
            for x, y in zip(pa, pb):
                self.assertLess(abs(float(x[1]) - float(y[1])), 1e-4, f"{a} / {b}")

    def test_new_conversation_on_the_same_system_prompt(self):
        e = self.spawn()
        a = e.ask("a1", chat((b"user", Q1)), max_tok=6)
        b = e.ask("b1", chat((b"user", Q2)), max_tok=6)
        log = e.close()
        self.assertIn("prompt tokens from the store", log, "the new conversation took nothing from the store")
        self.same(log, 0, chat((b"user", Q1)), a)
        self.same(log, 1, chat((b"user", Q2)), b)

    def test_first_conversation_comes_back(self):
        e = self.spawn()
        a = e.ask("a1", chat((b"user", Q1)), max_tok=6)
        e.ask("b1", chat((b"user", Q2)), max_tok=6)
        p2 = chat((b"user", Q1), (b"assistant", a), (b"user", Q3))
        a2 = e.ask("a2", p2, max_tok=6)
        log = e.close()
        hits = [l for l in log.splitlines() if "prompt tokens from the store" in l]
        self.assertGreaterEqual(len(hits), 2, log[-3000:])
        self.same(log, 2, p2, a2)

    def test_two_slots_the_other_slot(self):
        e = self.spawn(slots=2)
        a = e.ask("a1", chat((b"user", Q1)), max_tok=6, slot=0)
        e.ask("b1", chat((b"user", Q2)), max_tok=6, slot=1)
        p2 = chat((b"user", Q1), (b"assistant", a), (b"user", Q3))
        a2 = e.ask("a2", p2, max_tok=6, slot=1)   # the first conversation, on the slot the other held
        log = e.close()
        self.assertIn("prompt tokens from the store", log)
        self.same(log, 2, p2, a2, slots=2)

    def test_budget_too_small_keeps_nothing(self):
        e = self.spawn(COLI_PREFIX_STORE_MB="0")
        e.ask("a1", chat((b"user", Q1)), max_tok=6)
        b = e.ask("b1", chat((b"user", Q2)), max_tok=6)
        log = e.close()
        self.assertNotIn("from the store", log)
        self.same(log, 1, chat((b"user", Q2)), b)

    def test_store_off(self):
        e = self.spawn(COLI_PREFIX_STORE="0")
        e.ask("a1", chat((b"user", Q1)), max_tok=6)
        e.ask("b1", chat((b"user", Q2)), max_tok=6)
        log = e.close()
        self.assertNotIn("from the store", log)
        self.assertNotIn("[PREFIX] store:", log)

    def test_read_out_reads_everything(self):
        e = self.spawn()
        e.ask("a1", chat((b"user", Q1)), max_tok=6)
        p = chat((b"user", Q2))
        header = f"SUBMIT lp 0 {len(p)} 2 0 1 logprobs=2\n".encode()
        e.p.stdin.write(header + p + b"\n")
        e.p.stdin.flush()
        while True:
            line = e.p.stdout.readline()
            self.assertTrue(line, "engine closed")
            kind = line.split(b" ", 1)[0]
            if kind in (b"DATA", b"ECHO"):
                e.p.stdout.read(int(line.split()[2]))
                e.p.stdout.readline()
            elif kind in (b"DONE", b"ERROR"):
                break
        log = e.close()
        self.assertEqual(log.count("prompt tokens from the store"), 0, "a read-out took a stored prefix")


if __name__ == "__main__":
    unittest.main()
