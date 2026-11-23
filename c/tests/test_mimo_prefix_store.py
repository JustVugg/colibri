"""The prefix store on MiMo-V2.6 (prefix_store.h, mimo.c's mimo_ps_*): prompts read are kept by content,
and a later request that starts with one reads only what follows -- with the logits a fresh read gives.

Most layers attend a sliding window kept as a ring, which cannot be wound back, so the slot resumed
only from the end of what it held (or a photo whose full layers' rows must still be in the slot). The
scenarios, each against a fresh engine answering the same request alone, every step's top five logits:

- a new conversation on the first one's system prompt takes the checkpoint where its message starts;
- the first conversation back after another one on ANOTHER system prompt took the slot: its turn-end
  checkpoint;
- a budget too small for anything, COLI_PREFIX_STORE=0 and COLI_KV_PREFIX=0 keep and take nothing;
- a read-out (logprobs) reads its whole prompt.

The fixture is tools/make_mimo_tiny.py's (its window 8 tokens: every prompt outgrows it); the test
gives it a byte tokenizer whose id 300 is <|im_start|>, the token that opens a message, and a text to
every other id past the bytes, so any reply comes back as the same ids. On the CPU the fresh engine
has the store off and the logits are its own to the last digit; on a chain (MIMO_PS_BACKEND=vulkan) it
keeps an empty store, which cuts the prompt where the session did: the same top five, within 1e-3 (the
fixture's head is scaled for wide greedy margins, its logits run to 300: tests/vulkan_chain_serve.py
compares its logprobs within 1e-3 too).
"""
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine, byte_level_vocab  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("mimo.exe" if sys.platform == "win32" else "mimo")
FIXTURE = Path(os.environ.get("MIMO_TINY", HERE / "mimo_tiny"))

OPEN = b"<|im_start|>"
SYSTEM = b"(12+34=46; 7*8=56; 9-3=6; 5*5=25; 3+3=6; 12+34=46;)"
OTHER = b"(40-8=32; 6*6=36; 2+9=11; 8/4=2; 1+1=2; 40-8=32;)"
Q1, Q2, Q3 = b" 1+1=2; 2+2=4; 3+3=", b" 9*9=81; 8*8=64; 7*7=", b" 5-1=4; 6-1=5; 7-1="


def chat(*turns, system=SYSTEM):
    """A prompt: the system message, then the messages, each opened by <|im_start|>, then the reply's
    opening."""
    out = OPEN + system
    for text in turns:
        out += OPEN + text
    return out + OPEN + b"="


def ntok(prompt):
    """The prompt's tokens: a byte each, an added token (<|...|>) one."""
    return len(re.sub(rb"<\|[a-z0-9_]+\|>", b"#", prompt))


def logits(log, k=None):
    """Every step's [LOGITS] line, or request k's (0-based): a request starts where the engine says
    what it reuses."""
    reqs, cur = [], None
    for l in log.splitlines():
        if l.startswith("[PREFIX] reusing") or l.startswith("[PREFIX] no reuse"):
            cur = []
            reqs.append(cur)
        elif l.startswith("[LOGITS]") and cur is not None:
            cur.append(l)
    if k is None:
        return [l for r in reqs for l in r]
    return reqs[k] if k < len(reqs) else []


@unittest.skipUnless(ENGINE.exists(), "mimo is not built")
@unittest.skipUnless((FIXTURE / "model.safetensors.index.json").exists(),
                     "tiny MiMo fixture is absent (tools/make_mimo_tiny.py)")
class MimoPrefixStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="mimops_"))
        for f in FIXTURE.iterdir():
            if f.name != "tokenizer.json" and not f.name.startswith(".coli_"):
                os.symlink(f.resolve(), cls.dir / f.name)
        # every id past the bytes gets a text of its own, so any reply comes back as the same ids
        vocab = byte_level_vocab(256)
        added = []
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

    def spawn(self, **env):
        be = os.environ.get("MIMO_PS_BACKEND", "")
        dev = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "COLI_VK_TIER_SYNC": "1", "COLI_VK_TIER_BALANCE": "0"} if be == "vulkan" else {}
        base = {"SNAP": str(self.dir), "SERVE": "1", "OMP_NUM_THREADS": "2",
                "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0", "COLI_CUDA": "0",
                "COLI_PREFIX_STORE_BLOCK": "16", "COLI_PARK_MIN": "16", "COLI_LOGIT_DUMP": "1"}
        base.update(dev)
        base.update(env)
        e = ServeEngine(ENGINE, ["8"], base, log_prefix=True)
        timer = threading.Timer(120, e.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(e.close)
        return e

    def fresh(self, prompt, max_tok, **env):
        """A fresh engine's bytes and every step's top logits for this request alone: the store off on
        the CPU, on a chain an empty store (which cuts the prompt where the session did)."""
        if os.environ.get("MIMO_PS_BACKEND", "") != "vulkan" and "COLI_PREFIX_STORE" not in env:
            env = dict(env, COLI_PREFIX_STORE="0")
        e = self.spawn(**env)
        out = e.ask("fresh", prompt, max_tok=max_tok)
        return out, logits(e.close(), 0)

    def same(self, log, k, prompt, out, max_tok=6, **env):
        """Request k (0-based) of a session gave the bytes and the logits a fresh read of it does."""
        f_out, f_lo = self.fresh(prompt, max_tok, **env)
        self.assertEqual(out, f_out, "bytes differ from a fresh read")
        lo = logits(log, k)
        self.assertTrue(lo, "no logits for the request")
        if os.environ.get("MIMO_PS_BACKEND", "") != "vulkan":
            self.assertEqual(lo, f_lo, "logits differ from a fresh read")
            return
        self.assertEqual(len(lo), len(f_lo))
        for a, b in zip(lo, f_lo):
            pa = [t.split(":") for t in a.split()[1:]]
            pb = [t.split(":") for t in b.split()[1:]]
            self.assertEqual([x[0] for x in pa], [x[0] for x in pb], f"top five differ: {a} / {b}")
            for x, y in zip(pa, pb):
                self.assertLess(abs(float(x[1]) - float(y[1])), 1e-3, f"{a} / {b}")

    def hits(self, log):
        return [(int(a), int(b)) for a, b in re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from the store", log)]

    def test_new_conversation_on_the_same_system_prompt(self):
        e = self.spawn()
        a = e.ask("a1", chat(Q1), max_tok=6)
        b = e.ask("b1", chat(Q2), max_tok=6)
        log = e.close()
        self.assertEqual([h[0] for h in self.hits(log)], [ntok(OPEN + SYSTEM)], log[-3000:])
        self.same(log, 0, chat(Q1), a)
        self.same(log, 1, chat(Q2), b)

    def test_back_after_another_system_prompt(self):
        e = self.spawn()
        p1 = chat(Q1)
        a = e.ask("a1", p1, max_tok=6)
        e.ask("o1", chat(Q2, system=OTHER), max_tok=6)    # another system prompt: the rows are its own now
        p2 = p1 + a + OPEN + Q3 + OPEN + b"="
        a2 = e.ask("a2", p2, max_tok=6)
        log = e.close()
        self.assertGreaterEqual(self.hits(log)[-1][0], ntok(p1 + a) - 1, log[-3000:])   # the first turn as it ended
        self.same(log, 2, p2, a2)

    def test_nothing_kept_or_taken(self):
        for env in ({"COLI_PREFIX_STORE_MB": "0"}, {"COLI_PREFIX_STORE": "0"}, {"COLI_KV_PREFIX": "0"}):
            with self.subTest(**env):
                e = self.spawn(**env)
                e.ask("a1", chat(Q1), max_tok=6)
                b = e.ask("b1", chat(Q2), max_tok=6)
                log = e.close()
                self.assertNotIn("from the store", log)
                self.same(log, 1, chat(Q2), b, **env)

    def test_read_out_reads_everything(self):
        e = self.spawn()
        e.ask("a1", chat(Q1), max_tok=6)
        p = chat(Q2)
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
