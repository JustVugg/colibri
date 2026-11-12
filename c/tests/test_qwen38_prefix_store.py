"""The prefix store on Qwen3.8 (prefix_store.h, qwen38.c's q38_ps_*): prompts read are kept by content,
and a later request that starts with one reads only what follows -- with the logits a fresh read gives.

qwen38 already resumes from the live state, the prompt-end snapshot, a pin of the system prompt and a
parked conversation; the pins hold only the recurrent state, so their rows must still be where they
were. The store keeps the rows too. The scenarios, each against a fresh engine with the store off
answering the same request alone, every step's top five logits to the last digit:

- a new conversation on the first one's system prompt takes the checkpoint where its message starts;
- the first conversation back after another one on ANOTHER system prompt took the slot (its rows
  gone): its turn-end checkpoint, rows and state;
- two slots: the first conversation's next turn sent to the slot the other held;
- the MTP head on (its drafts verified, its rows in the store with the model's): the same bytes;
- a budget too small for anything, COLI_PREFIX_STORE=0 and COLI_KV_PREFIX=0 keep and take nothing;
- a read-out (logprobs) reads its whole prompt.

The fixture has 64 token ids, the bytes below 64 (tools/make_edge_tiny_tokenizer.py): the prompts
are digits and punctuation, and <|im_start|> takes byte 1's id, so the engine finds where a prompt's
messages start. QWEN38_PS_BACKEND=vulkan runs both engines on the dense chain, the fresh one with an
empty store (it cuts the prompt where the session did): the same top five, within 1e-4.
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
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("QWEN38_BINARY", HERE / ("qwen38.exe" if sys.platform == "win32" else "qwen38")))
FIXTURE = Path(os.environ.get("QWEN38_TINY", HERE / "qwen38_tiny"))
FIXTURE_MTP = Path(os.environ.get("QWEN38_TINY_MTP", HERE / "qwen38_tiny_mtp"))

OPEN = b"<|im_start|>"
SYSTEM = OPEN + b"(12+34=46; 7*8=56; 9-3=6; 5*5=25;)"
OTHER = OPEN + b"(40-8=32; 6*6=36; 2+9=11; 8/4=2;)"
Q1, Q2, Q3 = b" 1+1=2; 2+2=4; 3+3=", b" 9*9=81; 8*8=64; 7*7=", b" 5-1=4; 6-1=5; 7-1="


def logits(log, k=None):
    """Every step's [LOGITS] line, or request k's (0-based): a request starts where the engine says
    what it reuses (COLI_PREFIX_LOG)."""
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


def chat(*turns, system=SYSTEM):
    """A prompt: the system prompt, then the user's and the assistant's messages in turn (an
    assistant's opens with "<|im_start|>="), then the reply's opening."""
    out = system
    for i, text in enumerate(turns):
        out += OPEN + (b"=" if i % 2 else b"") + text
    return out + OPEN + b"="


def ntok(prompt):
    return len(prompt) - prompt.count(OPEN) * (len(OPEN) - 1)


def tokenizer(fixture, out):
    tok = json.loads((fixture / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = {sym: i for sym, i in tok["model"]["vocab"].items() if i != 1}
    vocab["<|im_start|>"] = 1
    tok["model"]["vocab"] = vocab
    tok["added_tokens"] = list(tok.get("added_tokens") or []) + [{"id": 1, "content": "<|im_start|>", "special": True}]
    (out / "tokenizer.json").write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")


@unittest.skipUnless(ENGINE.exists(), "qwen38 is not built")
@unittest.skipUnless((FIXTURE / "tokenizer.json").exists(),
                     "tiny qwen38 fixture or its tokenizer is absent "
                     "(tools/make_qwen38_tiny.py, tools/make_edge_tiny_tokenizer.py)")
class Qwen38PrefixStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dirs = {}
        for name, fx in (("plain", FIXTURE), ("mtp", FIXTURE_MTP)):
            if not (fx / "tokenizer.json").exists():
                continue
            d = Path(tempfile.mkdtemp(prefix="q38ps_"))
            for f in fx.iterdir():
                if f.name != "tokenizer.json" and not f.name.startswith(".coli_"):
                    os.symlink(f.resolve(), d / f.name)
            tokenizer(fx, d)
            cls.dirs[name] = d

    @classmethod
    def tearDownClass(cls):
        for d in cls.dirs.values():
            shutil.rmtree(d, ignore_errors=True)

    def spawn(self, slots=1, fixture="plain", **env):
        be = os.environ.get("QWEN38_PS_BACKEND", "")
        dev = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "COLI_VK_TIER_SYNC": "1", "COLI_VK_TIER_BALANCE": "0"} if be == "vulkan" else {}
        d = self.dirs[fixture]
        base = {"SNAP": str(d), "SERVE": "1", "TOK": str(d / "tokenizer.json"), "KV_SLOTS": str(slots),
                "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0", "Q38_TRUNK_GPU": "0",
                "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0", "Q38_MAXT": "256", "Q38_MTP": "0",
                "COLI_PREFIX_STORE_BLOCK": "16", "COLI_PARK_MIN": "16", "COLI_LOGIT_DUMP": "1", "COLI_LOOKUP": "0"}
        base.update(dev)
        base.update(env)
        e = ServeEngine(ENGINE, ["1", "8"], base, log_prefix=True)
        timer = threading.Timer(120, e.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(e.close)
        return e

    def fresh(self, prompt, max_tok, slots=1, **env):
        """A fresh engine's bytes and every step's top logits for this request alone: the store off on
        the CPU, on a chain an empty store (which cuts the prompt where the session did)."""
        if os.environ.get("QWEN38_PS_BACKEND", "") != "vulkan" and "COLI_PREFIX_STORE" not in env:
            env = dict(env, COLI_PREFIX_STORE="0")
        e = self.spawn(slots=slots, **env)
        out = e.ask("fresh", prompt, max_tok=max_tok)
        return out, logits(e.close())

    def same(self, log, k, prompt, out, max_tok=6, slots=1, logit_check=True, **env):
        """Request k (0-based) of a session gave the bytes and the logits a fresh read of it does."""
        f_out, f_lo = self.fresh(prompt, max_tok, slots, **env)
        self.assertEqual(out, f_out, "bytes differ from a fresh read")
        if not logit_check:
            return
        lo = logits(log, k)
        if os.environ.get("QWEN38_PS_BACKEND", "") != "vulkan":
            self.assertEqual(lo, f_lo, "logits differ from a fresh read")
            return
        self.assertEqual(len(lo), len(f_lo))
        for a, b in zip(lo, f_lo):
            pa = [t.split(":") for t in a.split()[1:]]
            pb = [t.split(":") for t in b.split()[1:]]
            self.assertEqual([x[0] for x in pa], [x[0] for x in pb], f"top five differ: {a} / {b}")
            for x, y in zip(pa, pb):
                self.assertLess(abs(float(x[1]) - float(y[1])), 1e-4, f"{a} / {b}")

    def hits(self, log):
        return [(int(a), int(b)) for a, b in re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from the store", log)]

    def test_new_conversation_on_the_same_system_prompt(self):
        e = self.spawn()
        a = e.ask("a1", chat(Q1), max_tok=6)
        b = e.ask("b1", chat(Q2), max_tok=6)
        log = e.close()
        self.assertEqual([h[0] for h in self.hits(log)], [ntok(SYSTEM)], log[-3000:])
        # every save says whether it was kept (COLI_PREFIX_LOG). The first request's, logged before the second
        # one reuses: where the system prompt ends, and where its turn ended (the reply fed but for a 6th token,
        # which max_tok stops before it is fed)
        first = log[:log.index("[PREFIX] reusing")]
        saves = [(int(p), k) for p, k in re.findall(r"\[PREFIX\] store: checkpoint at (\d+) (kept|not kept)", first)]
        end = ntok(chat(Q1)) + len(a) - (len(a) == 6)
        self.assertEqual(saves, [(ntok(SYSTEM), "kept"), (end, "kept")], log[-3000:])
        self.same(log, 0, chat(Q1), a)
        self.same(log, 1, chat(Q2), b)

    def test_back_after_another_system_prompt(self):
        e = self.spawn()
        p1 = chat(Q1)
        a = e.ask("a1", p1, max_tok=6)
        e.ask("o1", chat(Q2, system=OTHER), max_tok=6)    # another system prompt: the rows are its own now
        p2 = chat(Q1, a, Q3)
        a2 = e.ask("a2", p2, max_tok=6)
        log = e.close()
        self.assertGreaterEqual(self.hits(log)[-1][0], ntok(p1) + len(a) - 1, log[-3000:])   # the first turn as it ended
        self.same(log, 2, p2, a2)

    def test_two_slots_the_other_slot(self):
        e = self.spawn(slots=2)
        p1 = chat(Q1)
        a = e.ask("a1", p1, max_tok=6, slot=0)
        e.ask("b1", chat(Q2), max_tok=6, slot=1)
        p2 = chat(Q1, a, Q3)
        a2 = e.ask("a2", p2, max_tok=6, slot=1)   # the first conversation, on the slot the other held
        log = e.close()
        self.assertGreaterEqual(self.hits(log)[-1][0], ntok(p1) + len(a) - 1, log[-3000:])   # its turn's end
        self.same(log, 2, p2, a2, slots=2)

    def test_mtp_head_on(self):
        if "mtp" not in self.dirs:
            self.skipTest("tiny qwen38 MTP fixture is absent (tools/make_qwen38_tiny.py --mtp)")
        e = self.spawn(fixture="mtp", Q38_MTP="1")
        p1 = chat(Q1)
        a = e.ask("a1", p1, max_tok=8)
        e.ask("o1", chat(Q2, system=OTHER), max_tok=8)
        p2 = chat(Q1, a, Q3)
        a2 = e.ask("a2", p2, max_tok=8)
        log = e.close()
        self.assertTrue(self.hits(log), log[-3000:])
        self.same(log, 2, p2, a2, max_tok=8, fixture="mtp", Q38_MTP="1", logit_check=False)

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
