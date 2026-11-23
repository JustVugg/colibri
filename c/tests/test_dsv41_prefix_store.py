"""The prefix store on DeepSeek V4.1 (prefix_store.h, deepseek_v41.c's v41_ps_*): prompts read are kept
by content, and a later request that starts with one reads only what follows -- with the logits a cold
read of the same prompt gives.

V4.1's state is more than rows by position: every layer's window ring and its positions, the
compressor's filling group, what an index source published last. None of it can be wound back, and
resuming the live state is opt-in here (a decoded position attends otherwise than the same text
prefilled, test_dsv41_prefix_serve.py), so without the store a new conversation on the same system
prompt and a conversation's next turn both read their whole prompt again. The store keeps only what
a prompt read put there, so taking it is exact. The scenarios, each against a fresh engine with the
store off answering the same request alone, every step's top five logits to the last digit:

- a new conversation on the first one's system prompt takes the checkpoint where its message starts;
- the first conversation's next turn takes the checkpoint where its last prompt ended;
- two slots: the first conversation's next turn on the slot the other held;
- the speculative decoder on (its drafts verified): the same bytes;
- a budget too small for anything, COLI_PREFIX_STORE=0 and COLI_KV_PREFIX=0 keep and take nothing;
- a request that resumed the live state (COLI_KV_PREFIX=1: decoded positions in it) keeps nothing;
- a read-out (logprobs) reads its whole prompt.

The fixture's byte tokenizer gains the tokens that open a message (<｜User｜> as id 1, <｜Assistant｜>
as id 3, bytes no prompt here holds), so the engine finds where a prompt's messages start.
DSV41_PS_BACKEND=vulkan|cuda runs both engines of a comparison on that chain, the fresh one with an
empty store (it cuts the prompt where the session did: a chain adds a prompt's rows in another
order past a cut): the same top five, within 1e-4.
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
ENGINE = HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))

USER, ASSISTANT = "<｜User｜>".encode(), "<｜Assistant｜>".encode()
SYSTEM = b"You are a careful agent. Read every file before you edit it, run the tests, keep commits small. " \
         b"Explain each change."
Q1, Q2, Q3 = b"List the files to read first.", b"Write a commit message now.", b"Now run the tests."


def logits(log):
    return [l for l in log.splitlines() if l.startswith("[LOGITS]")]


def chat(*turns):
    """A prompt: the system prompt, then user / assistant messages, then the reply's opening."""
    out = SYSTEM
    for role, text in turns:
        out += (USER if role == "user" else ASSISTANT) + text
    return out + ASSISTANT


def ntok(prompt):
    """The prompt's tokens: a byte each, a message's opening one."""
    return len(prompt) - prompt.count(USER) * (len(USER) - 1) - prompt.count(ASSISTANT) * (len(ASSISTANT) - 1)


def checkpoints(log):
    m = re.search(r"\[PREFIX\] store: .* in (\d+) blocks and (\d+) checkpoints", log)
    return int(m.group(2)) if m else 0


@unittest.skipUnless(ENGINE.exists(), "deepseek_v41 is not built")
@unittest.skipUnless((FIXTURE / "model.safetensors").exists() or (FIXTURE / "gguf.json").exists(),
                     "tiny dsv41 fixture is absent (tools/make_dsv41_tiny.py, or a GGUF's: tests/vulkan_engines.sh's v41_gguf_tiny)")
class Dsv41PrefixStoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="dsv41ps_"))
        for f in FIXTURE.iterdir():
            if f.name != "tokenizer.json":
                os.symlink(f.resolve(), cls.dir / f.name)
        # a byte for every id below 256 (the GGUF fixture's own vocabulary is <0xNN> pieces), the
        # two openings in place of bytes 1 and 3
        vocab = {sym: i for sym, i in byte_level_vocab(256).items() if i not in (1, 3)}
        added = []
        for text, tid in (("<｜User｜>", 1), ("<｜Assistant｜>", 3)):
            vocab[text] = tid
            added.append({"id": tid, "content": text, "special": True})
        (cls.dir / "tokenizer.json").write_text(json.dumps({
            "version": "1.0", "added_tokens": added,
            "model": {"type": "BPE", "vocab": vocab, "merges": []},
        }, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def spawn(self, slots=1, **env):
        be = os.environ.get("DSV41_PS_BACKEND", "")
        dev = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "COLI_VK_TIER_SYNC": "1", "COLI_VK_TIER_BALANCE": "0"} if be == "vulkan" else \
              {"COLI_CUDA": "1", "COLI_CUDA_CHAIN": "1", "COLI_GPUS": "0"} if be == "cuda" else {}
        base = {"SNAP": str(self.dir), "SERVE": "1", "KV_SLOTS": str(slots), "OMP_NUM_THREADS": "2",
                "V41_DSPARK": "0", "COLI_PREFIX_STORE_BLOCK": "16", "COLI_LOGIT_DUMP": "1",
                "COLI_VULKAN": "0", "COLI_VK_CHAIN": "0", "COLI_CUDA": "0", "COLI_CUDA_CHAIN": "0"}
        base.update(dev)
        base.update(env)
        e = ServeEngine(ENGINE, ["8"], base, log_prefix=True, reuse=True)
        timer = threading.Timer(120, e.p.kill)
        timer.daemon = True
        timer.start()
        self.addCleanup(timer.cancel)
        self.addCleanup(e.close)
        return e

    def fresh(self, prompt, max_tok, slots=1, **env):
        """A fresh engine's bytes and every step's top logits for this request alone: the store off on
        the CPU, on a chain an empty store (which cuts the prompt where the session did)."""
        if os.environ.get("DSV41_PS_BACKEND", "") not in ("vulkan", "cuda") and "COLI_PREFIX_STORE" not in env:
            env = dict(env, COLI_PREFIX_STORE="0")
        e = self.spawn(slots=slots, **env)
        out = e.ask("fresh", prompt, max_tok=max_tok)
        return out, logits(e.close())

    def same(self, log, k, prompt, out, max_tok=6, slots=1, **env):
        """Request k (0-based) of a session gave the bytes and the logits a fresh read of it does."""
        f_out, f_lo = self.fresh(prompt, max_tok, slots, **env)
        self.assertEqual(out, f_out, "bytes differ from a fresh read")
        if env.get("V41_DSPARK", "0") != "0":
            return
        lo = logits(log)[k * max_tok:(k + 1) * max_tok]
        if os.environ.get("DSV41_PS_BACKEND", "") not in ("vulkan", "cuda"):
            self.assertEqual(lo, f_lo, "logits differ from a fresh read")
            return
        self.assertEqual(len(lo), len(f_lo))
        for a, b in zip(lo, f_lo):
            pa = [t.split(":") for t in a.split()[1:]]
            pb = [t.split(":") for t in b.split()[1:]]
            self.assertEqual([x[0] for x in pa], [x[0] for x in pb], f"top five differ: {a} / {b}")
            for x, y in zip(pa, pb):
                self.assertLess(abs(float(x[1]) - float(y[1])), 1e-4, f"{a} / {b}")

    def test_new_conversation_on_the_same_system_prompt(self):
        e = self.spawn()
        a = e.ask("a1", chat(("user", Q1)), max_tok=6)
        b = e.ask("b1", chat(("user", Q2)), max_tok=6)
        log = e.close()
        hits = re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from the store", log)
        self.assertEqual(len(hits), 1, log[-3000:])
        self.assertEqual(int(hits[0][0]), len(SYSTEM), "the checkpoint where the new conversation's message starts")
        self.same(log, 0, chat(("user", Q1)), a)
        self.same(log, 1, chat(("user", Q2)), b)

    def test_next_turn_takes_where_the_last_prompt_ended(self):
        e = self.spawn()
        p1 = chat(("user", Q1))
        a = e.ask("a1", p1, max_tok=6)
        e.ask("b1", chat(("user", Q2)), max_tok=6)
        p2 = chat(("user", Q1), ("assistant", a), ("user", Q3))
        a2 = e.ask("a2", p2, max_tok=6)
        log = e.close()
        hits = re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from the store", log)
        self.assertEqual(int(hits[-1][0]), ntok(p1), log[-3000:])
        self.same(log, 2, p2, a2)

    def test_two_slots_the_other_slot(self):
        e = self.spawn(slots=2)
        p1 = chat(("user", Q1))
        a = e.ask("a1", p1, max_tok=6, slot=0)
        e.ask("b1", chat(("user", Q2)), max_tok=6, slot=1)
        p2 = chat(("user", Q1), ("assistant", a), ("user", Q3))
        a2 = e.ask("a2", p2, max_tok=6, slot=1)   # the first conversation, on the slot the other held
        log = e.close()
        hits = re.findall(r"\[PREFIX\] (\d+) of (\d+) prompt tokens from the store", log)
        self.assertEqual(int(hits[-1][0]), ntok(p1), log[-3000:])
        self.same(log, 2, p2, a2, slots=2)

    def test_speculative_decoder_on(self):
        e = self.spawn(V41_DSPARK="1")
        e.ask("a1", chat(("user", Q1)), max_tok=8)
        b = e.ask("b1", chat(("user", Q2)), max_tok=8)
        log = e.close()
        self.assertIn("prompt tokens from the store", log)
        self.same(log, 1, chat(("user", Q2)), b, max_tok=8, V41_DSPARK="1")

    def test_nothing_kept_or_taken(self):
        for env in ({"COLI_PREFIX_STORE_MB": "0"}, {"COLI_PREFIX_STORE": "0"}, {"COLI_KV_PREFIX": "0"}):
            with self.subTest(**env):
                e = self.spawn(**env)
                e.ask("a1", chat(("user", Q1)), max_tok=6)
                b = e.ask("b1", chat(("user", Q2)), max_tok=6)
                log = e.close()
                self.assertNotIn("from the store", log)
                self.same(log, 1, chat(("user", Q2)), b)

    def test_live_resume_keeps_nothing(self):
        """COLI_KV_PREFIX=1: the next turn resumes the live state (a decoded reply in it, in general);
        what it reads stays out of the store, whose checkpoints are all a prompt read's. The first turn
        asks one token, which is never fed back, so the live state is its prompt whatever the fixture's
        vocabulary and the resume fires."""
        p1 = chat(("user", Q1))
        e = self.spawn(COLI_KV_PREFIX="1")
        e.ask("a1", p1, max_tok=1)
        log1 = e.close()
        e = self.spawn(COLI_KV_PREFIX="1")
        e.ask("a1", p1, max_tok=1)
        e.ask("a2", p1 + USER + Q3 + ASSISTANT, max_tok=6)
        log2 = e.close()
        self.assertIn("[PREFIX] reusing", log2, "the live state was not resumed: nothing is proven")
        self.assertGreater(checkpoints(log1), 0)
        self.assertEqual(checkpoints(log2), checkpoints(log1), "a turn that resumed the live state kept checkpoints")

    def test_read_out_reads_everything(self):
        e = self.spawn()
        e.ask("a1", chat(("user", Q1)), max_tok=6)
        p = chat(("user", Q2))
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
