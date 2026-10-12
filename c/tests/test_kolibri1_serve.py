#!/usr/bin/env python3
"""The kolibri1 SERVE loop, driven over its stdin/stdout protocol on the tiny fixture.

Needs numpy (fixture generator), the ./kolibri1 binary (make kolibri1) and
kolibri1_tiny/ref.json (the oracle). No gateway here: the frames are written by hand, as
openai_server.py's Engine writes them, so every protocol corner can be hit directly. The
gateway end to end is tests/test_kolibri1_gateway_real_engine.py.

Test-only engine hooks used (both print a loud line on stderr): K1_TEST_SERVE_IDS=1 (a
payload "ids:1,2,3" is the prompt as token ids, so the oracle's own ids can be sent) and
K1_SERVE_LOG=2 (one stderr line per request with prompt tokens, how much was reused, the ids
generated).

What is checked:
  * READY / CAPS / STAT handshake
  * a greedy request is token-exact the oracle's greedy continuation, for the 3 cases, for
    K1_CHUNK 64 / 3, an evicting expert cache, prefix reuse on and off
  * isolation: A, B, C, A, B in one process give the oracle's tokens every time; a NaN in the
    previous request's rows would surface (reset poisons what was written)
  * prefix reuse: snapshot path (prompt = previous prompt + more), live path (prompt = state
    + more), all-or-nothing (a partial prefix reuses nothing); the continuation equals the
    oracle's and equals the run with reuse off
  * stop ids (config eos, K1_STOP_IDS), max_tokens, context limits, STOP, CANCEL, the error replies
  * non-finite logits are an ERROR reply (head row NaN, or one embedding row NaN), never a
    sampled token, and the process keeps serving afterwards
  * seeded sampling is reproducible (same seed, in one process and across processes), differs
    for another seed, top_k 1 equals greedy when there is no tie at the maximum; the sampler itself against a reference
    (tests/test_kolibri1_serve_sampler.c)
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

try:
    import numpy as np  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kolibri1_serve_fixture as FX  # noqa: E402

TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
REF = Path(os.environ.get("K1_REF", ROOT / "kolibri1_tiny" / "ref.json"))
HOOKS = ("K1_LOGITS", "K1_CAPTURE", "K1_LOGITS_FIFO", "K1_ATTN_KEYS", "K1_FEED", "K1_REPLAY_IDS", "K1_PROBE",
         "K1_CAP", "K1_CHUNK", "K1_CTX", "K1_TOP_K", "K1_SEED", "SEED", "K1_STOP_IDS", "K1_SERVE_PREFIX",
         "COLI_KV_PREFIX", "K1_TEST_SERVE_SCRIPT", "K1_RELEASE", "K1_LAYERS")


class Serve:
    """one engine process speaking the protocol"""

    def __init__(self, model, env=None, cap=8, tmp=None):
        e = {k: v for k, v in os.environ.items() if k not in HOOKS}
        e.update({"SNAP": str(model), "SERVE": "1", "OMP_NUM_THREADS": "2", "K1_SERVE_LOG": "2",
                  "K1_TEST_SERVE_IDS": "1", "CTX": "1024"})
        e.update(env or {})
        self.err_path = Path(tempfile.mkstemp(prefix="k1serve-", suffix=".err", dir=tmp)[1])
        self.err = open(self.err_path, "wb")
        self.p = subprocess.Popen([str(BINARY), str(cap)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=self.err, env=e)
        self.seq = 0
        self.handshake = [self.p.stdout.readline() for _ in range(3)]

    def line(self):
        return self.p.stdout.readline()

    def send(self, data):
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def submit(self, ids_or_bytes, max_tokens=8, temperature=0.0, top_p=1.0, slot=0, ext="", rid=None):
        self.seq += 1
        rid = rid or f"r{self.seq}"
        payload = (b"ids:" + ",".join(map(str, ids_or_bytes)).encode()) if isinstance(ids_or_bytes, (list, tuple)) \
            else ids_or_bytes
        self.send(f"SUBMIT {rid} {slot} {len(payload)} {max_tokens} {temperature:.8g} {top_p:.8g}{ext}\n".encode()
                  + payload + b"\n")
        return rid

    def read_event(self):
        h = self.line()
        if not h:
            return ("EOF",)
        f = h.decode().split()
        if f[0] == "DATA":
            n = int(f[2])
            data = self.p.stdout.read(n)
            assert self.p.stdout.read(1) == b"\n"
            return ("DATA", data)
        if f[0] == "ACCEPT":
            return ("ACCEPT", int(f[2]))
        if f[0] == "DONE":
            return ("DONE", f[3:])
        if f[0] == "ERROR":
            return ("ERROR", " ".join(f[2:]))
        return (f[0], f[1:])

    def run(self, ids_or_bytes, **kw):
        """submit and read to DONE / ERROR. Returns (events, terminal)"""
        rid = self.submit(ids_or_bytes, **kw)
        events = []
        while True:
            ev = self.read_event()
            events.append(ev)
            if ev[0] in ("DONE", "ERROR", "EOF"):
                return rid, events, ev

    def log(self, rid, timeout=10):
        """(info line fields, generated ids) the engine logged for request rid"""
        info = gen = None
        end = time.time() + timeout
        while time.time() < end:
            text = self.err_path.read_text(errors="replace")
            # only newline-terminated lines: the engine may still be writing the last one
            # (reading half of a "generated:" line was the source of this test's flakiness)
            for ln in text.split("\n")[:-1]:
                if ln.startswith(f"[kolibri1] serve req {rid}: "):
                    info = dict(kv.split("=", 1) for kv in ln.split(": ", 1)[1].split())
                elif ln.startswith(f"[kolibri1] serve req {rid} generated:"):
                    gen = [int(x) for x in ln.split("generated:")[1].split()]
            if info is not None and gen is not None:
                return info, gen
            time.sleep(0.02)
        raise AssertionError(f"no log for {rid}:\n{self.err_path.read_text(errors='replace')[-2000:]}")

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        try:
            self.p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.err.close()
        self.p.stdout.close()
        return self.p.returncode


def text_of(events):
    return b"".join(e[1] for e in events if e[0] == "DATA")


@unittest.skipUnless(HAVE_NUMPY and BINARY.exists() and REF.exists(),
                     "needs numpy, the kolibri1 binary and kolibri1_tiny/ref.json")
class Kolibri1Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.base = cls.t / "fx"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.base), "--layout", "release", "--force"],
                       check=True, capture_output=True)
        cls.ref = json.loads(REF.read_text())
        shards = json.loads((cls.base / "fixture.json").read_text())["shard_sha256"]
        assert shards == cls.ref["fixture_shard_sha256"], "the fixture differs from the one the oracle ran on"
        FX.add_tokenizer_only(cls.base)
        cls.cases = cls.ref["cases"]
        cls.opened = []

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def serve(self, model=None, **kw):
        s = Serve(model or self.base, tmp=self.t, **kw)
        self.addCleanup(s.close)
        return s

    # ---- protocol ---------------------------------------------------------------------

    def test_handshake(self):
        s = self.serve()
        self.assertEqual(s.handshake[0], b"\x01\x01READY\x01\x01\n")
        self.assertEqual(s.handshake[1], b"CAPS vision=0\n")
        self.assertTrue(s.handshake[2].startswith(b"STAT "))
        self.assertIsNone(s.p.poll())
        self.assertEqual(s.close(), 0)

    def test_idle_cancel_and_unknown_frames(self):
        s = self.serve()
        s.send(b"CANCEL nothing\n")
        self.assertEqual(s.read_event(), ("ERROR", "NOT_FOUND"))
        s.send(b"STOP nothing\n")                       # ignored
        c = self.cases[0]
        _, ev, last = s.run(c["prompt"], max_tokens=3)
        self.assertEqual(last[0], "DONE")

    # ---- the oracle through SERVE -------------------------------------------------------

    def greedy(self, s, c, **kw):
        rid, ev, last = s.run(c["prompt"], max_tokens=c["ngen"], **kw)
        self.assertEqual(last[0], "DONE", ev)
        info, gen = s.log(rid)
        return rid, ev, last, info, gen

    def test_greedy_serve_equals_the_oracle(self):
        for tag, env in (("default", {}), ("chunk3", {"K1_CHUNK": "3"}), ("cap1", {"K1_CAP": "1"}),
                         ("noprefix", {"K1_SERVE_PREFIX": "0"}), ("chunk1", {"K1_CHUNK": "1"})):
            s = self.serve(env=env, cap=1 if tag == "cap1" else 8)
            for c in self.cases:
                with self.subTest(tag=tag, case=c["name"]):
                    rid, ev, last, info, gen = self.greedy(s, c)
                    self.assertEqual(gen, c["generated"])
                    self.assertEqual(int(info["emitted"]), c["ngen"])
                    self.assertEqual(last[1][0], str(c["ngen"]))          # DONE STAT <completion tokens>
                    self.assertEqual(last[1][5], "1")                        # length_limited: the budget ended it
                    self.assertEqual(int(last[1][4]), len(c["prompt"]))
                    self.assertLessEqual(len([e for e in ev if e[0] == "DATA"]), c["ngen"])   # a frame per token that decodes to bytes

    def test_isolation_between_requests(self):
        for env in ({}, {"K1_SERVE_PREFIX": "0"}):
            s = self.serve(env=env)
            order = [0, 1, 2, 0, 1, 2, 2, 0]
            for k, ci in enumerate(order):
                c = self.cases[ci]
                with self.subTest(env=env, k=k):
                    _, _, _, info, gen = self.greedy(s, c)
                    self.assertEqual(gen, c["generated"])
                    self.assertEqual(info["reused"].split("(")[0], "0")     # unrelated prompts reuse nothing

    def test_prefix_reuse_snapshot_live_and_all_or_nothing(self):
        c = self.cases[0]
        P, G = c["prompt"], c["generated"]
        runs = {}
        for tag, env in (("reuse", {}), ("off", {"K1_SERVE_PREFIX": "0"})):
            s = self.serve(env=env)
            out = []
            # 0,1: the prompt twice (the second needs a fresh token, so it reuses nothing).
            # 2: the prompt + 8 generated ids (snapshot path: longer than the prompt, shorter than the state).
            # 3: the whole state (prompt + the 16-token reply, 15 of them fed) + one more id: the live path.
            # 4: only 25 of the ids shared: nothing. 5: the first prompt again.
            plan = [(P, 24), (P, 24), (P + G[:8], 16), (P + G[:8] + G[8:24] + [7], 6), (P[:25] + [3, 4, 5], 6), (P, 24)]
            for ids, n in plan:
                rid, ev, last = s.run(ids, max_tokens=n)
                self.assertEqual(last[0], "DONE", ev)
                out.append(s.log(rid))
            runs[tag] = out
        reuse, off = runs["reuse"], runs["off"]
        for k in range(len(reuse)):
            self.assertEqual(reuse[k][1], off[k][1], f"request {k}: the reused state changed the tokens")
        self.assertEqual(reuse[0][1], G)
        self.assertEqual(reuse[1][1], G)
        self.assertEqual(reuse[2][1], G[8:24])                      # the oracle's own continuation
        how = [r[0]["reused"] for r in reuse]
        self.assertEqual(how[0], "0(none)")
        self.assertEqual(how[1], "0(none)")                          # the same prompt again recomputes
        self.assertEqual(how[2], f"{len(P)}(snapshot)")
        self.assertEqual(how[3], f"{len(P) + 8 + 15}(live)")
        self.assertEqual(how[4], "0(none)")                          # a partial prefix: the ring cannot be rewound
        self.assertEqual(how[5], "0(none)")
        self.assertEqual([r[0]["reused"] for r in off], ["0(none)"] * 6)

    # ---- stop, budget, control ----------------------------------------------------------

    def test_stop_ids_max_tokens_and_context(self):
        c = self.cases[0]
        G = c["generated"]
        s = self.serve(env={"K1_STOP_IDS": str(G[5])})
        rid, ev, last = s.run(c["prompt"], max_tokens=24)
        self.assertEqual(s.log(rid)[1], G[:5])                       # the stop token itself is not emitted
        self.assertEqual(last[1][0], "5")
        self.assertEqual(last[1][5], "0")                            # not length limited
        # the checkpoint's own eos (config.json eos_token_id)
        cfg_dir = self.t / "fx_eos"
        shutil.copytree(self.base, cfg_dir)
        cfg = json.loads((cfg_dir / "config.json").read_text())
        cfg["eos_token_id"] = G[3]
        (cfg_dir / "config.json").write_text(json.dumps(cfg))
        s2 = self.serve(model=cfg_dir)
        rid, ev, last = s2.run(c["prompt"], max_tokens=24)
        self.assertEqual(s2.log(rid)[1], G[:3])
        self.assertEqual(last[1][5], "0")
        # max_tokens
        s3 = self.serve()
        rid, ev, last = s3.run(c["prompt"], max_tokens=5)
        self.assertEqual(s3.log(rid)[1], G[:5])
        self.assertEqual(last[1][5], "1")
        # context: 40-token prompt, CTX 45 -> room for 5 tokens whatever max_tokens says; CTX 40 -> no room
        s4 = self.serve(env={"CTX": "45"})
        rid, ev, last = s4.run(c["prompt"], max_tokens=50)
        self.assertEqual(s4.log(rid)[1], G[:5])
        self.assertEqual(last[1][5], "1")
        s5 = self.serve(env={"CTX": "40"})
        rid, ev, last = s5.run(c["prompt"], max_tokens=4)
        self.assertEqual(last[0], "ERROR")
        self.assertTrue(last[1].startswith("CONTEXT_EXCEEDED"), last)
        rid, ev, last = s5.run(c["prompt"][:30], max_tokens=4)       # and the process still serves
        self.assertEqual(last[0], "DONE")
        rid, ev, last = s5.run(list(range(10, 60)), max_tokens=4)
        self.assertTrue(last[1].startswith("CONTEXT_EXCEEDED"), last)

    def test_stop_frame_ends_the_turn_cleanly(self):
        c = self.cases[0]
        s = self.serve(env={"CTX": "1024"})
        rid = s.submit(c["prompt"], max_tokens=900)
        first = None
        seen = 0
        while True:
            ev = s.read_event()
            if ev[0] == "DATA":
                seen += 1
                if first is None:
                    first = ev
                    s.send(f"STOP {rid}\n".encode())
            if ev[0] in ("DONE", "ERROR"):
                break
        self.assertEqual(ev[0], "DONE", ev)
        self.assertEqual(ev[1][5], "0")
        self.assertLess(seen, 900)
        _, _, last = s.run(c["prompt"], max_tokens=2)               # the next request is served
        self.assertEqual(last[0], "DONE")

    def test_cancel_frame_is_acknowledged_and_state_is_not_reused_wrongly(self):
        c = self.cases[1]
        s = self.serve()
        rid = s.submit(c["prompt"], max_tokens=900)
        seen = 0
        while True:
            ev = s.read_event()
            if ev[0] == "DATA":
                seen += 1
                if seen == 1:
                    s.send(f"CANCEL {rid}\n".encode())
            if ev[0] in ("DONE", "ERROR"):
                break
        self.assertEqual(ev, ("ERROR", "CANCELLED"))
        self.assertLess(seen, 900)
        # a request after the cancel: the oracle's tokens (the aborted state is consistent or reset, never mixed)
        for ci in (1, 0):
            _, _, _, info, gen = self.greedy(s, self.cases[ci])
            self.assertEqual(gen, self.cases[ci]["generated"])

    def test_bad_requests_leave_the_engine_serving(self):
        c = self.cases[2]
        s = self.serve()
        cases = [
            (dict(slot=1), "BAD_SLOT"),
            (dict(ext=" logprobs=3"), "LOGPROBS_UNSUPPORTED"),
            (dict(top_p=-0.5, temperature=1.0), "BAD_SAMPLING"),     # no private negative-top_p encoding: refused
            (dict(top_p=-1.0, temperature=0.0), "BAD_SAMPLING"),
        ]
        for kw, want in cases:
            rid, ev, last = s.run(c["prompt"], max_tokens=3, **kw)
            self.assertEqual(last[0], "ERROR")
            self.assertTrue(last[1].startswith(want), last)
        rid, ev, last = s.run(b"", max_tokens=3)
        self.assertEqual(last, ("ERROR", "EMPTY_PROMPT"))
        rid, ev, last = s.run([1, 2, 99999], max_tokens=3)
        self.assertEqual(last[0], "ERROR")
        self.assertTrue(last[1].startswith("BAD_TOKEN"), last)
        s.send(b"SUBMIT bad 0 notanumber 3 0 1\nxyz\n")             # header refused, framing intact?
        ev = s.read_event()
        self.assertEqual(ev[0], "ERROR")
        _, _, _, info, gen = self.greedy(s, c)
        self.assertEqual(gen, c["generated"])
        self.assertEqual(s.close(), 0)

    # ---- non-finite logits ----------------------------------------------------------------

    def nan_embed_fixture(self, token):
        import kolibri1_tiny_mutants as M
        dst = self.t / f"nan_embed_{token}"
        if dst.exists():
            return dst
        shutil.copytree(self.base, dst)

        def fn(entry):
            name, meta, blob = entry
            b = bytearray(blob)
            H = meta["shape"][1]
            for j in range(H):
                struct.pack_into("<H", b, (token * H + j) * 2, 0x7FC0)
            return name, meta, bytes(b)
        M.TL.edit(dst, "model.embed_tokens.weight", fn)
        return dst

    def test_non_finite_logit_is_an_error_reply_never_a_token(self):
        import kolibri1_tiny_mutants as M
        c = self.cases[0]
        head = self.t / "nan_head"
        M.make_nan_head(self.base, head, row=7)
        FX.add_tokenizer_only(head)
        s = self.serve(model=head)
        for _ in range(2):                                           # and again: the engine did not die
            rid, ev, last = s.run(c["prompt"], max_tokens=6)
            self.assertEqual(last[0], "ERROR")
            self.assertTrue(last[1].startswith("NONFINITE_LOGITS"), last)
            self.assertIn("vocabulary id 7", last[1])
            self.assertEqual(text_of(ev), b"")                       # nothing was streamed
        self.assertIsNone(s.p.poll())
        s.close()
        self.assertIn("non-finite logit", s.err_path.read_text())
        # one embedding row NaN: the router refuses a non-finite activation before any logit exists (a hard, fail-closed
        # exit of the process, as in every other mode); nothing is streamed
        used = set(sum((x["prompt"] + x["generated"] for x in self.cases), []))
        token = next(t for t in range(300, 5, -1) if t not in used)
        bad = self.nan_embed_fixture(token)
        FX.add_tokenizer_only(bad)
        s = self.serve(model=bad)
        _, _, _, info, gen = self.greedy(s, c)
        self.assertEqual(gen, c["generated"])                        # a clean request is exact
        rid, ev, last = s.run(c["prompt"][:10] + [token] + c["prompt"][10:], max_tokens=4)
        self.assertEqual(text_of(ev), b"")
        self.assertNotEqual(s.close(), 0)
        self.assertIn("non-finite", s.err_path.read_text())

    def test_nan_mid_request_is_an_error_and_the_next_request_is_clean(self):
        """K1_TEST_SERVE_NAN_REQ=2: the second request's logits get a NaN before its third token is drawn"""
        A, B = self.cases[0], self.cases[1]
        s = self.serve(env={"K1_TEST_SERVE_NAN_REQ": "2"})
        _, _, _, info, gen = self.greedy(s, A)
        self.assertEqual(gen, A["generated"])
        rid, ev, last = s.run(B["prompt"], max_tokens=24)
        self.assertEqual(last[0], "ERROR")
        self.assertTrue(last[1].startswith("NONFINITE_LOGITS"), last)
        self.assertEqual(len([e for e in ev if e[0] == "DATA"]), 2)  # two tokens went out before the bad row; the third was never drawn
        info, gen = s.log(rid)
        self.assertEqual(info["status"], "error")
        self.assertEqual(gen, B["generated"][:2])
        # the failed request left nothing behind: A (a prompt that would have matched its snapshot/state) and B recompute
        for c in (A, B, A):
            _, _, _, info, gen = self.greedy(s, c)
            self.assertEqual(gen, c["generated"])
            self.assertEqual(info["reused"], "0(none)")
        self.assertEqual(s.close(), 0)

    # ---- sampling -------------------------------------------------------------------------

    def sample_run(self, env, **kw):
        s = self.serve(env=env)
        c = self.cases[0]
        out = []
        for _ in range(2):
            rid, ev, last = s.run(c["prompt"], max_tokens=24, **kw)
            self.assertEqual(last[0], "DONE", ev)
            out.append(s.log(rid)[1])
        return out

    def test_seeded_sampling_is_reproducible(self):
        wide = {"temperature": 2.0, "top_p": 1.0}
        a1, a2 = self.sample_run({"K1_SEED": "7", "K1_TOP_K": "0"}, **wide)
        self.assertEqual(a1, a2)                                     # the same seed re-seeds every request
        (b1, _) = self.sample_run({"K1_SEED": "7", "K1_TOP_K": "0"}, **wide)
        self.assertEqual(a1, b1)                                     # and a new process
        (c1, _) = self.sample_run({"K1_SEED": "8", "K1_TOP_K": "0"}, **wide)
        self.assertNotEqual(a1, c1)
        # SEED works the way mimo's does; K1_SEED wins over it
        (d1, _) = self.sample_run({"SEED": "7", "K1_TOP_K": "0"}, **wide)
        self.assertEqual(a1, d1)
        (e1, _) = self.sample_run({"SEED": "99", "K1_SEED": "7", "K1_TOP_K": "0"}, **wide)
        self.assertEqual(a1, e1)
        self.assertNotEqual(a1, self.cases[0]["generated"])          # it really sampled
        # top_k 1 is greedy whatever the temperature when the maximum is unique (ties: the vendor stack keeps them all, see the sampler test); temperature 0 is greedy whatever the seed
        (g1, _) = self.sample_run({"K1_SEED": "7", "K1_TOP_K": "1"}, **wide)
        self.assertEqual(g1, self.cases[0]["generated"])
        (z1, _) = self.sample_run({"K1_SEED": "7"}, temperature=0.0, top_p=0.5)
        self.assertEqual(z1, self.cases[0]["generated"])
        # without a seed the stream continues, so two identical requests need not agree (and the engine starts)
        u1, u2 = self.sample_run({"K1_TOP_K": "0"}, **wide)
        self.assertEqual(len(u1), 24)


if __name__ == "__main__":
    unittest.main()
