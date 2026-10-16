#!/usr/bin/env python3
"""Benchmark channel of the kolibri1 SERVE loop (K1_BENCH_TOKENS / K1_BENCH_FILE / K1_IGNORE_EOS), CPU only, tiny fixture.

    KOLIBRI1_BIN=./kolibri1 python3 tests/test_kolibri1_bench_channel.py        (needs numpy for the fixture; skipped without it or the binary)

What it pins (fixed-length generation and per-token timing for benchmarking):
  * default SERVE is untouched: no K1BENCH line, same DONE frame fields, K1_IGNORE_EOS without K1_BENCH_TOKENS and K1_BENCH_FILE without it refuse to start;
  * one `tok` line per sampled token with its id, a monotonic non-decreasing nanosecond timestamp, start <= prefill_end <= first token <= end;
  * the stop id is reported (stop_id=1) when it stops the request, and with K1_IGNORE_EOS=1 a max_tokens = D request yields exactly D tokens, the stop ids
    included as ordinary tokens;
  * numerics: the sampled ids with the channel on (with and without K1_IGNORE_EOS, up to the first stop id) are the ids of the default run, token for token;
  * K1_BENCH_FILE moves the lines out of stderr; the context needed for prompt + D is exact.
Runs on any machine with a CPU build (it needs neither a GPU nor the checkpoint)."""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
BIN = os.environ.get("KOLIBRI1_BIN") or str(HERE.parent / "kolibri1")
try:
    import numpy  # noqa: F401
    from kolibri1_serve_fixture import make_serve_fixture, IM_END
    HAVE = True
except ImportError:
    HAVE = False

READY = b"\x01\x01READY\x01\x01"
SANITISED = ("K1_BENCH_TOKENS", "K1_BENCH_FILE", "K1_IGNORE_EOS", "K1_TEST_SERVE_SCRIPT", "K1_STOP_IDS", "K1_SERVE_LOG", "K1_SEED", "SEED", "K1_CTX", "CTX",
             "K1_SERVE_PREFIX", "COLI_KV_PREFIX", "K1_TOP_K", "K1_CHUNK", "K1_CAP", "K1_TEST_SERVE_IDS", "K1_LAYERS", "COLI_CUDA")
LINE = re.compile(r"K1BENCH (\w+) (.*)")


def parse_bench(text):
    out = []
    for ln in text.splitlines():
        m = LINE.search(ln)
        if m:
            kv = dict(x.split("=", 1) for x in m.group(2).split())
            out.append((m.group(1), kv))
    return out


def serve(fixture, prompts, env=None, timeout=300):
    """prompts: list of (id, 'ids:..' payload, max_tokens), sent one at a time (stdin stays open until the last reply: EOF on stdin ends a request, as in the gateway).
    -> (replies, stderr, returncode); replies: id -> {'data': bytes, 'done': str}"""
    import threading
    e = {k: v for k, v in os.environ.items() if k not in SANITISED}
    e.update({"SNAP": str(fixture), "SERVE": "1", "OMP_NUM_THREADS": "1", "K1_TEST_SERVE_IDS": "1", "K1_SERVE_PREFIX": "0", "K1_SERVE_LOG": "2", **(env or {})})
    p = subprocess.Popen([BIN], env=e, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    th = threading.Thread(target=lambda: err.append(p.stderr.read())); th.start()
    replies = {}
    try:
        if p.poll() is None:
            while True:                                   # the READY sentinel (an engine that refused to start closes stdout instead)
                line = p.stdout.readline()
                if not line or line.strip() == READY:
                    break
            for rid, payload, mx in prompts:
                if p.poll() is not None and not line:
                    break
                body = payload.encode()
                p.stdin.write(f"SUBMIT {rid} 0 {len(body)} {mx} 0 1\n".encode() + body + b"\n"); p.stdin.flush()
                while True:
                    line = p.stdout.readline()
                    if not line:
                        break
                    f = line.split()
                    if not f:
                        continue
                    if f[0] == b"DATA":
                        n = int(f[2]); replies.setdefault(f[1].decode(), {"data": b"", "done": None})["data"] += p.stdout.read(n + 1)[:n]
                    elif f[0] in (b"DONE", b"ERROR") and f[1].decode() == rid:
                        replies.setdefault(rid, {"data": b"", "done": None})["done"] = line.decode().strip()
                        break
    except (BrokenPipeError, OSError):
        pass
    try:
        p.stdin.close()
    except OSError:
        pass
    p.wait(timeout=timeout); th.join(); p.stdout.close(); p.stderr.close()
    return replies, (err[0] if err else b"").decode(errors="replace"), p.returncode


def generated(stderr, rid):
    for ln in stderr.splitlines():
        m = re.search(rf"serve req {rid} generated:(.*)", ln)
        if m:
            return [int(x) for x in m.group(1).split()]
    return None


@unittest.skipUnless(HAVE and os.path.exists(BIN), "needs numpy (fixture) and a built kolibri1 (KOLIBRI1_BIN)")
class BenchChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fx = make_serve_fixture(Path(cls.tmp.name) / "fx", gen_config=False)    # no generation_config: the only stop id is config.json eos (292, inside the tiny vocabulary)
        cls.prompt = "ids:5,6,7,8,9,10,11,12"
        cls.script = Path(cls.tmp.name) / "script.txt"
        cls.script.write_text("abc\n")          # request 1 is sampled as the tokens of "abc", then the stop id (test hook)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- default behaviour
    def test_default_serve_has_no_channel(self):
        rep, err, rc = serve(self.fx, [("1", self.prompt, 6)])
        self.assertEqual(rc, 0, err); self.assertNotIn("K1BENCH", err); self.assertTrue(rep["1"]["done"].startswith("DONE 1 STAT"))

    def test_ignore_eos_and_bench_file_need_the_channel(self):
        for env in ({"K1_IGNORE_EOS": "1"}, {"K1_BENCH_FILE": str(Path(self.tmp.name) / "x.txt")}):
            rep, err, rc = serve(self.fx, [("1", self.prompt, 4)], env=env)
            self.assertNotEqual(rc, 0, env); self.assertIn("K1_BENCH_TOKENS", err)

    # ---- the channel
    def test_every_sampled_token_has_an_id_and_a_monotonic_timestamp(self):
        rep, err, rc = serve(self.fx, [("1", self.prompt, 6)], env={"K1_BENCH_TOKENS": "1"})
        self.assertEqual(rc, 0, err)
        ev = parse_bench(err)
        kinds = [k for k, _ in ev]
        ev = [x for x in ev if x[0] != "counters"]; kinds = [k for k, _ in ev]            # the counters lines are checked in their own test
        self.assertEqual(kinds[0], "start"); self.assertEqual(kinds[1], "prefill_end"); self.assertEqual(kinds[-1], "end")
        toks = [kv for k, kv in ev if k == "tok"]
        self.assertGreaterEqual(len(toks), 1)
        t = [int(ev[0][1]["t_ns"]), int(ev[1][1]["t_ns"])] + [int(x["t_ns"]) for x in toks] + [int(ev[-1][1]["t_ns"])]
        self.assertEqual(t, sorted(t)); self.assertTrue(all(x > 0 for x in t))
        self.assertEqual([int(x["i"]) for x in toks], list(range(len(toks))))
        end = ev[-1][1]
        self.assertIn(int(end["emitted"]), (len(toks), len(toks) - 1))          # the stop id, if sampled, is a tok line but not an emitted token

    def test_cumulative_counters_at_start_and_end(self):
        rep, err, rc = serve(self.fx, [("1", self.prompt, 6), ("2", self.prompt, 6)], env={"K1_BENCH_TOKENS": "1", "K1_IGNORE_EOS": "1"})
        self.assertEqual(rc, 0, err)
        c = {(kv["req"], kv["at"]): kv for k, kv in parse_bench(err) if k == "counters"}
        self.assertEqual(set(c), {("1", "start"), ("1", "end"), ("2", "start"), ("2", "end")})
        loads = [int(c[x]["ec_loads"]) for x in (("1", "start"), ("1", "end"), ("2", "start"), ("2", "end"))]
        self.assertEqual(loads, sorted(loads)); self.assertGreater(loads[1], loads[0])          # a request reads experts through the engine's cache
        self.assertNotIn("sel_total", c[("1", "end")])
        self.assertEqual(int(c[("2", "start")]["ec_loads"]), int(c[("1", "end")]["ec_loads"]))   # nothing read between two requests

    def test_numerics_are_the_default_runs(self):
        base, err0, rc0 = serve(self.fx, [("1", self.prompt, 12)])
        ref = generated(err0, "1")
        self.assertEqual(rc0, 0, err0); self.assertIsNotNone(ref)
        for env in ({"K1_BENCH_TOKENS": "1"}, {"K1_BENCH_TOKENS": "1", "K1_IGNORE_EOS": "1"}):
            rep, err, rc = serve(self.fx, [("1", self.prompt, 12)], env=env)
            self.assertEqual(rc, 0, err)
            toks = [int(kv["id"]) for k, kv in parse_bench(err) if k == "tok"]
            self.assertEqual(toks[:len(ref)], ref, env)                              # the default run's ids, token for token
            if "K1_IGNORE_EOS" not in env:
                self.assertEqual(rep["1"]["data"], base["1"]["data"])

    # ---- the stop id and the fixed length
    def test_stop_id_is_reported_and_stops_by_default(self):
        rep, err, rc = serve(self.fx, [("1", self.prompt, 10)], env={"K1_BENCH_TOKENS": "1", "K1_TEST_SERVE_SCRIPT": str(self.script)})
        self.assertEqual(rc, 0, err)
        toks = [kv for k, kv in parse_bench(err) if k == "tok"]
        self.assertEqual(toks[-1]["stop_id"], "1"); self.assertEqual(int(toks[-1]["id"]), IM_END)
        self.assertTrue(all(x["stop_id"] == "0" for x in toks[:-1]))
        self.assertEqual(int([kv for k, kv in parse_bench(err) if k == "end"][0]["emitted"]), len(toks) - 1)
        self.assertIn(" 0\n", rep["1"]["done"] + "\n")                             # limited=0 in DONE: a stop, not the token limit

    def test_ignore_eos_gives_exactly_max_tokens(self):
        for d in (1, 5, 10):
            rep, err, rc = serve(self.fx, [("1", self.prompt, d)], env={"K1_BENCH_TOKENS": "1", "K1_IGNORE_EOS": "1", "K1_TEST_SERVE_SCRIPT": str(self.script)})
            self.assertEqual(rc, 0, err)
            toks = [kv for k, kv in parse_bench(err) if k == "tok"]
            self.assertEqual(len(toks), d)
            end = [kv for k, kv in parse_bench(err) if k == "end"][0]
            self.assertEqual((int(end["emitted"]), end["limited"], end["status"]), (d, "1", "ok"))
            if d > 3:
                self.assertEqual(toks[-1]["stop_id"], "1")                          # past the script the stop id keeps being sampled and recorded, not obeyed

    def test_context_prompt_plus_d_is_exact(self):
        d = 7
        n_prompt = len(self.prompt[4:].split(","))
        rep, err, rc = serve(self.fx, [("1", self.prompt, d)], env={"K1_BENCH_TOKENS": "1", "K1_IGNORE_EOS": "1", "K1_CTX": str(n_prompt + d), "K1_TEST_SERVE_SCRIPT": str(self.script)})
        self.assertEqual(rc, 0, err)
        self.assertEqual(len([1 for k, _ in parse_bench(err) if k == "tok"]), d)

    def test_two_requests_each_have_their_own_events(self):
        rep, err, rc = serve(self.fx, [("1", self.prompt, 3), ("2", self.prompt, 4)], env={"K1_BENCH_TOKENS": "1", "K1_IGNORE_EOS": "1"})
        self.assertEqual(rc, 0, err)
        ev = parse_bench(err)
        for rid, n in (("1", 3), ("2", 4)):
            self.assertEqual(len([1 for k, kv in ev if k == "tok" and kv["req"] == rid]), n)
            self.assertEqual([k for k, kv in ev if kv.get("req") == rid and k not in ("tok", "counters")], ["start", "prefill_end", "end"])

    def test_bench_file_takes_the_lines_off_stderr(self):
        f = Path(self.tmp.name) / "bench.txt"
        if f.exists():
            f.unlink()
        rep, err, rc = serve(self.fx, [("1", self.prompt, 4)], env={"K1_BENCH_TOKENS": "1", "K1_BENCH_FILE": str(f), "K1_IGNORE_EOS": "1"})
        self.assertEqual(rc, 0, err); self.assertNotIn("K1BENCH ", err)
        ev = parse_bench(f.read_text())
        self.assertEqual(len([1 for k, _ in ev if k == "tok"]), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
