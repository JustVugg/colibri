#!/usr/bin/env python3
"""kolibri1 attention-geometry / cache / capture hooks on the tiny fixture.  numpy + the binary only.

The comparison of the same hooks against the reference implementation on real weights is
maintainer-side tooling and is not part of this repository; this file is the CI-sized part: it needs no torch.

  * K1_ATTN_KEYS: every query of every layer visits exactly the keys SEM:89 allows (window 17: n = min(p+1, 17), first = max(0, p-16),
    last = p on sliding layers; everything from 0 on the full layer), and the ring holds the newest `rows` positions after the run
  * the same capture (x_out, z, ids, weights) and the same logits are produced, bit for bit, by one block, chunks 3 / 64, every row S = 1
    and by K1_DECODE_TAIL (prefill + single-token steps across the ring wrap)
  * capture integrity: chunks.jsonl hashes every 128-position chunk of every file, the engine writes ENGINE.json only at a clean end
    and never DONE.json, a killed run leaves no ENGINE.json, a wrong K1_FEED_SHA256 is refused
  * the logits stream: frames cover every position exactly once, <= 128 rows, and equal the K1_LOGITS dump bit for bit
  * request isolation: A;B == A alone, B alone; without reset B continues after A; a prefix snapshot reproduces the rest of a request
  * K1_LAYERS truncates loudly
"""
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
REF = Path(os.environ.get("K1_REF", ROOT / "kolibri1_tiny" / "ref.json"))
V = 320


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Capture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [w for w, ok in (("numpy", HAVE_NUMPY), (f"the binary {BINARY}", BINARY.exists()), (f"the oracle {REF}", REF.exists())) if not ok]
        if missing:
            msg = "the capture checks need " + ", ".join(missing)
            if os.environ.get("K1_REQUIRE_FORWARD") == "1":
                raise AssertionError(msg + " -- K1_REQUIRE_FORWARD=1 forbids skipping")
            raise unittest.SkipTest(msg)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.fx = cls.t / "fx"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.fx), "--layout", "release", "--force"], check=True, capture_output=True)
        cls.ref = json.loads(REF.read_text())
        cfg = json.loads((cls.fx / "config.json").read_text())
        cls.window = cfg["sliding_window"]
        cls.full = [t == "full_attention" for t in cfg["layer_types"]]
        c = cls.ref["cases"][0]
        cls.ids = (c["prompt"] + c["generated"])
        cls.T = len(cls.ids)
        cls.feed = cls.t / "feed.bin"
        np.asarray(cls.ids, dtype=np.int32).tofile(cls.feed)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_engine(self, args=(), env=None, ok=True):
        e = {**os.environ, "OMP_NUM_THREADS": "2", **(env or {})}
        for k in ("K1_LOGITS", "K1_CAPTURE", "K1_LOGITS_FIFO", "K1_ATTN_KEYS", "K1_FEED", "K1_TEST_KILL_AT", "K1_LAYERS"):
            if k not in (env or {}):
                e.pop(k, None)
        r = subprocess.run([str(BINARY), str(self.fx), *args], capture_output=True, text=True, env=e, timeout=300)
        if ok:
            self.assertEqual(r.returncode, 0, r.stderr)
        return r

    def capture(self, name, env=None, fifo=False):
        d = self.t / name
        e = {"K1_CAPTURE": str(d), "K1_FEED": str(self.feed), **(env or {})}
        fp = None
        if fifo:
            fp = self.t / (name + ".stream"); fp.write_bytes(b"")
            e["K1_LOGITS_FIFO"] = str(fp)
        r = self.run_engine((), e, ok=True)
        return d, fp, r

    def frames(self, path):
        raw = Path(path).read_bytes(); off = 0; out = []
        while off < len(raw):
            magic, pos0, n, v = raw[off:off + 4], *struct.unpack("<III", raw[off + 4:off + 16])
            self.assertEqual(magic, b"K1LG"); self.assertEqual(v, V); self.assertTrue(1 <= n <= 128)
            out.append((pos0, n, np.frombuffer(raw, dtype=np.float32, count=n * v, offset=off + 16).reshape(n, v)))
            off += 16 + n * v * 4
        return out

    # ------------------------------------------------------------------ attention keys ----
    def test_attention_keys_and_ring(self):
        for tag, env in (("chunk64", {}), ("chunk3", {"K1_CHUNK": "3"}), ("chunk1", {"K1_CHUNK": "1"}), ("tail20", {"K1_DECODE_TAIL": "20"})):
            kp = self.t / f"keys_{tag}.txt"
            self.run_engine((), {"K1_FEED": str(self.feed), "K1_ATTN_KEYS": str(kp), **env})
            seen = set(); R = {}; F = []
            for ln in kp.read_text().splitlines():
                t = ln.split()
                if t[0] == "K":
                    l, pos, first, n, last = map(int, t[1:])
                    if self.full[l]:
                        exp = (0, pos + 1, pos)
                    else:
                        exp = (max(0, pos - (self.window - 1)), min(pos + 1, self.window), pos)
                    self.assertEqual((first, n, last), exp, f"{tag} layer {l} pos {pos}")
                    self.assertNotIn((l, pos), seen); seen.add((l, pos))
                elif t[0] == "R":
                    _, l, slot, pos = map(int, t[1:]); R.setdefault(l, {})[slot] = pos
                elif t[0] == "F":
                    F.append(tuple(map(int, t[1:])))
            L = len(self.full)
            self.assertEqual(len(seen), L * self.T, tag)
            self.assertGreater(self.T, 2 * self.window)           # the ring wraps more than once
            for l in range(L):
                if self.full[l]:
                    self.assertIn((0, l, self.T), F)
                    continue
                rows = min(self.window, self.T)
                self.assertEqual(len(R[l]), rows)
                for s in range(rows):
                    self.assertEqual(R[l][s], max(p for p in range(self.T) if p % rows == s), f"{tag} layer {l} slot {s}")

    # --------------------------------------------------------------- bitwise variants ----
    def test_capture_and_logit_stream_are_bit_identical_across_chunking_and_decode(self):
        base = None
        for tag, env in (("one-block", {"K1_CHUNK": str(self.T)}), ("chunk64", {}), ("chunk3", {"K1_CHUNK": "3"}), ("chunk1", {"K1_CHUNK": "1"}),
                         ("tail25", {"K1_DECODE_TAIL": "25"}), ("tail60", {"K1_DECODE_TAIL": "60", "K1_CHUNK": "7"})):
            d, fp, _ = self.capture("cap_" + tag, env, fifo=True)
            files = {p.name: sha(p.read_bytes()) for p in sorted(d.iterdir()) if p.name[0] == "L"}
            fr = self.frames(fp)
            pos = 0
            for p0, n, _lg in fr:
                self.assertEqual(p0, pos); pos += n
            self.assertEqual(pos, self.T)
            logits = np.concatenate([lg for _, _, lg in fr])
            self.assertTrue(np.isfinite(logits).all())
            rows = [sha(r.tobytes()) for r in logits]
            self.assertEqual(len(files), 4 * len(self.full))
            if base is None:
                base = (files, rows, logits)
            else:
                self.assertEqual(files, base[0], tag)
                self.assertEqual(rows, base[1], tag)
        # the stream equals the plain K1_LOGITS dump
        lp = self.t / "dump.f32"
        self.run_engine(("--ids", ",".join(map(str, self.ids))), {"K1_LOGITS": str(lp)})
        dump = np.fromfile(lp, dtype=np.float32).reshape(self.T, V)
        self.assertTrue(np.array_equal(dump, base[2]))

    # ------------------------------------------------------------------- integrity ----
    def test_chunk_manifest_and_completion_files(self):
        d, _, _ = self.capture("cap_int")
        names = {p.name for p in d.iterdir()}
        self.assertIn("ENGINE.json", names); self.assertIn("chunks.jsonl", names); self.assertNotIn("DONE.json", names)
        eng = json.loads((d / "ENGINE.json").read_text())
        self.assertEqual((eng["errors"], eng["positions"], eng["layers"]), (0, self.T, len(self.full)))
        self.assertEqual(eng["feed_sha256"], sha(self.feed.read_bytes()))
        recs = [json.loads(l) for l in (d / "chunks.jsonl").read_text().splitlines()]
        self.assertEqual(len(recs), 4 * len(self.full) * ((self.T + 127) // 128))
        for r in recs:
            raw = (d / r["file"]).read_bytes()[r["offset"]:r["offset"] + r["bytes"]]
            self.assertEqual(len(raw), r["bytes"]); self.assertEqual(sha(raw), r["sha256"])
        cfg = json.loads((self.fx / "config.json").read_text())
        rb = {"x_out.f64": cfg["hidden_size"] * 8, "z.f32": cfg["num_experts"] * 4, "sel.i16": cfg["num_experts_per_tok"] * 2, "w.f32": cfg["num_experts_per_tok"] * 4}
        for l in range(len(self.full)):
            for k, b in rb.items():
                self.assertEqual((d / f"L{l:02d}.{k}").stat().st_size, self.T * b)
        # a rerun in the same directory removes a stale DONE.json
        (d / "DONE.json").write_text("{}")
        self.capture("cap_int")
        self.assertFalse((d / "DONE.json").exists())

    def test_killed_run_leaves_no_engine_json(self):
        d = self.t / "cap_kill"
        r = self.run_engine((), {"K1_CAPTURE": str(d), "K1_FEED": str(self.feed), "K1_TEST_KILL_AT": "50"}, ok=False)
        self.assertEqual(r.returncode, -9)
        self.assertFalse((d / "ENGINE.json").exists()); self.assertFalse((d / "DONE.json").exists())
        self.assertLess(len((d / "chunks.jsonl").read_text().splitlines()), 4 * len(self.full))

    def test_wrong_feed_hash_is_refused(self):
        r = self.run_engine((), {"K1_FEED": str(self.feed), "K1_FEED_SHA256": "0" * 64, "K1_CAPTURE": str(self.t / "cap_bad")}, ok=False)
        self.assertNotEqual(r.returncode, 0); self.assertIn("K1_FEED_SHA256", r.stderr)

    def test_early_failures_invalidate_completion_markers(self):
        """an old ENGINE.json / DONE.json must not survive a rerun that fails BEFORE the capture opens
        (bad feed hash, missing or malformed feed, unusable checkpoint, K1_CAPTURE with several requests)."""
        feed_odd = self.t / "feed_odd.bin"; feed_odd.write_bytes(b"\x01\x02\x03")
        cases = {
            "bad feed sha": ((), {"K1_FEED": str(self.feed), "K1_FEED_SHA256": "0" * 64}, None),
            "feed missing": ((), {"K1_FEED": str(self.t / "nope.bin")}, None),
            "feed not int32": ((), {"K1_FEED": str(feed_odd)}, None),
            "unusable checkpoint": ((), {"K1_FEED": str(self.feed)}, str(self.t / "no_such_checkpoint")),
        }
        for name, (args, env, ckpt) in cases.items():
            d = self.t / ("inval_" + name.replace(" ", "_")); d.mkdir(exist_ok=True)
            for n in ("DONE.json", "ENGINE.json", "DONE.json.tmp", "ENGINE.json.tmp"):
                (d / n).write_text("{}")
            e = {**os.environ, "OMP_NUM_THREADS": "2", "K1_CAPTURE": str(d), **env}
            for k in ("K1_LOGITS", "K1_LOGITS_FIFO", "K1_ATTN_KEYS", "K1_TEST_KILL_AT", "K1_LAYERS"):
                e.pop(k, None)
            r = subprocess.run([str(BINARY), ckpt or str(self.fx), *args], capture_output=True, text=True, env=e, timeout=300)
            self.assertNotEqual(r.returncode, 0, name)
            for n in ("DONE.json", "ENGINE.json", "DONE.json.tmp", "ENGINE.json.tmp"):
                self.assertFalse((d / n).exists(), f"{name}: stale {n} survived a failed rerun")

    def test_capture_needs_one_teacher_forced_request(self):
        r = self.run_engine(("--ids", "1,2,3;4,5", "--ngen", "0"), {"K1_CAPTURE": str(self.t / "cap_two")}, ok=False)
        self.assertNotEqual(r.returncode, 0)

    def test_truncated_model_is_loud(self):
        d, _, r = self.capture("cap_l2", {"K1_LAYERS": "2"})
        self.assertIn("TRUNCATED MODEL", r.stderr)
        self.assertEqual(json.loads((d / "ENGINE.json").read_text())["layers"], 2)
        self.assertFalse((d / "L02.x_out.f64").exists())
        self.assertTrue(json.loads((d / "ENGINE.json").read_text())["truncated"])

    # -------------------------------------------------------------- cache and requests ----
    def logits(self, reqs, env=None):
        lp = self.t / "lg"
        arg = ";".join(",".join(map(str, r)) for r in reqs)
        r = self.run_engine(("--ids", arg, "--ngen", "0"), {"K1_LOGITS": str(lp), **(env or {})})
        paths = [Path(str(lp) + f".r{k}") for k in range(len(reqs))] if len(reqs) > 1 else [lp]
        out = [np.fromfile(p, dtype=np.float32).reshape(-1, V) for p in paths]
        for p in paths: p.unlink()
        return out, r

    def test_reset_isolation_continuation_and_prefix_reuse(self):
        ids = self.ids
        A, B = ids[:50], ids[20:20 + 40][::-1]
        P = 30
        A2 = A[:P] + B[:25]
        (a_,), _ = self.logits([A]); (b_,), _ = self.logits([B]); (a2_,), _ = self.logits([A2]); (ab_,), _ = self.logits([A + B])
        (r0, r1), _ = self.logits([A, B])
        self.assertTrue(np.array_equal(r0, a_) and np.array_equal(r1, b_))
        (s0, s1), _ = self.logits([B, A])
        self.assertTrue(np.array_equal(s0, b_) and np.array_equal(s1, a_))
        (c0, c1), _ = self.logits([A, B], {"K1_RESET": "0"})
        self.assertTrue(np.array_equal(c1, ab_[len(A):]))             # continues after A
        self.assertFalse(np.array_equal(c1, b_))                      # the cache carries state: the isolation test has teeth
        for ch in ("64", "7"):
            (u0, u1), r = self.logits([A, A2], {"K1_REUSE_PREFIX": str(P), "K1_CHUNK": ch})
            self.assertIn("prefix restored", r.stdout)
            self.assertTrue(np.array_equal(u0, a_)); self.assertEqual(u1.shape[0], len(A2) - P)
            self.assertTrue(np.array_equal(u1, a2_[P:]), ch)
        (n0, n1), r = self.logits([A, B], {"K1_REUSE_PREFIX": str(P)})
        self.assertNotIn("prefix restored", r.stdout); self.assertTrue(np.array_equal(n1, b_))


if __name__ == "__main__":
    unittest.main()
