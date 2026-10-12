#!/usr/bin/env python3
"""kolibri1 CPU forward on the tiny fixture against the committed oracle output.

Needs numpy, the ./kolibri1 binary (make kolibri1) and kolibri1_tiny/ref.json (written by
tools/make_kolibri1_ref.py from the vendored reference; CI needs no torch).
The fixture is regenerated into a temp dir and its shard hashes must equal the ones the
oracle was run on.

Checks (every comparison with the oracle is on the oracle's own numbers):
  * token-exact greedy continuation for 3 cases, each longer than the window (17) and
    wrapping the sliding ring (K1_CHUNK 64 / 3 / 1, an expert cache that evicts, both FP8 kernels)
  * teacher-forced argmax at EVERY position through K1_LOGITS, against the oracle's argmax
  * the oracle's last-row top-5 logits within 1e-4 of their RMS (a numerical gate that does
    not rely on argmax)
  * the logits (every bit) are identical for K1_CHUNK 64/3/1, for a cache of 1 slot per layer,
    for 1 and 4 threads, between two runs, and between prefill rows and decode rows
  * the mutation matrix: every mutation targets an exercised expert; loader mutations are
    refused before any forward ran (no output, no logits file); the numeric ones load and are
    caught by the logit gate (the layer/operator probes against the reference catch them too)
"""
import hashlib
import json
import os
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()
REF = Path(os.environ.get("K1_REF", ROOT / "kolibri1_tiny" / "ref.json"))


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def engine(directory, ids, ngen=0, env=None, logits=None, timeout=300):
    e = {**os.environ, "OMP_NUM_THREADS": "2", **(env or {})}
    e.pop("K1_LOGITS", None)
    if logits:
        e["K1_LOGITS"] = str(logits)
    r = subprocess.run([str(BINARY), str(directory), "--ids", ",".join(map(str, ids)), "--ngen", str(ngen)],
                       capture_output=True, text=True, timeout=timeout, env=e)
    gen = None
    for line in r.stdout.splitlines():
        if line.startswith("generated:"):
            gen = [int(t) for t in line.split()[1:]]
    return r, gen


def evictions(stderr):
    for line in stderr.splitlines():
        if "expert cache:" in line:
            return int(line.split("loads, ")[1].split(" evictions")[0])
    raise AssertionError(stderr)


class Kolibri1Forward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [w for w, ok in (("numpy", HAVE_NUMPY), (f"the binary {BINARY} (make kolibri1)", BINARY.exists()),
                                   (f"the oracle {REF}", REF.exists())) if not ok]
        if missing:
            msg = "the forward checks need " + ", ".join(missing)
            if os.environ.get("K1_REQUIRE_FORWARD") == "1":      # the required Makefile target: a missing prerequisite is a FAILURE
                raise AssertionError(msg + " -- K1_REQUIRE_FORWARD=1 forbids skipping")
            raise unittest.SkipTest(msg)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.tmp.name) / "fx"
        subprocess.run([sys.executable, str(TOOL), "--out", str(cls.base), "--layout", "release", "--force"],
                       check=True, capture_output=True)
        cls.ref = json.loads(REF.read_text())
        shards = json.loads((cls.base / "fixture.json").read_text())["shard_sha256"]
        assert shards == cls.ref["fixture_shard_sha256"], "the fixture differs from the one the oracle ran on"
        assert sha256_file(cls.base / "config.json") == cls.ref["config_sha256"]
        cls.V = cls.ref["vocab"]
        cls.logits_cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tf(self, case, env=None, tag="default", directory=None):
        """teacher-forced logits [positions, V] of prompt+generated, ngen = 0"""
        key = (case["name"], tag, str(directory))
        if key in self.logits_cache:
            return self.logits_cache[key]
        ids = case["prompt"] + case["generated"]
        out = Path(self.tmp.name) / f"tf_{case['name']}_{tag}.f32"
        r, _ = engine(directory or self.base, ids, 0, env, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        a = np.fromfile(out, dtype=np.float32).reshape(len(ids), self.V)
        self.logits_cache[key] = a
        return a

    # ---- the oracle ---------------------------------------------------------------

    def test_cases_cross_the_window_and_the_oracle_is_decisive(self):
        w = self.ref["window"]
        for c in self.ref["cases"]:
            self.assertGreater(len(c["prompt"]), w)
            self.assertGreater(len(c["prompt"]) + len(c["generated"]), 2 * w)
        self.assertGreater(self.ref["min_margin_all_cases"], 1e-2)       # no near-tie argmax anywhere

    def test_greedy_token_exact(self):
        for c in self.ref["cases"]:
            for tag, env in (("chunk64", {}), ("chunk3", {"K1_CHUNK": "3"}), ("chunk1", {"K1_CHUNK": "1"}),
                             ("cap1", {"K1_CAP": "1"}), ("scalar", {"K1_FP8_KERNEL": "scalar"})):
                r, gen = engine(self.base, c["prompt"], c["ngen"], env)
                self.assertEqual(r.returncode, 0, r.stderr)
                self.assertEqual(gen, c["generated"], f"case {c['name']} {tag}")

    def test_teacher_forced_argmax_at_every_position(self):
        for c in self.ref["cases"]:
            lg = self.tf(c)
            self.assertTrue(np.isfinite(lg).all())
            self.assertEqual([int(t) for t in lg.argmax(-1)], c["argmax"], f"case {c['name']}")

    def test_last_row_top5_logits_match_the_oracle(self):
        for c in self.ref["cases"]:
            row = self.tf(c)[-1]
            ids, want = c["last_top5"]["ids"], np.array(c["last_top5"]["logits"])
            got = row[ids]
            tol = 1e-4 * float(np.sqrt((want ** 2).mean()))
            self.assertLessEqual(float(np.abs(got - want).max()), tol, f"case {c['name']}: {got} vs {want}")
            self.assertEqual([int(t) for t in np.argsort(-row)[:5]], ids)

    # ---- determinism and invariances -------------------------------------------------

    def test_logits_bit_identical_across_chunking_cache_threads_and_runs(self):
        c = self.ref["cases"][0]
        ref_lg = self.tf(c)
        for tag, env in (("chunk3", {"K1_CHUNK": "3"}), ("chunk1", {"K1_CHUNK": "1"}), ("chunk17", {"K1_CHUNK": "17"}),
                         ("chunk5000", {"K1_CHUNK": "5000"}), ("cap1", {"K1_CAP": "1"}), ("cap3", {"K1_CAP": "3"}),
                         ("thr1", {"OMP_NUM_THREADS": "1"}), ("thr4", {"OMP_NUM_THREADS": "4"}), ("rerun", {})):
            lg = self.tf(c, env, tag)
            self.assertTrue(np.array_equal(lg.view(np.uint32), ref_lg.view(np.uint32)), f"{tag}: logits differ bitwise")

    def test_cache_that_evicts_every_layer(self):
        c = self.ref["cases"][1]
        r, _ = engine(self.base, c["prompt"], 4, {"K1_CAP": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertGreaterEqual(evictions(r.stderr), 6 * 8)       # 8 exercised experts in each of the 6 layers, one slot each
        r, _ = engine(self.base, c["prompt"], 4, {})
        self.assertEqual(evictions(r.stderr), 0)

    def test_decode_rows_equal_prefill_rows_bitwise(self):
        c = self.ref["cases"][2]
        pre = self.tf(c)
        out = Path(self.tmp.name) / "decode.f32"
        r, gen = engine(self.base, c["prompt"], c["ngen"], {"K1_CHUNK": "7"}, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        dec = np.fromfile(out, dtype=np.float32).reshape(-1, self.V)
        self.assertEqual(gen, c["generated"])
        self.assertEqual(dec.shape[0], len(c["prompt"]) + c["ngen"] - 1)
        self.assertTrue(np.array_equal(dec.view(np.uint32), pre[:dec.shape[0]].view(np.uint32)))

    # ---- mutation matrix ----------------------------------------------------------------

    def mutant(self, kind):
        import kolibri1_tiny_mutants as M
        d = Path(self.tmp.name) / f"mut_{kind}"
        info = M.make_mutant(kind, self.base, d, self.ref)
        return d, info

    def test_non_finite_logit_is_a_diagnostic_and_a_nonzero_exit(self):
        """a NaN in one lm_head row (every position's logit for that token becomes NaN): the engine must stop with a
        diagnostic, exit non-zero, print no generated tokens and leave no usable logits; the clean fixture still works"""
        import kolibri1_tiny_mutants as M
        dst = Path(self.tmp.name) / "nan_head"
        M.make_nan_head(self.base, dst, row=7)
        ids = self.ref["cases"][0]["prompt"][:20]
        for env in ({}, {"K1_CHUNK": "3"}, {"K1_CHUNK": "1"}):
            r, gen = engine(dst, ids, 4, env)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("non-finite logit", r.stderr)
            self.assertIn("vocabulary id 7", r.stderr)
            self.assertIsNone(gen)
        r, gen = engine(self.base, ids, 4)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_mutations_target_exercised_experts(self):
        import kolibri1_tiny_mutants as M
        layer, e, n = M.pick_expert(self.ref)
        self.assertGreater(n, 0)
        self.assertIn(str(e), self.ref["exercised_experts"][str(layer)])

    def test_loader_mutations_are_refused_before_any_forward(self):
        import kolibri1_tiny_mutants as M
        c = self.ref["cases"][0]
        needles = {"scale_nonfinite": "non-finite scale", "swap_down_up": "shape", "grid_transposed_attn": "TRANSPOSED"}
        for kind, needle in needles.items():
            self.assertEqual(M.KINDS[kind], "loader")
            d, info = self.mutant(kind)
            out = Path(self.tmp.name) / f"{kind}.f32"
            r, gen = engine(d, c["prompt"], c["ngen"], {}, out)
            self.assertNotEqual(r.returncode, 0, kind)
            self.assertIn(needle, r.stderr, f"{kind}: {r.stderr}")
            self.assertIsNone(gen, kind)
            self.assertFalse(out.exists(), f"{kind}: a forward pass started")
            if "tensor" in info:
                self.assertIn(info["tensor"], r.stderr)

    def test_numeric_mutations_load_and_the_logit_gate_catches_them(self):
        import kolibri1_tiny_mutants as M
        c = self.ref["cases"][0]
        want = np.array(c["last_top5"]["logits"])
        tol = 1e-4 * float(np.sqrt((want ** 2).mean()))
        clean = self.tf(c)
        for kind in ("scale_exp_flip", "swap_gate_up", "grid_transposed_expert"):
            self.assertEqual(M.KINDS[kind], "numeric")
            d, info = self.mutant(kind)
            lg = self.tf(c, tag=kind, directory=d)          # loads and runs: no refusal
            dev = float(np.abs(lg[-1][c["last_top5"]["ids"]] - want).max())
            self.assertGreater(dev, tol, f"{kind}: the logit gate did not bite ({dev:.3g} <= {tol:.3g})")
            self.assertFalse(np.array_equal(lg.view(np.uint32), clean.view(np.uint32)), kind)
            tokens_differ = [int(t) for t in lg.argmax(-1)] != c["argmax"]
            print(f"  mutant {kind}: layer {info['layer']} expert {info['expert']} (used {info['expert_uses']}x): "
                  f"top-5 logit deviation {dev:.3g} (tolerance {tol:.2g}), argmax sequence changed: {tokens_differ}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
