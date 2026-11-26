"""tools/q38_logit_cmp.py on synthetic Q38_LOGITS_DUMP files, written the way qwen38.c's tf_nll
writes them (position, NLL, log Z, the top 32 as id:logit with %.6f), and, when the engine and the
tiny block-typed GGUF (make qwen38-tiny-gguf-generate) are there, on two real dumps of the same run.
"""
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import q38_logit_cmp as lc  # noqa: E402

ENGINE = HERE / ("qwen38.exe" if os.name == "nt" else "qwen38")
FIXTURE = Path(os.environ.get("QWEN38_TINY_GGUF_BLK", HERE / "qwen38_tiny_gguf_blk"))


def dump_lines(logits, forced, start=10):
    """The C writer: for each row a line, top 32 best first, ties by the lower id first."""
    out = []
    for k, (row, tok) in enumerate(zip(logits, forced)):
        row = row.astype(np.float32).astype(np.float64)
        mx = row.max(); z = mx + math.log(np.exp(row - mx).sum())
        order = sorted(range(row.size), key=lambda v: (-row[v], v))[:32]
        out.append(f"{start + k} {z - row[tok]:.6f} {z:.6f} " + " ".join(f"{v}:{row[v]:.6f}" for v in order))
    return "\n".join(out) + "\n"


def run_cli(*args):
    so, se = io.StringIO(), io.StringIO()
    with redirect_stdout(so), redirect_stderr(se):
        rc = lc.main(list(args))
    return rc, so.getvalue(), se.getvalue()


class LogitCmpTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        rng = np.random.default_rng(7)
        self.base = rng.standard_normal((200, 320)) * 3.0
        self.forced = rng.integers(0, 320, 200)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, logits, forced=None, start=10):
        p = self.dir / name
        p.write_text(dump_lines(logits, self.forced if forced is None else forced, start))
        return str(p)

    def test_identical_dumps_pass_strict(self):
        a = self.write("a.txt", self.base); b = self.write("b.txt", self.base)
        rc, out, _ = run_cli(a, b, "--criterion", "strict")
        self.assertEqual(rc, 0, out)
        _, s = lc.compare([(lc.parse_dump(a), lc.parse_dump(b))])
        self.assertEqual(s["top1_agree"], 1.0)
        self.assertEqual(s["max_kl"], 0.0)
        self.assertEqual(s["max_dnll"], 0.0)
        self.assertEqual(s["min_shared"], 32)

    def test_kl_matches_a_direct_computation(self):
        rng = np.random.default_rng(3)
        cand = self.base + rng.standard_normal(self.base.shape) * 0.05
        a, b = lc.parse_dump(self.write("a.txt", self.base)), lc.parse_dump(self.write("b.txt", cand))
        rows, _ = lc.compare([(a, b)])
        for r in rows[:20]:
            ta, tb = dict(a[r["pos"]][2]), dict(b[r["pos"]][2])
            ids = [i for i, _ in a[r["pos"]][2] if i in tb]
            pa = np.array([ta[i] for i in ids]); pb = np.array([tb[i] for i in ids])
            pa = np.exp(pa - pa.max()); pa /= pa.sum(); pb = np.exp(pb - pb.max()); pb /= pb.sum()
            self.assertAlmostEqual(r["kl"], float(np.sum(pa * np.log(pa / pb))), places=9)
            self.assertEqual(r["shared"], len(ids))

    def test_rounding_noise_passes_strict(self):
        rng = np.random.default_rng(5)
        cand = self.base + rng.standard_normal(self.base.shape) * 1e-3
        rc, out, _ = run_cli(self.write("a.txt", self.base), self.write("b.txt", cand))
        self.assertEqual(rc, 0, out)

    def test_moderate_noise_fails_strict_passes_loose(self):
        rng = np.random.default_rng(11)
        cand = self.base + rng.standard_normal(self.base.shape) * 0.09
        a, b = self.write("a.txt", self.base), self.write("b.txt", cand)
        _, s = lc.compare([(lc.parse_dump(a), lc.parse_dump(b))])
        self.assertTrue(1e-3 < s["mean_kl"] < 5e-3, s["mean_kl"])   # the case the two criteria part on
        rc_strict, out, _ = run_cli(a, b, "--criterion", "strict", "--top1", "0")
        rc_loose, out2, _ = run_cli(a, b, "--criterion", "loose", "--top1", "0")
        self.assertEqual(rc_strict, 1, out)
        self.assertEqual(rc_loose, 0, out2)

    def test_top1_flips_fail(self):
        cand = self.base.copy()
        for k in range(0, 200, 20):   # 10 of 200 rows (5%): the runner-up becomes the best
            row = cand[k]; o = np.argsort(-row); row[o[0]], row[o[1]] = row[o[1]], row[o[0]]
        a, b = self.write("a.txt", self.base), self.write("b.txt", cand)
        rc, out, _ = run_cli(a, b, "--criterion", "loose", "--mean-kl", "1")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL top-1", out)
        _, s = lc.compare([(lc.parse_dump(a), lc.parse_dump(b))])
        self.assertAlmostEqual(s["top1_agree"], 0.95)
        self.assertEqual(len(s["top1_disagree_at"]), 10)

    def test_one_bad_position_trips_max_kl(self):
        cand = self.base.copy()
        cand[37] = np.random.default_rng(1).standard_normal(320) * 3.0   # one position computed wrong
        rc, out, _ = run_cli(self.write("a.txt", self.base), self.write("b.txt", cand), "--top1", "0", "--mean-kl", "1")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL max KL", out)
        self.assertIn("pos 47", out)   # rows start at position 10

    def test_pairs_pool_and_json(self):
        a1, b1 = self.write("a1.txt", self.base[:100]), self.write("b1.txt", self.base[:100])
        a2, b2 = self.write("a2.txt", self.base[100:], self.forced[100:], 5), self.write("b2.txt", self.base[100:], self.forced[100:], 5)
        rc, out, _ = run_cli(a1, b1, a2, b2, "--json")
        self.assertEqual(rc, 0)
        j = json.loads(out)
        self.assertTrue(j["pass"])
        self.assertEqual(j["summary"]["positions"], 200)

    def test_positions_must_line_up(self):
        a = self.write("a.txt", self.base[:50]); b = self.write("b.txt", self.base[:49])
        rc, _, err = run_cli(a, b)
        self.assertEqual(rc, 2)
        self.assertIn("positions differ", err)

    def test_malformed_and_odd_arguments(self):
        p = self.dir / "bad.txt"; p.write_text("12 0.5 1.0 3:0.1 junk\n")
        rc, _, err = run_cli(str(p), str(p))
        self.assertEqual(rc, 2)
        self.assertIn("not a Q38_LOGITS_DUMP line", err)
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                lc.main([str(p)])

    def test_disjoint_support_is_infinite_kl(self):
        a = {5: (1.0, 2.0, [(1, 3.0), (2, 2.0)])}; b = {5: (1.0, 2.0, [(3, 3.0), (4, 2.0)])}
        _, s = lc.compare([(a, b)])
        self.assertTrue(math.isinf(s["max_kl"]))
        self.assertFalse(all(ok for _, ok, _ in lc.verdict(s, lc.CRITERIA["loose"])))

    def test_small_vocab_rows(self):
        """fewer than 32 entries a line (a vocabulary under 32): still compared"""
        rng = np.random.default_rng(2)
        base = rng.standard_normal((10, 20))
        rc, out, _ = run_cli(self.write("a.txt", base, rng.integers(0, 20, 10)), self.write("b.txt", base, rng.integers(0, 20, 10)))
        self.assertEqual(rc, 0, out)


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "gguf.json").exists() and (FIXTURE / "ref.json").exists(),
                     "qwen38 or the tiny block-typed GGUF absent (make qwen38-tiny-gguf-generate)")
class LogitDumpEngineTest(unittest.TestCase):
    """Two PPL runs of the same CPU engine on the fixture: the dump parses and compares as equal."""

    def dump(self, path):
        ref = json.loads((FIXTURE / "ref.json").read_text())
        if len(ref["full_ids"]) <= len(ref["prompt_ids"]):
            self.skipTest("ref.json has no continuation to score")
        env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_CUDA", "COLI_VULKAN", "Q38_"))}
        env.update(SNAP=str(FIXTURE), OMP_NUM_THREADS="2", USAGE_SAVE="0", GQ_EXACT="1", Q38_MTP="0", PPL="1",
                   Q38_LOGITS_DUMP=str(path))
        p = subprocess.run([str(ENGINE), "16", "8", str(FIXTURE / "ref.json")], env=env, cwd=HERE,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace", timeout=300)
        self.assertEqual(p.returncode, 0, p.stdout[-2000:])
        self.assertIn("TF-NLL:", p.stdout)
        return len(ref["full_ids"]) - len(ref["prompt_ids"])

    def test_same_run_twice(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            n = self.dump(a); self.dump(b)
            da = lc.parse_dump(str(a))
            self.assertEqual(len(da), n)
            self.assertTrue(all(len(top) == 32 for _, _, top in da.values()))
            _, s = lc.compare([(da, lc.parse_dump(str(b)))])
            self.assertEqual(s["max_kl"], 0.0)
            self.assertEqual(s["top1_agree"], 1.0)


if __name__ == "__main__":
    unittest.main()
