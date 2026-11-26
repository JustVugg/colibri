"""Q38_TF_TOPK (qwen38.c tf_nll(), default off) and its analyzer tools/q38_tf_kl.py.

- The analyzer's self-test: synthetic dumps written the way the engine writes them; the KL is exact when
  vocab <= K and within 5 % on top-heavy rows otherwise; top-1, the paired gap and the TF-NLL line; ties; a
  truncated dump and another text are refused.
- The engine on the tiny fixture (make qwen38-tiny-generate, or QWEN38_TINY=<dir>) with ref.json and PPL=1:
  the run with the dump prints what the run without it prints (timing aside); the dump's target log-probs
  rebuild the printed TF-NLL line; every row's top-1 is the reply token (ref.json's reply is the reference
  model's greedy continuation, which the engine reproduces); the top-K is descending and holds the target's
  log-prob bit for bit.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import q38_tf_kl as kl  # noqa: E402

ENGINE = Path(os.environ.get("QWEN38_BINARY", HERE / ("qwen38.exe" if sys.platform == "win32" else "qwen38")))
FIXTURE = Path(os.environ.get("QWEN38_TINY", HERE / "qwen38_tiny"))


class AnalyzerTest(unittest.TestCase):
    def test_self_test(self):
        self.assertEqual(kl.self_test(), 0)


@unittest.skipUnless(ENGINE.exists(), "qwen38 is not built")
@unittest.skipUnless((FIXTURE / "ref.json").exists(), "tiny qwen38 fixture is absent (make qwen38-tiny-generate)")
class EngineDumpTest(unittest.TestCase):
    def run_engine(self, **extra):
        env = {k: v for k, v in os.environ.items() if k != "Q38_TF_TOPK"}
        env.update(PPL="1", OMP_NUM_THREADS="2", SNAP=str(FIXTURE), **extra)
        p = subprocess.run([str(ENGINE), "4", "8", str(FIXTURE / "ref.json")], env=env,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        return p.stdout, p.stderr

    def test_dump_matches_the_printed_nll(self):
        off, _ = self.run_engine()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "tf.topk.bin")
            on, err = self.run_engine(Q38_TF_TOPK=path)
            dump = kl.read(path)
        timing = re.compile(r"^Speed:")
        self.assertEqual([l for l in off.splitlines() if not timing.match(l)],
                         [l for l in on.splitlines() if not timing.match(l)])
        self.assertIn("[tf-topk]", err)
        printed = re.search(r"^TF-NLL: \S+ nats/token over \d+ tokens", on, re.M).group(0)
        self.assertEqual(kl.nll_line(dump), printed)
        ref = json.loads((FIXTURE / "ref.json").read_text(encoding="utf-8"))
        prompt, full = ref["prompt_ids"], ref["full_ids"]
        self.assertEqual((dump["np"], dump["nfull"], dump["k"]), (len(prompt), len(full), min(32, dump["vocab"])))
        for pos, (p, target, tlp, logz, ids, lps) in zip(range(len(prompt), len(full)), dump["rows"]):
            self.assertEqual((p, target), (pos, full[pos]))
            self.assertEqual(ids[0], target, f"row {pos}: top-1 {ids[0]} is not the greedy reply token {target}")
            self.assertEqual(list(lps), sorted(lps, reverse=True))
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(lps[ids.index(target)], tlp)
            self.assertLessEqual(sum(map(math.exp, lps)), 1 + 1e-9)


if __name__ == "__main__":
    unittest.main()
