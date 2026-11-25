"""TF=1: teacher forcing on qwen36, every position's argmax against the reference's tf_pred.

`TF=1 SNAP=<container> ./qwen36 <cap> <bits> <ref.json>` feeds the reference's full_ids
through the model as a prompt and reads every row's logits out: the argmax at position s
is the engine's prediction for full_ids[s+1], the prompt's own positions included, which
the greedy oracle never looks at. Each is compared with `tf_pred`, the torch model's
argmax over one forward of the same sequence (make_qwen36_tiny.py --emit-ref and
make_qwen36_oracle.py write it). On the generated 128-token fixture:

  * every position matches and the run exits 0, printing "TEACHER-FORCING: N/N";
  * `DUMP=<path>` gets one row a position, raw float32, whose argmax is the prediction;
  * a reference with one wrong tf_pred entry fails (exit 1) naming the position, passes
    with `ORACLE_TF_MAX_MISMATCHES=1`, and `DEBUG_LOGITS=1` prints its top five logits;
  * a reference without tf_pred, or with a tf_pred of another length, fails;
  * without TF the greedy oracle runs as before.

Recipe (CPU is enough; the fixture is generated, not committed):

    make -C c qwen36 qwen36-tiny-v128-generate
    QWEN36_TINY=./c/qwen36_tiny_v128_c QWEN36_TINY_REF=./c/qwen36_tiny_v128/ref_full.json \\
        python3 -m unittest c.tests.test_qwen36_tf
"""
import array
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

ENGINE = Path(os.environ.get("QWEN36_ENGINE", HERE.parent / "qwen36"))
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE.parent / "qwen36_tiny_v128_c"))
REFERENCE = Path(os.environ.get("QWEN36_TINY_REF", HERE.parent / "qwen36_tiny_v128" / "ref_full.json"))
REQUIRED = os.environ.get("QWEN36_TINY_REQUIRED") == "1"
VOCAB = 128


def _argmax(row):
    best = 0
    for i in range(1, len(row)):
        if row[i] > row[best]:
            best = i
    return best


class TeacherForcing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = []
        if not ENGINE.exists():
            missing.append(f"{ENGINE} (make -C c qwen36)")
        if not (FIXTURE / "config.json").exists() or not REFERENCE.exists():
            missing.append(f"{FIXTURE} and {REFERENCE} (make -C c qwen36-tiny-v128-generate)")
        if missing:
            if REQUIRED:
                raise AssertionError("missing: " + ", ".join(missing))
            raise unittest.SkipTest("missing: " + ", ".join(missing))
        cls.ref = json.loads(REFERENCE.read_text())
        if "tf_pred" not in cls.ref:
            raise AssertionError(f"{REFERENCE} has no tf_pred: regenerate it (make -C c qwen36-tiny-v128-generate)")
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _run(self, ref, **env):
        path = Path(self.tmp.name) / "ref.json"
        path.write_text(json.dumps(ref))
        # COLI_DENSE_I8=0: the torch reference keeps the dense weights in float, the
        # engine's default int8 trunk flips near ties on random weights (the CI gate
        # runs the greedy oracle the same way)
        environment = {**os.environ, "SNAP": str(FIXTURE), "COLI_DENSE_I8": "0", **env}
        for k in ("TF", "DUMP", "DEBUG_LOGITS", "ORACLE_TF_MAX_MISMATCHES"):
            if k not in env:
                environment.pop(k, None)
        done = subprocess.run([str(ENGINE), "8", "8", str(path)], env=environment,
                              capture_output=True, text=True, timeout=600)
        return done.returncode, done.stdout + done.stderr

    def test_every_position_matches(self):
        rc, out = self._run(self.ref, TF="1")
        n = len(self.ref["full_ids"])
        self.assertIn(f"TEACHER-FORCING: {n}/{n} positions match the reference", out)
        self.assertNotIn("[tf] mismatch", out)
        self.assertEqual(rc, 0, out)

    def test_dump_holds_one_row_a_position(self):
        dump = Path(self.tmp.name) / "tf.f32"
        rc, out = self._run(self.ref, TF="1", DUMP=str(dump))
        self.assertEqual(rc, 0, out)
        data = array.array("f", dump.read_bytes())
        n = len(self.ref["full_ids"])
        self.assertEqual(len(data), n * VOCAB, "one row of vocab floats a position")
        self.assertIn(f"[dump] wrote {n} rows of {VOCAB} logits", out)
        rows = [data[i:i + VOCAB] for i in range(0, len(data), VOCAB)]
        self.assertEqual([_argmax(r) for r in rows], self.ref["tf_pred"])

    def test_a_wrong_prediction_fails_and_names_the_position(self):
        bad = copy.deepcopy(self.ref)
        pos = 2   # inside the prompt: a position the greedy oracle never compares
        bad["tf_pred"][pos] = (bad["tf_pred"][pos] + 1) % VOCAB
        n = len(bad["full_ids"])
        rc, out = self._run(bad, TF="1")
        self.assertEqual(rc, 1, out)
        self.assertIn(f"TEACHER-FORCING: {n - 1}/{n} positions", out)
        self.assertIn(f"[tf] mismatch pos={pos} expected={bad['tf_pred'][pos]} got={self.ref['tf_pred'][pos]}", out)
        self.assertNotIn("[LOGITS]", out)

        rc, out = self._run(bad, TF="1", ORACLE_TF_MAX_MISMATCHES="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("1 allowed", out)

        rc, out = self._run(bad, TF="1", DEBUG_LOGITS="1")
        self.assertEqual(rc, 1, out)
        self.assertIn(f"[LOGITS] pos={pos} top1-top2 gap=", out)
        self.assertIn(f"got={self.ref['tf_pred'][pos]} (", out)

    def test_allowance_must_be_a_count(self):
        rc, out = self._run(self.ref, TF="1", ORACLE_TF_MAX_MISMATCHES="x")
        self.assertEqual(rc, 1, out)
        self.assertIn("ORACLE_TF_MAX_MISMATCHES must be an integer", out)

    def test_a_reference_without_tf_pred_fails(self):
        bare = {k: v for k, v in self.ref.items() if k != "tf_pred"}
        rc, out = self._run(bare, TF="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("has no tf_pred", out)
        self.assertIn("TF pred  : " + " ".join(str(t) for t in self.ref["tf_pred"]), out)
        short = copy.deepcopy(self.ref)
        short["tf_pred"] = short["tf_pred"][:-1]
        rc, out = self._run(short, TF="1")
        self.assertEqual(rc, 1, out)
        self.assertIn("not the same sequence", out)

    def test_without_tf_the_greedy_oracle_runs(self):
        rc, out = self._run(self.ref)
        self.assertEqual(rc, 0, out)
        self.assertNotIn("TEACHER-FORCING", out)
        n_new = len(self.ref["full_ids"]) - len(self.ref["prompt_ids"])
        self.assertIn(f"Matching tokens: {n_new}/{n_new}", out)


if __name__ == "__main__":
    unittest.main()
