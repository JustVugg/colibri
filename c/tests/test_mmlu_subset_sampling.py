"""Cached MMLU sample counts must be satisfiable before any model is run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/eval_mmlu_subset.py"
QUESTION = {"subject": "logic", "question": "Which letter is second?",
            "choices": ["A", "B", "C", "D"], "answer": 1}


class MmluSubsetSamplingTest(unittest.TestCase):
    def test_impossible_counts_fail_before_engine_execution(self):
        for pool, count in (([QUESTION], 2), ([], 1), ([QUESTION], 0), ([QUESTION], -1)):
            with self.subTest(pool_size=len(pool), count=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cache = root / "questions.json"
                cache.write_text(json.dumps(pool))
                output = root / "result.json"
                result = subprocess.run([sys.executable, str(SCRIPT), "--model", "unused",
                                         "--cache", str(cache), "--n", str(count),
                                         "--binary", str(root / "must-not-run"), "--out", str(output)],
                                        capture_output=True, text=True, timeout=2)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--n", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    @unittest.skipIf(os.name == "nt", "fake executable uses a POSIX shebang")
    def test_exact_pool_count_visits_each_cached_question_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "questions.json"
            cache.write_text(json.dumps([QUESTION, {**QUESTION, "subject": "math"}]))
            engine = root / "fake-engine"
            engine.write_text(f"#!{sys.executable}\nprint('Answer: B\\n---')\n")
            engine.chmod(0o700)
            output = root / "result.json"
            result = subprocess.run([sys.executable, str(SCRIPT), "--model", "unused",
                                     "--cache", str(cache), "--n", "2", "--seed", "3",
                                     "--binary", str(engine), "--out", str(output)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(output.read_text())
            self.assertEqual((record["n"], record["answered"], record["acc"]), (2, 2, 1.0))
            self.assertEqual(sorted(r["subject"] for r in record["results"]), ["logic", "math"])


if __name__ == "__main__":
    unittest.main()
