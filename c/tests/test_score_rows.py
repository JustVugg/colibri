"""Real-process SCORE rows must not silently invent or ignore token ids."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

C_DIR = Path(__file__).resolve().parents[1]
ENGINE = C_DIR / ("colibri.exe" if os.name == "nt" else "colibri")
SNAP = C_DIR / "glm_tiny"


class ScoreRowsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not ENGINE.is_file() or not (SNAP / "model.safetensors").is_file():
            raise unittest.SkipTest("build colibri and generate the GLM tiny fixture first")

    def run_score(self, rows):
        keep = {"PATH", "HOME", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "USERPROFILE",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
        env = {k: v for k, v in os.environ.items() if k in keep}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.txt"
            path.write_bytes(rows.encode() if isinstance(rows, str) else rows)
            env.update(SNAP=str(SNAP), SCORE=str(path), SCORE_PREFIX="0", OMP_NUM_THREADS="4")
            return subprocess.run([str(ENGINE), "64", "16", "16"], cwd=C_DIR,
                                  env=env, capture_output=True, text=True, timeout=30)

    def test_valid_rows_keep_scores_and_empty_continuations(self):
        result = self.run_score("1 1 3 14\n1 0 3\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = result.stdout.splitlines()[-2:]
        self.assertEqual(rows[0].split()[1:], ["1", "0"])
        self.assertAlmostEqual(float(rows[0].split()[0]), -5.152055, places=4)
        self.assertEqual(rows[1], "0.000000 0 1")

    def test_missing_negative_out_of_vocabulary_and_extra_ids(self):
        for row in ("1 1 3", "1 1 3 -1", "1 1 3 10000000", "1 1 3 14 7"):
            with self.subTest(row=row):
                result = self.run_score(row + "\n")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("[SCORE] invalid request at line 1", result.stderr)

    def test_invalid_lengths_and_noninteger_fields(self):
        for row in ("0 1 3", "1 -1", "1.0 1 3 14", "1 1 3 14junk",
                    "2147483647 1", "999999999999999999999 0", "1 1 3 14\0ignored"):
            with self.subTest(row=row):
                result = self.run_score(row + "\n")
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("[SCORE] invalid request at line 1", result.stderr)

    def test_whole_file_validation_precedes_any_score(self):
        result = self.run_score("1 1 3 14\n1 1 3\n")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("[SCORE] invalid request at line 2", result.stderr)
        self.assertNotIn("-5.152055 1 0", result.stdout)


if __name__ == "__main__":
    unittest.main()
