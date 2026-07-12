import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


class CliOutputLanguageTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=HERE,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_help_is_english(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run GLM-5.2 locally", result.stdout)
        self.assertIn("automatically apply the RAM/VRAM plan", result.stdout)
        self.assertNotIn("modello", result.stdout.lower())
        self.assertNotIn("motore", result.stdout.lower())

    def test_info_status_is_english(self):
        with tempfile.TemporaryDirectory() as model:
            result = self.run_cli("info", "--model", model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config.json is missing", result.stdout)
        self.assertIn("disk", result.stdout)
        self.assertIn("engine", result.stdout)

    def test_missing_model_error_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing-model")
            result = self.run_cli("run", "--model", missing_model, "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)
        self.assertIn("set COLI_MODEL or use --model", result.stderr)


if __name__ == "__main__":
    unittest.main()
