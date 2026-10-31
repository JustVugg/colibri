"""The engram preparation CLI explains missing optional packages before work."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "prepare_dsv41.py"


class PrepareDsv41DependenciesTest(unittest.TestCase):
    def run_prepare(self, config):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model / "config.json").write_text(json.dumps(config), encoding="utf-8")
            result = subprocess.run([sys.executable, "-S", str(SCRIPT), "--model", directory],
                                    capture_output=True, text=True, timeout=30)
            return result, (model / "dsv41_engram.json").exists()

    def test_missing_packages_produce_install_guidance(self):
        result, sidecar_exists = self.run_prepare({"engram_layer_ids": [1]})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("transformers", result.stderr)
        self.assertIn("tokenizers", result.stderr)
        self.assertIn("numpy", result.stderr)
        self.assertIn("sympy", result.stderr)
        self.assertIn("python -m pip install", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(sidecar_exists)

    def test_no_engram_does_not_require_optional_packages(self):
        result, sidecar_exists = self.run_prepare({"engram_layer_ids": []})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("nothing to prepare", result.stdout)
        self.assertFalse(sidecar_exists)


if __name__ == "__main__":
    unittest.main()
