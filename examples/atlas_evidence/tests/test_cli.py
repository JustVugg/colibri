"""CLI process tests: offline demo, explicit sources and failed checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.atlas_evidence.core import digest

ROOT = Path(__file__).resolve().parents[3]


class CLITests(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "examples.atlas_evidence", *map(str, args)], cwd=ROOT, capture_output=True, text=True)

    def test_offline_demo_and_nonzero_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "demo"
            result = self.run_cli("demo", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(all(summary["checks"].values()))
            self.assertEqual(summary["model_requests"], 0)
            check = self.run_cli("check", out / "capture.json", "--sources", out / "changed-sources.json", "--out", out / "recheck.json")
            self.assertEqual(check.returncode, 1, check.stderr)
            self.assertFalse(json.loads((out / "recheck.json").read_text())["passed"])
            self.assertTrue((out / "review.html").is_file())
            rendered = self.run_cli("render", out / "capture.json", "--sources", out / "changed-sources.json", "--out", out / "changed-review.html")
            self.assertEqual(rendered.returncode, 1, rendered.stderr)
            self.assertTrue((out / "changed-review.html").is_file())
            repeated = self.run_cli("demo", "--out", out)
            self.assertEqual(repeated.returncode, 2)

    def test_snapshot_hashes_exact_bytes_and_preserves_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = base / "policy.txt"
            source.write_bytes("Café policy.\r\nSecond line.".encode())
            output = base / "sources.json"
            result = self.run_cli("snapshot", "--id", "policy", "--file", source, "--out", output, "--path", "sources/policy.txt")
            self.assertEqual(result.returncode, 0, result.stderr)
            value = json.loads(output.read_text())
            self.assertEqual(value[0]["text"], "Café policy.\r\nSecond line.")
            prior = output.read_bytes()
            result = self.run_cli("snapshot", "--id", "policy", "--file", source, "--out", output)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(output.read_bytes(), prior)

    def test_tampered_capture_has_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "demo"
            self.assertEqual(self.run_cli("demo", "--out", out).returncode, 0)
            path = out / "capture.json"
            value = json.loads(path.read_text())
            value["response"]["choices"][0]["message"]["content"] = "{}"
            path.write_text(json.dumps(value))
            result = self.run_cli("check", path, "--out", out / "invalid.json")
            self.assertEqual(result.returncode, 2)
            self.assertFalse((out / "invalid.json").exists())


if __name__ == "__main__":
    unittest.main()
