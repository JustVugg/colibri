"""Decision checkpoint diagnostics reject incomplete or malformed weight layouts."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from doctor import run_decision_doctor


class DecisionWeightsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model = Path(self.temp.name)
        (self.model / "encoder").mkdir()
        (self.model / "tokenizer").mkdir()
        (self.model / "rl_agent_config.json").write_text('{"head_layers":2}')
        (self.model / "encoder/config.json").write_text(
            '{"model_type":"modernbert","hidden_size":1024,"num_hidden_layers":28}')
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (self.model / "tokenizer" / name).write_text("{}")
        self.engine = self.model / "laya"
        self.engine.write_text("owned diagnostic fixture; never executed")
        self.engine.chmod(0o700)

    def write_weights(self, header, payload=b""):
        encoded = json.dumps(header).encode("utf-8")
        (self.model / "model.safetensors").write_bytes(
            struct.pack("<Q", len(encoded)) + encoded + payload)

    def test_incomplete_and_malformed_weights_are_failed_checks(self):
        cases = [
            ("missing payload", {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}, b""),
            ("non-object header", [], b""),
            ("negative shape", {"w": {"dtype": "F32", "shape": [-1], "data_offsets": [0, 4]}}, b"abcd"),
            ("shape span mismatch", {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}, b"abcd"),
            ("invalid dtype", {"w": {"dtype": [], "shape": [1], "data_offsets": [0, 4]}}, b"abcd"),
        ]
        for name, header, payload in cases:
            with self.subTest(case=name):
                self.write_weights(header, payload)
                report = run_decision_doctor(self.model, self.engine, available_memory=1 << 30)
                checks = [c for c in report["checks"] if c["id"] == "model.weights"]
                self.assertEqual(len(checks), 1)
                self.assertEqual(checks[0]["status"], "fail")
                self.assertEqual(report["status"], "error")
                self.assertFalse(any(c["id"] == "memory.ram" for c in report["checks"]))

    def test_valid_weight_dtypes_preserve_f32_residency_estimate(self):
        for dtype, byte_count in (("F32", 16), ("BF16", 8), ("F16", 8)):
            with self.subTest(dtype=dtype):
                self.write_weights({"w": {"dtype": dtype, "shape": [4],
                                          "data_offsets": [0, byte_count]}}, bytes(byte_count))
                report = run_decision_doctor(self.model, self.engine, available_memory=1 << 30)
                check = next(c for c in report["checks"] if c["id"] == "memory.ram")
                self.assertEqual(check["status"], "pass")
                self.assertEqual(check["details"]["weight_bytes"], 16)
                self.assertEqual(check["details"]["params"], 4)
                constrained = run_decision_doctor(self.model, self.engine, available_memory=8)
                self.assertEqual(next(c for c in constrained["checks"]
                                      if c["id"] == "memory.ram")["status"], "fail")

    def test_public_doctor_preserves_valid_weight_dtypes(self):
        cli = Path(__file__).resolve().parents[1] / "coli"
        for dtype, byte_count in (("F32", 16), ("BF16", 8), ("F16", 8)):
            with self.subTest(dtype=dtype):
                self.write_weights({"w": {"dtype": dtype, "shape": [4],
                                          "data_offsets": [0, byte_count]}}, bytes(byte_count))
                result = subprocess.run(
                    [sys.executable, str(cli), "doctor", "--model", str(self.model), "--json"],
                    capture_output=True, text=True, timeout=15,
                    env=dict(os.environ, COLI_ENGINE=str(cli.parent / "colibri")))
                report = json.loads(result.stdout)
                check = next(c for c in report["checks"] if c["id"] == "memory.ram")
                self.assertEqual(check["status"], "pass")
                self.assertEqual(check["details"]["weight_bytes"], 16)
                self.assertFalse(any(c["id"] == "model.weights" and c["status"] == "fail"
                                     for c in report["checks"]))
                self.assertNotIn("Traceback", result.stderr)

    def test_public_doctor_json_reports_truncated_weights_without_traceback(self):
        self.write_weights({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}})
        cli = Path(__file__).resolve().parents[1] / "coli"
        result = subprocess.run(
            [sys.executable, str(cli), "doctor", "--model", str(self.model), "--json"],
            capture_output=True, text=True, timeout=15,
            env=dict(os.environ, COLI_ENGINE=str(cli.parent / "colibri")))
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertTrue(any(c["id"] == "model.weights" and c["status"] == "fail"
                            for c in report["checks"]))
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
