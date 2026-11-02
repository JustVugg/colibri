"""Smoke verdicts must distinguish a completed child from partial output.

The real Python children are deterministic process fixtures, not models. They
emit a response-shaped stream to exercise status handling without inference.
"""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import diag_harness


class DiagnosticSmokeStatusTests(unittest.TestCase):
    def run_smoke(self, response, exit_status=0, stall=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            engine = root / "response-probe.py"
            engine.write_text(
                "import sys, time\n"
                f"print('question:{response}\\n---', flush=True)\n"
                + ("time.sleep(60)\n" if stall else "")
                + f"raise SystemExit({exit_status})\n")
            args = types.SimpleNamespace(
                snap=str(root), glm=str(engine), out=str(root / "logs"),
                ram=0, cuda=False, gpu=None, timeout=0.5 if stall else 5,
                cap=64, ngen=8)
            actual_popen = subprocess.Popen
            children = []

            def launch_probe(command, *args, **kwargs):
                if str(command[0]) == str(engine):
                    command = [sys.executable, str(engine), *command[1:]]
                child = actual_popen(command, *args, **kwargs)
                children.append(child)
                return child

            prompt = {"id": "status", "cat": "fixture", "prompt": "question:",
                      "expect": "Paris"}
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch.object(subprocess, "Popen", side_effect=launch_probe), \
                    mock.patch.object(diag_harness, "PROMPTS", [prompt]), \
                    contextlib.redirect_stdout(io.StringIO()):
                harness = diag_harness.DiagnosticHarness(args)
                result = harness.phase_smoke()
                self.assertEqual(harness.results["phases"]["smoke"], result)
            self.assertEqual(len(children), 1)
            self.assertIsNotNone(children[0].returncode)
            self.assertEqual(children[0].wait(timeout=1), children[0].returncode)
            return result

    def test_nonzero_exit_cannot_pass_with_expected_partial_text(self):
        result = self.run_smoke("Paris", exit_status=7)
        row = result["prompts"][0]
        self.assertEqual(row["generated"], "Paris")
        self.assertEqual(row["rc"], 7)
        self.assertFalse(row["pass"])
        self.assertEqual(result["pass"], 0)
        self.assertIn("7", row["fail_reason"])

    def test_timeout_cannot_pass_with_expected_partial_text(self):
        result = self.run_smoke("Paris", stall=True)
        row = result["prompts"][0]
        self.assertEqual(row["generated"], "Paris")
        self.assertEqual(row["rc"], -1)
        self.assertFalse(row["pass"])
        self.assertEqual(result["pass_rate"], 0)
        self.assertIn("timed out", row["fail_reason"].lower())

    def test_completed_expected_text_passes(self):
        result = self.run_smoke("Paris")
        self.assertEqual(result["pass"], 1)
        self.assertEqual(result["prompts"][0]["fail_reason"], "")

    def test_completed_wrong_answer_still_fails(self):
        result = self.run_smoke("Berlin")
        self.assertEqual(result["pass"], 0)
        self.assertIn("not found", result["prompts"][0]["fail_reason"])

    def test_completed_empty_answer_still_fails(self):
        result = self.run_smoke("")
        self.assertEqual(result["pass"], 0)
        self.assertIn("no tokens", result["prompts"][0]["fail_reason"])


if __name__ == "__main__":
    unittest.main()
