"""Legitimate early EOS must not crash or strand an interference experiment."""
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import benchmark_serving_interference as bench


class InterferenceEosTest(unittest.TestCase):
    def run_benchmark(self, transcript):
        proc = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(transcript),
                               stderr=io.BytesIO(), wait=lambda timeout=None: 0)
        output = io.StringIO()
        with patch.object(sys, "argv", ["bench", "--engine", "fixture", "--model", "fixture", "--decode-tokens", "1"]), \
             patch.object(bench.subprocess, "Popen", return_value=proc), \
             contextlib.redirect_stdout(output):
            bench.main()
        return json.loads(output.getvalue())

    def test_single_token_completion_has_no_intertoken_latency(self):
        result = self.run_benchmark(
            b"READY\nDATA 1 1\na\nDONE 1 stop\n"
            b"DATA 2 1\nb\nACCEPT 3 1\nDATA 3 1\nc\nDONE 2 stop\nDONE 3 stop\n")
        for phase in ("control", "treatment"):
            self.assertEqual(result[phase]["tokens"], 1)
            self.assertIsNone(result[phase]["median_tbt_s"])
            self.assertIsNone(result[phase]["max_tbt_s"])

    def test_interfering_prefill_eos_without_output_has_no_ttft(self):
        result = self.run_benchmark(
            b"READY\nDATA 1 1\na\nDONE 1 stop\n"
            b"DATA 2 1\nb\nACCEPT 3 1\nDONE 2 stop\nDONE 3 stop\n")
        self.assertIsNone(result["interfering_prefill"]["ttft_s"])

    def test_treatment_eos_without_output_fails_before_another_read(self):
        with self.assertRaisesRegex(RuntimeError, "treatment.*no.*token"):
            self.run_benchmark(b"READY\nDATA 1 1\na\nDONE 1 stop\nDONE 2 stop\n")


if __name__ == "__main__":
    unittest.main()
