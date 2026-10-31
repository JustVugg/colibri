"""Decode-only throughput counts work after the prefill timing boundary."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import benchmark_continuous_batch as bench


class DecodeMeasurementTest(unittest.TestCase):
    def phase(self, payload, times, batch=1, tokens=2):
        proc = SimpleNamespace(stdin=io.BytesIO(), stdout=io.BytesIO(payload))
        with patch.object(bench.time, "perf_counter", side_effect=times):
            return bench.run_phase(proc, batch, tokens, 1)

    def test_first_token_marks_boundary_without_counting_prefill_work(self):
        result = self.phase(b"DATA 1001 1\na\nDATA 1001 1\nb\nDONE 1001 stop\n", [10., 11., 12.])
        self.assertEqual(result["measured_tokens"], 1)
        self.assertEqual(result["aggregate_tok_s"], .5)

    def test_single_token_phase_has_no_measured_decode(self):
        result = self.phase(b"DATA 1001 1\na\nDONE 1001 stop\n", [10., 11.], tokens=1)
        self.assertEqual(result["measured_tokens"], 0)
        self.assertEqual(result["aggregate_tok_s"], 0.)

    def test_prior_slot_tokens_before_boundary_are_excluded(self):
        result = self.phase(b"DATA 1001 1\na\nDATA 1002 1\nb\nDATA 1001 1\nc\nDATA 1002 1\nd\nDONE 1001 stop\nDONE 1002 stop\n", [9., 10., 11., 12., 13., 14.], batch=2)
        self.assertEqual(result["measured_tokens"], 2)
        self.assertEqual(result["aggregate_tok_s"], .5)


if __name__ == "__main__":
    unittest.main()
