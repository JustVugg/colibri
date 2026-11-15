"""Held-out routing shares require valid, nonzero routing evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("placement_crossval", Path(__file__).resolve().parents[1] / "tools/placement_crossval.py")
placement = importlib.util.module_from_spec(spec)
spec.loader.exec_module(placement)


class RoutingCountValidationTest(unittest.TestCase):
    def read(self, content):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "category_1.txt"
            path.write_text(content)
            return placement.read_run(path)

    def test_negative_counts_cannot_produce_negative_or_overfull_shares(self):
        with self.assertRaisesRegex(ValueError, "category_1.txt:2:.*non-negative"):
            self.read("0 0 10\n0 1 -5\n")

    def test_negative_expert_is_not_an_expert_identity(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.read("0 -1 10\n")

    def test_zero_only_run_is_rejected_before_normalization(self):
        with self.assertRaisesRegex(ValueError, "no positive routing counts"):
            self.read("0 0 0\n0 1 0\n")

    def test_unused_experts_and_usage_headers_remain_valid(self):
        run = self.read("-1 3 42\n0 0 0\n0 1 10\n")
        self.assertEqual(run, {(0, 0): 0, (0, 1): 10})
        self.assertEqual(placement.category_shares([run]), {(0, 0): 0, (0, 1): 1})


if __name__ == "__main__":
    unittest.main()
