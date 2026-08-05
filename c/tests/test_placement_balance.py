import tempfile
import unittest
from pathlib import Path

from tools.placement_balance import read_counts, solve


class PlacementBalanceTest(unittest.TestCase):
    def test_reads_unique_counts_in_descending_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stats"
            path.write_text("3 0 4\n3 1 8\n3 0 6\n")
            self.assertEqual(read_counts(path), [8, 6])

    def test_aggregates_independent_stats_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "a.txt").write_text("3 0 4\n3 1 8\n")
            (path / "b.txt").write_text("3 0 6\n3 1 1\n")
            self.assertEqual(read_counts(path), [10, 9])

    def test_balances_critical_tier_instead_of_filling(self):
        plan = solve([40, 30, 20, 10], 4, 2.0, 1.0)
        self.assertEqual(plan.slots, 2)
        self.assertEqual(plan.gpu_share, 0.7)
        self.assertEqual(plan.cpu_seconds, 60.0)
        self.assertEqual(plan.gpu_seconds, 70.0)

    def test_capacity_floor_prevents_nonresident_plan(self):
        plan = solve([40, 30, 20, 10], 4, 2.0, 1.0, min_slots=3)
        self.assertEqual(plan.slots, 3)
        self.assertEqual(plan.gpu_share, 0.9)

    def test_rejects_capacity_floor_above_gpu_limit(self):
        with self.assertRaises(ValueError):
            solve([1, 1], 1, 2.0, 1.0, min_slots=2)

    def test_rejects_invalid_cost(self):
        with self.assertRaises(ValueError):
            solve([1], 1, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
