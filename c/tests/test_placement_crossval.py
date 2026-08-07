import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "tools" / "placement_crossval.py"
SPEC = importlib.util.spec_from_file_location("placement_crossval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlacementCrossvalTest(unittest.TestCase):
    def test_category_holdout_and_rank_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = {
                "alpha_0.txt": "0 0 9\n0 1 1\n",
                "alpha_1.txt": "0 0 8\n0 1 2\n",
                "beta_0.txt": "0 0 7\n0 2 3\n",
                "beta_1.txt": "0 0 6\n0 2 4\n",
                "gamma_0.txt": "0 0 5\n0 3 5\n",
                "gamma_1.txt": "0 0 6\n0 3 4\n",
            }
            for name, body in fixtures.items():
                (root / name).write_text(body)

            categories = MODULE.read_categories(root)
            pooled_mean, pooled_worst, _ = MODULE.cross_validate(
                categories, slots=1, penalty=None
            )
            mean, worst, detail = MODULE.cross_validate(categories, slots=1, penalty=0)
            self.assertEqual([name for name, _ in detail], ["alpha", "beta", "gamma"])
            self.assertGreaterEqual(pooled_mean, pooled_worst)
            self.assertGreaterEqual(mean, worst)
            self.assertAlmostEqual(worst, 0.55)

            ranked = MODULE.rank_experts(categories, penalty=0)
            output = root / "ranked.stats"
            MODULE.write_ranked(output, ranked)
            self.assertTrue(output.read_text().startswith("0 0 "))

    def test_rejects_empty_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "alpha_0.txt"
            path.write_text("")
            with self.assertRaises(ValueError):
                MODULE.read_run(path)


if __name__ == "__main__":
    unittest.main()
