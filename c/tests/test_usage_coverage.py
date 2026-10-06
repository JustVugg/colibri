import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.usage_coverage import Curve, analyze, gini, main, read_history

PROFILE = Path(__file__).parents[1] / "profiles" / "qwen36-35b.coli_usage"


def history(text, name="h.coli_usage"):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / name
        path.write_text(text)
        return read_history([path])


# 1 layer x 4 experts: 50 + 30 + 15 + 5 = 100 selections
SIMPLE = "-1 1 4\n-2 1 3815245270\n0 0 50\n0 1 30\n0 2 15\n0 3 5\n"


class ReadHistoryTest(unittest.TestCase):
    def test_header_gives_dimensions_and_is_not_counted(self):
        h = history(SIMPLE)
        self.assertEqual((h.n_layers, h.n_experts), (1, 4))
        self.assertEqual(h.total, 100)
        self.assertNotIn((-2, 1), h.counts)

    def test_sums_files_and_infers_dimensions_without_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("0 0 4\n1 2 6\n")
            (Path(tmp) / "b.txt").write_text("0 0 5\n")
            h = read_history([Path(tmp)])
        self.assertEqual(h.counts, {(0, 0): 9, (1, 2): 6})
        self.assertEqual((h.n_layers, h.n_experts), (2, 3))

    def test_rejects_mismatched_dimensions_and_bad_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.txt").write_text("-1 1 4\n0 0 1\n")
            (Path(tmp) / "b.txt").write_text("-1 2 4\n0 0 1\n")
            with self.assertRaises(ValueError):
                read_history([Path(tmp)])
        for bad in ("0 0\n", "-1 1 4\n0 9 1\n", "-1 1 4\n", "0 -1 3\n"):
            with self.assertRaises(ValueError):
                history(bad)


class CurveTest(unittest.TestCase):
    def test_targets_knee_and_budget_fit(self):
        curve = Curve(history(SIMPLE), lambda layer: 1.0)
        self.assertEqual([curve.coverage(k) for k in range(5)], [0, 0.5, 0.8, 0.95, 1.0])
        self.assertEqual(curve.for_target(0.7), 2)   # 2 experts reach 70%, not 3
        self.assertEqual(curve.for_target(0.8), 2)   # exactly on the boundary
        self.assertEqual(curve.fit(2.5), 2)
        # coverage - share of bytes: 0.25, 0.30, 0.20, 0.0 -> knee after 2 experts
        self.assertEqual(curve.knee(), 2)
        # marginal points per expert: 50, 30, 15, 5
        self.assertEqual(curve.lambda_point(20, 1.0), 2)
        self.assertEqual(curve.lambda_point(1, 1.0), 4)

    def test_unseen_experts_take_space_but_cover_nothing(self):
        curve = Curve(history("-1 1 4\n0 0 10\n"), lambda layer: 1.0)
        self.assertEqual(curve.n, 4)
        self.assertEqual(curve.coverage(1), 1.0)

    def test_ranks_by_selections_per_byte_when_widths_differ(self):
        # layer 1 experts are twice as wide: 60 selections in 2 units loses to 40 in 1
        h = history("-1 1 2\n0 0 40\n1 0 60\n")
        by_density = Curve(h, lambda layer: 2.0 if layer == 1 else 1.0)
        by_count = Curve(h, lambda layer: 2.0 if layer == 1 else 1.0, by="count")
        self.assertEqual(by_density.items[0][:2], (0, 0))
        self.assertEqual(by_count.items[0][:2], (1, 0))
        self.assertEqual(by_density.coverage(by_density.fit(1.0)), 0.4)
        self.assertEqual(by_count.coverage(by_count.fit(1.0)), 0.0)

    def test_gini(self):
        self.assertEqual(gini([5, 5, 5, 5]), 0.0)
        self.assertAlmostEqual(gini([0, 0, 0, 12]), 0.75)


class AnalyzeTest(unittest.TestCase):
    def test_plan_fills_vram_less_reserve_then_ram(self):
        # 4 experts of 1 GB each
        r = analyze(history(SIMPLE), expert_mb=1000, vram_gb=4, reserve_gb=2, ram_gb=1)
        self.assertEqual(r["plan"]["vram"]["experts"], 2)
        self.assertEqual(r["plan"]["vram_plus_ram"]["experts"], 3)
        self.assertAlmostEqual(r["plan"]["vram_plus_ram"]["coverage_pct"], 95.0)

    def test_plan_reports_what_stopping_at_the_knee_frees(self):
        r = analyze(history(SIMPLE), expert_mb=1000, vram_gb=5, reserve_gb=1)
        self.assertEqual(r["knee"]["experts"], 2)
        self.assertAlmostEqual(r["plan"]["vram_to_knee"]["freed_gb"], 2.0)
        self.assertAlmostEqual(r["plan"]["vram_to_knee"]["coverage_cost_pct"], 20.0)

    def test_holdout_scores_the_training_ranking(self):
        h = history(SIMPLE)
        held = history("-1 1 4\n0 0 10\n0 3 90\n")
        r = analyze(h, holdout=held, targets=(0.5,))
        self.assertAlmostEqual(r["targets"]["50pct"]["holdout_coverage_pct"], 10.0)
        with self.assertRaises(ValueError):
            analyze(h, holdout=history("-1 2 4\n0 0 1\n"))

    def test_row_widths_need_expert_size_and_plan_needs_sizes(self):
        with self.assertRaises(ValueError):
            analyze(history(SIMPLE), row_mb={0: 2.0})
        with self.assertRaises(ValueError):
            analyze(history(SIMPLE), vram_gb=8)


class CommandLineTest(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            main([str(a) for a in argv])
        return out.getvalue()

    def test_shipped_profile(self):
        r = json.loads(self.run_main(PROFILE, "--expert-mb", "1.7", "--json"))
        self.assertEqual(r["concentration"]["selections"], 796160)
        self.assertEqual(r["concentration"]["experts"], 40 * 256)
        coverage = [row["coverage_pct"] for row in r["curve"]]
        self.assertEqual(coverage, sorted(coverage))
        self.assertGreaterEqual(r["targets"]["70pct"]["coverage_pct"], 70.0)

    def test_text_report(self):
        text = self.run_main(PROFILE, "--vram-gb", "6", "--expert-mb", "1.7", "--min-gain", "5")
        for heading in ("curve", "smallest budget", "knee", "lambda point", "plan", "lower bound"):
            self.assertIn(heading, text)


if __name__ == "__main__":
    unittest.main()
