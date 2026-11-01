"""Failed benchmark conversion must not publish an empty completion marker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class FetchBenchmarksTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "bench"
        self.output.mkdir()
        # Replace only the optional network dataset boundary; execute the real
        # CLI, formatters, filtering, publication, and per-task failure logic.
        (self.root / "datasets.py").write_text(
            "import json, os\n"
            "def load_dataset(path, config, split):\n"
            "    return json.loads(os.environ['COLIBRI_OWNED_DATASETS'])[path]\n",
            encoding="utf-8",
        )
        self.script = Path(__file__).resolve().parents[1] / "tools" / "fetch_benchmarks.py"

    def run_fetch(self, datasets, tasks="hellaswag"):
        environment = dict(os.environ, PYTHONPATH=str(self.root),
                           COLIBRI_OWNED_DATASETS=json.dumps(datasets))
        return subprocess.run(
            [sys.executable, str(self.script), "--out", str(self.output),
             "--tasks", tasks, "--limit", "2", "--tries", "1"],
            env=environment, capture_output=True, text=True, timeout=10,
        )

    @staticmethod
    def hellaswag(label):
        return dict(activity_label="Fixture", ctx_a="A person", ctx_b="looks around",
                    endings=["walks away", "sits down"], label=label)

    def test_empty_or_unusable_dataset_is_not_published(self):
        for rows in ([], [self.hellaswag(-1)], [{"unrecognized": "schema"}]):
            with self.subTest(rows=rows):
                result = self.run_fetch({"Rowan/hellaswag": rows})
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertFalse((self.output / "hellaswag.jsonl").exists())
                self.assertFalse((self.output / "hellaswag.jsonl.part").exists())

    def test_failed_conversion_preserves_existing_dataset_and_continues(self):
        original = b'{"ctx":"existing","choices":["valid"],"gold":0}\n'
        existing = self.output / "hellaswag.jsonl"
        existing.write_bytes(original)
        result = self.run_fetch(
            {"Rowan/hellaswag": [], "ybisk/piqa": [
                dict(goal="Fixture goal", sol1="first", sol2="second", label=1)
            ]}, "hellaswag,piqa",
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(existing.read_bytes(), original)
        rows = [json.loads(line) for line in (self.output / "piqa.jsonl").read_text().splitlines()]
        self.assertEqual(rows, [dict(ctx="Question: Fixture goal\nAnswer:",
                                     choices=[" first", " second"], gold=1)])

    def test_valid_rows_still_publish_and_skip_invalid_rows(self):
        result = self.run_fetch({"Rowan/hellaswag": [
            self.hellaswag(1), self.hellaswag(-1), {"bad": "row"}
        ]})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in (self.output / "hellaswag.jsonl").read_text().splitlines()]
        self.assertEqual(rows, [dict(ctx="Fixture: A person Looks around",
                                     choices=[" walks away", " sits down"], gold=1)])


if __name__ == "__main__":
    unittest.main()
