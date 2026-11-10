"""Long-prompt benchmark driver tests; no model or network required."""
from argparse import Namespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from tools import benchmark_long_prompts as driver


class LongPromptDriverTest(unittest.TestCase):
    def test_runs_each_workload_to_a_separate_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workload_32k = root / "32k.jsonl"
            workload_64k = root / "64k.jsonl"
            workload_32k.write_text('{"messages":[{"role":"user","content":"32k"}]}\n')
            workload_64k.write_text('{"messages":[{"role":"user","content":"64k"}]}\n')
            runner = Mock(side_effect=[Mock(returncode=0), Mock(returncode=0)])
            args = Namespace(
                workload_32k=str(workload_32k), workload_64k=str(workload_64k),
                output_dir=str(root / "reports"), overwrite=False,
                base_url="http://127.0.0.1:8001/v1", model="qwen36",
                concurrency=1, repeats=2, warmup_requests=0, max_tokens=128,
                temperature=0, timeout=14400, api_key_env="OPENAI_API_KEY")

            self.assertEqual(driver.run_matrix(args, runner), 0)

            commands = [call.args[0] for call in runner.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertEqual(Path(commands[0][commands[0].index("--workload") + 1]),
                             workload_32k.resolve())
            self.assertEqual(Path(commands[0][commands[0].index("--output") + 1]),
                             (root / "reports" / "prompt-32k.json").resolve())
            self.assertEqual(Path(commands[1][commands[1].index("--workload") + 1]),
                             workload_64k.resolve())
            self.assertEqual(Path(commands[1][commands[1].index("--output") + 1]),
                             (root / "reports" / "prompt-64k.json").resolve())
            self.assertEqual(commands[0][commands[0].index("--repeats") + 1], "2")
            self.assertEqual(commands[1][commands[1].index("--timeout") + 1], "14400")

    def test_runs_both_sizes_and_reports_failure_if_either_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workloads = []
            for name in ("32k", "64k"):
                path = root / f"{name}.jsonl"
                path.write_text('{"messages":[{"role":"user","content":"prompt"}]}\n')
                workloads.append(str(path))
            args = Namespace(
                workload_32k=workloads[0], workload_64k=workloads[1],
                output_dir=str(root / "reports"), overwrite=False,
                base_url="http://127.0.0.1:8001/v1", model="qwen36",
                concurrency=1, repeats=1, warmup_requests=0, max_tokens=128,
                temperature=0, timeout=14400, api_key_env="OPENAI_API_KEY")
            runner = Mock(side_effect=[Mock(returncode=1), Mock(returncode=0)])

            self.assertEqual(driver.run_matrix(args, runner), 1)
            self.assertEqual(runner.call_count, 2)

    def test_refuses_same_workload_for_both_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            workload = Path(temp) / "same.jsonl"
            workload.write_text('{"messages":[{"role":"user","content":"prompt"}]}\n')
            args = Namespace(workload_32k=str(workload), workload_64k=str(workload),
                             output_dir=str(Path(temp) / "reports"), overwrite=False)

            with self.assertRaisesRegex(ValueError, "must be different files"):
                driver.run_matrix(args, Mock())

    def test_refuses_to_overwrite_reports_without_opt_in(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workloads = []
            for name in ("32k", "64k"):
                path = root / f"{name}.jsonl"
                path.write_text('{"messages":[{"role":"user","content":"prompt"}]}\n')
                workloads.append(str(path))
            output_dir = root / "reports"
            output_dir.mkdir()
            (output_dir / "prompt-32k.json").write_text("{}\n")
            args = Namespace(workload_32k=workloads[0], workload_64k=workloads[1],
                             output_dir=str(output_dir), overwrite=False)

            with self.assertRaisesRegex(ValueError, "report already exists"):
                driver.run_matrix(args, Mock())


if __name__ == "__main__":
    unittest.main()
