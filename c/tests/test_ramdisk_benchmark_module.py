"""Direct contracts for the callback-injected benchmark module."""

import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import benchmark as benchmark_support  # noqa: E402
import ramdisk  # noqa: E402


class BenchmarkModuleTest(unittest.TestCase):
    def test_runtime_knob_normalization_accepts_the_complete_vocabulary(self):
        core_count = mock.Mock(return_value=96)
        plan = {"placement": {"cpu_list": "0-95"}}

        result = benchmark_support._normalized_runtime_knobs(
            plan,
            {
                "PIPE": "1",
                "DIRECT": 0,
                "URING": "0",
                "PIPE_WORKERS": "96",
                "OMP_NUM_THREADS": "96",
                "OMP_PROC_BIND": "spread",
            },
            node=3,
            node_core_count=core_count,
        )

        self.assertEqual(
            result,
            {
                "PIPE": 1,
                "DIRECT": 0,
                "URING": 0,
                "PIPE_WORKERS": 96,
                "OMP_NUM_THREADS": 96,
                "OMP_PROC_BIND": "spread",
            },
        )
        core_count.assert_called_once_with(plan, 3)

    def test_runtime_knob_normalization_enforces_each_range(self):
        plan = {}
        invalid = (
            ({"PIPE": 2}, "PIPE benchmark knob must be 0 or 1"),
            (
                {"PIPE_WORKERS": 0},
                "PIPE_WORKERS benchmark knob is outside its safe range",
            ),
            (
                {"PIPE_WORKERS": 65},
                "PIPE_WORKERS benchmark knob is outside its safe range",
            ),
            (
                {"OMP_NUM_THREADS": 0},
                "OMP_NUM_THREADS=0 exceeds the 4-core benchmark target",
            ),
            (
                {"OMP_NUM_THREADS": 5},
                "OMP_NUM_THREADS=5 exceeds the 4-core benchmark target",
            ),
            (
                {"OMP_PROC_BIND": "master"},
                "OMP_PROC_BIND benchmark knob must be close or spread",
            ),
            (
                {"UNKNOWN": 1},
                "unsupported managed benchmark knob: UNKNOWN",
            ),
        )

        for knobs, message in invalid:
            with self.subTest(knobs=knobs), self.assertRaisesRegex(
                benchmark_support.RamdiskError,
                message,
            ):
                benchmark_support._normalized_runtime_knobs(
                    plan,
                    knobs,
                    node_core_count=lambda _plan, _node: 4,
                )

        self.assertEqual(
            benchmark_support._normalized_runtime_knobs(
                plan,
                {"PIPE_WORKERS": 64, "OMP_NUM_THREADS": 4},
                node_core_count=lambda _plan, _node: 4,
            ),
            {"PIPE_WORKERS": 64, "OMP_NUM_THREADS": 4},
        )
        self.assertEqual(
            benchmark_support._normalized_runtime_knobs(
                plan,
                {"OMP_PROC_BIND": "close"},
                node_core_count=lambda _plan, _node: 4,
            ),
            {"OMP_PROC_BIND": "close"},
        )

    def test_facade_injects_its_live_core_count_seam(self):
        plan = mock.sentinel.plan
        with mock.patch.object(
            ramdisk,
            "_node_core_count",
            return_value=2,
        ) as core_count:
            result = ramdisk._normalized_runtime_knobs(
                plan,
                {"OMP_NUM_THREADS": 2},
                node=7,
            )

        self.assertEqual(result, {"OMP_NUM_THREADS": 2})
        core_count.assert_called_once_with(plan, 7)

    def test_system_score_skips_mount_usage_without_statvfs(self):
        manifest = {
            "created_at": "2026-01-01T00:00:00+00:00",
            "ready_at": "2026-01-01T00:00:02+00:00",
            "mounts": [
                {
                    "path": "C:/colibri-ram",
                    "node": None,
                    "numa_allocation": {"0": 4},
                }
            ],
        }

        result = benchmark_support._system_score(
            manifest,
            [],
            10,
            12,
            meminfo=lambda: {
                "Shmem": 100,
                "ShmemPmdMapped": 25,
            },
            statvfs=None,
        )

        self.assertEqual(result["stage_seconds"], 2.0)
        self.assertEqual(result["shmem_bytes"], 0)
        self.assertIsNone(
            result["numa_page_placement"][0]["allocated_bytes"]
        )

    def test_import_is_lazy_and_does_not_load_lifecycle_layers(self):
        script = """
import sys
sys.path.insert(0, sys.argv[1])
import ramdisk_support.benchmark
unexpected = {
    "openai_server",
    "ramdisk_support.mounts",
    "ramdisk_support.processes",
    "ramdisk_support.state",
}.intersection(sys.modules)
assert not unexpected, sorted(unexpected)
"""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(C_DIR),
            ],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
