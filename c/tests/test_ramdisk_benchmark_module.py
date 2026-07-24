"""Direct contracts for the callback-injected benchmark module."""

import subprocess
import sys
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import benchmark as benchmark_support  # noqa: E402


class BenchmarkModuleTest(unittest.TestCase):
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
