"""Direct import and Linux measurement contracts for the benchmark module."""

import subprocess
import sys
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))

from ramdisk_support import benchmark  # noqa: E402


class BenchmarkModuleTest(unittest.TestCase):
    def test_status_rss_and_numa_parsers_keep_required_measurements(self):
        rss = benchmark._parse_process_status(
            "RssAnon:\t10 kB\nRssFile:\t20 kB\n"
            "RssShmem:\t30 kB\nVmSwap:\t40 kB\n"
        )
        placement = benchmark._parse_numa_maps(
            "00400000 default file=/tmp/model N0=3 N1=5\n"
            "00600000 default anon=2 dirty=2 N0=2\n",
            page_size=4096,
        )

        self.assertEqual(
            rss,
            {
                "anonymous_bytes": 10 * 1024,
                "file_bytes": 20 * 1024,
                "shmem_bytes": 30 * 1024,
                "process_swap_bytes": 40 * 1024,
            },
        )
        self.assertEqual(placement["by_node"], {"0": 5, "1": 5})
        self.assertEqual(placement["bytes_by_node"], {"0": 5 * 4096, "1": 5 * 4096})

    def test_missing_dram_collector_preflight_is_explicitly_unavailable(self):
        collector = benchmark.preflight_dram_collector(
            environ={},
            which=lambda _name: None,
        )
        self.assertFalse(collector.available)
        sample = collector.finish(collector.start(123, {"id": "cell"}))
        self.assertFalse(sample["available"])
        self.assertIn("unavailable", sample["error"])

    def test_numa_gate_tolerates_only_predeclared_incidental_pages(self):
        local = {
            "mode": "local",
            "nodes": [0],
            "min_expected_fraction": 0.95,
            "max_imbalance": 0.25,
        }
        interleaved = {
            "mode": "interleave",
            "nodes": [0, 1],
            "min_expected_fraction": 0.95,
            "max_imbalance": 0.25,
        }

        self.assertTrue(benchmark._placement_verified({"0": 950, "1": 50}, local))
        self.assertFalse(benchmark._placement_verified({"0": 800, "1": 200}, local))
        self.assertTrue(
            benchmark._placement_verified(
                {"0": 480, "1": 470, "2": 50},
                interleaved,
            )
        )
        self.assertFalse(
            benchmark._placement_verified(
                {"0": 700, "1": 250, "2": 50},
                interleaved,
            )
        )

    def test_import_is_lazy_and_frontend_free(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "openai_server",
    "ramdisk_ui",
    "ramdisk_textual",
    "ramdisk_support.curses_ui",
    "ramdisk_support.lifecycle",
    "ramdisk_support.processes",
    "ramdisk_support.state",
    "textual",
}

class RejectHeavyLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith("textual."):
            raise AssertionError("eager benchmark import: " + fullname)
        return None

sys.meta_path.insert(0, RejectHeavyLayers())
sys.path.insert(0, sys.argv[1])
import ramdisk_support.benchmark
assert not (blocked & set(sys.modules)), sorted(blocked & set(sys.modules))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
