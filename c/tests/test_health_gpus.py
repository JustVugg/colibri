"""/health's system.gpus: the NVIDIA cards as nvidia-smi sees them, read at most once a second, and
nothing at all (and no second look) on a machine without nvidia-smi."""
import os
import sys
import tempfile
import unittest

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WINDOWS = os.name == "nt"


@unittest.skipIf(WINDOWS, "the fake nvidia-smi is a shell script")
class HealthGpusTest(unittest.TestCase):
    def test_gpus_come_from_nvidia_smi_once_a_second(self):
        sys.path.insert(0, C_DIR)
        import openai_server
        with tempfile.TemporaryDirectory() as d:
            smi = os.path.join(d, "nvidia-smi")
            with open(smi, "w") as f:
                f.write("#!/bin/sh\necho 'NVIDIA Test Card, 37, 2048, 8192, 55, 31.50, 70.00'\n")
            os.chmod(smi, 0o755)
            path = os.environ.get("PATH", "")
            os.environ["PATH"] = d + os.pathsep + path
            try:
                openai_server._GPU_LAST.clear()
                openai_server._GPU_LAST.update({"t": 0.0, "v": None})
                gpus = openai_server._gpu_stats()
            finally:
                os.environ["PATH"] = path
                openai_server._GPU_LAST.clear()
                openai_server._GPU_LAST.update({"t": 0.0, "v": None})
        self.assertEqual(gpus, [{"name": "NVIDIA Test Card", "util": 37.0, "mem_used_gb": 2.0, "mem_total_gb": 8.0,
                                 "temp": 55.0, "power_w": 31.5, "power_max_w": 70.0}])

    def test_no_nvidia_smi_no_block(self):
        sys.path.insert(0, C_DIR)
        import openai_server
        path = os.environ.get("PATH", "")
        with tempfile.TemporaryDirectory() as d:
            os.environ["PATH"] = d
            try:
                openai_server._GPU_LAST.clear()
                openai_server._GPU_LAST.update({"t": 0.0, "v": None})
                self.assertIsNone(openai_server._gpu_stats())
            finally:
                os.environ["PATH"] = path
                openai_server._GPU_LAST.clear()
                openai_server._GPU_LAST.update({"t": 0.0, "v": None})


if __name__ == "__main__":
    unittest.main()
