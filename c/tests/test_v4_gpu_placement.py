"""Run production V4 GPU placement against a mock backend without CUDA."""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "uses ELF section GC")
class GpuPlacementTests(unittest.TestCase):
    def test_cache_ownership_and_cleanup(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        if not shutil.which(cc[0]):
            self.skipTest("C compiler unavailable")
        source = Path(__file__).with_suffix(".c")
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "gpu-placement")
            subprocess.run(cc + ["-D_GNU_SOURCE", "-O1", "-ffunction-sections",
                                 "-fdata-sections", str(source), "-Wl,--gc-sections",
                                 "-pthread", "-lm", "-o", binary], check=True,
                           capture_output=True, text=True)
            result = subprocess.run([binary], check=True, capture_output=True, text=True)
            self.assertIn("test_v4_gpu_placement: ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
