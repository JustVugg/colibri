"""Verify speculative head dispatch and error handling without model artifacts."""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "uses ELF section GC")
class HeadBatchTests(unittest.TestCase):
    def test_dispatch_verification_and_fallback(self):
        cc = shlex.split(os.environ.get("CC", "cc"))
        if not shutil.which(cc[0]):
            self.skipTest("C compiler unavailable")
        if "avx2" in Path("/proc/cpuinfo").read_text():
            cc += ["-mavx2"]
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / "head-batch")
            subprocess.run(cc + ["-D_GNU_SOURCE", "-O1", "-ffunction-sections",
                                 "-fdata-sections", str(Path(__file__).with_suffix(".c")),
                                 "-Wl,--gc-sections", "-pthread", "-lm", "-o", binary],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([binary], check=True, capture_output=True, text=True)
            self.assertIn("test_v4_head_batch: ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
