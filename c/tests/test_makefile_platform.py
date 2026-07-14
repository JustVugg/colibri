import os
import shutil
import subprocess
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")


@unittest.skipUnless(MAKE, "make is required")
class MakefilePlatformTests(unittest.TestCase):
    def test_windows_nt_without_uname_selects_mingw_build(self):
        env = os.environ.copy()
        env["OS"] = "Windows_NT"
        env["PATH"] = ""

        result = subprocess.run(
            [MAKE, "--no-print-directory", "-B", "-n", "glm"],
            cwd=C_DIR,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("-o glm.exe", result.stdout)
        self.assertIn("-fopenmp", result.stdout)
        self.assertIn("-static", result.stdout)


if __name__ == "__main__":
    unittest.main()
