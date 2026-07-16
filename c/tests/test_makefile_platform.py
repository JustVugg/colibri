import os
import shutil
import subprocess
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
MAKE = shutil.which("make")


@unittest.skipUnless(MAKE, "make is required")
class MakefilePlatformTests(unittest.TestCase):
    def _dry_run(self, target, triplet, **variables):
        args = [
            MAKE,
            "--no-print-directory",
            "-B",
            "-n",
            target,
            f"TRIPLET={triplet}",
        ]
        args.extend(f"{name}={value}" for name, value in variables.items())
        return subprocess.run(
            args,
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=True,
        )

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

    def test_portable_build_uses_target_architecture(self):
        cases = (
            ("x86_64-unknown-linux-gnu", "-march=x86-64-v3"),
            ("aarch64-unknown-linux-gnu", "-march=armv8-a"),
            ("powerpc64le-unknown-linux-gnu", "-mcpu=power8"),
            ("ppc64le-unknown-linux-gnu", "-mcpu=power8"),
        )

        for triplet, expected_flag in cases:
            with self.subTest(triplet=triplet):
                result = self._dry_run("portable", triplet)
                self.assertIn(expected_flag, result.stdout)

    def test_darwin_portable_build_does_not_force_x86_architecture(self):
        missing_libomp = "/colibri-test/missing-libomp"
        result = self._dry_run(
            "portable", "arm64-apple-darwin", OMPDIR=missing_libomp
        )

        self.assertIn("clang -O3", result.stdout)
        self.assertNotIn("-mcpu=x86-64-v3", result.stdout)
        self.assertNotIn(missing_libomp, result.stdout)
        self.assertNotIn("-fopenmp", result.stdout)


if __name__ == "__main__":
    unittest.main()
