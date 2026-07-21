import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
ROOT = C_DIR.parent
MAKE = shutil.which("make")


class RamdiskPackagingTest(unittest.TestCase):
    @unittest.skipUnless(MAKE, "make is required")
    def test_staged_install_dry_run_includes_support_module(self):
        with tempfile.TemporaryDirectory() as stage:
            result = subprocess.run(
                [
                    MAKE,
                    "--no-print-directory",
                    "-n",
                    "install",
                    f"DESTDIR={stage}",
                    "PREFIX=/usr",
                ],
                cwd=C_DIR,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("ramdisk.py", result.stdout)
        self.assertIn("/usr/libexec/colibri/", result.stdout)

    def test_nix_package_copies_module_and_runs_python_tests(self):
        flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
        self.assertIn("c/ramdisk.py", flake)
        self.assertIn("make test\n", flake)
        self.assertNotIn("make test-c\n", flake)

    def test_environment_reference_documents_rammap_contract(self):
        reference = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
        for variable in (
            "COLI_WEIGHTS_DIR",
            "COLI_STATE_DIR",
            "COLI_RAMMAP",
            "COLI_RAM_PREFAULT",
        ):
            self.assertIn(f"`{variable}`", reference)
        self.assertIn("Incompatible with `COLI_MMAP=1`", reference)
        self.assertIn("volatile weight mounts never hold runtime state", reference)


if __name__ == "__main__":
    unittest.main()
