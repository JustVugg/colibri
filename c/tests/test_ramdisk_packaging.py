import shutil
import subprocess
import sys
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

    def test_nix_package_uses_current_engine_and_runs_python_tests(self):
        flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
        self.assertIn("c/ramdisk.py", flake)
        self.assertIn('make -C c colibri ARCH="$ARCH"', flake)
        self.assertIn("cp c/colibri", flake)
        self.assertIn('COLI_ENGINE "$out/lib/colibri/colibri"', flake)
        self.assertIn('program = "${colibri}/bin/glm";', flake)
        self.assertIn("nativeCheckInputs = [pythonEnv];", flake)
        self.assertIn("make test\n", flake)
        self.assertNotIn("make test-c\n", flake)

    def test_release_builds_current_engine_and_includes_ramdisk_module(self):
        workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8"
        )
        self.assertIn("make colibri ${{ matrix.make_args }}", workflow)
        self.assertIn("cp c/colibri${{ matrix.ext }}", workflow)
        self.assertIn("cp c/ramdisk.py dist/", workflow)
        self.assertNotIn("make glm ${{ matrix.make_args }}", workflow)

    def test_clean_removes_current_and_legacy_engine_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [root / name for name in ("colibri", "colibri.exe", "glm", "glm.exe")]
            for artifact in artifacts:
                artifact.write_bytes(b"binary")

            subprocess.run(
                [sys.executable, str(C_DIR / "tools" / "clean.py")],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertTrue(all(not artifact.exists() for artifact in artifacts))

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
