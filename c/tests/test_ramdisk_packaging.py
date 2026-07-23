import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


C_DIR = Path(__file__).resolve().parents[1]
ROOT = C_DIR.parent
MAKE = shutil.which("make")
SETUPTOOLS_AVAILABLE = importlib.util.find_spec("setuptools") is not None
SUPPORT_MODULES = (
    "version.py",
    "resource_plan.py",
    "doctor.py",
    "openai_server.py",
    "ramdisk.py",
    "ramdisk_ui.py",
    "ramdisk_textual.py",
)


def copy_support(destination, exclude=()):
    for name in SUPPORT_MODULES:
        if name not in exclude:
            shutil.copy2(C_DIR / name, destination / name)


class RamdiskPackagingTest(unittest.TestCase):
    @unittest.skipUnless(SETUPTOOLS_AVAILABLE, "setuptools is required to build a wheel")
    def test_wheel_contains_runnable_ramdisk_control_plane(self):
        """The pip entry point must not depend on files left in a source clone."""
        with tempfile.TemporaryDirectory() as stage:
            stage_root = Path(stage)
            source = stage_root / "source"
            wheel_dir = stage_root / "wheel"
            installed = stage_root / "installed"
            runtime = source / "c"
            source.mkdir()
            wheel_dir.mkdir()
            runtime.mkdir()

            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT / name, source / name)
            shutil.copytree(ROOT / "colibri", source / "colibri")
            for name in (
                "__init__.py",
                "coli",
                "requirements-tui.txt",
                "download_fp8.py",
                *SUPPORT_MODULES,
            ):
                shutil.copy2(C_DIR / name, runtime / name)
            shutil.copytree(C_DIR / "tools", runtime / "tools")

            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import setuptools.build_meta as backend, sys; "
                        "print(backend.build_wheel(sys.argv[1]))"
                    ),
                    str(wheel_dir),
                ],
                cwd=source,
                text=True,
                capture_output=True,
                check=True,
            )

            wheels = tuple(wheel_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                members = set(wheel.namelist())
                for name in ("coli", "requirements-tui.txt", *SUPPORT_MODULES):
                    self.assertIn(f"c/{name}", members)
                wheel.extractall(installed)

            smoke = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.path.insert(0, sys.argv[1]); "
                        "from colibri.cli import main; "
                        "sys.argv = ['coli', 'ramdisk', '--help']; "
                        "main()"
                    ),
                    str(installed),
                ],
                cwd=stage_root,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("interleaved = one shared model copy", smoke.stdout)
        self.assertIn("prepare", smoke.stdout)

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
        self.assertIn("ramdisk_ui.py", result.stdout)
        self.assertIn("ramdisk_textual.py", result.stdout)
        self.assertIn("requirements-tui.txt", result.stdout)
        self.assertIn("version.py", result.stdout)
        self.assertIn("coli.libexec", result.stdout)
        self.assertIn("/usr/libexec/colibri/", result.stdout)

    def test_staged_installed_cli_can_load_ramdisk_support_modules(self):
        with tempfile.TemporaryDirectory() as stage:
            prefix = Path(stage) / "usr"
            bin_dir = prefix / "bin"
            libexec_dir = prefix / "libexec" / "colibri"
            bin_dir.mkdir(parents=True)
            libexec_dir.mkdir(parents=True)
            shutil.copy2(C_DIR / "coli", bin_dir / "coli")
            copy_support(libexec_dir)

            result = subprocess.run(
                [sys.executable, str(bin_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("interleaved = one shared model copy", result.stdout)
        self.assertIn("prepare", result.stdout)

    def test_custom_install_layout_uses_its_recorded_support_directory(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            bin_dir = root / "custom-bin"
            support_dir = root / "unusual" / "python-and-engine"
            bin_dir.mkdir()
            support_dir.mkdir(parents=True)
            shutil.copy2(C_DIR / "coli", bin_dir / "coli")
            (bin_dir / "coli.libexec").write_text(
                str(support_dir) + "\n", encoding="utf-8"
            )
            copy_support(support_dir)

            result = subprocess.run(
                [sys.executable, str(bin_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("interleaved = one shared model copy", result.stdout)

    def test_recorded_install_layout_wins_over_stale_colocated_bundle(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            bin_dir = root / "bin"
            support_dir = root / "support"
            bin_dir.mkdir()
            support_dir.mkdir()
            shutil.copy2(C_DIR / "coli", bin_dir / "coli")
            (bin_dir / "coli.libexec").write_text(
                str(support_dir) + "\n", encoding="utf-8"
            )
            copy_support(bin_dir)
            copy_support(support_dir)
            (bin_dir / "ramdisk_ui.py").write_text(
                'raise RuntimeError("stale colocated UI loaded")\n', encoding="utf-8"
            )

            result = subprocess.run(
                [sys.executable, str(bin_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("interleaved = one shared model copy", result.stdout)

    def test_colocated_release_modules_win_over_stale_sibling_install(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            release_dir = root / "release"
            stale_dir = root / "libexec" / "colibri"
            release_dir.mkdir()
            stale_dir.mkdir(parents=True)
            shutil.copy2(C_DIR / "coli", release_dir / "coli")
            copy_support(release_dir)
            copy_support(stale_dir)
            (stale_dir / "version.py").write_text(
                '__version__ = "stale"\n', encoding="utf-8"
            )
            (stale_dir / "ramdisk_ui.py").write_text(
                'raise RuntimeError("stale UI loaded")\n', encoding="utf-8"
            )

            result = subprocess.run(
                [sys.executable, str(release_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("interleaved = one shared model copy", result.stdout)

    def test_installed_support_wins_over_stale_engine_override_bundle(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            bin_dir = root / "usr" / "bin"
            support_dir = root / "usr" / "libexec" / "colibri"
            stale_engine_dir = root / "stale-engine-bundle"
            bin_dir.mkdir(parents=True)
            support_dir.mkdir(parents=True)
            stale_engine_dir.mkdir()
            shutil.copy2(C_DIR / "coli", bin_dir / "coli")
            copy_support(support_dir)
            copy_support(stale_engine_dir)
            (stale_engine_dir / "version.py").write_text(
                '__version__ = "stale"\n', encoding="utf-8"
            )
            (stale_engine_dir / "ramdisk_ui.py").write_text(
                'raise RuntimeError("stale engine UI loaded")\n', encoding="utf-8"
            )
            environment = dict(os.environ)
            environment["COLI_ENGINE"] = str(stale_engine_dir / "colibri")

            result = subprocess.run(
                [sys.executable, str(bin_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=True,
                env=environment,
            )

        self.assertIn("interleaved = one shared model copy", result.stdout)

    def test_partial_flat_bundle_cannot_mix_with_complete_sibling_install(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            release_dir = root / "release"
            sibling_dir = root / "libexec" / "colibri"
            release_dir.mkdir()
            sibling_dir.mkdir(parents=True)
            shutil.copy2(C_DIR / "coli", release_dir / "coli")
            copy_support(release_dir, exclude=("ramdisk_ui.py",))
            copy_support(sibling_dir)

            result = subprocess.run(
                [sys.executable, str(release_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete colocated support bundle", result.stderr)
        self.assertIn("ramdisk_ui.py", result.stderr)

    def test_partial_launcher_directory_cannot_shadow_complete_engine_bundle(self):
        with tempfile.TemporaryDirectory() as stage:
            root = Path(stage)
            bin_dir = root / "bin"
            engine_dir = root / "engine-bundle"
            bin_dir.mkdir()
            engine_dir.mkdir()
            shutil.copy2(C_DIR / "coli", bin_dir / "coli")
            shutil.copy2(C_DIR / "ramdisk_ui.py", bin_dir / "ramdisk_ui.py")
            copy_support(engine_dir)
            environment = dict(os.environ)
            environment["COLI_ENGINE"] = str(engine_dir / "colibri")

            result = subprocess.run(
                [sys.executable, str(bin_dir / "coli"), "ramdisk", "--help"],
                text=True,
                capture_output=True,
                check=False,
                env=environment,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("incomplete colocated support bundle", result.stderr)
        self.assertIn("version.py", result.stderr)

    def test_nix_package_uses_current_engine_and_runs_python_tests(self):
        flake = (ROOT / "flake.nix").read_text(encoding="utf-8")
        self.assertIn("c/ramdisk.py", flake)
        self.assertIn("c/ramdisk_ui.py", flake)
        self.assertIn("c/ramdisk_textual.py", flake)
        self.assertIn("textual", flake)
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
        self.assertIn("cp c/ramdisk_ui.py dist/", workflow)
        self.assertIn("cp c/ramdisk_textual.py dist/", workflow)
        self.assertIn("cp c/requirements-tui.txt dist/", workflow)
        self.assertIn("python3 coli ramdisk --help", workflow)
        self.assertIn(
            "python3 -m py_compile ramdisk.py ramdisk_ui.py ramdisk_textual.py",
            workflow,
        )
        self.assertIn(
            "python3 -m pip install --disable-pip-version-check "
            "-r requirements-tui.txt",
            workflow,
        )
        self.assertIn("import ramdisk_textual", workflow)
        self.assertIn('grep -Fq "interleaved = one shared model copy"', workflow)
        self.assertNotIn("python3 coli info 2>&1 || true", workflow)
        self.assertNotIn("make glm ${{ matrix.make_args }}", workflow)

    def test_clean_removes_current_and_legacy_engine_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [root / name for name in ("colibri", "colibri.exe", "glm", "glm.exe")]
            tests_dir = root / "tests"
            tests_dir.mkdir()
            artifacts.extend(
                tests_dir / name
                for name in ("test_resource_masks", "test_e8_kernel")
            )
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
            "COLI_NUMA_NODES",
            "COLI_CPU_AFFINITY",
        ):
            self.assertIn(f"`{variable}`", reference)
        self.assertIn("Incompatible with `COLI_MMAP=1`", reference)
        self.assertIn("volatile weight mounts never hold runtime state", reference)


if __name__ == "__main__":
    unittest.main()
