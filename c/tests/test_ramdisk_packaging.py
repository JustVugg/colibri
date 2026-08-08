"""Source and packaging contracts for the physically headless control plane."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "c"

FORBIDDEN_PATHS = (
    "c/ramdisk_ui.py",
    "c/ramdisk_textual.py",
    "c/ramdisk_support/curses_ui.py",
    "c/requirements-tui.txt",
)
CORE_MODULES = (
    "__init__.py",
    "accelerator.py",
    "cli.py",
    "common.py",
    "contracts.py",
    "discovery.py",
    "lifecycle.py",
    "linux_ops.py",
    "model.py",
    "mounts.py",
    "planning.py",
    "platform_ops.py",
    "presentation.py",
    "presets.py",
    "processes.py",
    "state.py",
    "tokens.py",
)
FUTURE_MODULES = ("benchmark.py", "runtime_monitor.py", "supervision.py")


class RamdiskPackagingTest(unittest.TestCase):
    def test_forbidden_frontend_paths_are_physically_absent(self):
        for relative in FORBIDDEN_PATHS:
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())

    def test_launcher_requires_exact_headless_inventory(self):
        tree = ast.parse((C_DIR / "coli").read_text(encoding="utf-8"))
        assignments = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name):
                    try:
                        assignments[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass

        self.assertEqual(
            assignments["_SUPPORT_MODULES"],
            (
                "resource_plan.py",
                "doctor.py",
                "autotune.py",
                "openai_server.py",
                "ramdisk.py",
            ),
        )
        self.assertEqual(
            assignments["_SUPPORT_PACKAGE_MODULES"],
            CORE_MODULES,
        )
        for module in FUTURE_MODULES:
            self.assertNotIn(module, assignments["_SUPPORT_PACKAGE_MODULES"])

    def test_every_packager_carries_core_without_frontend_or_runner(self):
        files = {
            "Makefile": (C_DIR / "Makefile").read_text(encoding="utf-8"),
            "flake.nix": (ROOT / "flake.nix").read_text(encoding="utf-8"),
            "release.yml": (ROOT / ".github/workflows/release.yml").read_text(
                encoding="utf-8"
            ),
        }
        for label, text in files.items():
            with self.subTest(packager=label):
                self.assertIn("ramdisk.py", text)
                self.assertIn("ramdisk_support", text)
                self.assertIn("tokens.py", text)
                self.assertNotIn("ramdisk_ui.py", text)
                self.assertNotIn("ramdisk_textual.py", text)
                self.assertNotIn("requirements-tui.txt", text)
                self.assertNotIn("curses_ui.py", text)

    def test_python_metadata_has_no_frontend_dependency_or_extra(self):
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('include = ["colibri*", "c", "c.tools", "c.ramdisk_support*"]', text)
        self.assertIn('"coli"', text)
        self.assertNotIn("textual", text.lower())
        self.assertNotIn("requirements-tui", text)

    def test_workflows_do_not_install_or_archive_frontend(self):
        for path in sorted((ROOT / ".github/workflows").glob("*.yml")):
            text = path.read_text(encoding="utf-8").lower()
            with self.subTest(path=path.name):
                self.assertNotIn("requirements-tui", text)
                self.assertNotIn("ramdisk_textual", text)
                self.assertNotIn("ramdisk_ui.py", text)
                self.assertNotIn("curses_ui.py", text)

    def test_bare_source_launcher_prints_headless_help_and_returns_two(self):
        result = subprocess.run(
            [sys.executable, str(C_DIR / "coli"), "ramdisk"],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("stage", result.stdout)
        self.assertIn("verify", result.stdout)
        self.assertNotIn("traceback", (result.stdout + result.stderr).lower())

    def test_missing_token_source_launcher_emits_json_only(self):
        result = subprocess.run(
            [
                sys.executable,
                str(C_DIR / "coli"),
                "ramdisk",
                "stage",
                "--yes",
                "--json",
            ],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stderr, "")
        self.assertEqual(
            json.loads(result.stdout)["schema"],
            "colibri.ramdisk.error.v1",
        )

    def test_make_install_headless_bundle_runs_without_extensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "c"
            prefix = root / "prefix"
            shutil.copytree(C_DIR, source)
            for engine in ("colibri", "olmoe"):
                path = source / engine
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(0o755)

            installed = subprocess.run(
                [
                    "make",
                    "install",
                    "COLI_V4_SUPPORTED=",
                    "PREFIX=%s" % prefix,
                ],
                cwd=source,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)

            support = prefix / "libexec" / "colibri" / "ramdisk_support"
            self.assertEqual(
                tuple(sorted(path.name for path in support.glob("*.py"))),
                tuple(sorted(CORE_MODULES)),
            )
            for relative in FORBIDDEN_PATHS:
                self.assertFalse((prefix / relative).exists())

            result = subprocess.run(
                [sys.executable, str(prefix / "bin" / "coli"), "ramdisk"],
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("stage", result.stdout)
            self.assertNotIn("traceback", (result.stdout + result.stderr).lower())

    def test_launcher_tolerates_known_future_extension_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "c"
            shutil.copytree(C_DIR, source)
            for module in FUTURE_MODULES:
                (source / "ramdisk_support" / module).write_text(
                    "# synthetic future extension\n",
                    encoding="utf-8",
                )

            result = subprocess.run(
                [sys.executable, str(source / "coli"), "ramdisk"],
                cwd=source,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("stage", result.stdout)
            self.assertNotIn("traceback", (result.stdout + result.stderr).lower())


    def test_slim_container_copies_the_complete_launcher_support_bundle(self):
        dockerfile = (ROOT / "docker" / "Dockerfile.slim").read_text(
            encoding="utf-8"
        )
        self.assertRegex(
            dockerfile,
            r"(?m)^COPY c/coli .*\bc/ramdisk\.py\b.* \./$",
        )
        self.assertRegex(
            dockerfile,
            r"(?m)^COPY c/ramdisk_support/ +\./ramdisk_support/$",
        )


if __name__ == "__main__":
    unittest.main()
