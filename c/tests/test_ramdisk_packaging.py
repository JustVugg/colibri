"""Source and packaging contracts for the physically headless control plane."""

import ast
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / "c"
PIP_AVAILABLE = importlib.util.find_spec("pip") is not None
SETUPTOOLS_AVAILABLE = importlib.util.find_spec("setuptools") is not None


def _setuptools_supports_spdx_license():
    """Whether the ambient backend supports the declared PEP 639 metadata."""
    if not SETUPTOOLS_AVAILABLE:
        return False
    try:
        version = importlib.metadata.version("setuptools")
        numeric = re.match(r"\d+(?:\.\d+)*", version)
        if numeric is None:
            return False
        parsed = tuple(
            int(part) for part in numeric.group(0).split(".")[:3]
        )
        while len(parsed) < 3:
            parsed += (0,)
    except Exception:
        return False
    return parsed >= (77, 0, 1)


SETUPTOOLS_SUPPORTS_SPDX = _setuptools_supports_spdx_license()
ISOLATED_PEP517_REQUESTED = (
    os.environ.get("COLIBRI_TEST_ISOLATED_PEP517") == "1"
)

FORBIDDEN_PATHS = (
    "c/ramdisk_ui.py",
    "c/ramdisk_textual.py",
    "c/ramdisk_support/curses_ui.py",
    "c/requirements-tui.txt",
    "c/tools/capture_ramdisk_tui.py",
    "c/tests/test_ramdisk_ui.py",
    "c/tests/test_ramdisk_textual.py",
    "c/tests/test_ramdisk_tui_capture.py",
    "c/tests/test_ramdisk_curses_ui_module.py",
    "docs/ramdisk-tui.md",
    "docs/ramdisk-tui-howto.md",
    "docs/media/ramdisk-tui/01-inspect.svg",
    "docs/media/ramdisk-tui/02-review.svg",
    "docs/media/ramdisk-tui/03-prepare-confirmation.svg",
    "docs/media/ramdisk-tui/04-ready.svg",
    "docs/media/ramdisk-tui/05-running.svg",
    "docs/media/ramdisk-tui/06-stopped.svg",
    "docs/media/ramdisk-tui/07-absent.svg",
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
ROOT_SUPPORT_MODULES = (
    "resource_plan.py",
    "doctor.py",
    "autotune.py",
    "openai_server.py",
    "version.py",
    "ramdisk.py",
)


class RamdiskPackagingTest(unittest.TestCase):
    def _run_packaging_command(self, command, *, cwd):
        result = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1",
            },
        )
        if result.returncode:
            self.fail(
                "packaging command failed (%s):\nstdout:\n%s\nstderr:\n%s"
                % (
                    " ".join(map(str, command)),
                    result.stdout,
                    result.stderr,
                )
            )
        return result

    def test_spdx_license_declares_a_compatible_setuptools_minimum(self):
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        minimum_match = re.search(
            r'setuptools>=(\d+(?:\.\d+)+)',
            pyproject,
        )
        self.assertIsNotNone(minimum_match, pyproject)
        self.assertGreaterEqual(
            tuple(map(int, minimum_match.group(1).split("."))),
            (77, 0, 1),
            "SPDX project metadata requires setuptools 77.0.1 or newer",
        )

    def _assert_wheel_contains_runnable_control_plane(self, *, isolated_build):
        with tempfile.TemporaryDirectory() as stage:
            stage_root = Path(stage)
            source = stage_root / "source"
            wheel_dir = stage_root / "wheel"
            installed = stage_root / "installed"
            runtime = source / "c"
            support = runtime / "ramdisk_support"
            source.mkdir()
            wheel_dir.mkdir()
            runtime.mkdir()
            support.mkdir()

            for name in ("pyproject.toml", "README.md", "LICENSE"):
                shutil.copy2(ROOT / name, source / name)
            shutil.copytree(ROOT / "colibri", source / "colibri")
            for name in ("__init__.py", "coli", *ROOT_SUPPORT_MODULES):
                shutil.copy2(C_DIR / name, runtime / name)
            for name in CORE_MODULES:
                shutil.copy2(C_DIR / "ramdisk_support" / name, support / name)
            shutil.copytree(C_DIR / "tools", runtime / "tools")

            if PIP_AVAILABLE:
                command = [
                    sys.executable,
                    "-m",
                    "pip",
                    "--isolated",
                    "wheel",
                    "--use-pep517",
                ]
                if not isolated_build:
                    command.extend(["--no-build-isolation", "--no-index"])
                command.extend(
                    [
                        "--verbose",
                        "--no-deps",
                        "--wheel-dir",
                        str(wheel_dir),
                        str(source),
                    ]
                )
                self._run_packaging_command(command, cwd=stage_root)
            else:
                if isolated_build:
                    self.fail("default isolated PEP 517 verification requires pip")
                self._run_packaging_command(
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
                )

            wheels = tuple(wheel_dir.glob("*.whl"))
            self.assertEqual(len(wheels), 1)

            if PIP_AVAILABLE:
                self._run_packaging_command(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--no-index",
                        "--no-deps",
                        "--target",
                        str(installed),
                        str(wheels[0]),
                    ],
                    cwd=stage_root,
                )

            with zipfile.ZipFile(wheels[0]) as wheel:
                members = set(wheel.namelist())
                self.assertIn("c/coli", members)
                for name in ROOT_SUPPORT_MODULES:
                    self.assertIn("c/%s" % name, members)
                for name in CORE_MODULES:
                    self.assertIn("c/ramdisk_support/%s" % name, members)
                for path in FORBIDDEN_PATHS:
                    self.assertNotIn(path, members)
                for name in FUTURE_MODULES:
                    self.assertNotIn("c/ramdisk_support/%s" % name, members)
                if not PIP_AVAILABLE:
                    wheel.extractall(installed)

            smoke = self._run_packaging_command(
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
            )

        self.assertIn("interleaved = one shared model copy", smoke.stdout)
        self.assertIn("stage", smoke.stdout)
        self.assertIn("verify", smoke.stdout)

    @unittest.skipUnless(
        SETUPTOOLS_AVAILABLE and SETUPTOOLS_SUPPORTS_SPDX,
        "non-isolated wheel build requires the declared setuptools>=77.0.1 "
        "(SPDX license); covered by the dedicated wheel lanes otherwise",
    )
    def test_wheel_contains_runnable_ramdisk_control_plane(self):
        """The selected ambient backend builds without package-index access."""
        self._assert_wheel_contains_runnable_control_plane(isolated_build=False)

    @unittest.skipUnless(
        PIP_AVAILABLE and ISOLATED_PEP517_REQUESTED,
        "set COLIBRI_TEST_ISOLATED_PEP517=1 for the networked build gate",
    )
    def test_default_isolated_pep517_wheel_installs_and_runs(self):
        """Default PEP 517 isolation resolves, builds, installs, and runs."""
        self._assert_wheel_contains_runnable_control_plane(isolated_build=True)

    def test_forbidden_frontend_paths_are_physically_absent(self):
        for relative in FORBIDDEN_PATHS:
            with self.subTest(path=relative):
                self.assertFalse((ROOT / relative).exists())

    def test_runner_modules_are_deferred_to_pr3(self):
        for name in FUTURE_MODULES:
            with self.subTest(module=name):
                self.assertFalse((C_DIR / "ramdisk_support" / name).exists())

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
                self.assertIn("ramdisk_support", text)
                for name in ROOT_SUPPORT_MODULES:
                    self.assertIn(name, text)
                for name in CORE_MODULES:
                    self.assertIn(name, text)
                self.assertNotIn("ramdisk_ui.py", text)
                self.assertNotIn("ramdisk_textual.py", text)
                self.assertNotIn("requirements-tui.txt", text)
                self.assertNotIn("curses_ui.py", text)
                for module in FUTURE_MODULES:
                    self.assertNotIn(module, text)

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
                self.assertNotIn("ram-disk ui", text)

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
            libexec = prefix / "libexec" / "colibri"
            for name in ("ramdisk_ui.py", "ramdisk_textual.py", "requirements-tui.txt"):
                self.assertFalse((libexec / name).exists())
            self.assertFalse((support / "curses_ui.py").exists())
            for name in FUTURE_MODULES:
                self.assertFalse((support / name).exists())

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
                    "raise AssertionError('future extension imported eagerly')\n",
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


    def test_clean_removes_makefile_outputs_and_legacy_engine_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = [
                root / name
                for name in (
                    ".build-config",
                    "colibri", "colibri.exe", "glm", "glm.exe",
                    "olmoe", "olmoe.exe", "inkling", "inkling.exe",
                    "kimi_k3", "kimi_k3.exe", "iobench", "iobench.exe",
                    "backend_cuda.o", "backend_cuda_ink.o", "backend_loader.o",
                    "backend_vulkan.o", "backend_cuda_test", "backend_cuda_test.exe",
                    "ragged_attention_test", "ragged_attention_test.exe",
                    "backend_cuda_bench", "backend_cuda_bench.exe", "backend_metal.o",
                    "backend_metal_test", "backend_metal_test.exe",
                    "gemm_largebatch_test", "gemm_largebatch_test.exe", "coli_cuda.dll",
                    "coli_cuda.lib", "coli_cuda.exp", "tools/libiq3.so",
                    "tools/libiq3.dylib", "tools/iq3.dll", "tools/librans_c.so",
                    "tools/librans_c.dylib", "tools/rans_c.dll",
                    "shaders/qmatmul_gate_up.spv", "shaders/attention_absorb.spv",
                    "shaders/rmsnorm.spv",
                    "coli_hip.dll", "coli_hip.lib", "coli_hip.exp", "coli_hip.pdb",
                    "deepseek_v4", "deepseek_v4.exe",
                    "native_quant.o", "native_quant_parallel.o",
                    "native_quant_dual.o", "native_quant_batch_avx512.o",
                    "native_quant_fp4_rows16.o",
                    "COLI_V4_UNIT_fixture.o",
                )
            ]
            tests_dir = root / "tests"
            tests_dir.mkdir()
            cache_dirs = [
                root / "__pycache__",
                root / "ramdisk_support" / "__pycache__",
                tests_dir / "__pycache__",
                root / "build" / "ownership",
            ]
            for cache_dir in cache_dirs:
                cache_dir.mkdir(parents=True)
                (cache_dir / "stale.pyc").write_bytes(b"bytecode")
            makefile = (C_DIR / "Makefile").read_text(encoding="utf-8")
            test_basenames = re.findall(
                r"^tests/(test_[a-z0-9_]+)\$\(EXE\):", makefile, re.MULTILINE
            )
            test_basenames.extend(
                (
                    "bench_topp",
                    "bench_dsa_select",
                    "bench_idot",
                    "bench_mla_simd",
                    "fuzz_rans",
                )
            )
            artifacts.extend(
                tests_dir / f"{name}{suffix}"
                for name in test_basenames
                for suffix in ("", ".exe")
            )
            for artifact in artifacts:
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(b"binary")

            subprocess.run(
                [sys.executable, str(C_DIR / "tools" / "clean.py")],
                cwd=root,
                text=True,
                capture_output=True,
                check=True,
            )
            remaining = [
                str(artifact)
                for artifact in artifacts
                if artifact.exists()
            ]
            self.assertFalse(
                remaining,
                "clean.py left %d artifact(s) behind: %s"
                % (len(remaining), remaining[:20]),
            )
            self.assertTrue(all(not cache_dir.exists() for cache_dir in cache_dirs))

    def test_environment_reference_documents_headless_state_contract(self):
        reference = (ROOT / "docs" / "ENVIRONMENT.md").read_text(encoding="utf-8")
        for variable in ("XDG_STATE_HOME", "COLI_RAMDISK_MANIFEST"):
            self.assertIn("`%s`" % variable, reference)
        self.assertIn("staging namespace is volatile", reference)
        self.assertRegex(
            reference,
            r"manifest, lifecycle lock, and\s+recovery record",
        )

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
