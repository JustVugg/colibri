"""Stable public and lazy-import contracts for the headless facade."""

import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import ramdisk  # noqa: E402


PUBLIC_SIGNATURES = {
    "discover_hardware": "()",
    "scan_model": "(model_dir)",
    "build_plan": "(args, hardware=None, model=None)",
    "configure_parser": "(parser, common_parent=None)",
    "prepare": (
        "(args, progress=None, display_plan=True, expected_plan_token=None, "
        "cancel_event=None)"
    ),
    "stage": (
        "(args, progress=None, display_plan=True, expected_plan_token=None, "
        "cancel_event=None)"
    ),
    "destroy": "(args, expected_manifest_token=None)",
    "status": "(deep=True)",
    "verify": "()",
    "dispatch": "(args, cli_path=None, engine_path=None, system=None)",
}


class RamdiskFacadeContractTest(unittest.TestCase):
    def test_public_exports_and_signatures_are_headless_and_stable(self):
        self.assertTrue(issubclass(ramdisk.RamdiskError, RuntimeError))
        for name, expected_signature in PUBLIC_SIGNATURES.items():
            with self.subTest(name=name):
                exported = getattr(ramdisk, name)
                self.assertIn(name, ramdisk.__all__)
                self.assertEqual(str(inspect.signature(exported)), expected_signature)

        for name in (
            "benchmark",
            "launch_tui",
            "BENCHMARK_SCHEMA",
            "BENCHMARK_PROMPT",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(ramdisk, name))
                self.assertNotIn(name, ramdisk.__all__)
        for name in ("start", "stop"):
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(ramdisk, name)))
                self.assertNotIn(name, ramdisk.__all__)

    def test_schema_names_and_versions_remain_stable(self):
        self.assertEqual(ramdisk.MANIFEST_VERSION, 1)
        self.assertEqual(ramdisk.PLAN_SCHEMA, "colibri.ramdisk.plan.v1")
        self.assertEqual(ramdisk.STATUS_SCHEMA, "colibri.ramdisk.status.v1")

    def test_import_rejects_frontend_and_runner_modules(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "ramdisk_ui",
    "ramdisk_textual",
    "ramdisk_support.curses_ui",
    "ramdisk_support.benchmark",
    "ramdisk_support.runtime_monitor",
    "ramdisk_support.supervision",
}

class RejectExcludedLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname == "textual" or fullname.startswith("textual."):
            raise AssertionError("excluded module imported: " + fullname)
        return None

sys.meta_path.insert(0, RejectExcludedLayers())
sys.path.insert(0, sys.argv[1])
import ramdisk

assert callable(ramdisk.build_plan)
assert not (blocked & set(sys.modules)), sorted(blocked & set(sys.modules))
assert not any(name == "textual" or name.startswith("textual.") for name in sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_keeps_heavy_headless_layers_lazy(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "ramdisk_support.presentation",
    "ramdisk_support.processes",
    "ssl",
    "urllib.request",
}

class RejectLazyLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("eager facade import: " + fullname)
        return None

sys.meta_path.insert(0, RejectLazyLayers())
sys.path.insert(0, sys.argv[1])
import ramdisk

assert callable(ramdisk.build_plan)
assert not (blocked & set(sys.modules)), sorted(blocked & set(sys.modules))
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_star_import_contains_headless_lifecycle_and_no_runner_api(self):
        namespace = {}
        exec("from ramdisk import *", namespace)

        for name in PUBLIC_SIGNATURES:
            self.assertIn(name, namespace)
        for name in (
            "benchmark",
            "launch_tui",
            "BENCHMARK_SCHEMA",
            "BENCHMARK_PROMPT",
            "start",
            "stop",
        ):
            self.assertNotIn(name, namespace)

    def test_historical_urllib_patch_seam_remains_lazy_and_forwarded(self):
        process_module = mock.Mock()
        with mock.patch(
            "ramdisk.urllib.request.urlopen",
            new=mock.sentinel.urlopen,
        ), mock.patch.object(
            ramdisk,
            "_processes_module",
            return_value=process_module,
        ):
            ramdisk._wait_managed_ready(mock.sentinel.record, 3.0)

        process_module._wait_managed_ready.assert_called_once_with(
            mock.sentinel.record,
            3.0,
            api_key=None,
            cancel_event=None,
            process_matches=ramdisk._process_matches,
            urlopen=mock.sentinel.urlopen,
        )


if __name__ == "__main__":
    unittest.main()
