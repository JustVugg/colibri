import argparse
import contextlib
import inspect
import io
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
    "start": "(args, cli_path=None, engine_path=None, cancel_event=None)",
    "stop": "(args=None)",
    "destroy": "(args, expected_manifest_token=None)",
    "status": "(deep=True)",
    "benchmark": "(args, cli_path=None, engine_path=None, cancel_event=None)",
    "dispatch": "(args, cli_path=None, engine_path=None, system=None)",
    "launch_tui": "(args, cli_path=None, engine_path=None, system=None)",
}

SCHEMA_CONSTANTS = {
    "MANIFEST_VERSION": 1,
    "PLAN_SCHEMA": "colibri.ramdisk.plan.v1",
    "STATUS_SCHEMA": "colibri.ramdisk.status.v1",
    "BENCHMARK_SCHEMA": "colibri.ramdisk.benchmark.v1",
}


class RamdiskFacadeContractTest(unittest.TestCase):
    def test_public_exports_and_signatures_remain_stable(self):
        self.assertTrue(issubclass(ramdisk.RamdiskError, RuntimeError))
        for name, expected_signature in PUBLIC_SIGNATURES.items():
            with self.subTest(name=name):
                exported = getattr(ramdisk, name)
                self.assertTrue(callable(exported))
                self.assertEqual(str(inspect.signature(exported)), expected_signature)

    def test_schema_names_and_versions_remain_stable(self):
        for name, expected in SCHEMA_CONSTANTS.items():
            with self.subTest(name=name):
                self.assertEqual(getattr(ramdisk, name), expected)

    def test_import_does_not_eagerly_load_optional_textual_frontend(self):
        script = """
import importlib.abc
import sys

class RejectTextual(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ramdisk_textual" or fullname == "textual" or fullname.startswith("textual."):
            raise AssertionError("optional Textual frontend loaded eagerly: " + fullname)
        return None

sys.meta_path.insert(0, RejectTextual())
sys.path.insert(0, sys.argv[1])
import ramdisk

loaded = sorted(
    name
    for name in sys.modules
    if name == "ramdisk_textual" or name == "textual" or name.startswith("textual.")
)
assert loaded == [], loaded
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_import_keeps_optional_facade_layers_lazy(self):
        script = r"""
import importlib.abc
import sys

blocked = {
    "ramdisk_textual",
    "ramdisk_ui",
    "ramdisk_support.benchmark",
    "ramdisk_support.curses_ui",
    "ramdisk_support.presentation",
    "ramdisk_support.processes",
    "ramdisk_support.runtime_monitor",
}

class RejectOptionalLayers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("eager optional facade import: " + fullname)
        return None

sys.meta_path.insert(0, RejectOptionalLayers())
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

    def test_import_does_not_initialize_the_stdlib_http_stack(self):
        script = r"""
import importlib.abc
import sys

blocked = {"ssl", "urllib.request"}

class RejectHttpStack(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked:
            raise AssertionError("eager network import: " + fullname)
        return None

sys.meta_path.insert(0, RejectHttpStack())
sys.path.insert(0, sys.argv[1])
import ramdisk

assert callable(ramdisk.status)
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

    def test_lazy_public_exports_remain_discoverable_and_star_importable(self):
        script = r"""
import sys

sys.path.insert(0, sys.argv[1])
import ramdisk

assert "BENCHMARK_PROMPT" in dir(ramdisk)
assert "urllib" in dir(ramdisk)
assert "ramdisk_support.benchmark" not in sys.modules
assert "urllib.request" not in sys.modules

namespace = {}
exec("from ramdisk import *", namespace)
assert isinstance(namespace["BENCHMARK_PROMPT"], str)
assert namespace["BENCHMARK_PROMPT"]
assert namespace["urllib"] is sys.modules["urllib"]
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(C_DIR)],
            cwd=C_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

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

    def test_non_linux_tui_fails_before_loading_ui_or_probing_hardware(self):
        stderr = io.StringIO()
        with (
            mock.patch.object(ramdisk.sys, "platform", "darwin"),
            mock.patch.object(
                ramdisk,
                "_load_textual_frontend",
                side_effect=AssertionError("Textual frontend should not load"),
            ) as load_textual,
            mock.patch.object(
                ramdisk,
                "discover_hardware",
                side_effect=AssertionError("Linux hardware should not be probed"),
            ) as discover_hardware,
            contextlib.redirect_stderr(stderr),
        ):
            result = ramdisk.launch_tui(argparse.Namespace())

        self.assertEqual(result, 2)
        self.assertIn("the TUI is supported only on Linux", stderr.getvalue())
        load_textual.assert_not_called()
        discover_hardware.assert_not_called()


if __name__ == "__main__":
    unittest.main()
