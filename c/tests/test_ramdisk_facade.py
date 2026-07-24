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
