"""The diagnostic CLI locates current engine names before starting a campaign."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"


class DiagnosticEngineDiscovery(unittest.TestCase):
    def exercise(self, name, location="engine", explicit=False, package_directory=False,
                 both=False, reported_platform=None):
        spec = importlib.util.spec_from_file_location("diag_harness", TOOLS / "diag_harness.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine_dir = root / "repository" / "c"
            tools = engine_dir / "tools"
            tools.mkdir(parents=True)
            unrelated = root / "elsewhere"
            unrelated.mkdir()
            cwd = engine_dir if location == "engine" else unrelated
            if package_directory:
                cwd = engine_dir.parent
                (cwd / "colibri").mkdir()
            engine = (cwd if name == "glm" else engine_dir) / name
            engine.write_bytes(b"fixture engine path; inference is validated separately\n")
            engine.chmod(0o755)
            if both:
                alternate = engine_dir / ("colibri.exe" if name == "colibri" else "colibri")
                alternate.write_bytes(b"other platform engine path\n")
                alternate.chmod(0o755)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            arguments = ["diag_harness", "--snap", str(snapshot), "--phase", "system"]
            if explicit:
                arguments += ["--glm", str(engine)]
            original_cwd = Path.cwd()
            try:
                os.chdir(cwd)
                with patch.object(sys, "argv", arguments), \
                     patch.object(module, "__file__", str(tools / "diag_harness.py")), \
                     patch.object(module.sys, "platform", reported_platform or sys.platform), \
                     patch.object(module, "DiagnosticHarness") as harness, \
                     contextlib.redirect_stderr(io.StringIO()):
                    module.main()
                    self.assertEqual(Path(harness.call_args.args[0].glm).resolve(), engine.resolve())
                    harness.return_value.run.assert_called_once_with()
            finally:
                os.chdir(original_cwd)

    def test_current_unix_engine_in_working_directory(self):
        self.exercise("colibri")

    def test_current_unix_engine_found_relative_to_script(self):
        self.exercise("colibri", "unrelated")

    def test_current_windows_engine_found_relative_to_script(self):
        self.exercise("colibri.exe", "unrelated")

    def test_repository_python_package_is_not_selected_as_binary(self):
        self.exercise("colibri", package_directory=True)

    def test_native_unix_binary_preferred_when_both_names_exist(self):
        self.exercise("colibri", both=True, reported_platform="linux")

    def test_native_windows_binary_preferred_when_both_names_exist(self):
        self.exercise("colibri.exe", both=True, reported_platform="win32")

    def test_legacy_engine_remains_supported(self):
        self.exercise("glm")

    def test_explicit_engine_remains_supported(self):
        self.exercise("custom-engine", explicit=True)


if __name__ == "__main__":
    unittest.main()
