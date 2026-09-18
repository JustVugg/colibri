"""Discovery stays with the Colibri copy containing the launcher."""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from colibri.launcher.domain import LauncherError
from colibri.launcher.installation import find_installation
from tests.launcher.test_adapter import make_release


PYTHON = Path(sys.executable)


class LocalInstallationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.local = make_release(self.base / "local")
        self.other = make_release(self.base / "other")
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.other)
        # Make a second real release discoverable through both old fallbacks.
        path_launcher = self.other / ("coli.exe" if os.name == "nt" else "coli")
        if path_launcher.name != "coli":
            shutil.copy2(self.other / "coli", path_launcher)
        path_launcher.chmod(0o755)
        environment = patch.dict(os.environ, {
            "COLIBRI_HOME": str(self.other),
            "PATH": str(self.other) + os.pathsep + os.environ.get("PATH", ""),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def test_frozen_launcher_uses_its_own_release_over_other_cwd_and_environment(self):
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "executable", str(self.local / "ColibriLauncher.exe")):
            installation = find_installation(python=PYTHON)

        self.assertEqual(installation.root, self.local.resolve())
        self.assertEqual(installation.version, "9.9.9")

    def test_frozen_launcher_subfolder_uses_its_immediate_colibri_parent(self):
        launcher_folder = self.local / "ColibriLauncher"
        launcher_folder.mkdir()
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "executable", str(launcher_folder / "ColibriLauncher.exe")):
            installation = find_installation(python=PYTHON)

        self.assertEqual(installation.root, self.local.resolve())

    def test_frozen_launcher_does_not_fall_back_to_a_different_colibri_copy(self):
        downloads = self.base / "downloads" / "ColibriLauncher"
        downloads.mkdir(parents=True)
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "executable", str(downloads / "ColibriLauncher.exe")):
            with self.assertRaisesRegex(LauncherError, "inside.*Colibri"):
                find_installation(python=PYTHON)

    def test_frozen_launcher_does_not_search_above_its_immediate_parent(self):
        nested = self.local / "tools" / "launcher"
        nested.mkdir(parents=True)
        with patch.object(sys, "frozen", True, create=True), \
                patch.object(sys, "executable", str(nested / "ColibriLauncher.exe")):
            with self.assertRaises(LauncherError):
                find_installation(python=PYTHON)

    def test_source_launcher_uses_the_working_colibri_folder_over_environment(self):
        os.chdir(self.local)
        with patch.object(sys, "frozen", False, create=True):
            installation = find_installation(python=PYTHON)

        self.assertEqual(installation.root, self.local.resolve())

    def test_source_launcher_does_not_fall_back_to_environment_or_path(self):
        empty = self.base / "empty"
        empty.mkdir()
        os.chdir(empty)
        with patch.object(sys, "frozen", False, create=True):
            with self.assertRaisesRegex(LauncherError, "Colibri folder"):
                find_installation(python=PYTHON)


if __name__ == "__main__":
    unittest.main()
