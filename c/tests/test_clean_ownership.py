"""Owner for what `make clean` is responsible for removing.

There was no test here, and the cost of that showed up in release engineering:
tools/clean.py carried a hand-written list of engine binaries that had never
gained qwen36. `make clean` left the old qwen36 binary in place, and the release
Package step copies whatever sits at c/qwen36 without asking where it came from,
so a locally staged release could ship an engine that was never rebuilt. CI did
not see it because CI starts from an empty runner.

The fix was to derive the engine names from family_registry instead of listing
them, and the point of this file is to keep that derivation honest: the first
test compares the two sets mechanically, so adding a family to the registry
cannot silently create a new uncleanable build product.

clean.py is executed in a sandbox rather than in the checkout -- running the real
thing against the real tree during `make check` would delete the binaries the
rest of the suite is using.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
C_DIR = os.path.dirname(HERE)
CLEAN = os.path.join(C_DIR, "tools", "clean.py")

sys.path.insert(0, C_DIR)
from family_registry import all_families  # noqa: E402


def packaged_engines():
    """What a release archive must contain -- the release job's own source.

    .github/workflows/release.yml verifies the packaged archive against exactly
    this registry, so it is the authority on which binaries get published.
    """
    return sorted({f.engine_artifact for f in all_families()})


class CleanOwnershipTest(unittest.TestCase):
    def _sandbox(self, files=(), dirs=()):
        """A throwaway c/ with clean.py, its import, and some build products."""
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        os.makedirs(os.path.join(tmp, "tools"))
        os.makedirs(os.path.join(tmp, "tests"))
        shutil.copy(CLEAN, os.path.join(tmp, "tools", "clean.py"))
        shutil.copy(os.path.join(C_DIR, "family_registry.py"), tmp)
        for rel in files:
            path = os.path.join(tmp, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(b"build product")
        for rel in dirs:
            os.makedirs(os.path.join(tmp, rel), exist_ok=True)
        return tmp

    def _run_clean(self, tmp):
        return subprocess.run([sys.executable, os.path.join("tools", "clean.py")],
                              cwd=tmp, capture_output=True, text=True, timeout=120)

    # ---- the invariant -----------------------------------------------------

    def test_every_packaged_engine_is_owned_by_clean(self):
        """PACKAGED_BUT_NOT_CLEANED must be 0, by construction not by vigilance."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("coli_clean", CLEAN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        owned = set(mod.FILES)
        missing = [e for e in packaged_engines()
                   if e not in owned or e + ".exe" not in owned]
        self.assertEqual(missing, [], "release-packaged but not cleaned: %s" % missing)

    def test_qwen36_specifically(self):
        """The one that was actually missing. Pinned by name so it stays fixed."""
        self.assertIn("qwen36", packaged_engines())
        import importlib.util
        spec = importlib.util.spec_from_file_location("coli_clean", CLEAN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIn("qwen36", mod.FILES)
        self.assertIn("qwen36.exe", mod.FILES)

    def test_importing_the_module_is_inert(self):
        """Reading FILES must not delete anything.

        This is not hypothetical: the first version of the test above imported
        clean.py with exec_module while `make check` had cwd=c/, and the
        module-level delete loop ran against the real tree, removing the
        binaries the rest of the suite was about to use. The action now lives
        behind __main__.
        """
        import importlib.util
        tmp = self._sandbox(files=["colibri.exe", "qwen36.exe"])
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            spec = importlib.util.spec_from_file_location("coli_clean_inert", CLEAN)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            os.chdir(cwd)
        for rel in ("colibri.exe", "qwen36.exe"):
            self.assertTrue(os.path.exists(os.path.join(tmp, rel)),
                            "importing clean.py deleted %s" % rel)

    # ---- behaviour ---------------------------------------------------------

    def test_clean_removes_engine_binaries(self):
        names = [e + ext for e in packaged_engines() for ext in ("", ".exe")]
        tmp = self._sandbox(files=names)
        self._run_clean(tmp)
        survivors = [n for n in names if os.path.exists(os.path.join(tmp, n))]
        self.assertEqual(survivors, [], "survived clean: %s" % survivors)

    def test_clean_removes_the_physical_probe(self):
        """A stale probe reports PASS for code that is no longer in the tree."""
        tmp = self._sandbox(files=["tests/xdna_physical_probe",
                                   "tests/xdna_physical_probe.exe"])
        self._run_clean(tmp)
        for rel in ("tests/xdna_physical_probe", "tests/xdna_physical_probe.exe"):
            self.assertFalse(os.path.exists(os.path.join(tmp, rel)), rel)

    def test_clean_does_not_delete_sources_or_unrelated_files(self):
        keep = ["tests/xdna_physical_probe.c", "tests/test_something.c",
                "tests/test_fixture.json", "tests/notes.md", "colibri.c",
                "my_notes.txt", "tests/data.bin", "tools/clean.py"]
        tmp = self._sandbox(files=[k for k in keep if k != "tools/clean.py"],
                            dirs=["tests/fixtures"])
        self._run_clean(tmp)
        for rel in keep:
            self.assertTrue(os.path.exists(os.path.join(tmp, rel)),
                            "clean deleted %s" % rel)
        self.assertTrue(os.path.isdir(os.path.join(tmp, "tests", "fixtures")))

    def test_clean_is_idempotent(self):
        tmp = self._sandbox(files=["colibri.exe", "qwen36.exe",
                                   "tests/test_x.exe"])
        first = self._run_clean(tmp)
        self.assertEqual(first.returncode, 0, first.stderr)
        before = sorted(os.listdir(tmp))
        second = self._run_clean(tmp)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(sorted(os.listdir(tmp)), before)
        self.assertIn("removed 0", second.stdout)


if __name__ == "__main__":
    unittest.main()
