"""What `make install` puts on a machine has to be what the launcher needs.

Two invariants from the field report in #1689:

- the launcher derives its libexec directory from its own path, so it must
  resolve symlinks first: a merged-/usr system exposes /bin/coli as a link to
  /usr/bin/coli, and abspath() kept the alias, deriving /libexec/colibri;
- every root-level Python module the launcher reaches (tools/pack_python.py
  computes that closure for the release archive) must be in the handwritten
  `make install` list, or installed `coli serve` fails on an import the
  source tree satisfied (v41_dsml.py in 1.12.0).
"""
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "tools"))
import pack_python  # noqa: E402


def installed_root_modules(makefile):
    """The .py files the install recipe copies into $(LIBEXECDIR)/ itself."""
    text = makefile.read_text(encoding="utf-8")
    body = text[text.index("\ninstall:"):]
    body = body[:body.index("\nuninstall:")]
    body = body.replace("\\\n", " ")
    names = set()
    for line in body.splitlines():
        if "$(INSTALL) -m 644" in line and line.rstrip().endswith("$(DESTDIR)$(LIBEXECDIR)/"):
            names |= set(re.findall(r"\b([A-Za-z0-9_]+\.py)\b", line))
    return names


class InstallClosure(unittest.TestCase):
    def test_every_reached_root_module_is_installed(self):
        reached = {p.name for p in pack_python.needed(HERE)
                   if p.parent == HERE and p.suffix == ".py"}
        installed = installed_root_modules(HERE / "Makefile")
        self.assertTrue(reached, "pack_python reached no root module; the parser is broken")
        self.assertEqual(reached - installed, set(),
                         "reached by the launcher but missing from `make install`")

    @unittest.skipIf(sys.platform == "win32", "symlinks need privileges on Windows")
    def test_launcher_resolves_a_merged_usr_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "usr" / "bin").mkdir(parents=True)
            (root / "usr" / "libexec").mkdir()
            launcher = root / "usr" / "bin" / "coli"
            launcher.write_bytes((HERE / "coli").read_bytes())
            launcher.chmod(0o755)
            # the support modules of this checkout stand in for the installed ones
            os.symlink(HERE, root / "usr" / "libexec" / "colibri")
            os.symlink(root / "usr" / "bin", root / "bin")
            env = {k: v for k, v in os.environ.items() if k not in ("COLI_ENGINE", "PYTHONPATH")}
            for alias in (root / "bin" / "coli", root / "usr" / "bin" / "coli"):
                with self.subTest(alias=str(alias.relative_to(root))):
                    result = subprocess.run([sys.executable, str(alias), "--version"],
                                            capture_output=True, text=True, env=env,
                                            cwd=tmp, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("colibri", result.stdout)
                    self.assertNotIn("ModuleNotFoundError", result.stderr)


if __name__ == "__main__":
    unittest.main()
