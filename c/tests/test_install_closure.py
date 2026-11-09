"""What `make install` puts on a machine has to be what the launcher needs.

Two invariants from the field report in #1689:

- the launcher derives its libexec directory from its own path, so it must
  resolve symlinks first: a merged-/usr system exposes /bin/coli as a link to
  /usr/bin/coli, and abspath() kept the alias, deriving /libexec/colibri;
- every root-level Python module the launcher reaches (tools/pack_python.py
  computes that closure for the release archive) must be in the handwritten
  `make install` list, or installed `coli serve` fails on an import the
  source tree satisfied (v41_dsml.py in 1.12.0).

A staged install must also retain the scripts' runtime data and serve the
chat UI after its source tree has been removed.
"""
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
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


@unittest.skipIf(sys.platform == "win32", "make install uses POSIX install")
@unittest.skipUnless(shutil.which("make") and shutil.which("install"), "make and install required")
class InstalledRuntime(unittest.TestCase):
    def test_staged_install_keeps_runtime_assets_outside_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "c"
            source.mkdir(parents=True)
            shutil.copy2(HERE / "Makefile", source / "Makefile")
            shutil.copy2(HERE / "coli", source / "coli")
            for module in HERE.glob("*.py"):
                shutil.copy2(module, source / module.name)
            (source / "tools").mkdir()
            for module in (HERE / "tools").glob("*.py"):
                shutil.copy2(module, source / "tools" / module.name)
            data = [p for p in pack_python.needed(HERE) if p.suffix != ".py"]
            for asset in data:
                shutil.copy2(asset, source / asset.relative_to(HERE))
            shutil.copytree(HERE.parent / "ui", source.parent / "ui",
                            ignore=shutil.ignore_patterns("__pycache__"))
            # This exercises the actual install recipe, not the engine build:
            # no model or compiler is needed to serve the chat's static page.
            engines = ("colibri-tui", "colibri", "glm53", "inkling", "kimi_k3",
                       "olmoe", "qwen36", "qwen38", "deepseek_v41", "mimo",
                       "qwenimage", "laya", "gliner_decide")
            for name in engines:
                (source / name).write_text("unused engine fixture\n", encoding="utf-8")
            prefix = root / "installed"
            result = subprocess.run(
                [shutil.which("make"), "--no-print-directory", "install",
                 "COLI_V4_SUPPORTED=0", "EXE=", f"PREFIX={prefix}",
                 *[f"--old-file={name}" for name in engines]],
                cwd=source, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # Removing the fixture source ensures neither the launcher's path
            # search nor the UI's imports can fall back to checkout files.
            shutil.rmtree(source.parent)
            libexec = prefix / "libexec" / "colibri"
            for asset in data:
                installed = libexec / asset.relative_to(HERE)
                self.assertTrue(installed.is_file(), f"missing installed runtime data: {installed.name}")
                self.assertEqual(installed.read_bytes(), asset.read_bytes())
            env = {k: v for k, v in os.environ.items()
                   if k not in ("PYTHONPATH", "COLI_ENGINE", "COLI_MODEL")}
            env["PYTHONUNBUFFERED"] = "1"
            for module in ("server.py", "models.py"):
                result = subprocess.run(
                    [sys.executable, str(libexec / "ui" / module), "--help"],
                    cwd=root, env=env, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            # The Models page also invokes coli for GGUF import and serving.
            result = subprocess.run(
                [sys.executable, "-c",
                 "import runpy, subprocess, sys; m = runpy.run_path(sys.argv[1]); "
                 "subprocess.run([sys.executable, m['COLI'], '--version'], check=True)",
                 str(libexec / "ui" / "models.py")],
                cwd=root, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("colibri", result.stdout)
            # Pick a free loopback port for the installed launcher's real server.
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, str(prefix / "bin" / "coli"), "ui", "--no-browser",
                     "--port", str(port), "--backend", "http://192.0.2.1:8000"],
                    cwd=root, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    # Allow for slow UI startup on shared CI runners.
                    # A loopback check must also ignore any inherited proxy.
                    deadline = time.monotonic() + 60
                    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                    page = None
                    last_error = None
                    while time.monotonic() < deadline and process.poll() is None:
                        try:
                            with opener.open(f"http://127.0.0.1:{port}/", timeout=1) as response:
                                page = response.read()
                            break
                        except urllib.error.URLError as error:
                            last_error = str(error)
                            time.sleep(0.05)
                    log.seek(0)
                    self.assertIsNotNone(page, f"UI returncode={process.poll()}, "
                                         f"last HTTP error={last_error}\n{log.read()}")
                    self.assertEqual(page, (HERE.parent / "ui" / "index.html").read_bytes())
                finally:
                    # coli launches a Python UI child; stop the whole test
                    # session so terminating the launcher cannot orphan it.
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
