import contextlib
import importlib.machinery
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("coli_stop_scope", str(C_DIR / "coli"))
spec = importlib.util.spec_from_loader(loader.name, loader)
coli = importlib.util.module_from_spec(spec)
loader.exec_module(coli)


class StopScopeTests(unittest.TestCase):
    def cmdline(self, *args):
        return b"\0".join(arg.encode() for arg in args) + b"\0"

    def test_explicit_port_does_not_match_another_server(self):
        raw = self.cmdline("python3", "/opt/colibri/coli", "serve",
                           "--model", "/models/a", "--port", "9000")
        self.assertTrue(coli._serve_cmdline_matches_port(raw, 9000))
        self.assertFalse(coli._serve_cmdline_matches_port(raw, 8000))

    def test_equals_port_syntax(self):
        raw = self.cmdline("/opt/colibri/coli", "serve", "--port=9000")
        self.assertTrue(coli._serve_cmdline_matches_port(raw, 9000))

    def test_serve_as_an_argument_is_not_a_server(self):
        raw = self.cmdline("python3", "/opt/colibri/coli", "chat",
                           "--prompt", "serve", "--port", "9000")
        self.assertFalse(coli._serve_cmdline_matches_port(raw, 9000))

    def test_default_port_matches_only_8000(self):
        raw = self.cmdline("python3", "/opt/colibri/coli", "serve",
                           "--model", "/models/a")
        self.assertTrue(coli._serve_cmdline_matches_port(raw, 8000))
        self.assertFalse(coli._serve_cmdline_matches_port(raw, 9000))

    def test_engine_tag_is_port_specific(self):
        raw = b"SNAP=/models/a\0SERVE=1\0COLI_SERVE_PORT=9000\0"
        self.assertTrue(coli._serve_environ_matches_port(raw, 9000))
        self.assertFalse(coli._serve_environ_matches_port(raw, 8000))

    def test_serve_tags_the_engine_environment(self):
        args = types.SimpleNamespace(port=9123)
        with mock.patch.object(coli, "env_for_engine", return_value={"X": "1"}):
            env = coli._serve_engine_env(args, "glm")
        self.assertEqual(env, {"X": "1", "COLI_SERVE_PORT": "9123"})

    def test_stop_handles_platform_without_proc(self):
        args = types.SimpleNamespace(port=9123, dry_run=True)
        with mock.patch.object(coli, "banner"), \
             mock.patch.object(coli, "serve_pidfile", return_value="/not/a/pidfile"), \
             mock.patch.object(coli.os, "listdir", side_effect=OSError):
            coli.cmd_stop(args)

    @unittest.skipUnless(sys.platform.startswith("linux") and Path("/proc").is_dir(),
                         "process discovery uses Linux /proc")
    def test_stop_kills_only_the_requested_server(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "coli"
            script.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
            own = subprocess.Popen([sys.executable, str(script), "serve",
                                    "--port", "9123"])
            other = subprocess.Popen([sys.executable, str(script), "serve",
                                      "--port", "9124"])
            try:
                for _ in range(50):
                    try:
                        if b"serve" in Path(f"/proc/{own.pid}/cmdline").read_bytes():
                            break
                    except OSError:
                        pass
                    time.sleep(0.02)
                args = types.SimpleNamespace(port=9123, dry_run=False)
                with mock.patch.object(coli, "banner"), \
                     mock.patch.object(coli, "serve_pidfile",
                                       return_value=os.path.join(tmp, "missing.pid")), \
                     mock.patch.object(coli.time, "sleep"), \
                     contextlib.redirect_stdout(io.StringIO()):
                    coli.cmd_stop(args)
                own.wait(timeout=5)
                self.assertIsNone(other.poll(), "stop killed a server on another port")
            finally:
                for process in (own, other):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
