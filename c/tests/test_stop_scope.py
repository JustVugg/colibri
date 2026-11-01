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

    def test_web_is_a_server(self):
        # `coli web` is `coli serve` plus a browser, and what `coli start` runs.
        raw = self.cmdline("python3", "/opt/colibri/coli", "web", "--model", "/models/a",
                           "--host", "127.0.0.1", "--port", "9000", "--no-browser")
        self.assertTrue(coli._serve_cmdline_matches_port(raw, 9000))
        self.assertFalse(coli._serve_cmdline_matches_port(raw, 8000))

    def test_stop_without_pidfile_drains_a_web_launcher(self):
        # Without the launcher in the table, its engine was killed as a stray and
        # the launcher's watchdog started the engine again behind "stopped".
        table = [(4242, self.cmdline("python3", "/opt/colibri/coli", "web",
                                     "--port", "9123"), "python3")]
        args = types.SimpleNamespace(port=9123, dry_run=True)
        out = io.StringIO()
        with mock.patch.object(coli, "banner"), \
             mock.patch.object(coli, "serve_pidfile", return_value="/not/a/pidfile"), \
             mock.patch.object(coli, "_stop_proc_table", return_value=table), \
             contextlib.redirect_stdout(out):
            coli.cmd_stop(args)
        self.assertIn("would stop 4242: coli serve (cmdline, port 9123)", out.getvalue())

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

    def test_serve_that_cannot_bind_keeps_the_running_serves_pidfile(self):
        # A second `coli serve` on a busy port fails to bind; the pidfile must
        # still name the serve that holds the port, for coli stop and coli status.
        running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                pidfile = Path(tmp, "coli-serve-9123.pid")
                pidfile.write_text(f"{running.pid} /models/a\n")
                family = types.SimpleNamespace(id="glm", has_gateway_adapter=True,
                                               limits=types.SimpleNamespace(max_kv_slots=8))
                args = types.SimpleNamespace(model="/models/a", host="127.0.0.1", port=9123,
                                             api_key=None, cap=None, cors_origin=None,
                                             max_queue=8, queue_timeout=300, kv_slots=1,
                                             allowed_host=[], temp=None,
                                             cluster_coordinator=None)
                with mock.patch.object(coli, "need_model"), \
                     mock.patch.object(coli, "resolve_model",
                                       return_value=types.SimpleNamespace(descriptor=family)), \
                     mock.patch.object(coli, "engine_for", return_value="/engines/glm"), \
                     mock.patch.object(coli, "serve_model_id", return_value="glm"), \
                     mock.patch.object(coli, "_serve_engine_env", return_value={}), \
                     mock.patch.object(coli, "serve_pidfile", return_value=str(pidfile)), \
                     mock.patch.object(coli, "serve_stopfile",
                                       return_value=os.path.join(tmp, "coli-serve-9123.stop")), \
                     mock.patch("openai_server.serve",
                                side_effect=OSError(48, "Address already in use")):
                    with self.assertRaises(OSError):
                        coli.cmd_serve(args)
                self.assertEqual(pidfile.read_text(), f"{running.pid} /models/a\n")
        finally:
            running.terminate()
            running.wait(timeout=5)

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
                     mock.patch.object(coli, "serve_stopfile",
                                       return_value=os.path.join(tmp, "coli-serve-9123.stop")), \
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
