"""`coli stop` that works without rights on the process, and says so when it fails (#2091).

On Windows a serve started from an administrator console cannot be ended from a normal
prompt: TerminateProcess and taskkill are both denied. `coli stop` swallowed the error and
printed "stopped -- RAM released" with the engine still resident. Now:
  1. the serve watches for a stop request next to its pidfile and ends itself the way its
     signal handlers do (the engine drained), which needs no rights on it at all;
  2. a serve that does not take the request (one from before it existed) is ended as before,
     on Windows with the one right TerminateProcess needs, then taskkill;
  3. whatever still runs at the end is named, with the reason, and the exit code is 1.
Real processes on every platform the CI runs (Linux, macOS, Windows): the request travels
through the same coli functions the serve and `coli stop` use.
"""
import contextlib
import importlib.machinery
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import types
import unittest
from pathlib import Path
from unittest import mock

C_DIR = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader("coli_stop_request", str(C_DIR / "coli"))
spec = importlib.util.spec_from_loader(loader.name, loader)
coli = importlib.util.module_from_spec(spec)
loader.exec_module(coli)

# A serve in miniature: coli's own watcher on a stop file, and the handler openai_server.serve
# installs (SIGBREAK on Windows, SIGTERM elsewhere), which here records the drain and exits.
SERVE = textwrap.dedent("""
    import importlib.machinery, importlib.util, os, signal, sys, threading, time
    loader = importlib.machinery.SourceFileLoader("coli_child", sys.argv[1])
    spec = importlib.util.spec_from_loader(loader.name, loader)
    coli = importlib.util.module_from_spec(spec); loader.exec_module(coli)
    stopfile, marker = sys.argv[2], sys.argv[3]
    def drained(*_):
        with open(marker, "w") as f: f.write("drained")
        os._exit(0)
    signal.signal(signal.SIGBREAK if sys.platform == "win32" else signal.SIGTERM, drained)
    threading.Thread(target=coli._watch_stop_file, args=(stopfile, coli._stop_self, 0.05),
                     daemon=True).start()
    print("ready", flush=True)
    while True: time.sleep(0.05)
""")


class FakeClock:
    """time.time and time.sleep without the waiting."""
    def __init__(self):
        self.now = 1000.0

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class StopRequest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stopfile = os.path.join(self.tmp.name, "coli-serve-9123.stop")
        self.pidfile = os.path.join(self.tmp.name, "coli-serve-9123.pid")
        self.marker = os.path.join(self.tmp.name, "drained")
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=10)

    def serve(self):
        script = os.path.join(self.tmp.name, "serve.py")
        Path(script).write_text(SERVE, encoding="utf-8")
        child = subprocess.Popen([sys.executable, script, str(C_DIR / "coli"), self.stopfile, self.marker],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.children.append(child)
        self.assertEqual(child.stdout.readline().strip(), "ready", child.stderr.read() if child.poll() else "")
        return child

    def decoy(self):
        """A serve from before the request: it watches nothing."""
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        self.children.append(child)
        return child

    def stop(self, pid, **patches):
        Path(self.pidfile).write_text(f"{pid} /models/x\n")
        out = io.StringIO()
        with mock.patch.object(coli, "banner"), \
             mock.patch.object(coli, "serve_pidfile", return_value=self.pidfile), \
             mock.patch.object(coli, "serve_stopfile", return_value=self.stopfile), \
             mock.patch.object(coli, "_stop_proc_table", return_value=[]), \
             contextlib.ExitStack() as stack, contextlib.redirect_stdout(out):
            for name, value in patches.items():
                stack.enter_context(mock.patch.object(coli, name, value))
            try:
                coli.cmd_stop(types.SimpleNamespace(port=9123, dry_run=False))
                code = 0
            except SystemExit as exit_:
                code = exit_.code
        return code, out.getvalue()

    def test_the_request_ends_the_serve_through_its_own_handler(self):
        child = self.serve()
        with mock.patch.object(coli, "serve_stopfile", return_value=self.stopfile):
            self.assertTrue(coli._request_stop(9123, wait=10.0), "the serve did not take the request")
        self.assertEqual(child.wait(timeout=10), 0)
        self.assertEqual(Path(self.marker).read_text(), "drained", "ended without its drain")
        self.assertFalse(os.path.exists(self.stopfile), "the request was left behind")

    def test_coli_stop_ends_a_serve_that_watches(self):
        child = self.serve()
        code, said = self.stop(child.pid)
        self.assertEqual(code, 0, said)
        self.assertIn("stopped", said)
        self.assertEqual(child.wait(timeout=10), 0)
        self.assertEqual(Path(self.marker).read_text(), "drained")

    def test_coli_stop_still_ends_a_serve_that_does_not(self):
        child = self.decoy()
        code, said = self.stop(child.pid)
        self.assertEqual(code, 0, said)
        self.assertIsNotNone(child.wait(timeout=10))
        self.assertFalse(os.path.exists(self.stopfile), "an unanswered request stays for the next serve")

    def test_a_process_it_cannot_end_is_named_and_the_exit_code_says_so(self):
        child = self.decoy()
        clock = FakeClock()
        # SIGTERM and the forced end both refused, as for a serve with more rights than us
        with mock.patch.object(coli.time, "time", clock.time), mock.patch.object(coli.time, "sleep", clock.sleep), \
             mock.patch.object(coli.os, "kill", lambda *a: None):
            code, said = self.stop(child.pid, _terminate=lambda pid: "Access is denied. (WinError 5)",
                                   _request_stop=lambda port, wait=2.0: False)
        self.assertEqual(code, 1, said)
        self.assertIn(f"could not stop {child.pid}", said)
        self.assertIn("Access is denied. (WinError 5)", said)
        self.assertNotIn("✓ stopped", said)
        if sys.platform == "win32":
            self.assertIn("more rights than this prompt", said)

    def test_the_watcher_ends_with_the_serve(self):
        # cmd_serve's watcher, left running in a process that goes on (the test runner
        # once), ended that process at the next `coli stop` of the same port
        import threading
        done = threading.Event()
        stopped = []
        watcher = threading.Thread(target=coli._watch_stop_file,
                                   args=(self.stopfile, lambda: stopped.append(1), 0.05),
                                   kwargs={"done": done}, daemon=True)
        watcher.start()
        done.set()
        watcher.join(timeout=5)
        self.assertFalse(watcher.is_alive(), "the watcher outlived its serve")
        Path(self.stopfile).write_text("1\n")
        time.sleep(0.2)
        self.assertEqual(stopped, [], "a request after the serve ended still stopped this process")

    @unittest.skipUnless(hasattr(os, "getuid"), "the temp folder is per user on Windows")
    def test_another_user_s_file_is_not_a_request(self):
        Path(self.stopfile).write_text("1\n")
        self.assertTrue(coli._stop_file_is_ours(self.stopfile))
        with mock.patch.object(coli.os, "getuid", return_value=os.getuid() + 1):
            self.assertFalse(coli._stop_file_is_ours(self.stopfile))

    @unittest.skipUnless(sys.platform == "win32", "TerminateProcess")
    def test_terminate_with_the_one_right_it_needs(self):
        child = self.decoy()
        self.assertIsNone(coli._win_terminate(child.pid))
        self.assertIsNotNone(child.wait(timeout=10))
        self.assertIsNone(coli._win_terminate(child.pid), "an ended process is not an error")

    @unittest.skipIf(sys.platform == "win32", "no zombies: an ended process is gone")
    def test_a_zombie_is_not_alive(self):
        # Linux reads the state in /proc, macOS (no /proc) in ps: there both stop tests
        # above failed with "could not stop" on their own ended, not yet reaped, child
        child = subprocess.Popen([sys.executable, "-c", "pass"])
        self.children.append(child)
        for _ in range(500):
            if coli._proc_state(child.pid) == "Z":
                break
            time.sleep(0.01)
        self.assertEqual(coli._proc_state(child.pid), "Z", "the ended child's state could not be read")
        self.assertFalse(coli._pid_alive(child.pid), "an exited child not yet reaped counted as running")
        self.assertTrue(coli._pid_alive(os.getpid()), "this very process counted as gone")


if __name__ == "__main__":
    unittest.main()
