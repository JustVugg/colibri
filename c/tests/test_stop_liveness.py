"""#1049: `coli stop` must actually find a live serve, on every platform.

The bug these cover: `os.kill(pid, 0)` is the POSIX liveness idiom, but on
Windows os.kill() has no signal semantics and raises OSError [WinError 87] for
a pid the caller did not spawn. cmd_stop swallowed that as "not running", so
`coli stop` reported "nothing running" while the port was still LISTENING and
~5 GB of engines were resident.
"""
import importlib.machinery, importlib.util, os, subprocess, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COLI = os.path.join(os.path.dirname(HERE), "coli")


def load_coli():
    loader = importlib.machinery.SourceFileLoader("coli_mod", COLI)
    spec = importlib.util.spec_from_loader("coli_mod", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


class StopLivenessTest(unittest.TestCase):
    def setUp(self):
        self.coli = load_coli()

    def test_probe_exists(self):
        """cmd_stop must not use the raw POSIX idiom for liveness."""
        self.assertTrue(hasattr(self.coli, "_pid_alive"))
        with open(COLI, encoding="utf-8") as f: src = f.read()
        stop = src[src.index("def cmd_stop("):]
        stop = stop[:stop.index("\ndef ")]
        self.assertNotIn("os.kill(pid,0)", stop)
        self.assertIn("_pid_alive(pid)", stop)

    def test_self_is_alive(self):
        self.assertTrue(self.coli._pid_alive(os.getpid()))

    def test_reaped_child_is_not_alive(self):
        """A process we started and waited on must read as dead."""
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()
        self.assertFalse(self.coli._pid_alive(p.pid))

    def test_sigkill_lookup_is_guarded(self):
        """signal.SIGKILL is absent on win32 and AttributeError is not OSError."""
        with open(COLI, encoding="utf-8") as f: src = f.read()
        stop = src[src.index("def cmd_stop("):]
        stop = stop[:stop.index("\ndef ")]
        # the *call* must not name it directly; the explanatory comment may.
        self.assertNotIn("os.kill(pid, signal.SIGKILL)", stop)
        self.assertIn('getattr(signal, "SIGKILL"', stop)


class OrphanJobTest(unittest.TestCase):
    """The engine re-execs for OMP tuning and orphans itself; a Job Object with
    KILL_ON_JOB_CLOSE is what ties it to the server's lifetime on Windows."""

    def test_engine_takes_a_job_handle(self):
        with open(os.path.join(os.path.dirname(HERE), "openai_server.py"),
                  encoding="utf-8") as f:
            src = f.read()
        self.assertIn("_win_kill_on_close_job", src)
        self.assertIn("JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE", src)
        self.assertIn("_win_kill_on_close_job(getattr(self.process,", src)
        # the non-Windows path must not even read .pid: the suite drives Engine
        # with a fake process object that has none.
        self.assertIn('if sys.platform == "win32":', src)

    @staticmethod
    def _server():
        sys.path.insert(0, os.path.dirname(HERE))
        try:
            import openai_server
            return openai_server
        finally:
            sys.path.pop(0)

    @unittest.skipIf(sys.platform == "win32",
                     "on win32 it returns a real job handle; see the test below")
    def test_helper_is_a_noop_off_windows(self):
        self.assertIsNone(self._server()._win_kill_on_close_job(os.getpid()))

    @unittest.skipUnless(sys.platform == "win32", "Job Objects are a Windows mechanism")
    def test_job_kills_the_child_when_the_handle_closes(self):
        """The whole point: closing the job must take the process with it.

        Never pass os.getpid() here. Assigning the test runner itself to a
        KILL_ON_JOB_CLOSE job means the first CloseHandle -- ours, or a GC of a
        stray reference -- terminates the runner. An earlier version of this
        file did exactly that and CI only survived because a raw HANDLE is a
        plain int that Python never closes for you.
        """
        import ctypes
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        job = None
        try:
            job = self._server()._win_kill_on_close_job(child.pid)
            self.assertIsNotNone(job, "job creation failed on a live child")
            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
            job = None
            child.wait(timeout=15)          # raises TimeoutExpired if it survived
            self.assertIsNotNone(child.returncode)
        finally:
            if job:
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(job)
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
