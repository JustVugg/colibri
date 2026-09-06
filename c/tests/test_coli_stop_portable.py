"""`coli stop` must work on platforms without /proc.

`cmd_stop` scanned `/proc` unguarded. On macOS and Windows that directory does not exist, so
`os.listdir("/proc")` raised FileNotFoundError *after* the pidfile target had been collected but
*before* anything was stopped -- and it raised for `--dry-run` too, so there was no safe way to
even inspect what would be killed.

That mattered more than a crash usually does. The docstring on `cmd_stop` explains that the engine
re-execs itself for OMP tuning and therefore does NOT carry the name you started it with, which is
why `pkill -x glm` silently killed nothing and let two ghost engines OOM a box. `coli stop` is the
command that exists to prevent exactly that, and on macOS it could not run at all.

So these tests assert both halves:
  1. the command runs and exits cleanly on this platform;
  2. it still FINDS a `coli serve`-shaped process, i.e. discovery was ported, not merely guarded.
"""
import os
import subprocess
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COLI = os.path.join(os.path.dirname(HERE), "coli")


def run_stop(*extra):
    return subprocess.run(
        [sys.executable, COLI, "stop", "--dry-run", *extra],
        capture_output=True, text=True, timeout=60,
    )


class ColiStopPortable(unittest.TestCase):
    def test_dry_run_does_not_crash(self):
        """Exits 0 with no traceback, on any platform."""
        r = run_stop("--port", "8000")
        self.assertNotIn("Traceback", r.stderr, f"cmd_stop raised:\n{r.stderr}")
        self.assertNotIn("FileNotFoundError", r.stderr)
        self.assertEqual(r.returncode, 0, f"rc={r.returncode}\nstdout={r.stdout}\nstderr={r.stderr}")

    def test_dry_run_kills_nothing(self):
        """--dry-run must be inspection-only: a decoy it identifies is still alive afterwards."""
        decoy = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", "coli", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.0)
            run_stop("--port", "8000")
            self.assertIsNone(decoy.poll(), "--dry-run terminated a process")
        finally:
            decoy.kill()
            decoy.wait(timeout=10)

    def test_discovers_a_coli_serve_process(self):
        """Discovery must be PORTED, not just guarded.

        The decoy's command line contains both 'coli' and ' serve', which is the same signature
        cmd_stop looks for on Linux via /proc/<pid>/cmdline. If the /proc scan were merely wrapped
        in try/except, this test would fail: the command would exit 0 and find nothing, which is
        the silent-no-op failure the original docstring warns about.
        """
        decoy = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", "coli", "serve"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.0)
            r = run_stop("--port", "8000")
            self.assertIn(str(decoy.pid), r.stdout,
                          f"cmd_stop did not discover pid {decoy.pid}\nstdout={r.stdout}")
        finally:
            decoy.kill()
            decoy.wait(timeout=10)


    def test_does_not_match_the_word_server(self):
        """`" serve" in cmd` also matches "server", so an unrelated process is a kill target.

        This is not hypothetical. Running `coli stop --dry-run` from a shell whose command line
        contained the phrase "LIVE server" listed that shell as something to stop -- and the
        non-dry-run path SIGTERMs then SIGKILLs every target. A command whose stated purpose is
        to avoid the collateral damage of `pkill` must not itself kill bystanders.

        The decoy below carries 'coli' and 'server' but never the word 'serve'.
        """
        decoy = subprocess.Popen(
            # A SPACE before "server" is what makes `" serve" in cmd` true. A hyphen does not,
            # so the decoy must use separate argv entries or it fails to reproduce the bug at all.
            [sys.executable, "-c", "import time; time.sleep(30)", "coli", "LIVE", "server"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(1.0)
            r = run_stop("--port", "8000")
            self.assertNotIn(str(decoy.pid), r.stdout,
                             "cmd_stop targeted a process matching 'server', not 'serve'\n"
                             f"stdout={r.stdout}")
        finally:
            decoy.kill()
            decoy.wait(timeout=10)


if __name__ == "__main__":
    unittest.main()
