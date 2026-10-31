"""Mux benchmark failures must release their real engine subprocess."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"


class BenchmarkShutdownTest(unittest.TestCase):
    def run_failure(self, tool, stalled):
        spec = importlib.util.spec_from_file_location(tool, TOOLS / (tool + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary) / "engine.py"
            engine.write_text(
                "#!" + sys.executable + "\n"
                "import signal,sys,time\n" +
                ("signal.signal(signal.SIGTERM,signal.SIG_IGN)\n" if stalled == "ignore_term" else "") +
                "print('READY',flush=True)\n"
                "sys.stdin.buffer.readline()\n"
                "print('ERROR fixture failure',flush=True)\n" +
                ("time.sleep(60)\n" if stalled else ""), encoding="utf-8")
            engine.chmod(0o755)
            processes = []
            real_spawn = subprocess.Popen

            def spawn(*args, **kwargs):
                command = [sys.executable, *args[0]]
                process = real_spawn(command, *args[1:], **kwargs)
                processes.append(process)
                real_wait = process.wait
                # Exercise real child lifetime with a short test grace period,
                # rather than spending the production 30 seconds per fixture.
                process.wait = lambda timeout=None: real_wait(
                    min(timeout, .1) if timeout is not None else None)
                return process

            arguments = [tool, "--engine", str(engine), "--model", temporary]
            if tool == "benchmark_kv_prefix":
                arguments += ["--share", "0"]
            try:
                with patch.object(sys, "argv", arguments), \
                     patch.object(module.subprocess, "Popen", side_effect=spawn), \
                     contextlib.redirect_stdout(io.StringIO()), \
                     contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, "ERROR fixture failure"):
                        module.main()
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll(), "benchmark leaked its engine")
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream:
                            stream.close()

    def test_continuous_preserves_error_and_reaps_stalled_engine(self):
        self.run_failure("benchmark_continuous_batch", True)

    @unittest.skipUnless(os.name == "posix", "Windows termination is already forced")
    def test_continuous_kills_engine_that_ignores_termination(self):
        self.run_failure("benchmark_continuous_batch", "ignore_term")

    def test_continuous_preserves_cooperative_shutdown(self):
        self.run_failure("benchmark_continuous_batch", False)

    def test_kv_prefix_preserves_error_and_reaps_stalled_engine(self):
        self.run_failure("benchmark_kv_prefix", True)

    def test_interference_preserves_error_and_reaps_stalled_engine(self):
        self.run_failure("benchmark_serving_interference", True)

    @unittest.skipUnless(os.name == "posix", "Windows termination is already forced")
    def test_kv_prefix_kills_engine_that_ignores_termination(self):
        self.run_failure("benchmark_kv_prefix", "ignore_term")

    @unittest.skipUnless(os.name == "posix", "Windows termination is already forced")
    def test_interference_kills_engine_that_ignores_termination(self):
        self.run_failure("benchmark_serving_interference", "ignore_term")

    def test_kv_prefix_preserves_cooperative_shutdown(self):
        self.run_failure("benchmark_kv_prefix", False)

    def test_interference_preserves_cooperative_shutdown(self):
        self.run_failure("benchmark_serving_interference", False)


if __name__ == "__main__":
    unittest.main()
