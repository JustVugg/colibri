"""Engine.close drain contract: stdin EOF is the engine's graceful exit.

``Engine.close`` must give the engine its one portable graceful path — the
serve loop reads requests from stdin, so closing that pipe lets the process
run its ``atexit`` teardown (``qt_shutdown`` writes HEAT_FILE) and exit 0.
Only a process that outlives the drain window falls through to the existing
terminate/kill ladder.

The tests drive ``close()`` against real subprocesses of ``python -c``:
one that blocks on ``stdin.read()`` (the well-behaved engine) and one that
ignores EOF and sleeps (the hung engine). No model, GPU, or engine build is
involved — the contract under test is the *shutdown handshake*.
"""

import os
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openai_server  # noqa: E402

EOF_ENGINE = "import sys\nsys.stdin.read()\n"   # blocks until EOF, exits 0
HANG_ENGINE = "import time\ntime.sleep(120)\n"  # ignores stdin entirely


def bare_engine(code):
    """An Engine with only what close() touches, around a real subprocess."""
    engine = object.__new__(openai_server.Engine)
    engine.closed = False
    engine.pending_lock = threading.Lock()
    engine._fail_pending = lambda *args, **kwargs: None
    engine.dispatcher = threading.current_thread()   # skip the join branch
    engine.process = subprocess.Popen(
        [sys.executable, "-c", code], stdin=subprocess.PIPE)
    return engine


class EngineCloseDrain(unittest.TestCase):
    def tearDown(self):
        saved = openai_server._ENGINE_DRAIN_S
        self.addCleanup(lambda: setattr(openai_server, "_ENGINE_DRAIN_S", saved))

    def test_eof_exit_is_graceful(self):
        openai_server._ENGINE_DRAIN_S = 10.0
        engine = bare_engine(EOF_ENGINE)
        self.addCleanup(engine.process.kill)
        engine.process.stdin.write(b"x")   # a buffered request byte, like a turn
        engine.process.stdin.flush()
        started = time.perf_counter()
        engine.close()
        elapsed = time.perf_counter() - started
        self.assertEqual(engine.process.returncode, 0,
                         "engine should exit 0 on stdin EOF, not by signal")
        self.assertLess(elapsed, openai_server._ENGINE_DRAIN_S,
                        "graceful exit must not consume the drain window")

    def test_hung_engine_falls_back_to_hard_stop(self):
        openai_server._ENGINE_DRAIN_S = 0.5
        engine = bare_engine(HANG_ENGINE)
        self.addCleanup(engine.process.kill)
        started = time.perf_counter()
        engine.close()
        elapsed = time.perf_counter() - started
        self.assertIsNotNone(engine.process.returncode,
                              "fallback ladder must still stop the process")
        self.assertNotEqual(engine.process.returncode, 0,
                            "a killed engine must not look like a clean exit")
        self.assertLess(elapsed, 20, "fallback should be prompt")


if __name__ == "__main__":
    unittest.main()
