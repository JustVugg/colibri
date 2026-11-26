"""The RAM fill on DeepSeek V4.1 (V41_RAM_FILL), against the tiny fixture.

A start leaves the expert cache empty and every expert is read the first time a
request routes to it. Where the cache holds every expert and the machine has the
RAM, a few reader threads read the rest in the background and the serving thread
places them in the layers' slots. That must change when an expert is read and
nothing else: the same requests on an engine that fills and on one that does not
(V41_RAM_FILL=0) must write the same bytes, with one conversation and with
several at once, with the requests sent while the fill runs and after it.

And the checks the fill makes before it starts are the operator's protection: a
cache that cannot hold every expert, or a RAM budget the fill would exceed, must
leave the engine as it was and say so in one line.

Runs against the tiny random-init fixture (tools/make_dsv41_tiny.py): 6 layers
of 8 routed experts, so a cap of 8 holds a layer's experts and one of 4 does not.
"""
import os
import re
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("DSV41_ENGINE", HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")))
THREADS = os.environ.get("DSV41_TEST_THREADS", "2")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))
PROMPTS = [bytes((7 * i + 3 * k) % 251 for i in range(24)) for k in range(6)]


@unittest.skipUnless(ENGINE.exists() and (FIXTURE / "model.safetensors").exists()
                     and sys.platform.startswith(("linux", "darwin", "freebsd")),
                     "built V4.1 engine and tiny fixture required")
class Dsv41RamFillTest(unittest.TestCase):
    def spawn(self, cap=8, **env):
        base = {"SNAP": str(FIXTURE), "SERVE": "1", "CTX": "64", "OMP_NUM_THREADS": THREADS, "V41_DSPARK": "0"}
        base.update({k: str(v) for k, v in env.items()})
        engine = ServeEngine(ENGINE, [str(cap)], base, log_prefix=False, reuse=False)
        self.addCleanup(engine.p.stdout.close)
        self.addCleanup(engine.p.stderr.close)
        return engine

    def run_turns(self, cap=8, slots=1, **env):
        if slots > 1:
            env["KV_SLOTS"] = slots
        engine = self.spawn(cap, **env)
        out = []
        try:
            for i, prompt in enumerate(PROMPTS):
                out.append(engine.ask(f"r{i}", prompt, max_tok=8, slot=i % slots))
        finally:
            log = engine.close()
        return out, log

    def test_same_bytes_with_and_without_the_fill(self):
        plain, log_off = self.run_turns(V41_RAM_FILL=0)
        self.assertNotIn("RAM fill", log_off)
        for readers in (1, 4):
            filled, log = self.run_turns(V41_RAM_FILL_READERS=readers)
            self.assertEqual(filled, plain)
            self.assertRegex(log, r"RAM fill: 48 experts .* in the background")

    def test_requests_while_a_slow_fill_runs(self):
        """A reader pausing before each expert keeps the fill running under the requests: they read
        what they need themselves, the fill's copies of those are dropped, the bytes are the same."""
        plain, _ = self.run_turns(V41_RAM_FILL=0)
        for slots in (1, 2):
            want = plain if slots == 1 else self.run_turns(slots=2, V41_RAM_FILL=0)[0]
            for readers in (1, 3):
                engine_env = {"V41_RAM_FILL_PACE_US": 15000, "V41_RAM_FILL_READERS": readers}
                filled, log = self.run_turns(slots=slots, **engine_env)
                self.assertEqual(filled, want)
                self.assertIn("RAM fill: 48 experts", log)

    def test_fill_before_ready_reads_every_expert(self):
        plain, _ = self.run_turns(V41_RAM_FILL=0)
        filled, log = self.run_turns(V41_RAM_FILL_WAIT=1)
        self.assertEqual(filled, plain)
        done = re.search(r"RAM fill: done, (\d+) experts .* (\d+) of the model's (\d+) experts in RAM", log)
        self.assertIsNotNone(done, log[-2000:])
        self.assertEqual(done.group(1, 2, 3), ("48", "48", "48"))

    def test_several_conversations(self):
        plain, _ = self.run_turns(slots=2, V41_RAM_FILL=0)
        for wait in (0, 1):
            filled, log = self.run_turns(slots=2, V41_RAM_FILL_WAIT=wait)
            self.assertEqual(filled, plain)
            self.assertIn("RAM fill: 48 experts", log)

    def test_an_idle_engine_completes_the_fill(self):
        engine = self.spawn()
        try:
            first = engine.ask("a", PROMPTS[0], max_tok=8)
            deadline = time.time() + 30
            while time.time() < deadline and "RAM fill: done" not in "".join(engine._err):
                time.sleep(0.05)
            again = engine.ask("b", PROMPTS[0], max_tok=8)
        finally:
            log = engine.close()
        self.assertIn("RAM fill: done", log)
        self.assertIn("48 of the model's 48 experts in RAM", log)
        self.assertEqual(first, again)

    def test_a_cache_too_small_is_left_alone(self):
        plain, _ = self.run_turns(cap=4, V41_RAM_FILL=0)
        out, log = self.run_turns(cap=4)
        self.assertEqual(out, plain)
        self.assertRegex(log, r"RAM fill: off, the expert cache holds 4 of a layer's 8 experts")
        self.assertNotIn("RAM fill: done", log)

    def test_a_ram_budget_is_kept(self):
        out, log = self.run_turns(RAM_GB="0.000001")
        self.assertRegex(log, r"RAM fill: off, .* over the RAM budget")
        self.assertNotIn("RAM fill: done", log)
        plain, _ = self.run_turns(V41_RAM_FILL=0)
        self.assertEqual(out, plain)


if __name__ == "__main__":
    unittest.main()
