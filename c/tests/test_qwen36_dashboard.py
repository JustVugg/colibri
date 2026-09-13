"""The dashboard's Brain and Profile tabs read four lines from the engine's
stdout: EMAP, HITS, PROF, TIERS. On Qwen3.6 they were empty, not because
anything needed activating but because qwen36.c emitted none of them. This
test drives the engine the way the dashboard does -- through `coli serve`
and the /experts and /profile endpoints -- and asserts the data arrives in
the shape the server parses, so a family that stops emitting a line fails
here rather than as a blank tab on a user's screen.

It needs a Qwen3.6 container with a tokenizer.json and a built `qwen36`.
Set COLI_QWEN36_FIXTURE to the container directory; without it the test is
skipped, with the reason on the record. The CI job builds the tiny fixture it
already uses for the token-exact oracle and adds the byte tokenizer from
tools/make_edge_tiny_tokenizer.py, which is all `coli serve` needs.
"""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FIXTURE = os.environ.get("COLI_QWEN36_FIXTURE")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


@unittest.skipUnless(FIXTURE and Path(FIXTURE, "config.json").is_file()
                     and Path(FIXTURE, "tokenizer.json").is_file(),
                     "COLI_QWEN36_FIXTURE not set to a Qwen3.6 container with a tokenizer")
@unittest.skipUnless((HERE / "qwen36").exists() or (HERE / "qwen36.exe").exists(),
                     "qwen36 engine is not built")
class Qwen36DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, str(HERE / "coli"), "serve", "--model", FIXTURE,
             "--port", str(cls.port), "--cap", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                get_json(f"http://127.0.0.1:{cls.port}/health", timeout=2)
                break
            except Exception:
                if cls.proc.poll() is not None:
                    raise RuntimeError("coli serve exited: " + cls.proc.stdout.read()[-2000:])
                time.sleep(1)
        else:
            raise RuntimeError("coli serve did not come up in 180 s")
        model = get_json(f"http://127.0.0.1:{cls.port}/v1/models")["data"][0]["id"]
        body = json.dumps({"model": model, "prompt": "abcabcabc", "max_tokens": 3}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{cls.port}/v1/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            cls.completion = json.load(r)
        cls.experts = get_json(f"http://127.0.0.1:{cls.port}/experts")
        cls.profile = get_json(f"http://127.0.0.1:{cls.port}/profile")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_brain_has_a_grid(self):
        """EMAP: one row per layer, one column per expert, two hex digits per
        cell. This engine's boot reader expects READY then STAT, so the grid
        goes after both: emitted between them it is an invalid engine status,
        emitted before READY it is discarded. Both mistakes end here as an
        empty grid."""
        e = self.experts
        self.assertGreater(e.get("rows", 0), 0, "no EMAP reached the server")
        self.assertGreater(e.get("cols", 0), 0)
        self.assertEqual(len(e["map"]), e["rows"] * e["cols"] * 2)

    def test_brain_lights_up_after_a_turn(self):
        """HITS: a bitmap of the experts this turn touched, packed 8 per hex
        pair, refreshed every turn. Every layer of this model is a MoE layer,
        so a prompt through it must touch experts somewhere."""
        e = self.experts
        self.assertGreaterEqual(e.get("seq", 0), 1, "no HITS reached the server")
        bits = e["rows"] * e["cols"]
        self.assertEqual(len(e["hits"]), ((bits + 7) // 8) * 2)
        self.assertNotEqual(int(e["hits"], 16), 0,
                            "a 9-token prompt through MoE layers touched no expert")

    def test_profile_records_the_turn_with_its_phases(self):
        """PROF: wall, prompt/completion tokens, five phase timings, forwards.
        The forwards are COUNTED where step() is called, not derived from the
        token count: a turn that stops at max_tokens does not run a forward
        for the token it will never sample."""
        turns = self.profile.get("turns", [])
        self.assertEqual(len(turns), 1, "exactly one turn was made")
        t = turns[0]
        for key in ("wall_s", "prompt_tokens", "completion_tokens", "expert_disk_s",
                    "expert_wait_s", "expert_matmul_s", "attention_s", "lm_head_s", "forwards"):
            self.assertIn(key, t)
        self.assertEqual(t["completion_tokens"], 3)
        self.assertEqual(t["forwards"], 3, "prefill plus one forward per token after the first")
        phases = t["expert_disk_s"] + t["expert_matmul_s"] + t["attention_s"] + t["lm_head_s"]
        # Not > 0: PROF prints milliseconds and a 3-token turn on a tiny model
        # finishes under a millisecond on a fast runner, so every phase rounds
        # to 0.000 and the sum is legitimately zero (CI, 2026-09-10). What the
        # contract holds is that phases never exceed the wall.
        self.assertGreaterEqual(phases, 0.0)
        self.assertGreater(t["wall_s"], 0.0)
        self.assertLessEqual(phases, t["wall_s"] * 1.05 + 0.01,
                             "phase timings exceed the wall clock: a timer is double-counting")


if __name__ == "__main__":
    unittest.main()
