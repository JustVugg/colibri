"""The dashboard's Brain and Profile tabs on DeepSeek V4.1 Flash.

Same contract as every other family: EMAP after READY and STAT, HITS and PROF after
DONE, over the rows and columns the grid declares. This drives the engine the way the
gateway does -- through `coli serve` -- so a family that stops emitting one of the four
lines fails here rather than as a blank tab.

Needs a V4.1-shaped container with a tokenizer and a built `deepseek_v41`. Set
COLI_DSV41_FIXTURE to the container; without it the test skips with the reason on
the record. CI builds the fixture with tools/make_dsv41_tiny.py.
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
FIXTURE = Path(os.environ.get("COLI_DSV41_FIXTURE", ""))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file()
                     and (FIXTURE / "tokenizer.json").is_file(),
                     "COLI_DSV41_FIXTURE not set to a V4.1 container with a tokenizer")
@unittest.skipUnless((HERE / "deepseek_v41").exists() or (HERE / "deepseek_v41.exe").exists(),
                     "deepseek_v41 engine is not built")
class Dsv41DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        # Drafts off: one forward per token is what the profile assertion below is
        # about, and a draft head that got lucky would fold two tokens into one
        # forward. tests/test_dsv41_dspark_serve.py covers the other path.
        cls.proc = subprocess.Popen(
            [sys.executable, str(HERE / "coli"), "serve", "--model", str(FIXTURE),
             "--port", str(cls.port), "--cap", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            env={**os.environ, "V41_DSPARK": "0"})
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
        request = urllib.request.Request(f"http://127.0.0.1:{cls.port}/v1/completions",
                                         data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            cls.completion = json.load(response)
        cls.experts = get_json(f"http://127.0.0.1:{cls.port}/experts")
        cls.profile = get_json(f"http://127.0.0.1:{cls.port}/profile")
        cls.config = json.loads((FIXTURE / "config.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(10)
        except subprocess.TimeoutExpired:
            cls.proc.kill()

    def test_brain_grid_matches_the_model(self):
        text = self.config.get("text_config", self.config)
        experts = self.experts
        self.assertEqual(experts["rows"], text["num_hidden_layers"])
        self.assertEqual(experts["cols"], text["n_routed_experts"])
        self.assertEqual(len(experts["map"]), experts["rows"] * experts["cols"] * 2)

    def test_brain_lights_up_and_shows_residents(self):
        experts = self.experts
        bits = experts["rows"] * experts["cols"]
        self.assertEqual(len(experts["hits"]), ((bits + 7) // 8) * 2)
        self.assertNotEqual(int(experts["hits"], 16), 0,
                            "a 9-token prompt routed no expert at all")
        self.assertNotEqual(int(experts["map"], 16), 0,
                            "after a turn the cache holds experts: the grid must show them")

    def test_profile_records_the_turn(self):
        turns = self.profile.get("turns", [])
        self.assertEqual(len(turns), 1)
        turn = turns[0]
        for key in ("wall_s", "prompt_tokens", "completion_tokens", "expert_disk_s",
                    "expert_wait_s", "expert_matmul_s", "attention_s", "lm_head_s", "forwards"):
            self.assertIn(key, turn)
        self.assertEqual(turn["completion_tokens"], 3)
        self.assertEqual(turn["forwards"], 3, "prefill plus one forward per token after the first")
        phases = (turn["expert_disk_s"] + turn["expert_matmul_s"] +
                  turn["attention_s"] + turn["lm_head_s"])
        self.assertGreaterEqual(phases, 0.0)
        self.assertLessEqual(phases, turn["wall_s"] * 1.05 + 0.01,
                             "phase timings exceed the wall clock: a timer double-counts")


if __name__ == "__main__":
    unittest.main()
