"""The dashboard's Brain and Profile tabs read four lines from the engine's
stdout: EMAP, HITS, PROF, TIERS. On GLM-5.3-Flash they were empty, not because
anything needed activating but because glm53.c emitted none of them. This
test drives the engine the way the dashboard does -- through `coli serve`
and the /experts and /profile endpoints -- and asserts the data arrives in
the shape the server parses, so a family that stops emitting a line fails
here rather than as a blank tab on a user's screen.

It needs a GLM-5.3-Flash container and a built `glm53`. Set
COLI_GLM53_FIXTURE to the container directory; without it the test is
skipped, with the reason on the record.
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
FIXTURE = os.environ.get("COLI_GLM53_FIXTURE")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_json(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


@unittest.skipUnless(FIXTURE and Path(FIXTURE, "config.json").is_file(),
                     "COLI_GLM53_FIXTURE not set to a GLM-5.3-Flash container")
@unittest.skipUnless((HERE / "glm53").exists() or (HERE / "glm53.exe").exists(),
                     "glm53 engine is not built")
class Glm53DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.proc = subprocess.Popen(
            [sys.executable, str(HERE / "coli"), "serve", "--model", FIXTURE,
             "--port", str(cls.port), "--cap", "2"],
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
        """EMAP: one row per sparse layer served, one column per expert, two
        hex digits per cell. Emitted after READY: the boot reader discards
        everything before the sentinel, which is exactly the mistake the first
        draft of this feature made."""
        e = self.experts
        self.assertGreater(e.get("rows", 0), 0, "no EMAP reached the server")
        self.assertGreater(e.get("cols", 0), 0)
        self.assertEqual(len(e["map"]), e["rows"] * e["cols"] * 2)

    def test_brain_lights_up_after_a_turn(self):
        """HITS: a bitmap of the experts this turn touched, packed 8 per hex
        pair, refreshed every turn even when nothing was touched."""
        e = self.experts
        self.assertGreaterEqual(e.get("seq", 0), 1, "no HITS reached the server")
        bits = e["rows"] * e["cols"]
        self.assertEqual(len(e["hits"]), ((bits + 7) // 8) * 2)
        self.assertNotEqual(int(e["hits"], 16), 0,
                            "a 9-token prompt through a sparse layer touched no expert")

    def test_profile_records_the_turn_with_its_phases(self):
        """PROF: wall, prompt/completion tokens, five phase timings, forwards.
        expert_wait_s is 0 by construction (glm53 reads synchronously), the
        others must add up to something real."""
        turns = self.profile.get("turns", [])
        self.assertEqual(len(turns), 1, "exactly one turn was made")
        t = turns[0]
        for key in ("wall_s", "prompt_tokens", "completion_tokens", "expert_disk_s",
                    "expert_wait_s", "expert_matmul_s", "attention_s", "lm_head_s", "forwards"):
            self.assertIn(key, t)
        self.assertEqual(t["completion_tokens"], 3)
        self.assertEqual(t["forwards"], 3, "one forward per generated token")
        self.assertEqual(t["expert_wait_s"], 0.0)
        phases = t["expert_disk_s"] + t["expert_matmul_s"] + t["attention_s"] + t["lm_head_s"]
        self.assertGreater(phases, 0.0)
        self.assertLessEqual(phases, t["wall_s"] * 1.05 + 0.01,
                             "phase timings exceed the wall clock: a timer is double-counting")


if __name__ == "__main__":
    unittest.main()
