"""Drafting changes the speed of a turn and nothing else.

Off by default since the first measurement on the real checkpoint, where it
accepted half its proposals and still cost 32% more wall clock; V41_DSPARK=1
turns it on, and this drives both paths.

DSpark proposes tokens and the main model verifies them in one forward. The whole
design rests on one property: what comes out has to be what would have come out
without it. This drives `coli serve` twice over the same prompt, once with drafts on
and once with V41_DSPARK=0, and compares the text.

The tiny fixture's draft head is random, so almost nothing it proposes is accepted --
which is the interesting half of the path, not the boring one: every round drafts,
verifies, rejects and rolls back, and if the rollback left the caches describing
positions that were never committed, the two runs would part company within a few
tokens.

Needs a V4.1 container with a DSpark head and a tokenizer. CI builds one with
tools/make_dsv41_tiny.py; set COLI_DSV41_FIXTURE to it.
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
PROMPT = "abcabcabcabcabc"
TOKENS = 12


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def has_draft_head():
    if not (FIXTURE.name and (FIXTURE / "config.json").is_file()):
        return False
    text = json.loads((FIXTURE / "config.json").read_text())
    text = text.get("text_config", text)
    # dspark_block_size, not n_mtp_layers: the released checkpoint declares the
    # first and omits the second, and the fixture now matches it. The engine
    # counts the stages by probing the shards for the same reason.
    return int(text.get("dspark_block_size", 0)) > 0


def completion(drafts):
    """One turn through the gateway, with drafting on or off. Returns (text, log)."""
    port = free_port()
    environment = {**os.environ, "V41_DSPARK": "1" if drafts else "0"}
    process = subprocess.Popen(
        [sys.executable, str(HERE / "coli"), "serve", "--model", str(FIXTURE),
         "--port", str(port), "--cap", "4"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=environment)
    try:
        deadline = time.time() + 180
        while True:
            if time.time() > deadline:
                raise RuntimeError("coli serve did not come up in 180 s")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                    model = json.load(r)["data"][0]["id"]
                break
            except Exception:
                if process.poll() is not None:
                    raise RuntimeError("coli serve exited early")
                time.sleep(1)
        body = json.dumps({"model": model, "prompt": PROMPT,
                           "max_tokens": TOKENS, "temperature": 0.0}).encode()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
                                         data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            answer = json.load(response)
    finally:
        process.terminate()
        try:
            output = process.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate()[0]
    return answer["choices"][0]["text"], output or ""


@unittest.skipUnless(has_draft_head(), "COLI_DSV41_FIXTURE has no DSpark head")
@unittest.skipUnless((HERE / "deepseek_v41").exists() or (HERE / "deepseek_v41.exe").exists(),
                     "deepseek_v41 engine is not built")
class Dsv41DsparkServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.drafted, cls.drafted_log = completion(True)
        cls.plain, cls.plain_log = completion(False)

    def test_drafting_does_not_change_the_answer(self):
        self.assertTrue(self.plain, "the turn produced no text at all")
        self.assertEqual(self.drafted, self.plain,
                         "the same prompt gave different text with drafts on: a "
                         "rejected round left state behind")

    def test_the_engine_says_which_mode_it_is_in(self):
        self.assertIn("DSpark on", self.drafted_log)
        self.assertIn("DSpark drafts off", self.plain_log)

    def test_drafts_were_actually_proposed(self):
        """Otherwise the comparison above is two identical runs of the same path."""
        self.assertIn("DSpark:", self.drafted_log.replace("DSpark on", ""),
                      "no round reported: nothing was drafted, so nothing was verified")


if __name__ == "__main__":
    unittest.main()
