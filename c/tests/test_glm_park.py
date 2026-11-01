"""GLM-5.2 (colibri.c): a short request between two turns keeps the one slot's conversation.

A client's title or summary request between two turns of a conversation (Hermes asks for the
title after the first) shares a few tokens with the conversation and wrote its rows over the
slot's from there: the conversation's next turn read everything again. The slot's first rows
are parked for it (park.h) and come back with the next request: the conversation continues
from the whole prefix it held, and both answer as a cold engine does.

Through the multiplexed protocol the gateway speaks (SERVE=1 SERVE_BATCH=1, one KV slot),
against the tiny GLM oracle model (tools/make_glm_oracle.py), with the byte-level tokenizer
test_glm_prompt_cancel.py writes into a scratch copy.
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import ServeEngine  # noqa: E402
from test_glm_prompt_cancel import BINARY, FIXTURE, SLOT_PREFIX, byte_tokenizer  # noqa: E402


@unittest.skipUnless(BINARY, "colibri is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "model.safetensors").is_file(),
                     "COLI_ORACLE_FIXTURE not set to the tiny GLM oracle (tools/make_glm_oracle.py)")
class GlmParkTest(unittest.TestCase):
    OPENING = b"The capital of France is Paris, the capital of Italy is Rome, and the capital of"
    SIDE = b"Five times five is"
    SECOND = OPENING + b" Spain is Madrid. The capital of Germany is"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="glm_park_")
        cls.snap = Path(cls.tmp) / "snap"
        shutil.copytree(FIXTURE, cls.snap)
        (cls.snap / "tokenizer.json").write_text(json.dumps(byte_tokenizer()), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def spawn(self, **env):
        for kept in self.snap.glob(".coli_kv*"):   # every engine from nothing (see test_glm_prompt_cancel)
            kept.unlink()
        return ServeEngine(BINARY, ["0"],
                           dict({"SNAP": str(self.snap), "SERVE": "1", "SERVE_BATCH": "1",
                                 "KV_SLOTS": "1", "CTX": "256", "COLI_TEMP": "0",
                                 "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1"}, **env),
                           header_tail=" 0")

    def cold(self, prompt):
        engine = self.spawn()
        try:
            return engine.ask("1", prompt, max_tok=8)
        finally:
            engine.close()

    def three_turns(self, **env):
        engine = self.spawn(COLI_PARK_MIN="16", **env)
        engine.ask("1", self.OPENING)
        side = engine.ask("2", self.SIDE, max_tok=8)
        got = engine.ask("3", self.SECOND, max_tok=8)
        return side, got, engine.close()

    def test_the_conversation_comes_back_after_a_short_request(self):
        side, got, log = self.three_turns()
        self.assertIn("conversation parked for a", log, "the short request did not park:\n" + log)
        self.assertIn("conversation is back", log, "the parked conversation did not come back:\n" + log)
        self.assertEqual(side, self.cold(self.SIDE), "the short request answered otherwise beside a parked conversation")
        self.assertEqual(got, self.cold(self.SECOND), "the conversation answered otherwise after it was parked")
        held = [(int(m.group(1)), int(m.group(2))) for m in SLOT_PREFIX.finditer(log)]
        self.assertGreaterEqual(held[-1][0], held[0][1], "the next turn read the parked conversation again:\n" + log)

    def test_park_off_reads_the_conversation_again(self):
        """COLI_PARK=0: the request takes the slot, and the next turn shares only its first bytes."""
        side, got, log = self.three_turns(COLI_PARK="0")
        self.assertNotIn("parked", log)
        held = [(int(m.group(1)), int(m.group(2))) for m in SLOT_PREFIX.finditer(log)]
        self.assertLess(held[-1][0], held[0][1], log)
        self.assertEqual(got, self.cold(self.SECOND))


if __name__ == "__main__":
    unittest.main()
