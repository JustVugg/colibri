"""Prefix reuse on Qwen3.8 in serve mode: a chat's next turn reads only its new text.

A chat client resends the transcript every turn: the previous prompt, the answer
the engine wrote, then the new message. Qwen3.8 kept a snapshot of its state at
the previous prompt's end only, so every turn re-read the whole previous answer
before its first token (15.8-24.5 s for a 1,712-token second turn on a CPU box).
It now continues from its live state, which covers the answer too, and keeps the
snapshot for a prompt that does not extend it (a regenerated answer).

The gate is the one tests/prefix_serve_harness.py states for every engine: the
reused turn must produce what the same prompt produces on a cold engine, and the
reuse must cover the GENERATED positions, or this proves nothing the snapshot did
not already do.

The tiny fixture has 64 token ids and the byte tokenizer CI gives it
(tools/make_edge_tiny_tokenizer.py) maps them to bytes 0..63, one id per byte.
So the prompts here use digits and punctuation only, and a reply of ids below
64 comes back as the same ids when the client appends it.
"""
import os
import re
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import PromptCancelContract, ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("QWEN38_BINARY", HERE / ("qwen38.exe" if sys.platform == "win32" else "qwen38")))
FIXTURE = Path(os.environ.get("QWEN38_TINY", HERE / "qwen38_tiny"))
REUSED = re.compile(r"\[PREFIX\] reusing (\d+) of (\d+) prompt tokens \(\d+%\) from (.+)")


@unittest.skipUnless(ENGINE.exists(), "qwen38 is not built")
@unittest.skipUnless((FIXTURE / "tokenizer.json").exists(),
                     "tiny qwen38 fixture or its tokenizer is absent "
                     "(tools/make_qwen38_tiny.py, tools/make_edge_tiny_tokenizer.py)")
class Qwen38PrefixServeTest(PromptCancelContract, unittest.TestCase):
    OPENING = b"12+34=46; 7*8=56; 9-3="
    QUESTION = b"; 81/9="
    # the fixture's ids are the bytes below 64; checks: before the segment, then before each
    # layer, so the midway cancel stops before layer 1 and puts the recurrent state back
    CANCEL_OPENING = OPENING
    CANCEL_MORE = b"6; 81/9=9; 5+5=10; 6*6="
    CANCEL_AFTER = 2

    def spawn(self, log_prefix=True, reuse=True):
        return ServeEngine(ENGINE, ["1", "8"],
                           {"SNAP": str(FIXTURE), "SERVE": "1", "TOK": str(FIXTURE / "tokenizer.json"),
                            "OMP_NUM_THREADS": "2", "COLI_NO_OMP_TUNE": "1", "COLI_CUDA": "0",
                            "Q38_TRUNK_GPU": "0", "Q38_MAXT": "128"},
                           log_prefix=log_prefix, reuse=reuse)

    def cold(self, prompt):
        engine = self.spawn(log_prefix=False)
        try:
            return engine.ask("1", prompt, max_tok=8)
        finally:
            engine.close()

    def reuses(self, log):
        return [(int(m.group(1)), int(m.group(2)), m.group(3)) for m in REUSED.finditer(log)]

    def test_the_next_turn_reuses_the_answer(self):
        """The chat shape: prompt, the reply, a new question. The second turn
        must reuse past the first prompt, into the reply's positions, and answer
        as a cold engine does."""
        warm = self.spawn()
        reply = warm.ask("1", self.OPENING, max_tok=8)
        self.assertTrue(reply and max(reply) < 64, "the fixture replied outside its byte ids: %r" % reply)
        second = self.OPENING + reply + self.QUESTION
        got = warm.ask("2", second, max_tok=8)
        log = warm.close()
        self.assertEqual(got, self.cold(second), "reusing the answer changed the next one")
        reused = self.reuses(log)
        self.assertEqual(len(reused), 1, "expected the second turn alone to reuse:\n" + log)
        n, total, source = reused[0]
        self.assertEqual(total, len(second))
        self.assertGreater(n, len(self.OPENING),
                           "the reuse stopped at the first prompt: the answer was read again\n" + log)
        self.assertEqual(source, "the live state")

    def test_a_prompt_read_in_segments_answers_as_one(self):
        """A served prompt is read in segments of COLI_PROMPT_SEGMENT rows so that a CANCEL
        can stop it between them: the attention rows, the recurrence and the routing are per
        row and in row order, so three-row segments answer what one call answers."""
        prompt = self.OPENING + self.CANCEL_MORE
        with mock.patch.dict(os.environ, {"COLI_PROMPT_SEGMENT": "3"}):
            engine = self.spawn(log_prefix=False)
        try:
            got = engine.ask("1", prompt, max_tok=8)
        finally:
            engine.close()
        self.assertEqual(got, self.cold(prompt))

    def test_three_turns_keep_reusing(self):
        warm = self.spawn()
        transcript = self.OPENING
        for turn, question in enumerate((self.QUESTION, b"; 5+5=", b"; 6*7="), 1):
            reply = warm.ask(str(turn), transcript, max_tok=8)
            transcript += reply + question
        last = warm.ask("4", transcript, max_tok=8)
        log = warm.close()
        self.assertEqual(last, self.cold(transcript), "three reused turns changed the answer")
        self.assertEqual([r[2] for r in self.reuses(log)], ["the live state"] * 3, log)

    def test_reuse_matches_the_same_engine_with_reuse_disabled(self):
        warm = self.spawn()
        reply = warm.ask("1", self.OPENING, max_tok=8)
        second = self.OPENING + reply + self.QUESTION
        with_reuse = warm.ask("2", second, max_tok=8)
        warm.close()
        off = self.spawn(reuse=False)
        off_reply = off.ask("1", self.OPENING, max_tok=8)
        without = off.ask("2", second, max_tok=8)
        log = off.close()
        self.assertEqual(off_reply, reply)
        self.assertNotIn("[PREFIX] reusing", log, "COLI_KV_PREFIX=0 did not disable reuse")
        self.assertEqual(with_reuse, without, "reuse changed the answer on the same engine")

    def test_a_regenerated_answer_comes_from_the_snapshot(self):
        """Asking the same prompt again (regenerate) cannot use the live state,
        which runs past it into the answer: the prompt-end snapshot serves it."""
        warm = self.spawn()
        first = warm.ask("1", self.OPENING, max_tok=8)
        again = warm.ask("2", self.OPENING, max_tok=8)
        log = warm.close()
        self.assertEqual(again, first, "the regenerated greedy answer differs")
        self.assertEqual(self.reuses(log), [(len(self.OPENING), len(self.OPENING), "the prompt-end snapshot")],
                         log)

    def test_a_diverging_prompt_starts_over(self):
        warm = self.spawn()
        warm.ask("1", self.OPENING, max_tok=8)
        other = b"5*5=25; 3+3="
        got = warm.ask("2", other, max_tok=8)
        log = warm.close()
        self.assertIn("[PREFIX] no reuse", log)
        self.assertEqual(got, self.cold(other), "a diverging prompt was answered from the old state")


if __name__ == "__main__":
    unittest.main()
