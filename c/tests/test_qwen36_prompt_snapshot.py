"""Qwen3.6's prompt-end snapshot (#2589), against the tiny fixture.

Three of every four layers are DeltaNet, whose recurrence cannot step back. A chat request
that does not extend the state the engine holds read the whole conversation again: a retry of
a reply (the state holds the prompt and the reply's tokens, the retry only the prompt), a last
answer edited or rendered otherwise (the state leaves the request after the last prompt's
end), and a client gone while the prompt was read (the read stopped with some layers advanced,
and the engine started over). The engine now keeps the state as the last prompt read to its
end left it, the K/V rows where they are, and goes back to it in all three; what it answers
must be what a cold engine answers.

The cancel scenario is the shared PromptCancelContract: the cancelled turn's CANCEL is held
back for two checks (COLI_PROMPT_BREAK_AFTER), so the read stops after the first layer, a
DeltaNet one, with its state advanced and the others not.
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import PromptCancelContract, ServeEngine, ensure_byte_tokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
FIXTURE = Path(os.environ.get("QWEN36_TINY", HERE / "qwen36_tiny_c"))
OPENING = b"The capital of France is"


def spawn(log_prefix=True, **env):
    return ServeEngine(ENGINE, ["8", "8"], dict({"SNAP": str(FIXTURE), "SERVE": "1", "COLI_DENSE_I8": "0"}, **env),
                       log_prefix=log_prefix)


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless((FIXTURE / "model-globals.safetensors").exists(),
                     "tiny qwen36 container is absent (tools/convert_qwen36.py)")
class Qwen36PromptSnapshotTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_byte_tokenizer(FIXTURE)

    def cold(self, prompt):
        eng = spawn(log_prefix=False)
        try:
            return eng.ask("1", prompt, max_tok=8)
        finally:
            eng.close()

    def test_a_retry_of_the_last_prompt_reads_nothing(self):
        warm = spawn()
        warm.ask("1", OPENING, max_tok=8)
        again = warm.ask("2", OPENING, max_tok=8)
        log = warm.close()
        self.assertIn(f"resuming from the prompt-end snapshot: {len(OPENING)} of {len(OPENING)} prompt tokens", log, log)
        self.assertEqual(again, self.cold(OPENING), "the retry answered otherwise than a cold engine")

    def test_an_edited_answer_resumes_where_the_last_prompt_ended(self):
        warm = spawn()
        warm.ask("1", OPENING, max_tok=8)
        edited = OPENING + b" Paris, said the edited answer. And the capital of Spain is"
        got = warm.ask("2", edited, max_tok=8)
        log = warm.close()
        self.assertIn(f"resuming from the prompt-end snapshot: {len(OPENING)} of {len(edited)} prompt tokens", log, log)
        self.assertEqual(got, self.cold(edited), "resuming from the snapshot changed the answer")

    def test_another_turn_after_a_retry_resumes_from_it_again(self):
        """The snapshot outlives the turn it served: a third turn that leaves the retry's reply
        resumes from it again, and a turn that extends what is held reuses it live."""
        warm = spawn()
        warm.ask("1", OPENING, max_tok=8)
        warm.ask("2", OPENING, max_tok=8)
        longer = OPENING + b" a question about geography, and the capital of Spain is"
        got = warm.ask("3", longer, max_tok=1)
        extended = warm.ask("4", longer + b" Madrid. And of Portugal it is", max_tok=8)
        log = warm.close()
        self.assertIn(f"resuming from the prompt-end snapshot: {len(OPENING)} of {len(longer)} prompt tokens", log, log)
        self.assertIn(f"[PREFIX] reusing {len(longer)} of", log, log)
        self.assertEqual(got, self.cold(longer)[:len(got)], "the turn after a resumed retry answered otherwise")
        self.assertEqual(extended, self.cold(longer + b" Madrid. And of Portugal it is"))

    def test_a_cancel_after_the_whole_prompt_keeps_its_end(self):
        """A client gone once the prompt was read whole (the check before its first token):
        the state is the prompt's end, and the same prompt sent again reads nothing."""
        warm = spawn(COLI_PROMPT_BREAK_AFTER="1000")   # only the check after the forward reads stdin
        prompt = OPENING + b" and the capital of Spain is"
        warm.ask("1", OPENING)
        frame, written = warm.ask_cancelled("2", prompt)
        again = warm.ask("3", prompt, max_tok=8)
        log = warm.close()
        self.assertEqual((frame, written), ("ERROR 2 CANCELLED", b""), log)
        self.assertIn(f"cancelled while reading the prompt: {len(prompt)} of {len(prompt)}", log, log)
        self.assertIn(f"resuming from the prompt-end snapshot: {len(prompt)} of {len(prompt)}", log, log)
        self.assertEqual(again, self.cold(prompt))

    def test_off_reads_the_conversation_again(self):
        warm = spawn(COLI_PROMPT_SNAPSHOT="0")
        warm.ask("1", OPENING, max_tok=8)
        again = warm.ask("2", OPENING, max_tok=8)
        log = warm.close()
        self.assertNotIn("prompt-end snapshot", log, log)
        self.assertEqual(log.count("[PREFIX] no reuse"), 2, "the retry did not read the conversation again:\n" + log)
        self.assertEqual(again, self.cold(OPENING))


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless((FIXTURE / "model-globals.safetensors").exists(),
                     "tiny qwen36 container is absent (tools/convert_qwen36.py)")
class Qwen36PromptCancelTest(PromptCancelContract, unittest.TestCase):
    # the check before the forward, then one before each layer: the third is before layer 1,
    # so the first (DeltaNet) layer has read the cancelled prompt's rows and the rest have not
    CANCEL_AFTER = 2
    PROMPT_CANCEL_KEEPS_PREFIX = True   # the prompt-end snapshot lies before the stopped read

    @classmethod
    def setUpClass(cls):
        ensure_byte_tokenizer(FIXTURE)

    def spawn(self, log_prefix=True):
        return spawn(log_prefix=log_prefix)

    def test_without_the_snapshot_a_stopped_read_starts_over(self):
        """The bite of the midway scenario: with the snapshot off the engine says it holds
        nothing after a read stopped in the middle (the state it had is gone)."""
        with mock.patch.dict(os.environ, dict(self.CANCEL_ENV, COLI_PROMPT_BREAK_AFTER=str(self.CANCEL_AFTER),
                                              COLI_PROMPT_SNAPSHOT="0")):
            warm = self.spawn()
        warm.ask("1", self.CANCEL_OPENING)
        frame, _ = warm.ask_cancelled("2", self.CANCEL_OPENING + self.CANCEL_MORE)
        log = warm.close()
        self.assertEqual(frame, "ERROR 2 CANCELLED", log)
        self.assertIn(f"cancelled while reading the prompt: 0 of {len(self.CANCEL_OPENING + self.CANCEL_MORE)}", log, log)


if __name__ == "__main__":
    unittest.main()
