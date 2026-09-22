"""KV prefix reuse on DeepSeek V4.1: the shared contract, against the tiny fixture.

V4.1 is the hardest case in the tree and the reason the contract is worth having.
A turn here does not build one tensor: it builds a window ring per layer with its
own position map, a compressed KV and index-key cache whose slot is position over
the compression ratio, the compressor's partial group, and the engram history.
All of them are position-indexed and append-only, which is exactly why reuse is
sound -- a state that stopped at position P is the state at P -- and also why a
wrong reuse length would corrupt five things at once, silently.

Speculative decoding (DSpark) is on by default here and rolls the caches back to
the last draft that verified. The record follows that rollback, so what it claims
is what the caches hold. Its acceptance rate may differ between a warm and a cold
engine; the emitted tokens may not, because every draft is verified by the main
model before a single one is written.

Runs against the tiny random-init fixture the oracle job already builds
(tools/make_dsv41_tiny.py): no checkpoint, no fast disk.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_serve_harness import PrefixReuseContract, ServeEngine  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("deepseek_v41.exe" if sys.platform == "win32" else "deepseek_v41")
FIXTURE = Path(os.environ.get("DSV41_TINY", HERE / "dsv41_tiny"))


@unittest.skipUnless(ENGINE.exists(), "deepseek_v41 is not built")
@unittest.skipUnless((FIXTURE / "model.safetensors").exists(),
                     "tiny dsv41 fixture is absent (tools/make_dsv41_tiny.py)")
class Dsv41PrefixServeTest(PrefixReuseContract, unittest.TestCase):
    # This fixture's vocabulary is the byte range, so a reply survives the round
    # trip and the replay scenario can require reuse to fire. It is also the
    # scenario that covers the speculative rollback: DSpark drafts through that
    # turn, the main model rejects some of them, and the record has to give those
    # positions back or the next turn finds a prefix the caches no longer hold.
    REPLAY_REUSES = True

    def spawn(self, log_prefix=True, reuse=True):
        # Reuse is opt-in on this engine (see the comment on kv_prefix_on), so
        # the arm that wants it has to ask. `reuse=False` still wins: the harness
        # sets COLI_KV_PREFIX=0 after this env is merged.
        return ServeEngine(ENGINE, ["8"],
                           {"SNAP": str(FIXTURE), "SERVE": "1", "COLI_KV_PREFIX": "1"},
                           log_prefix=log_prefix, reuse=reuse)

    def test_a_turn_that_replays_the_reply_is_never_wrong(self):
        """Overridden, because on this engine it is not true, and the reason is
        the architecture rather than the wiring.

        A query reads one set of index keys when its position is PREFILLED (its
        layer's own index owner, masked to the query's reach) and another when
        the position is DECODED (whatever was published last). Both are the
        vendor's: give the prefill the decode schedule and the tiny oracle drops
        to 7/8, give the decode the prefill owner and it drops to 1/8. So a
        position generated in an earlier turn does not attend the way the same
        text attends when prefilled cold, and a prefix that contains the reply
        cannot reproduce a cold engine's answer.

        What this asserts instead is what does hold, and what a user is entitled
        to: the reuse fires, the engine is deterministic about it, and turning
        reuse off returns the cold answer exactly. The exactness of a resumed
        PREFILL -- the part that is ours to get right -- is gated by
        test_a_resumed_prefill_matches_a_cold_one below.
        """
        opening = b"The capital of France is"

        warm = self.spawn()
        reply = warm.ask("1", opening, max_tok=8)
        second = opening + reply + b" and the capital of Spain is"
        got = warm.ask("2", second, max_tok=8)
        log = warm.close()

        again = self.spawn(log_prefix=False)
        again.ask("1", opening, max_tok=8)
        twice = again.ask("2", second, max_tok=8)
        again.close()

        off = self.spawn(log_prefix=False, reuse=False)
        off.ask("1", opening, max_tok=8)
        without = off.ask("2", second, max_tok=8)
        off.close()

        cold = self.spawn(log_prefix=False, reuse=False)
        fresh = cold.ask("1", second, max_tok=8)
        cold.close()

        self.assertIn("[PREFIX] reusing", log,
                      "the turn that replays the reply did not reuse: the record "
                      "does not describe the generated positions.\n"
                      "engine said:\n" + (log or "(nothing on stderr)"))
        self.assertEqual(got, twice, "the reused answer is not deterministic")
        self.assertEqual(without, fresh,
                         "with reuse off, the second turn must still be a cold prefill")

    def test_a_resumed_prefill_matches_a_cold_one(self):
        """The exactness that IS ours: a prefill that resumes mid-sequence has to
        compute what a cold prefill computes for the same positions.

        This is what the index-key schedule fix is for. It used to select the
        speculative batch by `start_pos > 0`, which a resumed prefill also
        matches, so its rows read another layer's published keys and picked a
        different index top-k. Every turn here asks for one token, so nothing
        generated enters the record and the whole prefix is prefilled text.
        """
        full = b"The capital of France is a question about geography and the capital of Spain is another one"
        cold = self.spawn(log_prefix=False)
        expected = cold.ask("1", full, max_tok=8)
        cold.close()

        for tail in (4, 12, 24, 48):
            with self.subTest(tail=tail):
                warm = self.spawn()
                warm.ask("1", full[:-tail])
                got = warm.ask("2", full, max_tok=8)
                log = warm.close()
                self.assertIn("[PREFIX] reusing", log, "no reuse, so nothing is proven")
                self.assertEqual(got, expected,
                                 f"a prefill resumed {tail} bytes from the end does not "
                                 f"reproduce the cold one")


if __name__ == "__main__":
    unittest.main()
