"""Closed-set scoring over the serve protocol: exact, and invisible to chat.

Two properties, and the whole feature rests on them.

EXACTNESS. Rewinding to a snapshot must give the SAME log probabilities as
recomputing the prompt from cold. If it does not, the engine is answering from
a state that is not the one it claims to hold -- and the answer still looks
plausible, so nothing else in the tree would catch it. The gate is a paired
comparison: the same option scored through the snapshot and through a fresh
engine must agree to the last digit printed.

INVISIBILITY. A request that does not ask for the channel must produce the
byte-identical frames it produced before the channel existed. Chat is the mode
everyone uses; the scoring channel is opt-in and must stay that way. The check
is a byte diff of the frames, not a reading of the text.

Runs against the tiny olmoe fixture the oracle job already builds
(tools/make_olmoe_tiny.py + convert_olmoe_merged.py), so it needs no
checkpoint. Skipped when the fixture is absent, like the other serve tests.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if sys.platform == "win32" else "olmoe")
FIXTURE = Path(os.environ.get("OLMOE_TINY", HERE / "olmoe_tiny"))
PROMPT = "Context: the release is late and the tests are red.\nQuestion: ship?\nAnswer:"
OPTION = " no"


class Engine:
    """One serve-mode engine, driven over the wire the gateway uses."""

    def __init__(self):
        env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1",
                   TOK=str(FIXTURE / "tokenizer.json"),
                   COLI_NO_OMP_TUNE="1", OMP_NUM_THREADS="2")
        self.p = subprocess.Popen([str(ENGINE)], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                  bufsize=0)
        while True:
            line = self.readline()
            if line is None:
                raise AssertionError("engine died before READY")
            if "READY" in line:
                return

    def readline(self):
        out = b""
        while True:
            ch = self.p.stdout.read(1)
            if not ch:
                return None
            if ch == b"\n":
                return out.decode("utf-8", "replace").replace("\x01", "")
            out += ch

    def submit(self, rid, text, max_tokens, extension=""):
        """Returns (frames, echoes). frames are raw lines, for byte comparison."""
        payload = text.encode("utf-8")
        self.p.stdin.write(
            f"SUBMIT {rid} 0 {len(payload)} {max_tokens} 0 1{extension}\n".encode())
        self.p.stdin.write(payload)
        self.p.stdin.write(b"\n")
        self.p.stdin.flush()
        frames, echoes = [], {}
        while True:
            line = self.readline()
            if line is None:
                return frames, echoes
            fields = line.split()
            if not fields:
                continue
            if fields[0] in ("ECHO", "DATA"):
                size = int(fields[2])
                if size:
                    self.p.stdout.read(size)
                self.p.stdout.read(1)
                if fields[0] == "ECHO" and fields[4] not in ("nan", "-nan"):
                    echoes[int(fields[3])] = float(fields[4])
            frames.append(line)
            if fields[0] in ("DONE", "ERROR"):
                return frames, echoes

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        self.p.terminate()
        self.p.wait(timeout=30)


def strip_volatile(frames):
    """Drop the lines that differ run to run whatever the request asked for:
    timings, hardware, and the routing telemetry that accumulates on disk."""
    return [f for f in frames
            if not f.startswith(("STAT", "PROF", "DONE", "TIERS", "HWINFO", "EMAP", "HITS"))]


@unittest.skipUnless((FIXTURE / "config.json").is_file(),
                     "tiny olmoe fixture is absent (tools/make_olmoe_tiny.py)")
@unittest.skipUnless(ENGINE.is_file(), "olmoe engine is not built")
class BrioServe(unittest.TestCase):
    def test_a_snapshot_scores_exactly_like_a_cold_recompute(self):
        warm = Engine()
        try:
            _, prefix = warm.submit(1, PROMPT, 0, " logprobs=1 pin=1")
            _, pinned = warm.submit(2, PROMPT + OPTION, 0, " logprobs=1")
        finally:
            warm.close()
        self.assertTrue(prefix, "the warm pass read nothing")
        tail_starts = max(prefix) + 1

        cold = Engine()
        try:
            _, fresh = cold.submit(1, PROMPT + OPTION, 0, " logprobs=1")
        finally:
            cold.close()

        wanted = {p for p in fresh if p >= tail_starts}
        self.assertTrue(wanted, "the cold run read no tail position")

        # SAME positions, not just the ones that happen to overlap. The first
        # fresh token is predicted by the snapshot's own logits; if those go
        # missing its ECHO carries "nan" and drops out, and comparing only the
        # intersection would call that a pass. It is not: one option token
        # would silently stop counting towards the score.
        got = {p for p in pinned if p >= tail_starts}
        self.assertEqual(got, wanted,
                         f"the snapshot scored positions {sorted(got)} where a cold "
                         f"recompute scored {sorted(wanted)} -- an option token lost "
                         f"its predictor")

        for position in sorted(wanted):
            self.assertEqual(
                pinned[position], fresh[position],
                f"position {position}: rewinding gave {pinned[position]}, "
                f"recomputing gave {fresh[position]} -- the snapshot is not the "
                f"state it claims to be")

    def test_a_stale_snapshot_is_not_restored(self):
        """A snapshot whose rows the state no longer holds must be refused.

        The snapshot carries the recurrent state and the ids, never the
        attention rows: those stay where they are, and between two options
        another conversation can overwrite them. Matching the snapshot's ids
        against the REQUEST is not enough then -- they also have to match what
        the state still says it holds, which is what kv_prefix_holds() asks.

        Without that check the engine restores a recurrent state that belongs
        to one conversation on top of attention rows that belong to another,
        and answers from a mixture of the two. The reply stays plausible, so
        only a paired comparison catches it: here a divergent prompt is sent
        between the snapshot and its use, and the score must still equal a
        cold recompute."""
        other = "Context: the invoice is overdue by ninety days.\nQuestion: escalate?\nAnswer:"

        engine = Engine()
        try:
            _, prefix = engine.submit(1, PROMPT, 0, " logprobs=1 pin=1")
            engine.submit(2, other, 4)          # overwrites the record with other ids
            _, after = engine.submit(3, PROMPT + OPTION, 0, " logprobs=1")
        finally:
            engine.close()
        self.assertTrue(prefix, "the warm pass read nothing")
        tail_starts = max(prefix) + 1

        cold = Engine()
        try:
            _, fresh = cold.submit(1, PROMPT + OPTION, 0, " logprobs=1")
        finally:
            cold.close()

        wanted = sorted(p for p in fresh if p >= tail_starts)
        self.assertTrue(wanted, "the cold run read no tail position")
        for position in wanted:
            self.assertIn(position, after,
                          f"position {position} was not scored after the detour")
            self.assertEqual(
                after[position], fresh[position],
                f"position {position}: {after[position]} after a divergent prompt "
                f"came in between, {fresh[position]} from cold -- a stale snapshot "
                f"was restored over someone else's attention rows")

    def test_the_snapshot_only_reads_the_fresh_tail(self):
        """The point of the snapshot: the shared prefix is not read again."""
        engine = Engine()
        try:
            _, prefix = engine.submit(1, PROMPT, 0, " logprobs=1 pin=1")
            _, tail = engine.submit(2, PROMPT + OPTION, 0, " logprobs=1")
        finally:
            engine.close()
        self.assertTrue(prefix, "the warm pass read nothing")
        self.assertTrue(tail, "the option read nothing")
        self.assertGreater(min(tail), max(prefix) - 1,
                           "the option re-read positions the snapshot already held")

    def test_a_request_without_the_keys_is_byte_identical(self):
        """Chat must not be able to tell the channel exists.

        Two fresh engines rather than two turns of one: the request id is part
        of every frame, so the same id has to be reused, and a second turn on a
        live engine would also start from a different cached state."""
        def frames_for(extension):
            engine = Engine()
            try:
                frames, _ = engine.submit(1, PROMPT, 4, extension)
            finally:
                engine.close()
            return strip_volatile(frames)

        plain = frames_for("")
        off = frames_for(" logprobs=0 pin=0")
        self.assertEqual(plain, off,
                         "the two ways of opting out disagree")

        # And opting out must see NOTHING of the channel. Comparing the two
        # opt-out spellings only proves they agree with each other: a defect
        # that turned the read-out on for everybody would move both arms
        # together and pass. These two assertions are the actual invariant.
        for frames in (plain, off):
            self.assertFalse([f for f in frames if f.startswith("ECHO")],
                             "a request that did not ask for the channel got ECHO frames")
            for line in frames:
                if line.startswith("DATA"):
                    self.assertEqual(
                        len(line.split()), 3,
                        f"a DATA frame carried the numeric tail unasked: {line!r}")

    def test_read_only_generates_nothing(self):
        """max_tokens=0 means read the prompt and stop: no DATA frame at all."""
        engine = Engine()
        try:
            frames, _ = engine.submit(1, PROMPT, 0, " logprobs=1")
        finally:
            engine.close()
        self.assertFalse([f for f in frames if f.startswith("DATA")],
                         "a read-only request still generated a token")
        self.assertTrue([f for f in frames if f.startswith("ECHO")],
                        "a read-only request read nothing")

    def test_read_only_needs_the_channel(self):
        """max_tokens=0 without logprobs stays a malformed request."""
        engine = Engine()
        try:
            frames, _ = engine.submit(1, PROMPT, 0)
        finally:
            engine.close()
        self.assertTrue([f for f in frames if f.startswith("ERROR")],
                        "max_tokens=0 was accepted without the logprobs channel")


if __name__ == "__main__":
    unittest.main()
