"""End-to-end proof that KV prefix reuse changes nothing but the time.

Two turns on one engine (the second reusing the first's state) must produce the
SAME tokens as the second turn alone on a cold engine. That is the whole licence
to ship the optimisation: if reuse alters even one token it is answering from a
state that belongs to a different conversation, and the reply would still look
plausible — nothing else in the tree would catch it.

Runs against the tiny random-init fixture the oracle job already builds
(tools/make_tiny_inkling.py), so it needs no checkpoint and no fast disk. The
real 469 GB model cannot serve this test on a developer machine: measured here,
a single token pulled 79 GB off disk in 9 minutes and had not finished. That is
why this gate lives in CI on a fixture rather than in a benchmark on hardware.
"""
import json
import os
import subprocess
import threading
import sys
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("inkling.exe" if sys.platform == "win32" else "inkling")
FIXTURE = Path(os.environ.get("INKLING_TINY", HERE / "tiny_inkling"))
MAXTOK = 4
READY = b"\x01\x01READY\x01\x01\n"


def byte_level_vocab(size):
    """The GPT-2 byte->unicode map that tok.h's tk_build_bytemap builds.

    Bytes 33..126, 161..172 and 174..255 keep their own codepoint; the rest are
    renumbered from 256 upward in byte order. Reproduced here rather than
    imported so a drift between the two shows up as a failing test.
    """
    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    vocab, spare = {}, 0
    for b in range(size):
        if b in direct:
            cp = b
        else:
            cp, spare = 256 + spare, spare + 1
        vocab[chr(cp)] = b
    return vocab


def ensure_tokenizer(fixture):
    """tools/make_tiny_inkling.py emits weights and a teacher-forcing oracle but
    no tokenizer: the oracle path feeds token ids directly. Serve mode goes
    through text, so it needs one. The fixture model is vocab_size=256 /
    unpadded 250, so one token per byte is not a simplification — it is the
    whole vocabulary."""
    path = fixture / "tokenizer.json"
    if path.exists():
        return
    path.write_text(json.dumps({
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": byte_level_vocab(250), "merges": []},
    }), encoding="utf-8")


class Engine:
    """A serve-mode inkling, speaking the protocol in docs/serve_protocol.md."""

    def __init__(self, log_prefix=True):
        env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1", NGEN=str(MAXTOK))
        if log_prefix:
            env["INK_PREFIX_LOG"] = "1"
        self.p = subprocess.Popen([str(ENGINE), "8"], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        # Drain stderr continuously. Reading it only at close() loses whatever
        # the engine wrote after the pipe filled, and an empty capture is
        # indistinguishable from an engine that said nothing -- which is exactly
        # the ambiguity that made the first failure here unreadable.
        self._err = []
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()
        deadline = time.time() + 300
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                err = self.p.stderr.read().decode(errors="replace")
                raise RuntimeError(f"engine exited before READY:\n{err[-2000:]}")
            if READY.strip() in line:
                return
        raise RuntimeError("engine never reported READY")

    def ask(self, rid, prompt):
        """Returns the generated payload as BYTES.

        Never as str. A random-init model emits arbitrary byte sequences that
        are mostly not valid UTF-8, so decoding with errors="replace" turns them
        into U+FFFD and re-encoding yields entirely different bytes. Feeding
        that back as the next turn's prefix made the prompt diverge from the
        state the engine actually held -- a fault in the harness that looked
        exactly like a fault in the engine.
        """
        assert isinstance(prompt, bytes), "prompts stay bytes end to end"
        header = f"SUBMIT {rid} 0 {len(prompt)} {MAXTOK} 0 1\n".encode()
        self.p.stdin.write(header + prompt + b"\n")
        self.p.stdin.flush()
        chunks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-request")
            text = line.decode("latin-1").rstrip("\n")
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                chunks.append(self.p.stdout.read(int(text.split()[2])))
                self.p.stdout.readline()          # the newline after the payload
            elif kind in ("DONE", "END"):
                return b"".join(chunks)
            elif kind == "ERROR":
                raise RuntimeError(f"engine error: {text}")

    def _drain(self):
        for line in iter(self.p.stderr.readline, b""):
            self._err.append(line.decode(errors="replace"))

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except Exception:
            self.p.kill()
        self._pump.join(timeout=10)
        return "".join(self._err)


@unittest.skipUnless(ENGINE.exists(), "inkling is not built")
@unittest.skipUnless((FIXTURE / "config.json").exists(),
                     "tiny inkling fixture is absent (tools/make_tiny_inkling.py)")
class InklingPrefixServeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_tokenizer(FIXTURE)

    def test_byte_level_vocab_matches_the_engines_map(self):
        """Guard the assumption the two tests below rest on: if this map ever
        disagrees with tok.h, they would fail for a reason that has nothing to
        do with prefix reuse, and the failure would be very hard to read."""
        vocab = byte_level_vocab(250)
        self.assertEqual(len(vocab), 250, "the map must be injective")
        self.assertEqual(vocab["A"], 65, "printable ASCII keeps its own byte")
        self.assertEqual(vocab["~"], 126, "top of the direct range")
        self.assertEqual(vocab[chr(256)], 0, "byte 0 is the first renumbered one")
        self.assertEqual(vocab[chr(256 + 32)], 32, "space is renumbered, not literal")

    def test_reused_prefix_yields_identical_tokens(self):
        warm = Engine()
        opening = b"The capital of France is"
        first = warm.ask("1", opening)
        # Turn 2 EXTENDS what the state already holds — the shape a chat client
        # produces when it resends the transcript with a new question appended.
        second_prompt = opening + first + b" and the capital of Spain is"
        reused = warm.ask("2", second_prompt)
        log = warm.close()

        cold = Engine(log_prefix=False)
        fresh = cold.ask("1", second_prompt)
        cold.close()

        self.assertIn("[PREFIX] reusing", log,
                      "the second turn did not reuse the first turn's state; "
                      "this test proves nothing unless it does.\n"
                      "engine said:\n" + (log or "(nothing on stderr)"))
        self.assertEqual(reused, fresh,
                         "reusing the prefix changed the output — the engine "
                         "answered from a state that is not this conversation")

    def test_a_diverging_prompt_is_not_reused(self):
        """The rejection path matters as much as the reuse path: a prompt that
        shares no prefix must start over, not splice itself onto stale state."""
        eng = Engine()
        eng.ask("1", b"The capital of France is")
        diverged = eng.ask("2", b"Completely different opening text here")
        log = eng.close()

        cold = Engine(log_prefix=False)
        fresh = cold.ask("1", b"Completely different opening text here")
        cold.close()

        self.assertEqual(diverged, fresh,
                         "a diverging prompt was contaminated by the previous turn")


if __name__ == "__main__":
    unittest.main()
