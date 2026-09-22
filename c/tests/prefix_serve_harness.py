"""Shared contract for KV prefix reuse, exercised through the serve protocol.

Reusing a previous turn's attention state must change the timing and nothing
else. If it alters even one token the engine is answering from a state that
belongs to a different conversation, and the reply still reads plausibly --
nothing else in the tree would catch it. So the gate is: the same two turns on
one engine must produce what the second turn alone produces on a cold one.

The scenarios live here rather than in each engine's file because the contract
is the same for every engine, and a contract copied per engine drifts per
engine. Each engine's test module supplies `spawn()` and its skip conditions;
everything below is identical for all of them.

The turns ask for ONE token each unless a scenario says otherwise. That is not
timidity: with max_tok=1 no sampled token is ever fed back, so the recorded
prefix is exactly the prompt and the next turn's prompt can be built as plain
text. Rebuilding it from the reply instead goes through decode-then-encode,
which a fixture whose vocabulary is wider than the byte range cannot round-trip
-- the prompt then diverges and the harness fault looks exactly like an engine
fault. One scenario does replay the reply and asserts the property that holds
either way: reuse or no reuse, the answer must match a cold engine's.

The turn being COMPARED always asks for eight. One token is not a gate: a
deliberately broken reuse that kept the attention rows and zeroed the recurrent
state still produced the same single argmax on the tiny Qwen3.6 fixture, and
only showed up once there were several tokens to disagree about.
"""
import json
import os
import subprocess
import threading
import time

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


def ensure_byte_tokenizer(fixture, size=256):
    """Give a fixture that ships none a one-token-per-byte tokenizer.

    Some tiny fixtures carry weights and a teacher-forcing oracle but no
    tokenizer, because the oracle path feeds token ids directly. Serve mode goes
    through text, so it needs one.
    """
    path = fixture / "tokenizer.json"
    if path.exists():
        return
    path.write_text(json.dumps({
        "version": "1.0",
        "added_tokens": [],
        "model": {"type": "BPE", "vocab": byte_level_vocab(size), "merges": []},
    }), encoding="utf-8")


class ServeEngine:
    """An engine in serve mode, speaking the protocol in docs/serve_protocol.md."""

    def __init__(self, engine, argv, env, log_prefix=True, reuse=True):
        env = dict(os.environ, **env)
        if log_prefix:
            env["COLI_PREFIX_LOG"] = "1"
        if not reuse:
            env["COLI_KV_PREFIX"] = "0"
        self.p = subprocess.Popen([str(engine)] + [str(a) for a in argv], env=env,
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, bufsize=0)
        # Drain stderr continuously: reading it only at close() loses whatever
        # the engine wrote after the pipe filled, and an empty capture is
        # indistinguishable from an engine that said nothing -- exactly the
        # ambiguity that makes a first failure here unreadable.
        self._err = []
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()
        deadline = time.time() + 300
        while time.time() < deadline:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before READY:\n"
                                   + "".join(self._err)[-2000:])
            if READY.strip() in line:
                return
        raise RuntimeError("engine never reported READY")

    def ask(self, rid, prompt, max_tok=1):
        """Returns the generated payload as BYTES.

        Never as str: a random-init fixture emits arbitrary byte sequences that
        are mostly not valid UTF-8, so decoding with errors="replace" and
        re-encoding yields entirely different bytes.
        """
        assert isinstance(prompt, bytes), "prompts stay bytes end to end"
        header = f"SUBMIT {rid} 0 {len(prompt)} {max_tok} 0 1\n".encode()
        self.p.stdin.write(header + prompt + b"\n")
        self.p.stdin.flush()
        chunks = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-request:\n"
                                   + "".join(self._err)[-2000:])
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


class PrefixReuseContract:
    """The five scenarios. Mix into a TestCase that provides spawn(**kwargs).

    Deliberately not a TestCase itself: unittest would collect and run the
    scenarios once more with no engine behind them.
    """

    # An engine whose fixture round-trips every token id through text can demand
    # the stronger property in the replay scenario below: that reuse actually
    # FIRES when the client appends the reply. A fixture with more ids than the
    # byte range cannot, because the reply comes back as different text and the
    # prompt legitimately diverges.
    REPLAY_REUSES = False

    def spawn(self, log_prefix=True, reuse=True):   # pragma: no cover - overridden
        raise NotImplementedError

    def test_reused_prefix_yields_identical_tokens(self):
        warm = self.spawn()
        opening = b"The capital of France is"
        warm.ask("1", opening)
        # Turn 2 EXTENDS what the state already holds -- the shape a chat client
        # produces when it resends the transcript with more appended.
        second = opening + b" a question about geography, and the capital of Spain is"
        reused = warm.ask("2", second, max_tok=8)
        log = warm.close()

        cold = self.spawn(log_prefix=False)
        fresh = cold.ask("1", second, max_tok=8)
        cold.close()

        self.assertIn("[PREFIX] reusing", log,
                      "the second turn did not reuse the first turn's state; "
                      "this test proves nothing unless it does.\n"
                      "engine said:\n" + (log or "(nothing on stderr)"))
        self.assertEqual(reused, fresh,
                         "reusing the prefix changed the output -- the engine "
                         "answered from a state that is not this conversation")

    def test_three_turns_keep_reusing(self):
        """Each turn holds a longer prefix than the last. On an engine that
        grows its cache this is also the test that covers the growth path: the
        rows and the record must survive the reallocation, per head, or the
        third turn answers from rows that belong to other positions."""
        warm = self.spawn()
        p1 = b"Streaming experts from disk"
        p2 = p1 + b" means the decode rate depends on the drive"
        p3 = p2 + b" and on how many experts stay resident in memory"
        warm.ask("1", p1)
        warm.ask("2", p2)
        third = warm.ask("3", p3, max_tok=8)
        log = warm.close()

        cold = self.spawn(log_prefix=False)
        fresh = cold.ask("1", p3, max_tok=8)
        cold.close()

        self.assertEqual(log.count("[PREFIX] reusing"), 2,
                         "expected turns 2 and 3 to reuse:\n" + log)
        self.assertEqual(third, fresh, "reuse across three turns changed the output")

    def test_reuse_matches_the_same_engine_with_reuse_disabled(self):
        """The A/B the kill switch exists for: same process lifetime, same two
        turns, reuse on and off. Stricter than the cold-engine comparison
        because it also holds the expert cache and the router state constant."""
        opening = b"Streaming experts from disk means"
        extended = opening + b" that the decode rate then depends on"

        warm = self.spawn()
        warm.ask("1", opening)
        with_reuse = warm.ask("2", extended, max_tok=8)
        warm.close()

        off = self.spawn(reuse=False)
        off.ask("1", opening)
        without = off.ask("2", extended, max_tok=8)
        log = off.close()

        self.assertNotIn("[PREFIX] reusing", log,
                         "COLI_KV_PREFIX=0 did not disable reuse")
        self.assertEqual(with_reuse, without,
                         "reuse changed the answer on the same engine")

    def test_a_diverging_prompt_is_not_reused(self):
        """The rejection path matters as much as the reuse path: a prompt that
        shares no prefix must start over, not splice itself onto stale state."""
        other = b"Completely different opening text here"
        eng = self.spawn()
        eng.ask("1", b"The capital of France is")
        diverged = eng.ask("2", other, max_tok=8)
        log = eng.close()

        cold = self.spawn(log_prefix=False)
        fresh = cold.ask("1", other, max_tok=8)
        cold.close()

        self.assertIn("[PREFIX] no reuse", log,
                      "a divergent prompt should have reported why it could not "
                      "reuse:\n" + (log or "(nothing on stderr)"))
        self.assertEqual(diverged, fresh,
                         "a diverging prompt was contaminated by the previous turn")

    def test_a_shorter_prompt_after_a_longer_one(self):
        """The record only ever grows its length, so a miss has to CLEAR it
        rather than overwrite it: a shorter prompt writes fewer positions than
        the last turn recorded, and a stale tail would claim coverage the cache
        no longer has -- which the next turn would then happily reuse."""
        short = b"Short prompt"
        eng = self.spawn()
        eng.ask("1", b"A considerably longer opening than the one that follows it")
        eng.ask("2", short)
        after = eng.ask("3", short + b" extended once more", max_tok=8)
        eng.close()

        cold = self.spawn(log_prefix=False)
        fresh = cold.ask("1", short + b" extended once more", max_tok=8)
        cold.close()

        self.assertEqual(after, fresh,
                         "a stale record survived a shorter prompt and was reused")

    def test_a_turn_that_replays_the_reply_is_never_wrong(self):
        """The real chat shape: the client appends the assistant's reply and a
        new question. Whether the reply survives this fixture's decode-encode
        round trip decides whether reuse FIRES, and on a fixture with more token
        ids than bytes that is allowed to vary -- what may never vary is the
        answer. Where it does round-trip (REPLAY_REUSES), firing is required:
        this is the one scenario whose prefix contains GENERATED positions, so
        it is what holds an engine to keeping its record honest about them."""
        warm = self.spawn()
        opening = b"The capital of France is"
        reply = warm.ask("1", opening, max_tok=8)
        second = opening + reply + b" and the capital of Spain is"
        got = warm.ask("2", second, max_tok=8)
        log = warm.close()

        cold = self.spawn(log_prefix=False)
        fresh = cold.ask("1", second, max_tok=8)
        cold.close()

        self.assertEqual(got, fresh, "replaying the reply changed the answer")
        if self.REPLAY_REUSES:
            self.assertIn("[PREFIX] reusing", log,
                          "the turn that replays the reply did not reuse: the "
                          "record does not describe the generated positions.\n"
                          "engine said:\n" + (log or "(nothing on stderr)"))
