"""Legacy run_serve refuses over-context prompts in both arms, never silently truncating.

The legacy protocol (what `coli chat` speaks, and the pre-mux `\\x02PROMPT` API
mode) has no ERROR frame: on an over-context turn the engine emits an empty
answer plus `STAT ... CONTEXT_EXCEEDED <cause>` (#2097). This drives the real
`colibri` binary with SERVE=1 against the tiny GLM oracle model
(`c/tools/make_glm_oracle.py`) with a small CTX and asserts the refusal in
both modes, the cause token for a filled history, and that a prompt that fits
is still answered.

Needs a built `colibri` and COLI_ORACLE_FIXTURE pointing at a directory with
the oracle's model.safetensors + config.json. The byte-level tokenizer.json is
written into a scratch copy: the oracle ships none, and one token per byte
keeps the token math exact.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
BINARY = next((path for path in (HERE / "colibri.exe", HERE / "colibri") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_ORACLE_FIXTURE", ""))
CTX = 64
NGEN = 4
OVER_LONG = b"abc" * 40          # 120 bytes -> 120 tokens, far past CTX


def byte_tokenizer():
    """One token per byte; byte 0 is skipped (the C JSON parser rejects NUL keys)."""
    return {"version": "1.0",
            "model": {"type": "BPE",
                      "vocab": {chr(b): b for b in range(1, 256)},
                      "merges": []},
            "added_tokens": []}


class ServeSession:
    """One SERVE=1 engine process speaking the legacy byte protocol."""

    def __init__(self, snap):
        keep = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE",
                "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
        env = {k: v for k, v in os.environ.items() if k in keep}
        env.update(SNAP=str(snap), SERVE="1", CTX=str(CTX), NGEN=str(NGEN),
                   COLI_TEMP="0", OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1")
        self.process = subprocess.Popen([str(BINARY), "0"], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, bufsize=0)
        while b"READY" not in self.readline():
            pass
        self.readline()  # startup STAT
        self.readline()  # TIERS

    def readline(self):
        line = self.process.stdout.readline()
        if not line:
            raise AssertionError("the engine closed the stream mid-turn")
        return line

    def turn(self, payload):
        """Send one interactive line; return (answer bytes, STAT fields)."""
        self.process.stdin.write(payload + b"\n")
        self.process.stdin.flush()
        return self._read_turn()

    def api_turn(self, prompt):
        """Send one \\x02PROMPT frame (the pre-mux API mode)."""
        self.process.stdin.write(f"\x02PROMPT {len(prompt)} {NGEN} 0.0 1.0\n".encode()
                                 + prompt + b"\n")
        self.process.stdin.flush()
        return self._read_turn()

    def _read_turn(self):
        answer = b""
        for _ in range(400):
            line = self.readline()
            if b"\x01\x01END\x01\x01" in line:
                break
            answer += line
        else:
            raise AssertionError("no END sentinel within 400 lines")
        stat = self.readline().decode("utf-8", "replace").strip().split()
        assert stat[0] == "STAT", stat
        return answer, stat

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(60)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


@unittest.skipUnless(BINARY, "colibri engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "model.safetensors").is_file()
                     and (FIXTURE / "config.json").is_file(),
                     "COLI_ORACLE_FIXTURE not set to a make_glm_oracle.py output dir")
class RunServeContextExceededTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = tempfile.TemporaryDirectory(prefix="colibri-ctx-")
        snap = Path(cls.scratch.name)
        shutil.copy2(FIXTURE / "model.safetensors", snap / "model.safetensors")
        shutil.copy2(FIXTURE / "config.json", snap / "config.json")
        (snap / "tokenizer.json").write_text(json.dumps(byte_tokenizer(), ensure_ascii=True),
                                             encoding="utf-8")
        cls.snap = snap

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def session(self):
        session = ServeSession(self.snap)
        self.addCleanup(session.close)
        return session

    def test_interactive_over_long_prompt_is_refused(self):
        answer, stat = self.session().turn(OVER_LONG)
        self.assertFalse(answer.strip(), "an over-context turn must have no answer")
        self.assertEqual(stat[7], "CONTEXT_EXCEEDED", stat)
        self.assertEqual(stat[8], "PROMPT_TOO_LONG", stat)

    def test_api_mode_over_long_prompt_is_refused(self):
        answer, stat = self.session().api_turn(OVER_LONG)
        self.assertFalse(answer.strip(), "an over-context turn must have no answer")
        self.assertEqual(stat[7], "CONTEXT_EXCEEDED", stat)
        self.assertEqual(stat[8], "PROMPT_TOO_LONG", stat)

    def test_a_filled_history_is_refused_with_the_reset_cause(self):
        session = self.session()
        answer, stat = session.turn(b"hi")
        self.assertTrue(answer.strip(), "a fitting turn must be answered")
        self.assertNotIn("CONTEXT_EXCEEDED", stat)
        # The second identical turn no longer fits CTX: refuse, do not wipe history.
        answer, stat = session.turn(b"hi")
        self.assertFalse(answer.strip(), "an over-context turn must have no answer")
        self.assertEqual(stat[7], "CONTEXT_EXCEEDED", stat)
        self.assertEqual(stat[8], "HISTORY_FULL", stat)

    def test_a_prompt_that_fits_is_still_answered(self):
        answer, stat = self.session().turn(b"hi")
        self.assertTrue(answer.strip(), "a fitting turn must be answered")
        self.assertNotIn("CONTEXT_EXCEEDED", stat)


if __name__ == "__main__":
    unittest.main()
