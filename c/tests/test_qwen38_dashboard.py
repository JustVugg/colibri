"""The dashboard's Brain tab reads two lines from the engine's stdout: EMAP
(the grid, after READY and STAT and again after every turn) and HITS (which
experts the turn routed, after DONE next to PROF). qwen38.c emitted neither,
so on Qwen3.8 the tab was empty. The chat page's live map reads a third, HITL
(the experts routed since the previous one, after the tokens it wrote). This
test drives the engine over stdio the way the gateway does and reads the turn
to its last line.

It needs a Qwen3.8 tiny fixture with a tokenizer.json and a built `qwen38`.
Set QWEN38_TINY to the fixture; without it the test is skipped with the
reason on the record. CI uses the oracle fixture and adds the byte tokenizer.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen38.exe" if sys.platform == "win32" else "qwen38")
FIXTURE = Path(os.environ.get("QWEN38_TINY", ""))
MAXTOK = 3


def read_until(p, kind, limit=400):
    """Lines up to and including the first `kind`; DATA payloads consumed."""
    lines = []
    for _ in range(limit):
        line = p.stdout.readline()
        if not line:
            raise RuntimeError("engine closed: " + p.stderr.read().decode(errors="replace")[-2000:])
        text = line.decode("latin-1").rstrip("\n")
        if text.split(" ", 1)[0] == "DATA":
            p.stdout.read(int(text.split()[2]))
            p.stdout.readline()
            continue
        lines.append(text)
        if text.startswith(kind + " ") or text.startswith(kind):
            return lines
    raise RuntimeError(f"no {kind} within {limit} lines:\n" + "\n".join(lines[-10:]))


def one_turn(prompt=b"abcabcabc", **extra):
    """stdin stays open for the whole turn: the engine polls it per token and
    an EOF there is a cancel, which is what a `printf | ./qwen38` pipe gets."""
    env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1", **extra)
    p = subprocess.Popen([str(ENGINE), "1", "8"], env=env, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    try:
        read_until(p, "\x01\x01READY")
        boot = read_until(p, "EMAP")
        p.stdin.write(f"SUBMIT 7 0 {len(prompt)} {MAXTOK} 0 1\n".encode() + prompt + b"\n")
        p.stdin.flush()
        turn = read_until(p, "HITS")
        after = read_until(p, "EMAP")
        return boot, turn, after
    finally:
        p.stdin.close()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


@unittest.skipUnless(ENGINE.exists(), "qwen38 is not built")
@unittest.skipUnless((FIXTURE / "config.json").is_file() and (FIXTURE / "tokenizer.json").is_file(),
                     "QWEN38_TINY not set to a Qwen3.8 fixture with a tokenizer")
class Qwen38DashboardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boot, cls.turn, cls.after = one_turn()
        cls.config = json.loads((FIXTURE / "config.json").read_text())

    @staticmethod
    def only(lines, kind):
        found = [l.split() for l in lines if l.startswith(kind + " ")]
        assert len(found) == 1, f"expected exactly one {kind}: {found}"
        return found[0]

    def test_grid_after_ready_and_stat(self):
        """The boot reader consumes READY then STAT; EMAP must come after
        both, and is one row per layer (every layer is MoE here)."""
        kinds = [l.split(" ", 1)[0] for l in self.boot]
        self.assertLess(kinds.index("STAT"), kinds.index("EMAP"))
        _, rows, cols, hexs = self.only(self.boot, "EMAP")
        self.assertEqual(int(rows), self.config["num_hidden_layers"])
        self.assertEqual(int(cols), self.config["num_experts"])
        self.assertEqual(len(hexs), int(rows) * int(cols) * 2)

    def test_grid_refreshed_after_the_turn_shows_residents(self):
        """Tier bit 6: after a turn the cache holds experts, so the grid is no
        longer all zero (cap 1 keeps one per layer)."""
        _, _, _, hexs = self.only(self.after, "EMAP")
        self.assertNotEqual(int(hexs, 16), 0, "no expert resident after a turn")

    def test_hits_matches_the_grid(self):
        kinds = [l.split(" ", 1)[0] for l in self.turn]
        self.assertLess(kinds.index("DONE"), kinds.index("HITS"))
        _, rows, cols, hexs = self.only(self.turn, "HITS")
        _, erows, ecols, _ = self.only(self.boot, "EMAP")
        self.assertEqual((rows, cols), (erows, ecols))
        self.assertEqual(len(hexs), ((int(rows) * int(cols) + 7) // 8) * 2)
        self.assertNotEqual(int(hexs, 16), 0, "a 9-token prompt routed no expert at all")

    def test_live_frames_during_the_turn(self):
        """HITL frames come with the tokens, before DONE: four hex digits an expert
        (layer * experts + expert), each one the turn's HITS has too."""
        kinds = [l.split(" ", 1)[0] for l in self.turn]
        frames = [l.split() for l in self.turn if l.startswith("HITL ")]
        self.assertTrue(frames, "no HITL frame in the turn")
        self.assertLess(kinds.index("HITL"), kinds.index("DONE"))
        _, rows, cols, hexs = self.only(self.turn, "HITS")
        n = int(rows) * int(cols)
        bits = bytes.fromhex(hexs)
        for f in frames:
            self.assertEqual(len(f), 2)
            self.assertEqual(len(f[1]) % 4, 0)
            for k in range(0, len(f[1]), 4):
                i = int(f[1][k:k + 4], 16)
                self.assertLess(i, n)
                self.assertTrue(bits[i >> 3] >> (i & 7) & 1, f"expert {i} lit but not in the turn's HITS")

    def test_live_frames_off(self):
        _, turn, _ = one_turn(COLI_HITS_LIVE="0")
        self.assertFalse([l for l in turn if l.startswith("HITL ")])
        self.assertTrue([l for l in turn if l.startswith("HITS ")])

    def test_prof_is_still_there(self):
        prof = self.only(self.turn, "PROF")
        self.assertEqual(len(prof), 10)
        self.assertEqual(int(prof[3]), MAXTOK)


if __name__ == "__main__":
    unittest.main()
