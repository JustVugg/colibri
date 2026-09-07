"""The dashboard's Brain tab lights up from a HITS line the engine prints
after every turn: which experts the turn routed, one bit each, packed 8 per
hex pair, over the same rows and columns as EMAP. olmoe.c printed EMAP (the
grid) but never HITS, so on OLMoE the grid stayed dark. This test drives the
engine over stdio the way the gateway does and reads the turn to its last
line.

It needs a converted OLMoE container with a tokenizer.json and a built
`olmoe`. Set OLMOE_TINY to the container; without it the test is skipped
with the reason on the record. CI builds the tiny checkpoint from
tools/make_olmoe_tiny.py, converts it, and adds the byte tokenizer.
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if sys.platform == "win32" else "olmoe")
FIXTURE = Path(os.environ.get("OLMOE_TINY", ""))
MAXTOK = 3


def one_turn(prompt=b"abcabcabc"):
    """Run the engine in serve mode, submit one prompt, return the protocol
    lines of the turn (DATA payloads consumed) up to and including HITS.
    stdin stays open for the whole turn: the engine polls it per token and an
    EOF there is a cancel, which is what a `printf | ./olmoe` pipe gets."""
    env = dict(os.environ, SNAP=str(FIXTURE), SERVE="1")
    p = subprocess.Popen([str(ENGINE), "4", "8"], env=env, stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    try:
        while True:
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before READY: "
                                   + p.stderr.read().decode(errors="replace")[-2000:])
            if b"READY" in line:
                break
        p.stdin.write(f"SUBMIT 7 0 {len(prompt)} {MAXTOK} 0 1\n".encode() + prompt + b"\n")
        p.stdin.flush()
        lines = []
        for _ in range(200):
            line = p.stdout.readline()
            if not line:
                raise RuntimeError("engine closed mid-turn: "
                                   + p.stderr.read().decode(errors="replace")[-2000:])
            text = line.decode("latin-1").rstrip("\n")
            kind = text.split(" ", 1)[0]
            if kind == "DATA":
                p.stdout.read(int(text.split()[2]))
                p.stdout.readline()
                continue
            lines.append(text)
            if kind == "HITS":
                return lines
        raise RuntimeError("no HITS within 200 lines after SUBMIT:\n" + "\n".join(lines[-10:]))
    finally:
        p.stdin.close()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()


@unittest.skipUnless(ENGINE.exists(), "olmoe is not built")
@unittest.skipUnless((FIXTURE / "config.json").is_file() and (FIXTURE / "tokenizer.json").is_file(),
                     "OLMOE_TINY not set to a converted OLMoE container with a tokenizer")
class OlmoeDashboardHitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lines = one_turn()
        cls.config = json.loads((FIXTURE / "config.json").read_text())

    def line(self, kind):
        found = [l for l in self.lines if l.startswith(kind + " ")]
        self.assertEqual(len(found), 1, f"expected exactly one {kind} line, got {found}")
        return found[0].split()

    def test_hits_closes_the_turn_after_done(self):
        kinds = [l.split(" ", 1)[0] for l in self.lines]
        self.assertIn("DONE", kinds)
        self.assertLess(kinds.index("DONE"), kinds.index("HITS"),
                        "HITS belongs to the turn it describes: after DONE, with PROF")

    def test_hits_has_the_grid_shape(self):
        """Rows and columns must be EMAP's, or the Brain tab paints the bits on
        the wrong cells: every layer here is a MoE layer."""
        _, rows, cols, hexs = self.line("HITS")
        self.assertEqual(int(rows), self.config["num_hidden_layers"])
        self.assertEqual(int(cols), self.config["num_experts"])
        self.assertEqual(len(hexs), ((int(rows) * int(cols) + 7) // 8) * 2)
        self.assertNotEqual(int(hexs, 16), 0,
                            "a 9-token prompt through MoE layers routed no expert at all")

    def test_prof_still_has_its_ten_fields(self):
        prof = self.line("PROF")
        self.assertEqual(len(prof), 10)
        self.assertEqual(int(prof[3]), MAXTOK, "completion tokens")
        self.assertEqual(int(prof[9]), MAXTOK + 1,
                         "prefill plus one forward per token: this engine steps the last one too")


if __name__ == "__main__":
    unittest.main()
