"""The dashboard's Brain tab lights up from a HITS line the engine prints
after every turn: which experts the turn routed, one bit each, packed 8 per
hex pair, over the same rows and columns as EMAP. inkling.c printed EMAP
(the grid, refreshed every turn) but never HITS, so on Inkling the grid
stayed dark. Same harness as the prefix-reuse test, one turn, read to the
last line of the turn.
"""
import unittest

from tests.test_inkling_prefix_serve import ENGINE, FIXTURE, MAXTOK, Engine, ensure_tokenizer


def read_until(engine, kind, limit=200):
    """Lines after the previous read, up to and including the first `kind`."""
    lines = []
    for _ in range(limit):
        line = engine.p.stdout.readline()
        if not line:
            raise RuntimeError("engine closed: " + "".join(engine._err)[-2000:])
        text = line.decode("latin-1").rstrip("\n")
        if text.split(" ", 1)[0] == "DATA":
            engine.p.stdout.read(int(text.split()[2]))
            engine.p.stdout.readline()
            continue
        lines.append(text)
        if text.startswith(kind + " "):
            return lines
    raise RuntimeError(f"no {kind} within {limit} lines:\n" + "\n".join(lines[-10:]))


@unittest.skipUnless(ENGINE.exists(), "inkling is not built")
@unittest.skipUnless((FIXTURE / "config.json").exists(),
                     "tiny inkling fixture is absent (tools/make_tiny_inkling.py)")
class InklingDashboardHitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_tokenizer(FIXTURE)
        engine = Engine(log_prefix=False)
        try:
            cls.boot = read_until(engine, "EMAP")        # the grid, printed once after READY
            engine.ask(1, b"abcabcabc")                  # returns at DONE
            cls.turn = read_until(engine, "HITS")        # PROF, then HITS, then the refreshed grid
        finally:
            engine.close()

    @staticmethod
    def only(lines, kind):
        found = [l.split() for l in lines if l.startswith(kind + " ")]
        assert len(found) == 1, f"expected exactly one {kind}: {found}"
        return found[0]

    def test_hits_matches_the_grid(self):
        """Rows and columns are EMAP's (the sparse layers), or the Brain tab
        paints the bits on the wrong cells."""
        _, rows, cols, hexs = self.only(self.turn, "HITS")
        _, erows, ecols, emap = self.only(self.boot, "EMAP")
        self.assertEqual((rows, cols), (erows, ecols))
        self.assertEqual(len(hexs), ((int(rows) * int(cols) + 7) // 8) * 2)
        self.assertNotEqual(int(hexs, 16), 0,
                            "a 9-token prompt through the sparse layers routed no expert at all")

    def test_prof_counts_forwards_not_tokens(self):
        """A turn stopped by max_tokens does not run a forward for the token it
        never samples: prefill plus one per token after the first."""
        prof = self.only(self.turn, "PROF")
        self.assertEqual(len(prof), 10)
        self.assertEqual(int(prof[3]), MAXTOK, "completion tokens")
        self.assertEqual(int(prof[9]), MAXTOK)


if __name__ == "__main__":
    unittest.main()
