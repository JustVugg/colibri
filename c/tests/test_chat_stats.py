"""#1475: the chat footer is a rendering of measured generation statistics,
not a hard-coded line. gen_stats() takes the server's exact completion_tokens
when it sent them and falls back to the chars/4 estimate; render_gen_stats()
lays the result out per --stats mode (COLI_CHAT_STATS sets the default)."""
import importlib.machinery
import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

CLI = Path(__file__).resolve().parent.parent / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_stats_t", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["coli"]
    try:
        loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


class ChatStatsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_cli()

    def test_estimate_full(self):
        st = self.m.gen_stats("x" * 1480, 313.0)
        self.assertEqual(st["tokens"], 370)
        self.assertFalse(st["exact"])
        self.assertEqual(self.m.render_gen_stats(st, "full"), "~370 tok · 313s · ~1.18 tok/s")

    def test_exact_full(self):
        st = self.m.gen_stats("x" * 1480, 313.0, exact_tokens=372)
        self.assertTrue(st["exact"])
        self.assertEqual(self.m.render_gen_stats(st, "full"), "372 tok · 313s · 1.19 tok/s")

    def test_compact_and_off(self):
        st = self.m.gen_stats("x" * 1480, 313.0)
        self.assertEqual(self.m.render_gen_stats(st, "compact"), "~370 tok · ~1.18 tok/s")
        self.assertEqual(self.m.render_gen_stats(st, "off"), "")

    def test_interrupted_suffix(self):
        st = self.m.gen_stats("x" * 120, 23.0, interrupted=True)
        self.assertEqual(self.m.render_gen_stats(st, "full"), "~30 tok · 23s · ~1.30 tok/s · ⏹ interrupted")

    def test_zero_elapsed_has_no_rate(self):
        st = self.m.gen_stats("abcd", 0.0)
        self.assertIsNone(st["tps"])
        self.assertEqual(self.m.render_gen_stats(st, "full"), "~1 tok · 0s")

    def test_bool_is_not_an_exact_count(self):
        self.assertFalse(self.m.gen_stats("abcdefgh", 1.0, exact_tokens=True)["exact"])
        self.assertFalse(self.m.gen_stats("abcdefgh", 1.0, exact_tokens=-1)["exact"])

    def test_mode_resolution(self):
        m = self.m
        self.assertEqual(m.chat_stats_mode(SimpleNamespace(stats="compact")), "compact")
        old = os.environ.get("COLI_CHAT_STATS")
        try:
            os.environ["COLI_CHAT_STATS"] = "off"
            self.assertEqual(m.chat_stats_mode(SimpleNamespace(stats=None)), "off")
            self.assertEqual(m.chat_stats_mode(SimpleNamespace(stats="full")), "full")   # the flag wins
            os.environ["COLI_CHAT_STATS"] = "bogus"
            self.assertEqual(m.chat_stats_mode(SimpleNamespace()), "full")
        finally:
            if old is None: os.environ.pop("COLI_CHAT_STATS", None)
            else: os.environ["COLI_CHAT_STATS"] = old


if __name__ == "__main__":
    unittest.main()
