"""env_for: default Windows misurati (DIRECT/PIPE/PILOT_REAL + blocco OMP).

Carica `coli` come modulo (ha la guardia __main__) e verifica il contratto:
- win32: i tre default I/O e il blocco OMP sono setdefault
- un override esplicito dell'utente vince sempre
- COLI_NO_OMP_TUNE spegne SOLO il blocco OMP, non i default I/O
- non-win32: env_for non tocca nulla di tutto questo
"""
import importlib.machinery
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_loader = importlib.machinery.SourceFileLoader("coli_cli", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


def args(**over):
    base = dict(model="X", policy="quality", ram=0, ngen=0, topp=0, topk=0,
                temp=None, repin=0, ctx=0, auto_tier=False, gpu=None, vram=0)
    base.update(over)
    return types.SimpleNamespace(**base)


class EnvDefaultsTest(unittest.TestCase):
    def env_for_with(self, environ, platform):
        with mock.patch.dict(os.environ, environ, clear=True), \
             mock.patch.object(sys, "platform", platform):
            return coli.env_for(args())

    def test_win32_sets_measured_defaults(self):
        e = self.env_for_with({}, "win32")
        self.assertEqual(e["DIRECT"], "1")
        self.assertEqual(e["PIPE"], "1")
        self.assertEqual(e["PILOT_REAL"], "1")
        self.assertEqual(e["OMP_WAIT_POLICY"], "active")
        self.assertNotIn("OMP_PROC_BIND", e)  # MinGW libgomp: niente affinity

    def test_explicit_override_wins(self):
        e = self.env_for_with({"DIRECT": "0", "PIPE": "0"}, "win32")
        self.assertEqual(e["DIRECT"], "0")
        self.assertEqual(e["PIPE"], "0")
        self.assertEqual(e["PILOT_REAL"], "1")  # non overridden -> default

    def test_kill_switch_scope_is_omp_only(self):
        e = self.env_for_with({"COLI_NO_OMP_TUNE": "1"}, "win32")
        self.assertNotIn("OMP_WAIT_POLICY", e)
        self.assertNotIn("OMP_NUM_THREADS", e)
        self.assertEqual(e["DIRECT"], "1")   # i default I/O restano attivi
        self.assertEqual(e["PIPE"], "1")

    def test_non_windows_untouched(self):
        e = self.env_for_with({}, "linux")
        for k in ("DIRECT", "PIPE", "PILOT_REAL", "OMP_WAIT_POLICY"):
            self.assertNotIn(k, e)


if __name__ == "__main__":
    unittest.main()
