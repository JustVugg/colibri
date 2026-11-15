"""qwen36 holds a GGUF's dense IQ3_S matrices as 4-bit codes for the CPU (Q36_IQ3S_CODES, #2704).

A small qwen35 dense GGUF whose DeltaNet projections are IQ3_S in three layers (both of layer 0,
layer 1's qkv, layer 2's gate) runs on the CPU with the codes and with the file's rows: the engine
says it holds them as codes only in the first run, both runs give the same tokens, and every
layer's residual of the last token (DUMP_LAYERS) stays within 1e-4 of the other run's. The
kernels' bits against their reference are tests/test_gq_r16.c's.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import numpy as np
    import gguf  # noqa: F401
    from test_gguf_import import write_model
except ImportError:  # pragma: no cover
    write_model = None

HELD = "IQ3_S matrices the CPU computes held as codes"


@unittest.skipUnless(ENGINE.exists(), "qwen36 is not built")
@unittest.skipUnless(write_model is not None, "the gguf package is not installed")
class Qwen36Iq3sCodesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="q36_iq3s_"))
        gg = cls.tmp / "m.gguf"
        write_model(gg, np.random.default_rng(23), dense=True,
                    dn_types={0: ("IQ3_S", "IQ3_S"), 1: ("IQ3_S", "Q8_0"), 2: ("Q8_0", "IQ3_S")})
        cls.model = cls.tmp / "m"
        r = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(gg), str(cls.model)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("import failed:\n" + r.stdout + r.stderr)
        (cls.model / "ref.json").write_text(
            '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_engine(self, codes):
        """(tokens, every layer's residual of the last token, the log) for one run"""
        dump = self.tmp / f"layers{codes}.f32"
        env = dict(os.environ, SNAP=str(self.model), OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1",
                   USAGE_SAVE="0", COLI_CUDA="0", COLI_VULKAN="0", Q36_IQ3S_CODES=codes, DUMP_LAYERS=str(dump))
        r = subprocess.run([str(ENGINE), "2", "8", str(self.model / "ref.json")], env=env,
                           capture_output=True, text=True, errors="replace", timeout=300)
        log = r.stdout + r.stderr
        tokens = re.findall(r"^C engine : (.*?)(?: +Text|$)", log, re.M)
        self.assertTrue(tokens, "no token line:\n" + log[-2000:])
        return tokens[-1].split(), np.fromfile(dump, dtype=np.float32), log

    def test_codes_give_the_file_rows_answer(self):
        with_codes, layers1, log1 = self.run_engine("1")
        rows, layers0, log0 = self.run_engine("0")
        if HELD not in log1 and "kept in the file's rows" not in log1:
            self.skipTest("this build has no interleaved kernels (no AVX2): the codes are never made")
        self.assertIn(HELD, log1, log1[-2000:])
        self.assertNotIn(HELD, log0, log0[-2000:])
        self.assertEqual(with_codes, rows, "the codes changed the tokens")
        # the layers' residuals: the same integers, other float roundings (test_gq_r16's gap)
        self.assertEqual(layers1.size, layers0.size)
        self.assertGreater(layers0.size, 0)
        gap = float(np.linalg.norm(layers1 - layers0) / max(np.linalg.norm(layers0), 1e-30))
        self.assertLess(gap, 1e-4, f"the residuals moved by {gap:.3e} of their norm")


if __name__ == "__main__":
    unittest.main()
