"""A GGUF matrix the CUDA tier holds gives its host rows back, and the CPU reads it back from the
file, the same bytes, if it ever needs it (qwen36.c q36_gq_drop / q36_gq_reload).

Before, every matrix the tier placed kept its host rows too: Qwen3.8-27B UD-Q4_K_XL with its whole
trunk on a 32 GB card held 15.4 GB of RAM it never read again.

tests/test_qwen36_gguf_hostdrop.c runs the whole engine against the fake CUDA backend, which takes
every block format and computes none, on a small dense qwen35 GGUF (one DeltaNet layer of two types,
K-quants the CPU holds interleaved, the tied head):
  - COLI_CUDA=0: the CPU's tokens and last logits, the reference;
  - the tier on, every device matmul failing: every placed matrix gave its rows back, the CPU read
    each one back at its first product, and the tokens and the logits are the reference's, bit for bit;
  - the tier on, the device answering: the rows were given back and none was read back.
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
EXE = HERE / "tests" / ("test_qwen36_gguf_hostdrop.exe" if sys.platform == "win32" else "test_qwen36_gguf_hostdrop")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import numpy as np
    import gguf  # noqa: F401
    from test_gguf_import import write_model
except ImportError:  # pragma: no cover
    write_model = None

LINE = re.compile(r"^HOSTDROP (\w+) private=(\d+) ws=(\d+) dropped=(\d+) bytes=(\d+) reloaded=(\d+)$", re.M)


def run(model, tmp, tag, **env):
    """(tokens, logits bytes, {when: (private, ws, dropped, bytes, reloaded)}, log) of one run"""
    dump = tmp / f"logits-{tag}.f32"
    e = dict(os.environ, SNAP=str(model), OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", USAGE_SAVE="0",
             COLI_VULKAN="0", COLI_CUDA_CHAIN="0", QT_NO_WARMSTART="1", COLI_LOOKUP="0", Q36_OFFER_SHEXP="1", DUMP=str(dump), **env)
    r = subprocess.run([str(EXE), "2", "8", str(model / "ref.json")], env=e, capture_output=True,
                       text=True, errors="replace", timeout=300)
    log = r.stdout + r.stderr   # the exit status is the reference's match (random weights: none)
    tokens = re.findall(r"^C engine : (.*?)(?: +Text|$)", log, re.M)
    if not tokens:
        raise AssertionError(f"{tag}: no token line\n" + log[-3000:])
    lines = {m.group(1): tuple(int(m.group(k)) for k in range(2, 7)) for m in LINE.finditer(log)}
    return tokens[-1].split(), dump.read_bytes(), lines, log


@unittest.skipUnless(EXE.exists(), "tests/test_qwen36_gguf_hostdrop is not built")
@unittest.skipUnless(write_model is not None, "the gguf package is not installed")
class Qwen36GgufHostDropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="q36_hostdrop_"))
        gg = cls.tmp / "m.gguf"
        write_model(gg, np.random.default_rng(31), dense=True, dn_types={1: ("Q8_0", "Q4_K")})
        cls.model = cls.tmp / "m"
        r = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(gg), str(cls.model)],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError("import failed:\n" + r.stdout + r.stderr)
        (cls.model / "ref.json").write_text(
            '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}')
        cls.cpu = run(cls.model, cls.tmp, "cpu", COLI_CUDA="0")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cpu_reads_dropped_rows_back_exactly(self):
        tokens, logits, lines, log = run(self.model, self.tmp, "fail", COLI_CUDA="1", FAKE_MATMUL_FAIL="1")
        self.assertIn("matrices on the CUDA tier gave their host rows back", log, log[-3000:])
        self.assertIn("its rows read back from the file", log, log[-3000:])
        dropped = lines["placed"][2]
        # every matrix of the trunk: 3 DeltaNet layers' qkv, z and out, the attention layer's
        # q, k, v and o, four MLPs' gate, up and down, and the head
        self.assertEqual(dropped, 3 * 3 + 4 + 4 * 3 + 1, log[-3000:])
        self.assertEqual(lines["placed"][4], 0, "a row was read back before any product")
        self.assertEqual(lines["exit"][4], dropped, "every placed matrix ran on the CPU, read back once")
        self.assertEqual(tokens, self.cpu[0])
        self.assertEqual(logits, self.cpu[1], "the read-back rows changed the logits")

    def test_device_run_reads_nothing_back(self):
        # QWEN_SHARED_BATCH=1: off the chain, a prompt's rows reach the MLP through
        # qwen_shared_experts_cpu's batch, which multiplies on the CPU whatever the tier holds
        # (and so reads the rows back); one row at a time it asks the device first
        _, _, lines, log = run(self.model, self.tmp, "dev", COLI_CUDA="1", FAKE_MATMUL_FAIL="0", QWEN_SHARED_BATCH="1")
        self.assertEqual(lines["placed"][2], 3 * 3 + 4 + 4 * 3 + 1, log[-3000:])
        self.assertGreater(lines["placed"][3], 0)
        self.assertEqual(lines["exit"][4], 0, "a matrix the device holds was read back:\n" + log[-3000:])
        self.assertNotIn("its rows read back from the file", log)


if __name__ == "__main__":
    unittest.main()
