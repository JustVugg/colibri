"""A GGUF matrix the CUDA tier holds gives its host rows back, and the CPU reads it back from the
file, the same bytes, if it ever needs it (qwen36.c q36_gq_drop / q36_gq_reload). And the loader
never holds a matrix the tier takes: the tier's upload reads its rows from the file a range at a
time through one staging buffer (q36_gq_stream), the loader the rest after the placement
(q36_gq_settle).

Before, every matrix the tier placed kept its host rows too: Qwen3.8-27B UD-Q4_K_XL with its whole
trunk on a 32 GB card held 15.4 GB of RAM it never read again; and the loader read all of them
first, 16.6 GB before the tier placed a byte.

tests/test_qwen36_gguf_hostdrop.c runs the whole engine against the fake CUDA backend, which takes
every block format and computes none, on a small dense qwen35 GGUF (one DeltaNet layer of two types,
K-quants the CPU holds interleaved, the tied head):
  - COLI_CUDA=0: the CPU's tokens and last logits, the reference;
  - the tier on, every device matmul failing: every placed matrix gave its rows back, the CPU read
    each one back at its first product, and the tokens and the logits are the reference's, bit for bit;
  - the tier on, the device answering: the rows were given back and none was read back;
  - a small staging buffer (several ranges a matrix, the rotary heads' rows un-permuted a head at a
    time): the device gets the bytes, in the order, that a backend without ranges gets whole;
and on a ~160 MiB fixture, the load's peak: below the file's size plus the staging buffer, and the
CPU run's less what the tier took.
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
    import test_gguf_import as tgi
    from test_gguf_import import write_model
except ImportError:  # pragma: no cover
    write_model = None

LINE = re.compile(r"^HOSTDROP (\w+) ((?:\w+=\w+ ?)+)$", re.M)
TRUNK = 3 * 3 + 4 + 4 * 3 + 1   # 3 DeltaNet layers' qkv, z and out, the attention layer's q, k, v and o, four MLPs, the head


def run(model, tmp, tag, args=("2", "8"), **env):
    """(tokens, logits bytes, {when: {field: value}}, log) of one run"""
    dump = tmp / f"logits-{tag}.f32"
    e = dict(os.environ, SNAP=str(model), OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", USAGE_SAVE="0",
             COLI_VULKAN="0", COLI_CUDA_CHAIN="0", QT_NO_WARMSTART="1", COLI_LOOKUP="0", Q36_OFFER_SHEXP="1", DUMP=str(dump), **env)
    r = subprocess.run([str(EXE), *args, str(model / "ref.json")], env=e, capture_output=True,
                       text=True, errors="replace", timeout=600)
    log = r.stdout + r.stderr   # the exit status is the reference's match (random weights: none)
    tokens = re.findall(r"^C engine : (.*?)(?: +Text|$)", log, re.M)
    if not tokens:
        raise AssertionError(f"{tag}: no token line\n" + log[-3000:])
    lines = {m.group(1): {k: int(v, 16) if k == "rows" else int(v) for k, v in (kv.split("=") for kv in m.group(2).split())}
             for m in LINE.finditer(log)}
    return tokens[-1].split(), dump.read_bytes(), lines, log


def import_model(gg, model, ref):
    r = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(gg), str(model)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("import failed:\n" + r.stdout + r.stderr)
    (model / "ref.json").write_text(ref)


@unittest.skipUnless(EXE.exists(), "tests/test_qwen36_gguf_hostdrop is not built")
@unittest.skipUnless(write_model is not None, "the gguf package is not installed")
class Qwen36GgufHostDropTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="q36_hostdrop_"))
        gg = cls.tmp / "m.gguf"
        write_model(gg, np.random.default_rng(31), dense=True, dn_types={1: ("Q8_0", "Q4_K")})
        cls.model = cls.tmp / "m"
        import_model(gg, cls.model, '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}')
        cls.cpu = run(cls.model, cls.tmp, "cpu", COLI_CUDA="0")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cpu_reads_dropped_rows_back_exactly(self):
        tokens, logits, lines, log = run(self.model, self.tmp, "fail", COLI_CUDA="1", FAKE_MATMUL_FAIL="1")
        self.assertIn("matrices on the CUDA tier gave their host rows back", log, log[-3000:])
        self.assertIn("its rows read back from the file", log, log[-3000:])
        self.assertIn("staging buffer", log, log[-3000:])
        dropped = lines["placed"]["dropped"]
        self.assertEqual(dropped, TRUNK, log[-3000:])
        self.assertEqual(lines["placed"]["reloaded"], 0, "a row was read back before any product")
        self.assertEqual(lines["exit"]["reloaded"], dropped, "every placed matrix ran on the CPU, read back once")
        self.assertEqual(tokens, self.cpu[0])
        self.assertEqual(logits, self.cpu[1], "the read-back rows changed the logits")

    def test_device_run_reads_nothing_back(self):
        # QWEN_SHARED_BATCH=1: off the chain, a prompt's rows reach the MLP through
        # qwen_shared_experts_cpu's batch, which multiplies on the CPU whatever the tier holds
        # (and so reads the rows back); one row at a time it asks the device first
        _, _, lines, log = run(self.model, self.tmp, "dev", COLI_CUDA="1", FAKE_MATMUL_FAIL="0", QWEN_SHARED_BATCH="1")
        self.assertEqual(lines["placed"]["dropped"], TRUNK, log[-3000:])
        self.assertGreater(lines["placed"]["bytes"], 0)
        self.assertEqual(lines["exit"]["reloaded"], 0, "a matrix the device holds was read back:\n" + log[-3000:])
        self.assertNotIn("its rows read back from the file", log)

    def test_streamed_rows_are_the_rows_uploaded_whole(self):
        # a 64 KiB staging buffer takes the DeltaNet's qkv ++ z (139 KiB) in three ranges, the
        # rotary heads' rows (HOSTDROP_ROPE_PERM=1) a head at a time; a backend without ranges
        # gets each matrix whole, read into RAM first as before (qdw_gq_rows): the same bytes
        for perm in ("0", "1"):
            with self.subTest(rope_perm=perm):
                env = dict(COLI_CUDA="1", FAKE_MATMUL_FAIL="0", QWEN_SHARED_BATCH="1", HOSTDROP_ROPE_PERM=perm)
                _, _, s, slog = run(self.model, self.tmp, "stream" + perm, Q36_GGUF_STAGE_KB="64", **env)
                _, _, w, wlog = run(self.model, self.tmp, "whole" + perm, FAKE_UPLOAD_ROWS="0", **env)
                s, w = s["placed"], w["placed"]
                self.assertIn("staging buffer", slog, slog[-3000:])
                self.assertNotIn("staging buffer", wlog, wlog[-3000:])
                self.assertEqual(w["ranges"], 0)
                self.assertGreater(s["ranges"], s["uploads"], "no matrix went up in several ranges")
                self.assertEqual((s["uploads"], s["dropped"], s["bytes"]), (w["uploads"], w["dropped"], w["bytes"]))
                self.assertEqual(s["rows"], w["rows"], "the device got other bytes through the staging buffer")
                # Q36_GGUF_STAGE_KB=0: every matrix read at load and uploaded whole, as before
                _, _, o, olog = run(self.model, self.tmp, "old" + perm, Q36_GGUF_STAGE_KB="0", **env)
                o = o["placed"]
                self.assertNotIn("staging buffer", olog, olog[-3000:])
                self.assertEqual(o["ranges"], 0)
                self.assertEqual((o["uploads"], o["dropped"], o["rows"]), (w["uploads"], w["dropped"], w["rows"]))


# ~160 MiB: Q8_0 embedding 32000 x 1024 (tied head), 8 layers (2 attention), MLP 4096
MID = dict(H=1024, LAYERS=8, I=4096, VOCAB=32000, QH=8, KVH=2, HD=128, KH=8, VH=16, KD=128)
STAGE_KB = 1024


def fast_blocks(name, shape, rng, std=0.02):
    """random blocks with tiny scales: no dequantize probe, quick at this size"""
    blck, size, f16 = tgi.TYPES[name]
    rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
    raw = rng.integers(0, 256, (rows * nb, size), dtype=np.uint8)
    for off in f16:
        raw[:, off:off + 2] = np.full(rows * nb, 1e-4, np.float16).view(np.uint8).reshape(-1, 2)
    return raw.reshape(*shape[:-1], nb * size)


@unittest.skipUnless(EXE.exists(), "tests/test_qwen36_gguf_hostdrop is not built")
@unittest.skipUnless(write_model is not None, "the gguf package is not installed")
class Qwen36GgufLoadPeakTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="q36_loadpeak_"))
        gg = cls.tmp / "m.gguf"
        saved = {k: getattr(tgi, k) for k in (*MID, "blocks")}
        try:
            for k, v in MID.items():
                setattr(tgi, k, v)
            tgi.blocks = fast_blocks
            write_model(gg, np.random.default_rng(7), dense=True, dn_types={1: ("Q8_0", "Q4_K")})
        finally:
            for k, v in saved.items():
                setattr(tgi, k, v)
        cls.size = gg.stat().st_size
        cls.model = cls.tmp / "m"
        import_model(gg, cls.model, '{"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8, 0, 0]}')
        env = dict(QWEN_SHARED_BATCH="1", Q36_GGUF_STAGE_KB=str(STAGE_KB))
        cls.cpu = run(cls.model, cls.tmp, "cpu", args=("2", "2"), COLI_CUDA="0", **env)
        cls.dev = run(cls.model, cls.tmp, "dev", args=("2", "2"), COLI_CUDA="1", FAKE_MATMUL_FAIL="0", **env)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_load_holds_the_cpus_matrices_and_the_staging_only(self):
        stage, slack = STAGE_KB << 10, 32 << 20   # slack: the tier's own allocations, allocator rounding
        placed, dev, cpu = self.dev[2]["placed"], self.dev[2]["exit"], self.cpu[2]["exit"]
        mib = lambda v: f"{v / 2**20:.1f} MiB"
        msg = (f"fixture {mib(self.size)}, staging {mib(stage)}; CPU run peak private {mib(cpu['peak_private'])} ws {mib(cpu['peak_ws'])}; "
               f"device run at the placement peak private {mib(placed['peak_private'])} ws {mib(placed['peak_ws'])}, "
               f"{placed['dropped']} matrices ({mib(placed['bytes'])}) on the device\n" + self.dev[3][-3000:])
        self.assertEqual(placed["dropped"], 8 * 3 + 6 * 3 + 2 * 4 + 1, msg)
        self.assertEqual(dev["reloaded"], 0, msg)
        self.assertLess(placed["peak_private"], self.size + stage, msg)
        self.assertLess(placed["peak_ws"], self.size + stage, msg)
        # the CPU run holds every matrix: the device run's peak is that less what the tier took, plus the buffer
        self.assertLess(dev["peak_private"] + placed["bytes"], cpu["peak_private"] + stage + slack, msg)
        print("\n" + msg.split("\n")[0])


if __name__ == "__main__":
    unittest.main()
