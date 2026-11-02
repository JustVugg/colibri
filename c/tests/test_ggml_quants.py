"""ggml_quants.h decodes every GGUF weight type exactly like the reference.

For each type: random blocks (with every fp16 field forced finite, some
subnormal, some zero) are decoded by the C header and by gguf-py
(gguf.quants.dequantize, the reference implementation of the format), and the
two float32 arrays must be identical BIT FOR BIT. The engine's GGUF paths are
built on these decoders, so any difference here is a wrong weight everywhere.
"""
import os
import subprocess
import struct
import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
# GQ_TEST_BIN_DIR / GQ_TEST_WRAP run the same checks on another build, e.g. an
# aarch64 one under qemu: GQ_TEST_BIN_DIR=/tmp/arm GQ_TEST_WRAP="qemu-aarch64 -cpu max"
def _bin(name):
    d = os.environ.get("GQ_TEST_BIN_DIR")
    return (Path(d) if d else HERE / "tests") / name
def _wrap(cmd):
    w = os.environ.get("GQ_TEST_WRAP")
    return (w.split() if w else []) + [str(c) for c in cmd]
BIN = _bin("test_ggml_quants.exe" if sys.platform == "win32" else "test_ggml_quants")

try:
    from gguf import quants as gq
    from gguf.constants import GGMLQuantizationType as QT
except ImportError:  # pragma: no cover
    gq = None

# type -> (block elems, block bytes, offsets of fp16 fields inside the block)
TYPES = {
    "Q4_0": (32, 18, [0]), "Q4_1": (32, 20, [0, 2]), "Q5_0": (32, 22, [0]),
    "Q5_1": (32, 24, [0, 2]), "Q8_0": (32, 34, [0]), "Q2_K": (256, 84, [80, 82]),
    "Q3_K": (256, 110, [108]), "Q4_K": (256, 144, [0, 2]), "Q5_K": (256, 176, [0, 2]),
    "Q6_K": (256, 210, [208]), "IQ2_XXS": (256, 66, [0]), "IQ2_XS": (256, 74, [0]),
    "IQ2_S": (256, 82, [0]), "IQ3_XXS": (256, 98, [0]), "IQ3_S": (256, 110, [0]),
    "IQ1_S": (256, 50, [0]), "IQ1_M": (256, 56, []), "IQ4_NL": (32, 18, [0]),
    "IQ4_XS": (256, 136, [0]), "MXFP4": (32, 17, []), "TQ1_0": (256, 54, [52]),
    "TQ2_0": (256, 66, [64]),
    "NVFP4": (64, 36, []), "Q1_0": (128, 18, [0]), "Q2_0": (64, 18, [0]),
}
TYPE_ID = {"NVFP4": 40, "Q1_0": 41, "Q2_0": 42}


def ref_dequant(name, raw):
    """gguf-py where it has the type; for Q1_0 / Q2_0, which it does not, the
    ggml C reference (dequantize_row_q1_0 / _q2_0) transcribed to numpy."""
    if name == "Q1_0":
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
        bits = (raw[:, 2:18, None] >> np.arange(8, dtype=np.uint8)) & 1
        return np.where(bits.reshape(-1, 128) == 1, d, -d).astype(np.float32).reshape(-1)
    if name == "Q2_0":
        d = raw[:, 0:2].copy().view(np.float16).astype(np.float32)
        q = (raw[:, 2:18, None] >> (np.arange(4, dtype=np.uint8) * 2)) & 3
        return ((q.reshape(-1, 64).astype(np.int32) - 1).astype(np.float32) * d).reshape(-1)
    return gq.dequantize(raw.reshape(-1), QT[name]).astype(np.float32).reshape(-1)


def finite_f16(rng, n):
    v = rng.uniform(-2.0, 2.0, n).astype(np.float16)
    sub = rng.random(n) < 0.05
    v[sub] = (rng.integers(1, 1023, int(sub.sum())).astype(np.uint16)).view(np.float16)
    v[rng.random(n) < 0.02] = 0
    return v


def make_blocks(name, n_blocks, rng):
    blck, size, f16 = TYPES[name]
    raw = rng.integers(0, 256, (n_blocks, size), dtype=np.uint8)
    for off in f16:
        raw[:, off:off + 2] = finite_f16(rng, n_blocks).view(np.uint8).reshape(n_blocks, 2)
    if name == "IQ1_M":
        # the block scale is an fp16 spread over the top nibbles of four
        # uint16: clear its exponent's top bit (bit 14 of the last one) so
        # it can never be inf/NaN
        raw[:, 55] &= 0xBF
    return raw


def run_c(type_id, n, data):
    p = subprocess.run(_wrap([BIN]), input=struct.pack("<iq", type_id, n) + data,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode:
        raise AssertionError(f"decoder exit {p.returncode}: {p.stderr.decode()}")
    return np.frombuffer(p.stdout, dtype=np.float32)


@unittest.skipIf(gq is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(BIN.is_file(), "tests/test_ggml_quants is not built (make tests/test_ggml_quants)")
class GgmlQuantsExact(unittest.TestCase):
    def check(self, name, n_blocks=64, seed=1):
        rng = np.random.default_rng(seed)
        raw = make_blocks(name, n_blocks, rng)
        ref = ref_dequant(name, raw)
        got = run_c(TYPE_ID.get(name) or int(QT[name]), ref.size, raw.tobytes())
        self.assertEqual(got.size, ref.size)
        diff = np.flatnonzero(got.view(np.uint32) != ref.view(np.uint32))
        self.assertEqual(diff.size, 0,
                         f"{name}: {diff.size}/{ref.size} values differ; first at {diff[:3]}: "
                         f"C {got[diff[:3]]} vs ref {ref[diff[:3]]}")

    def test_every_block_type_is_bit_exact(self):
        for name in TYPES:
            for seed in (1, 2, 3):
                with self.subTest(type=name, seed=seed):
                    self.check(name, seed=seed)

    def test_float_types(self):
        rng = np.random.default_rng(7)
        for name, dt in (("F16", np.float16), ("BF16", None), ("F32", np.float32)):
            with self.subTest(type=name):
                if name == "BF16":
                    raw = (rng.standard_normal(512).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
                    ref = (raw.astype(np.uint32) << 16).view(np.float32)
                else:
                    raw = finite_f16(rng, 512) if dt is np.float16 else rng.standard_normal(512).astype(np.float32)
                    ref = raw.astype(np.float32)
                got = run_c(int(QT[name]), ref.size, raw.tobytes())
                self.assertTrue(np.array_equal(got.view(np.uint32), ref.view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
