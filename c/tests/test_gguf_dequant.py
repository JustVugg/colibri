#!/usr/bin/env python3
"""tools/gguf_dequant.py: ggml block dequantization, pinned to ggml's math.

Two independent checks:
  * committed golden vectors (tests/fixtures/gguf_dequant_golden.npz), generated
    with llama.cpp's gguf-py reference on deterministic random blocks -- run in
    CI with only numpy;
  * if the gguf package is importable, a live comparison against gguf.quants.
    dequantize on fresh random blocks, so drift from the reference is caught.

Also covers the scalar types (F32/F16/BF16) and the refusal paths (unsupported
type, numel/byte-count mismatch).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gguf_reader
from gguf_reader import GGUFError
from gguf_dequant import (
    dequantize,
    dequantize_q2_K,
    dequantize_q3_K,
    dequantize_q6_K,
)

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "gguf_dequant_golden.npz"

BLOCK_CASES = [
    ("Q2_K", gguf_reader.GGML_TYPE_Q2_K, 84, dequantize_q2_K),
    ("Q3_K", gguf_reader.GGML_TYPE_Q3_K, 110, dequantize_q3_K),
    ("Q6_K", gguf_reader.GGML_TYPE_Q6_K, 210, dequantize_q6_K),
]


class GoldenDequantTest(unittest.TestCase):
    def setUp(self):
        if not GOLDEN.is_file():
            self.skipTest("golden fixture missing: %s" % GOLDEN)
        self.golden = np.load(str(GOLDEN))

    def test_block_types_match_golden(self):
        for name, ggml_type, block_bytes, _func in BLOCK_CASES:
            raw = self.golden["%s_raw" % name]
            expected = self.golden["%s_expected" % name].reshape(-1)
            got = dequantize(raw.tobytes(), ggml_type, expected.size)
            self.assertEqual(got.shape, expected.shape)
            np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True,
                                       err_msg="%s dequant mismatch" % name)

    def test_dispatcher_uses_same_functions(self):
        for name, ggml_type, _bb, func in BLOCK_CASES:
            raw = self.golden["%s_raw" % name]
            direct = func(raw).reshape(-1)
            dispatched = dequantize(raw.tobytes(), ggml_type, direct.size)
            np.testing.assert_array_equal(dispatched, direct)


class ScalarDequantTest(unittest.TestCase):
    def test_f32(self):
        values = np.array([0.0, 1.5, -2.25, 3.5e10], dtype=np.float32)
        got = dequantize(values.tobytes(), gguf_reader.GGML_TYPE_F32, values.size)
        np.testing.assert_array_equal(got, values)

    def test_f16(self):
        values = np.array([0.0, 1.5, -2.25, 0.125], dtype=np.float16)
        got = dequantize(values.tobytes(), gguf_reader.GGML_TYPE_F16, values.size)
        np.testing.assert_array_equal(got, values.astype(np.float32))

    def test_bf16(self):
        u16 = np.array([0x0000, 0x3FC0, 0xC040, 0x3E00], dtype=np.uint16)
        got = dequantize(u16.tobytes(), gguf_reader.GGML_TYPE_BF16, u16.size)
        expected = (u16.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got, expected)


class DequantRefusalTest(unittest.TestCase):
    def test_unsupported_type(self):
        with self.assertRaisesRegex(GGUFError, "unsupported ggml type"):
            dequantize(b"\x00" * 16, 100, 16)

    def test_numel_not_multiple_of_block(self):
        with self.assertRaisesRegex(GGUFError, "not a multiple of block"):
            dequantize(b"\x00" * 84, gguf_reader.GGML_TYPE_Q2_K, 255)

    def test_byte_count_mismatch(self):
        with self.assertRaisesRegex(GGUFError, "expected"):
            dequantize(b"\x00" * 84, gguf_reader.GGML_TYPE_Q2_K, 512)


class LiveReferenceTest(unittest.TestCase):
    def test_matches_gguf_py_on_random_blocks(self):
        try:
            import gguf
            from gguf.quants import dequantize as reference
        except ImportError as exc:
            self.skipTest("gguf package not installed: %s" % exc)

        rng = np.random.default_rng(20260825)
        mapping = {
            "Q2_K": (gguf.GGMLQuantizationType.Q2_K, gguf_reader.GGML_TYPE_Q2_K, 84),
            "Q3_K": (gguf.GGMLQuantizationType.Q3_K, gguf_reader.GGML_TYPE_Q3_K, 110),
            "Q6_K": (gguf.GGMLQuantizationType.Q6_K, gguf_reader.GGML_TYPE_Q6_K, 210),
        }
        for name, (ref_type, our_type, block_bytes) in mapping.items():
            for _ in range(25):
                nb = int(rng.integers(1, 5))
                raw = rng.integers(0, 256, size=nb * block_bytes, dtype=np.uint8)
                expected = reference(raw.reshape(nb, block_bytes).copy(), ref_type).reshape(-1)
                got = dequantize(raw.tobytes(), our_type, nb * 256)
                np.testing.assert_allclose(got, expected, rtol=0, atol=0, equal_nan=True,
                                           err_msg="%s live mismatch" % name)


if __name__ == "__main__":
    unittest.main()
