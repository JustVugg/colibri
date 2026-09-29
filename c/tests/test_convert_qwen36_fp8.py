import importlib.util
import sys
import unittest
from pathlib import Path

import torch


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_qwen36.py"
SPEC = importlib.util.spec_from_file_location("convert_qwen36", CONVERTER)
convert_qwen36 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = convert_qwen36
SPEC.loader.exec_module(convert_qwen36)


class Qwen36Fp8ConverterTest(unittest.TestCase):
    def test_merged_fp8_uses_e4m3_and_128_square_block_scales(self):
        gate = torch.linspace(-3.0, 3.0, 130 * 257).reshape(130, 257)
        up = torch.linspace(-2.0, 4.0, 130 * 257).reshape(130, 257)
        down = torch.linspace(-5.0, 1.0, 257 * 130).reshape(257, 130)

        merged, scales = convert_qwen36.make_merged_fp8(gate, up, down)

        block_scales = ((130 + 127) // 128) * ((257 + 127) // 128)
        self.assertEqual(merged.dtype, torch.float8_e4m3fn)
        self.assertEqual(merged.numel(), 3 * 130 * 257)
        self.assertEqual(scales.dtype, torch.float32)
        self.assertEqual(scales.numel(), 3 * block_scales)
        self.assertTrue(torch.isfinite(merged.float()).all())
        self.assertTrue((scales > 0).all())


if __name__ == "__main__":
    unittest.main()
