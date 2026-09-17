#!/usr/bin/env python3
"""Source contract for GLM-5.3 clamped-SwiGLU Metal support.

GLM-5.3 must not use the existing plain SwiGLU routed-expert kernel. This test
protects a separate clamped kernel/API while the engine wiring is added next.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HDR = (ROOT / "backend_metal.h").read_text()
SRC = (ROOT / "backend_metal.mm").read_text()


class Glm53MetalClampedSourceTests(unittest.TestCase):
    def test_separate_clamped_kernel_exists(self):
        self.assertIn("kernel void moe_silu_clamped", SRC)
        self.assertIn("float v=min(g[i], limit)", SRC)
        self.assertIn("float uv=clamp(u[i], -limit, limit)", SRC)
        self.assertIn("(v/(1.0f+exp(-v)))*uv", SRC)

    def test_separate_pipeline_is_compiled(self):
        self.assertIn("g_moe_silu_clamped", SRC)
        self.assertIn('P("moe_silu_clamped")', SRC)

    def test_public_clamped_moe_api_exists(self):
        self.assertIn("coli_metal_moe_block_clamped", HDR)
        self.assertIn("float swiglu_limit", HDR)
        self.assertIn('extern "C" int coli_metal_moe_block_clamped', SRC)

    def test_plain_moe_api_remains_unchanged(self):
        self.assertIn("int coli_metal_moe_block(int nb", HDR)
        self.assertIn("g_moe_silu];", SRC)


if __name__ == "__main__":
    unittest.main()
