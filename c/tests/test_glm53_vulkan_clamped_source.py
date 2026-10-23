#!/usr/bin/env python3
"""Source contract: Vulkan gate_up SwiGLU can clamp like the CPU glm53 path."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHADER = (ROOT / "shaders" / "qmatmul_gate_up.comp").read_text()
SRC = (ROOT / "backend_vulkan.c").read_text()
HDR = (ROOT / "backend_vulkan.h").read_text()
GLM = (ROOT / "glm53.c").read_text()


class Glm53VulkanClampedSourceTests(unittest.TestCase):
    def test_shader_has_limit_and_clamp(self):
        self.assertIn("float limit;", SHADER)
        self.assertIn("if (p.limit > 0.0)", SHADER)
        self.assertIn("if (gt > p.limit) gt = p.limit;", SHADER)
        self.assertIn("if (ut < -p.limit) ut = -p.limit;", SHADER)

    def test_host_push_constant_carries_limit(self):
        self.assertIn("struct PCGU", SRC)
        self.assertIn("coli_vk_set_swiglu_limit", HDR)
        self.assertIn("sizeof(struct PCGU)", SRC)

    def test_glm53_publishes_its_checkpoint_limit(self):
        self.assertIn("coli_vk_set_swiglu_limit(m->c.swiglu_limit)", GLM)

    def test_zero_limit_keeps_unclamped_path(self):
        self.assertIn("g_swiglu_limit = limit > 0.f ? limit : 0.f", SRC)


if __name__ == "__main__":
    unittest.main()
