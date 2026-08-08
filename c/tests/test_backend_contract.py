import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BackendContractTests(unittest.TestCase):
    def test_deepseek_v4_build_does_not_claim_vulkan_support(self):
        source = (ROOT / "Makefile.deepseek-v4").read_text(encoding="utf-8")
        runtime = (ROOT / "deepseek_v4.c").read_text(encoding="utf-8")
        self.assertNotIn("COLI_VULKAN", runtime)
        self.assertIn("does not currently include a Vulkan backend", source)

    def test_metal_batched_moe_accepts_grouped_int4(self):
        source = (ROOT / "backend_metal.mm").read_text(encoding="utf-8")
        header = (ROOT / "backend_metal.h").read_text(encoding="utf-8")
        self.assertIn("fmt != 4", source)
        self.assertIn("fmt == 4", source)
        self.assertIn("including fmt=4 grouped int4", header)

    def test_metal_moe_grouped_shader_has_per_group_scale_path(self):
        source = (ROOT / "backend_metal.mm").read_text(encoding="utf-8")
        self.assertIn("device const float* scales", source)
        self.assertIn("i/GS", source)

    def test_vulkan_primary_device_can_be_selected(self):
        source = (ROOT / "backend_vulkan.c").read_text(encoding="utf-8")
        self.assertIn('getenv("COLI_VK_DEV")', source)
        self.assertIn("invalid COLI_VK_DEV", source)


if __name__ == "__main__":
    unittest.main()
