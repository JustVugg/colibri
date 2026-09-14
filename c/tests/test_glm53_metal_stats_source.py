#!/usr/bin/env python3
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "c" / "glm53.c"


class Glm53MetalStatsSourceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = SRC.read_text()

    def test_counters_declared(self):
        for needle in (
            "g_metal_moe_attempt",
            "g_metal_moe_ok",
            "g_metal_moe_fallback",
            "g_metal_moe_rows",
        ):
            self.assertIn(needle, self.src)

    def test_attempt_counted_at_routed_dispatch(self):
        self.assertIn("g_metal_moe_attempt++;", self.src)
        self.assertIn("metal_done = coli_metal_moe_block_clamped(", self.src)

    def test_success_and_fallback_counted(self):
        self.assertIn("g_metal_moe_ok++;", self.src)
        self.assertIn("g_metal_moe_rows += (uint64_t)R;", self.src)
        self.assertIn("if (!metal_done) g_metal_moe_fallback++;", self.src)

    def test_verbose_summary_emitted(self):
        self.assertIn(
            'printf("metal moe attempts %llu ok %llu fallback %llu rows %llu\\n",',
            self.src,
        )


if __name__ == "__main__":
    unittest.main()
