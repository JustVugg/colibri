#!/usr/bin/env python3
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "glm53.c").read_text()


class Glm53MetalRoutedSourceTests(unittest.TestCase):
    def test_metal_slots_are_page_aligned_and_registered(self):
        self.assertIn("posix_memalign(&p, 16384, need)", SRC)
        self.assertIn("coli_metal_register(slot->own, slot->own_len)", SRC)
        self.assertIn("metal_slot = g_metal_ready", SRC)

    def test_cpu_mmap_path_is_preserved(self):
        self.assertIn("if (!metal_slot)", SRC)
        self.assertIn("st_map_shard_range", SRC)
        self.assertIn("if (mapped_ok) { slot->eid = eid; return; }", SRC)

    def test_routed_experts_use_clamped_batched_api(self):
        self.assertIn("coli_metal_moe_block_clamped(", SRC)
        self.assertIn("here, c->hidden, c->moe_inter, 4, 64", SRC)
        self.assertIn("out, tokens, c->swiglu_limit", SRC)

    def test_metal_failure_keeps_cpu_fallback(self):
        self.assertIn("if (!metal_done)", SRC)
        self.assertIn("mlp3(tmp, x + (size_t)t * c->hidden", SRC)

    def test_packed_rows_keep_router_weights(self):
        self.assertIn("rows[R] = t; rw[R] = scale; nr[i]++; R++;", SRC)
        self.assertIn("xoff[i] = R", SRC)


if __name__ == "__main__":
    unittest.main()
