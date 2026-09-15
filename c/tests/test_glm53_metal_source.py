#!/usr/bin/env python3
"""Source-level contract for the first GLM-5.3 Metal integration step.

This intentionally does not claim routed-expert acceleration yet.  It protects
build wiring, opt-in initialization, resident fmt=1/fmt=4 dispatch, CPU fallback,
and Metal-handle cleanup while the MoE path is implemented separately.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "glm53.c").read_text()
MAKE = (ROOT / "Makefile").read_text()


class Glm53MetalSourceTests(unittest.TestCase):
    def test_backend_is_opt_in(self):
        self.assertIn('#ifdef COLI_METAL\n#include "backend_metal.h"', SRC)
        self.assertIn('getenv("COLI_METAL")', SRC)
        self.assertIn('coli_metal_init()', SRC)
        self.assertIn('coli_metal_available()', SRC)

    def test_resident_matmul_has_metal_dispatch_and_cpu_fallback(self):
        start = SRC.index("static void mv(float *out, const Mat *w, const float *x)")
        end = SRC.index("static void rms(", start)
        body = SRC[start:end]
        self.assertIn("g_metal_ready", body)
        self.assertIn("coli_metal_matmul", body)
        self.assertIn("w->fmt == 1 || w->fmt == 4", body)
        self.assertIn("matmul_i4_grouped", body)
        self.assertIn("matmul_q", body)

    def test_metal_handle_is_released(self):
        start = SRC.index("static void mat_release(Mat *mat)")
        end = SRC.index("static void model_release", start)
        body = SRC[start:end]
        self.assertIn("coli_metal_tensor_free", body)

    def test_glm53_links_metal_object(self):
        start = MAKE.index("glm53$(EXE):")
        end = MAKE.index("kimi_k3$(EXE):", start)
        rule = MAKE[start:end]
        self.assertIn("backend_metal.h", rule)
        self.assertIn("$(METAL_OBJ)", rule)


if __name__ == "__main__":
    unittest.main()
