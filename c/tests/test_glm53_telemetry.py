import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GLM = (ROOT / "glm53.c").read_text()
TRACE = (ROOT / "route_trace.h").read_text()


class Glm53RoutingTelemetryTest(unittest.TestCase):
    def test_shared_route_trace_wiring(self):
        self.assertIn('#include "route_trace.h"', GLM)
        self.assertIn('rt_init("glm53", c->n_layers, c->n_experts);', GLM)
        self.assertIn('rt_route(index, t, mine, mine_w, topk);', GLM)
        self.assertIn('rt_trace_end();', GLM)

    def test_rows_and_usage_lifecycle(self):
        self.assertIn('for (int i = 0; i < c->first_dense; i++) rt_drop_row(i);', GLM)
        self.assertIn('rt_drop_row(c->n_layers);', GLM)
        self.assertIn('getenv("COLI_USAGE")', GLM)
        self.assertIn('rt_load(g_glm53_usage)', GLM)
        self.assertIn('rt_save(g_glm53_usage, 0)', GLM)

    def test_engine_identity_registered(self):
        self.assertIn('"glm53"', TRACE)

    def test_segment_open_does_not_attach_global_telemetry(self):
        start = GLM.index('static int glm53_segment_engine_open(')
        end = GLM.index('static void glm53_segment_engine_destroy(', start)
        segment_open = GLM[start:end]
        self.assertNotIn('glm53_telemetry_init', segment_open)
        self.assertNotIn('rt_init(', segment_open)


if __name__ == '__main__':
    unittest.main()
