"""Check fused decode bindings against the checkpoint tensor plan, without CUDA."""
from pathlib import Path
import re
import unittest


class SharedExpertBindingsTest(unittest.TestCase):
    def test_decode_fused_moe_receives_gate_up_down_shapes(self):
        source = (Path(__file__).resolve().parents[1] / "deepseek_v4.c").read_text()
        shapes = dict((name, (rows, columns)) for rows, columns, name in re.findall(
            r'add_fp8\(plan, (\w+), (\w+), "(ffn\.shared_experts\.w[123])"', source))
        bindings = re.findall(
            r'void \*(s[gud]) = coli_v4_layer_gpu\(weights, "([^"]+)"\);', source)
        # Gate and up project hidden -> moe; down projects moe -> hidden.
        expected = {"sg": ("moe", "hidden"), "su": ("moe", "hidden"),
                    "sd": ("hidden", "moe")}
        self.assertTrue(bindings, "no fused decode shared-expert bindings found")
        self.assertEqual(set(name for name, _ in bindings), set(expected))
        for name, tensor in bindings:
            with self.subTest(binding=name, tensor=tensor):
                self.assertEqual(shapes[tensor], expected[name])


if __name__ == "__main__":
    unittest.main()
