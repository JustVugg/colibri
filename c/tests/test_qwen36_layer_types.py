"""#1532: the qwen36 planner geometry needs the per-layer kinds. The HF config
carries `layer_types`, but a converted container may carry them only in
qwen36_meta.json (the engine reads that, so chat worked while doctor and plan
refused), and the upstream class derives them from `full_attention_interval`.
All three sources must satisfy the planner; none of them, a clear error."""
import json
import os
import tempfile
import unittest

from family_registry import _qwen36_geometry, _qwen36_layer_types

BASE = {"num_hidden_layers": 8, "num_key_value_heads": 2, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_key_head_dim": 128,
        "linear_num_value_heads": 32, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "num_experts": 256}
KINDS = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"]


class Qwen36LayerTypesTest(unittest.TestCase):
    def test_config_list_wins(self):
        self.assertEqual(_qwen36_layer_types(dict(BASE, layer_types=KINDS), 8, None), KINDS)

    def test_meta_file_fills_the_gap(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "qwen36_meta.json"), "w") as f:
                json.dump({"layer_types": KINDS, "n_layers": 8}, f)
            self.assertEqual(_qwen36_layer_types(dict(BASE), 8, d), KINDS)
            geometry = _qwen36_geometry(dict(BASE), 4096, d)
            self.assertGreater(geometry.context_state_bytes, 0)

    def test_interval_derives_the_kinds(self):
        self.assertEqual(_qwen36_layer_types(dict(BASE, full_attention_interval=4), 8, None), KINDS)

    def test_a_wrong_length_list_falls_through(self):
        self.assertEqual(_qwen36_layer_types(dict(BASE, layer_types=KINDS[:5], full_attention_interval=4), 8, None), KINDS)

    def test_nothing_known_is_a_clear_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as caught:
                _qwen36_layer_types(dict(BASE), 8, d)
        self.assertIn("full_attention_interval", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
