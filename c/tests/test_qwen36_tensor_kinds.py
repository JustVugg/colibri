"""The converter's tensor-name contract, pinned to the real checkpoints.

Torch-free: it imports only tools/qwen36_tensor_kinds.py. The name lists are
the safetensors indexes of the two shipped checkpoints with the layer/expert
indices collapsed, so a checkpoint that adds a tensor kind fails here before
anyone spends a terabyte finding out in the converter.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from qwen36_tensor_kinds import (  # noqa: E402
    GLOBAL_KINDS, LAYER_KINDS, UnknownTensor, classify, resolve_prefix, skip_reason)

# Qwen/Qwen3.8-2.4T-A95B, model.safetensors.index.json (1609 tensors, 213
# shards, fetched 2026-09-03). "N" stands for a layer index; 92 layers, of
# which 23 (i % 4 == 3) carry self_attn and 69 linear_attn.
QWEN38_2P4T = (
    "lm_head.weight",
    "model.embed_tokens.weight",
    "model.norm.weight",
    "model.layers.N.input_layernorm.weight",
    "model.layers.N.post_attention_layernorm.weight",
    "model.layers.N.linear_attn.A_log",
    "model.layers.N.linear_attn.conv1d.weight",
    "model.layers.N.linear_attn.dt_bias",
    "model.layers.N.linear_attn.in_proj_a.weight",
    "model.layers.N.linear_attn.in_proj_b.weight",
    "model.layers.N.linear_attn.in_proj_qkv.weight",
    "model.layers.N.linear_attn.in_proj_z.weight",
    "model.layers.N.linear_attn.norm.weight",
    "model.layers.N.linear_attn.out_proj.weight",
    "model.layers.N.mlp.experts.down_proj",
    "model.layers.N.mlp.experts.gate_up_proj",
    "model.layers.N.mlp.gate.weight",
    "model.layers.N.mlp.shared_expert.down_proj.weight",
    "model.layers.N.mlp.shared_expert.gate_proj.weight",
    "model.layers.N.mlp.shared_expert.up_proj.weight",
    "model.layers.N.mlp.shared_expert_gate.weight",
    "model.layers.N.self_attn.k_norm.weight",
    "model.layers.N.self_attn.k_proj.weight",
    "model.layers.N.self_attn.o_proj.weight",
    "model.layers.N.self_attn.q_norm.weight",
    "model.layers.N.self_attn.q_proj.weight",
    "model.layers.N.self_attn.v_proj.weight",
    # mtp_num_hidden_layers: 1 -- one head shaped like an attention layer
    "mtp.fc.weight",
    "mtp.norm.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.N.input_layernorm.weight",
    "mtp.layers.N.post_attention_layernorm.weight",
    "mtp.layers.N.mlp.experts.down_proj",
    "mtp.layers.N.mlp.experts.gate_up_proj",
    "mtp.layers.N.mlp.gate.weight",
    "mtp.layers.N.mlp.shared_expert.down_proj.weight",
    "mtp.layers.N.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.N.mlp.shared_expert.up_proj.weight",
    "mtp.layers.N.mlp.shared_expert_gate.weight",
    "mtp.layers.N.self_attn.k_norm.weight",
    "mtp.layers.N.self_attn.k_proj.weight",
    "mtp.layers.N.self_attn.o_proj.weight",
    "mtp.layers.N.self_attn.q_norm.weight",
    "mtp.layers.N.self_attn.q_proj.weight",
    "mtp.layers.N.self_attn.v_proj.weight",
)

# Qwen/Qwen3.6-35B-A3B (1045 tensors): a vision-language checkpoint. Same
# text tensors under model.language_model., plus the same mtp head and a
# visual tower (333 tensors, three shown; the prefix is what matters).
QWEN36_35B = tuple(
    n.replace("model.", "model.language_model.", 1) if n.startswith("model.") else n
    for n in QWEN38_2P4T
) + (
    "model.visual.patch_embed.proj.weight",
    "model.visual.blocks.N.attn.qkv.weight",
    "model.visual.merger.linear_fc1.weight",
)

# transformers save_pretrained on the text model (the tiny fixture): experts
# one tensor each, no mtp, no visual.
TINY_FIXTURE = tuple(
    n for n in QWEN38_2P4T
    if not n.startswith("mtp.") and ".mlp.experts." not in n
) + (
    "model.layers.N.mlp.experts.N.gate_proj.weight",
    "model.layers.N.mlp.experts.N.up_proj.weight",
    "model.layers.N.mlp.experts.N.down_proj.weight",
)


def _concrete(names):
    return [n.replace("N", "7") for n in names]


class TensorKindsTest(unittest.TestCase):
    def _classify_all(self, names):
        prefix = resolve_prefix(names)
        return prefix, [classify(n, prefix) for n in names]

    def test_every_2p4t_tensor_is_placed(self):
        prefix, placed = self._classify_all(_concrete(QWEN38_2P4T))
        self.assertEqual(prefix, "model.")
        kinds = {p[0] for p in placed}
        self.assertEqual(kinds, {"global", "layer", "skip"})
        self.assertEqual({p[1] for p in placed if p[0] == "skip"}, {"mtp"})
        self.assertEqual({p[1] for p in placed if p[0] == "global"}, set(GLOBAL_KINDS))
        self.assertEqual({p[2] for p in placed if p[0] == "layer"},
                         set(LAYER_KINDS) - set())

    def test_every_35b_tensor_is_placed(self):
        prefix, placed = self._classify_all(_concrete(QWEN36_35B))
        self.assertEqual(prefix, "model.language_model.")
        self.assertEqual({p[1] for p in placed if p[0] == "skip"}, {"mtp", "visual"})
        self.assertEqual(sum(p[0] == "layer" for p in placed),
                         sum(p[0] == "layer" for p in self._classify_all(
                             _concrete(QWEN38_2P4T))[1]))

    def test_tiny_fixture_per_expert_layout(self):
        prefix, placed = self._classify_all(_concrete(TINY_FIXTURE))
        self.assertEqual(prefix, "model.")
        self.assertNotIn("skip", {p[0] for p in placed})
        experts = [p for p in placed if p[0] == "layer" and p[2].startswith("mlp.experts.7.")]
        self.assertEqual(len(experts), 3)

    def test_layer_index_is_read(self):
        self.assertEqual(classify("model.layers.91.mlp.experts.down_proj", "model."),
                         ("layer", 91, "mlp.experts.down_proj"))
        self.assertEqual(classify("model.language_model.layers.3.self_attn.q_proj.weight",
                                  "model.language_model."),
                         ("layer", 3, "self_attn.q_proj.weight"))

    def test_unknown_names_stop_the_converter(self):
        for name in (
            "model.layers.0.mlp.experts.gate_up_proj.weight_scale_inv",  # FP8 sidecar
            "model.layers.0.linear_attn.in_proj_qkvz.weight",           # Qwen3-Next naming
            "model.layers.0.self_attn.q_proj.bias",                       # attention_bias
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
            "model.layers.0.mlp.experts.gate_up_proj_extra",
            "model.language_model.layers.0.mlp.gate.weight",  # wrong prefix for this call
            "audio.encoder.weight",
            "model.mtp.fc.weight",
        ):
            with self.subTest(name=name):
                with self.assertRaises(UnknownTensor):
                    classify(name, "model.")

    def test_skip_groups_have_a_stated_reason(self):
        for group in ("mtp", "visual"):
            self.assertTrue(skip_reason(group))
        self.assertEqual(skip_reason("nope"), "")

    def test_layer_kinds_are_exact_suffixes(self):
        # fragments would let "norm.weight" swallow "linear_attn.norm.weight"
        # AND "mlp.experts.0.norm.weight"; exact strings keep a new kind loud
        for kind in LAYER_KINDS:
            self.assertFalse(kind.startswith("."))
            self.assertFalse(kind.endswith("."))
            self.assertNotIn("N", kind)


if __name__ == "__main__":
    unittest.main()
