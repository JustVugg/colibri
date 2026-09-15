#!/usr/bin/env python3
"""tools/gguf_olmoe_profile.py: OLMoE name mapping, layout and int8 quantization.

The strongest check here is test_matches_reference_quantizer: our numpy
quantize_row is compared against the torch implementation in
tools/convert_olmoe_merged.py, which is the code the existing OLMoE container
was built with. If the two disagree, the new GGUF converter would produce a
container the engine reads differently from the known-good one.
"""

import sys
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise unittest.SkipTest("numpy not installed: %s" % exc)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import gguf_olmoe_profile as profile


class MappingTest(unittest.TestCase):
    def test_parse_expert(self):
        self.assertEqual(profile.parse_expert("blk.3.ffn_gate_exps.weight"), (3, "gate"))
        self.assertEqual(profile.parse_expert("blk.0.ffn_down_exps.weight"), (0, "down"))
        self.assertIsNone(profile.parse_expert("blk.0.attn_q.weight"))
        self.assertIsNone(profile.parse_expert("token_embd.weight"))

    def test_dense_target(self):
        self.assertEqual(profile.dense_target("token_embd.weight"),
                         "model.embed_tokens.weight")
        self.assertEqual(profile.dense_target("output.weight"), "lm_head.weight")
        self.assertEqual(profile.dense_target("output_norm.weight"), "model.norm.weight")
        self.assertEqual(profile.dense_target("blk.5.attn_q.weight"),
                         "model.layers.5.self_attn.q_proj.weight")
        self.assertEqual(profile.dense_target("blk.11.ffn_gate_inp.weight"),
                         "model.layers.11.mlp.gate.weight")
        self.assertEqual(profile.dense_target("blk.0.ffn_norm.weight"),
                         "model.layers.0.post_attention_layernorm.weight")
        self.assertIsNone(profile.dense_target("blk.0.some_unknown.weight"))


class LayoutTest(unittest.TestCase):
    def test_to_logical_reverses_dims(self):
        flat = np.arange(6, dtype=np.float32)
        logical = profile.to_logical(flat, [3, 2])
        self.assertEqual(logical.shape, (2, 3))
        np.testing.assert_array_equal(logical, flat.reshape(2, 3))

    def test_to_logical_1d_is_identity(self):
        flat = np.arange(5, dtype=np.float32)
        np.testing.assert_array_equal(profile.to_logical(flat, [5]), flat)

    def test_expert_matrix_shape(self):
        self.assertEqual(profile.expert_matrix_shape([2048, 1024, 64]), (1024, 2048))
        self.assertEqual(profile.expert_matrix_shape([1024, 2048, 64]), (2048, 1024))

    def test_restore_rope_layout_inverts_llama_permutation(self):
        def forward(w, n_head, n_kv=None):
            n = n_kv if (n_kv and n_head != n_kv) else n_head
            return np.ascontiguousarray(
                w.reshape((n, 2, w.shape[0] // n // 2) + w.shape[1:]).swapaxes(1, 2).reshape(w.shape))

        rng = np.random.default_rng(11)
        for (out, n_head, n_kv) in ((16, 4, 4), (24, 4, 2), (12, 2, None), (64, 4, None)):
            hf = rng.normal(size=(out, 3)).astype(np.float32)
            permuted = forward(hf, n_head, n_kv)
            if out // (n_kv if (n_kv and n_head != n_kv) else n_head) // 2 != 2:
                self.assertFalse(np.array_equal(permuted, hf))
            restored = profile.restore_rope_layout(permuted, n_head, n_kv)
            np.testing.assert_array_equal(restored, hf)

    def test_restore_rope_layout_matches_torch_reference(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest("torch unavailable: %s" % exc)
        for (out, n_head, n_kv) in ((16, 4, 4), (24, 4, 2), (12, 2, None)):
            rng = np.random.default_rng(out)
            hf = torch.from_numpy(rng.normal(size=(out, 7)).astype(np.float32))
            n = n_kv if (n_kv and n_head != n_kv) else n_head
            permuted = (hf.reshape(n, 2, out // n // 2, *hf.shape[1:])
                        .swapaxes(1, 2).reshape(hf.shape))
            restored = profile.restore_rope_layout(permuted.numpy(), n_head, n_kv)
            np.testing.assert_array_equal(restored, hf.numpy())


class QuantizationTest(unittest.TestCase):
    def test_quantize_row_math(self):
        w = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]], dtype=np.float32)
        q, s = profile.quantize_row(w)
        self.assertEqual(q.dtype, np.int8)
        self.assertEqual(q.shape, (2, 3))
        self.assertEqual(s.shape, (2,))
        # row 0: absmax 2 -> scale 2/127, q = round(w/scale)
        self.assertAlmostEqual(float(s[0]), 2.0 / 127.0, places=7)
        expected = np.clip(np.rint(w[0] / s[0]), -128, 127).astype(np.int8)
        np.testing.assert_array_equal(q[0], expected)
        # reconstruction W ~= q*scale
        np.testing.assert_allclose(q.astype(np.float32) * s[:, None], w, atol=2.0 / 127.0)

    def test_quantize_row_rejects_non_2d(self):
        with self.assertRaises(ValueError):
            profile.quantize_row(np.zeros(8, dtype=np.float32))

    def test_matches_reference_quantizer(self):
        try:
            import torch
            import convert_olmoe_merged as reference
        except ImportError as exc:
            self.skipTest("torch/convert_olmoe_merged unavailable: %s" % exc)

        rng = np.random.default_rng(20260825)
        for _ in range(20):
            rows = int(rng.integers(1, 8))
            cols = int(rng.integers(1, 40))
            w = rng.normal(0.0, 1.0, size=(rows, cols)).astype(np.float32)
            q_ref, s_ref = reference.quantize_row(torch.from_numpy(w))
            q_got, s_got = profile.quantize_row(w)
            np.testing.assert_array_equal(q_got, q_ref.cpu().numpy())
            np.testing.assert_allclose(s_got, s_ref.cpu().numpy(), rtol=0, atol=0)

    def test_merge_expert_layout(self):
        rng = np.random.default_rng(7)
        hidden, inter = 4, 6
        gate = rng.normal(0, 1, (inter, hidden)).astype(np.float32)
        up = rng.normal(0, 1, (inter, hidden)).astype(np.float32)
        down = rng.normal(0, 1, (hidden, inter)).astype(np.float32)
        merged, scales = profile.merge_expert(gate, up, down)

        qg, sg = profile.quantize_row(gate)
        qu, su = profile.quantize_row(up)
        qd, sd = profile.quantize_row(down)
        np.testing.assert_array_equal(
            merged, np.concatenate((qg.reshape(-1), qu.reshape(-1), qd.reshape(-1))))
        np.testing.assert_array_equal(
            scales, np.concatenate((sg, su, sd)))
        self.assertEqual(merged.dtype, np.int8)
        self.assertEqual(scales.dtype, np.float32)
        self.assertEqual(merged.size, inter * hidden * 2 + hidden * inter)
        self.assertEqual(scales.size, inter + inter + hidden)


class ConfigTest(unittest.TestCase):
    def _meta(self):
        return {
            "olmoe.embedding_length": 2048,
            "olmoe.block_count": 16,
            "olmoe.attention.head_count": 16,
            "olmoe.attention.head_count_kv": 16,
            "olmoe.expert_count": 64,
            "olmoe.expert_used_count": 8,
            "olmoe.feed_forward_length": 1024,
            "olmoe.rope.freq_base": 10000.0,
            "olmoe.attention.layer_norm_rms_epsilon": 1e-5,
            "tokenizer.ggml.tokens": ["a"] * 50304,
        }

    def test_build_config(self):
        config = profile.build_config(self._meta())
        self.assertEqual(config["hidden_size"], 2048)
        self.assertEqual(config["num_hidden_layers"], 16)
        self.assertEqual(config["num_attention_heads"], 16)
        self.assertEqual(config["num_key_value_heads"], 16)
        self.assertEqual(config["num_experts"], 64)
        self.assertEqual(config["num_experts_per_tok"], 8)
        self.assertEqual(config["intermediate_size"], 1024)
        self.assertEqual(config["vocab_size"], 50304)
        self.assertFalse(config["norm_topk_prob"])
        self.assertEqual(config["model_type"], "olmoe")
        self.assertEqual(config["architectures"], ["OlmoeForCausalLM"])

    def test_build_config_norm_topk_prob_override(self):
        config = profile.build_config(self._meta(), norm_topk_prob=True)
        self.assertTrue(config["norm_topk_prob"])

    def test_build_config_missing_key(self):
        meta = self._meta()
        del meta["olmoe.expert_count"]
        with self.assertRaises(ValueError):
            profile.build_config(meta)

    def test_build_config_missing_tokens(self):
        meta = self._meta()
        del meta["tokenizer.ggml.tokens"]
        with self.assertRaises(ValueError):
            profile.build_config(meta)


class TokenizerTest(unittest.TestCase):
    def _meta(self):
        return {
            "tokenizer.ggml.tokens": ["hello", " world", "<unk>", "Ġt", "a"],
            "tokenizer.ggml.merges": ["Ġ t", "Ġ Ġ"],
            "tokenizer.ggml.token_type": [1, 1, 3, 4, 6],
            "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.ggml.eos_token_id": 2,
            "tokenizer.ggml.padding_token_id": 0,
            "tokenizer.ggml.add_bos_token": False,
            "tokenizer.ggml.add_eos_token": False,
            "tokenizer.chat_template": "{{ bos_token }}...",
        }

    def test_vocab_and_merges(self):
        tokenizer = profile.build_tokenizer(self._meta())
        self.assertEqual(tokenizer["model"]["type"], "BPE")
        self.assertEqual(tokenizer["model"]["vocab"], {
            "hello": 0, " world": 1, "<unk>": 2, "Ġt": 3, "a": 4})
        self.assertEqual(tokenizer["model"]["merges"], ["Ġ t", "Ġ Ġ"])

    def test_added_tokens_only_special_and_user_defined(self):
        tokenizer = profile.build_tokenizer(self._meta())
        self.assertEqual(tokenizer["added_tokens"], [
            {"id": 2, "content": "<unk>", "special": True},
            {"id": 3, "content": "Ġt", "special": False},
        ])

    def test_special_ids(self):
        tokenizer = profile.build_tokenizer(self._meta())
        self.assertEqual(tokenizer["bos_token"], "<unk>")
        self.assertEqual(tokenizer["eos_token"], "<unk>")
        self.assertEqual(tokenizer["pad_token"], "hello")
        self.assertFalse(tokenizer["add_bos_token"])
        self.assertFalse(tokenizer["add_eos_token"])
        self.assertEqual(tokenizer["chat_template"], "{{ bos_token }}...")

    def test_missing_tokens_raises(self):
        meta = self._meta()
        del meta["tokenizer.ggml.tokens"]
        with self.assertRaises(ValueError):
            profile.build_tokenizer(meta)

    def test_byte_token_not_added(self):
        meta = self._meta()
        meta["tokenizer.ggml.token_type"] = [6, 6, 6, 6, 6]
        tokenizer = profile.build_tokenizer(meta)
        self.assertEqual(tokenizer["added_tokens"], [])

    def test_missing_token_type_defaults_to_normal(self):
        meta = self._meta()
        del meta["tokenizer.ggml.token_type"]
        tokenizer = profile.build_tokenizer(meta)
        self.assertEqual(tokenizer["added_tokens"], [])

    def test_partial_token_type_defaults_to_normal(self):
        meta = self._meta()
        meta["tokenizer.ggml.token_type"] = [3]  # only the first token is typed
        tokenizer = profile.build_tokenizer(meta)
        self.assertEqual(tokenizer["added_tokens"], [
            {"id": 0, "content": "hello", "special": True},
        ])


if __name__ == "__main__":
    unittest.main()
