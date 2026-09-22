#!/usr/bin/env python3
"""tools/convert_gguf_to_olmoe.py: end-to-end conversion of a tiny synthetic GGUF.

Builds a minimal OLMoE-shaped GGUF (all F16/F32, no K-quants) with the smallest
dims the engine accepts, runs the real CLI, and asserts the container c/olmoe.c
expects: config.json keys, tokenizer.json reconstruction, the dense name
mapping, and per-expert merged_weight (I8) + .qs (F32) at the exact sizes the
engine requires. Also covers --dry-run (writes nothing, creates no dir) and the
refusals for a non-OLMoE GGUF / unknown tensor; --remove-source-file removes
the source only after full success.
"""

import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import importlib

try:
    import numpy as np
    for _dependency in ("safetensors", "safetensors.numpy"):
        importlib.import_module(_dependency)
except ImportError as exc:
    raise unittest.SkipTest("numpy/safetensors not installed: %s" % exc)

TESTDIR = Path(__file__).resolve().parent
TOOLS = TESTDIR.parent / "tools"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(TESTDIR))

from gguf_fixture import (  # noqa: E402  (encoder helpers reused verbatim)
    METADATA_ARRAY,
    METADATA_BOOL,
    METADATA_FLOAT32,
    METADATA_STRING,
    METADATA_UINT32,
    _encode_string,
    _encode_value,
)

import convert_gguf_to_olmoe  # noqa: E402

# tiny but engine-valid dims
HIDDEN, INTER, N_EXPERTS, LAYERS, VOCAB, HEADS = 8, 4, 2, 1, 16, 2


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _f16_bytes(values):
    return np.asarray(values, dtype=np.float16).tobytes()


def _f32_bytes(values):
    return np.asarray(values, dtype=np.float32).tobytes()


def build_tiny_olmoe_gguf(path, arch="olmoe", extra_tensor=None, override=None):
    rng = np.random.default_rng(20260825)
    override = override or {}

    def m2d(hf_shape):
        return list(reversed(hf_shape))

    tensors = []
    # HF shape -> GGUF dims (reversed). Dense matrices F16, 1-D norms F32.
    def add_gguf(name, gguf_dims, ggml_type, payload):
        tensors.append({"name": name, "dims": gguf_dims,
                        "ggml_type": ggml_type, "payload": override.get(name, payload)})

    add_gguf("token_embd.weight", m2d([VOCAB, HIDDEN]), 1,
             _f16_bytes(rng.normal(size=VOCAB * HIDDEN)))
    add_gguf("output.weight", m2d([VOCAB, HIDDEN]), 1,
             _f16_bytes(rng.normal(size=VOCAB * HIDDEN)))
    add_gguf("output_norm.weight", [HIDDEN], 0, _f32_bytes(rng.normal(size=HIDDEN)))
    for layer in range(LAYERS):
        add_gguf("blk.%d.attn_norm.weight" % layer, [HIDDEN], 0,
                 _f32_bytes(rng.normal(size=HIDDEN)))
        add_gguf("blk.%d.ffn_norm.weight" % layer, [HIDDEN], 0,
                 _f32_bytes(rng.normal(size=HIDDEN)))
        for w in ("q", "k", "v", "output"):
            add_gguf("blk.%d.attn_%s.weight" % (layer, w), m2d([HIDDEN, HIDDEN]), 1,
                     _f16_bytes(rng.normal(size=HIDDEN * HIDDEN)))
        add_gguf("blk.%d.attn_q_norm.weight" % layer, [HIDDEN], 0,
                 _f32_bytes(rng.normal(size=HIDDEN)))
        add_gguf("blk.%d.attn_k_norm.weight" % layer, [HIDDEN], 0,
                 _f32_bytes(rng.normal(size=HIDDEN)))
        add_gguf("blk.%d.ffn_gate_inp.weight" % layer, m2d([N_EXPERTS, HIDDEN]), 1,
                 _f16_bytes(rng.normal(size=N_EXPERTS * HIDDEN)))
        add_gguf("blk.%d.ffn_gate_exps.weight" % layer, m2d([N_EXPERTS, INTER, HIDDEN]), 1,
                 _f16_bytes(rng.normal(size=N_EXPERTS * INTER * HIDDEN)))
        add_gguf("blk.%d.ffn_up_exps.weight" % layer, m2d([N_EXPERTS, INTER, HIDDEN]), 1,
                 _f16_bytes(rng.normal(size=N_EXPERTS * INTER * HIDDEN)))
        add_gguf("blk.%d.ffn_down_exps.weight" % layer, m2d([N_EXPERTS, HIDDEN, INTER]), 1,
                 _f16_bytes(rng.normal(size=N_EXPERTS * INTER * HIDDEN)))
    if extra_tensor:
        tensors.append(extra_tensor)

    metadata = [
        (METADATA_STRING, "general.architecture", arch),
        (METADATA_UINT32, "general.alignment", 32),
        (METADATA_UINT32, "olmoe.embedding_length", HIDDEN),
        (METADATA_UINT32, "olmoe.block_count", LAYERS),
        (METADATA_UINT32, "olmoe.attention.head_count", HEADS),
        (METADATA_UINT32, "olmoe.attention.head_count_kv", HEADS),
        (METADATA_UINT32, "olmoe.expert_count", N_EXPERTS),
        (METADATA_UINT32, "olmoe.expert_used_count", 2),
        (METADATA_UINT32, "olmoe.feed_forward_length", INTER),
        (METADATA_FLOAT32, "olmoe.rope.freq_base", 10000.0),
        (METADATA_FLOAT32, "olmoe.attention.layer_norm_rms_epsilon", 1e-5),
        (METADATA_ARRAY, "tokenizer.ggml.tokens",
         (METADATA_STRING, ["aa", "bb", "<unk>", "Ġt"])),
        (METADATA_ARRAY, "tokenizer.ggml.token_type",
         (METADATA_UINT32, [1, 1, 3, 4])),
        (METADATA_ARRAY, "tokenizer.ggml.merges",
         (METADATA_STRING, ["Ġ t"])),
        (METADATA_UINT32, "tokenizer.ggml.bos_token_id", 2),
        (METADATA_UINT32, "tokenizer.ggml.eos_token_id", 2),
        (METADATA_BOOL, "tokenizer.ggml.add_bos_token", False),
        (METADATA_BOOL, "tokenizer.ggml.add_eos_token", False),
    ]
    return write_gguf(path, metadata, tensors)


def write_gguf(path, metadata, tensors):
    alignment = 32
    out = b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))
    for value_type, key, value in metadata:
        out += _encode_string(key)
        out += struct.pack("<I", value_type)
        out += _encode_value(value_type, value)

    data_offset = 0
    data_base = _align(len(out) + sum(
        8 + len(t["name"].encode("utf-8")) + 4 + 8 * len(t["dims"]) + 4 + 8
        for t in tensors), alignment)
    for tensor in tensors:
        out += _encode_string(tensor["name"])
        out += struct.pack("<I", len(tensor["dims"]))
        for dim in tensor["dims"]:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", tensor["ggml_type"])
        out += struct.pack("<Q", data_offset)
        data_offset += len(tensor["payload"])
        data_offset = _align(data_offset, alignment)

    out += b"\x00" * (data_base - len(out))
    for tensor in tensors:
        out += tensor["payload"]
        out += b"\x00" * (_align(len(tensor["payload"]), alignment) - len(tensor["payload"]))
    Path(path).write_bytes(out)


def run_converter(input_path, output_path, *extra):
    return subprocess.run(
        [sys.executable, str(TOOLS / "convert_gguf_to_olmoe.py"),
         "--input", str(input_path), "--output", str(output_path), *extra],
        cwd=str(TOOLS.parent), capture_output=True, text=True,
        timeout=120)


def _rope_permute(w, n_head, n_kv_head=None):
    """llama.cpp's q/k RoPE permutation (conversion/llama.py), for the fixture."""
    n = n_kv_head if (n_kv_head and n_head != n_kv_head) else n_head
    return np.ascontiguousarray(
        w.reshape((n, 2, w.shape[0] // n // 2) + w.shape[1:]).swapaxes(1, 2).reshape(w.shape))


def expected_names():
    names = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    for layer in range(LAYERS):
        base = "model.layers.%d" % layer
        names.update({
            base + ".input_layernorm.weight",
            base + ".post_attention_layernorm.weight",
            base + ".self_attn.q_proj.weight",
            base + ".self_attn.k_proj.weight",
            base + ".self_attn.v_proj.weight",
            base + ".self_attn.o_proj.weight",
            base + ".self_attn.q_norm.weight",
            base + ".self_attn.k_norm.weight",
            base + ".mlp.gate.weight",
        })
    for layer in range(LAYERS):
        for expert in range(N_EXPERTS):
            names.add(base_merge_name(layer, expert))
            names.add("model.layers.%d.mlp.experts.%d.qs" % (layer, expert))
    return names


def base_merge_name(layer, expert):
    return "model.layers.%d.mlp.experts.%d.merged_weight" % (layer, expert)


class ConverterEndToEndTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.gguf = self.dir / "model.gguf"
        self.out = self.dir / "out"
        build_tiny_olmoe_gguf(str(self.gguf))

    def test_converts_full_container(self):
        result = run_converter(self.gguf, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)

        config = json.loads((self.out / "config.json").read_text())
        self.assertEqual(config["hidden_size"], HIDDEN)
        self.assertEqual(config["num_experts"], N_EXPERTS)
        self.assertEqual(config["vocab_size"], 4)
        self.assertFalse(config["norm_topk_prob"])

        tokenizer = json.loads((self.out / "tokenizer.json").read_text())
        self.assertEqual(tokenizer["model"]["vocab"], {"aa": 0, "bb": 1, "<unk>": 2, "Ġt": 3})

        present = set()
        from safetensors import safe_open
        for shard in sorted(self.out.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="np") as handle:
                present.update(handle.keys())

        self.assertEqual(present, expected_names())

        with safe_open(str(next(self.out.glob("model-*.safetensors"))), framework="np") as handle:
            merged = handle.get_tensor(base_merge_name(0, 0))
            qs = handle.get_tensor("model.layers.0.mlp.experts.0.qs")
            self.assertEqual(merged.dtype, np.int8)
            self.assertEqual(qs.dtype, np.float32)
            self.assertEqual(merged.size, INTER * HIDDEN * 2 + HIDDEN * INTER)
            self.assertEqual(qs.size, INTER + INTER + HIDDEN)

    def test_dry_run_creates_nothing(self):
        result = run_converter(self.gguf, self.out, "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.out.exists())

    def test_refuses_wrong_architecture(self):
        wrong = self.dir / "wrong.gguf"
        build_tiny_olmoe_gguf(str(wrong), arch="llama")
        result = run_converter(wrong, self.out)
        self.assertEqual(result.returncode, 1)
        self.assertIn("general.architecture", result.stderr)
        self.assertFalse(self.out.exists())

    def test_refuses_unknown_tensor(self):
        weird = self.dir / "weird.gguf"
        build_tiny_olmoe_gguf(str(weird), extra_tensor={
            "name": "blk.0.not_a_real_tensor",
            "dims": [HIDDEN],
            "ggml_type": 0,
            "payload": _f32_bytes(np.zeros(HIDDEN)),
        })
        result = run_converter(weird, self.out)
        self.assertEqual(result.returncode, 1)
        self.assertIn("unrecognized tensors", result.stderr)

    def test_remove_source_only_on_success(self):
        result = run_converter(self.gguf, self.out, "--remove-source-file")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.gguf.exists())
        self.assertTrue((self.out / "config.json").is_file())

    def test_qk_left_as_stored_by_default(self):
        rng = np.random.default_rng(99)
        hf_q_proj = rng.normal(size=(HIDDEN, HIDDEN)).astype(np.float16)
        permuted_q_proj = _rope_permute(hf_q_proj, HEADS)
        self.assertFalse(np.array_equal(permuted_q_proj, hf_q_proj))
        build_tiny_olmoe_gguf(str(self.gguf), override={
            "blk.0.attn_q.weight": permuted_q_proj.tobytes(),
        })

        result = run_converter(self.gguf, self.out)
        self.assertEqual(result.returncode, 0, result.stderr)
        got = self._read("model.layers.0.self_attn.q_proj.weight")
        # no flag: the stored (permuted) layout is passed through unchanged
        np.testing.assert_array_equal(got, permuted_q_proj)

    def test_qk_permuted_flag_undoes_rope_permutation(self):
        rng = np.random.default_rng(99)
        hf_q_proj = rng.normal(size=(HIDDEN, HIDDEN)).astype(np.float16)
        hf_k_proj = rng.normal(size=(HIDDEN, HIDDEN)).astype(np.float16)
        build_tiny_olmoe_gguf(str(self.gguf), override={
            "blk.0.attn_q.weight": _rope_permute(hf_q_proj, HEADS).tobytes(),
            "blk.0.attn_k.weight": _rope_permute(hf_k_proj, HEADS).tobytes(),
        })

        result = run_converter(self.gguf, self.out, "--qk-permuted")
        self.assertEqual(result.returncode, 0, result.stderr)
        np.testing.assert_array_equal(
            self._read("model.layers.0.self_attn.q_proj.weight"), hf_q_proj)
        np.testing.assert_array_equal(
            self._read("model.layers.0.self_attn.k_proj.weight"), hf_k_proj)

    def _read(self, name):
        from safetensors import safe_open
        for shard in sorted(self.out.glob("model-*.safetensors")):
            with safe_open(str(shard), framework="np") as handle:
                if name in handle.keys():
                    return handle.get_tensor(name)
        raise KeyError(name)

    def test_overwrite_rewrites_existing_container(self):
        self.assertEqual(run_converter(self.gguf, self.out).returncode, 0)
        (self.out / "model-99999.safetensors").write_bytes(b"stale")
        result = run_converter(self.gguf, self.out, "--overwrite")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.out / "model-99999.safetensors").exists())

    def test_refuses_existing_container_without_flag(self):
        self.assertEqual(run_converter(self.gguf, self.out).returncode, 0)
        before = {p.name: p.read_bytes() for p in self.out.glob("model-*.safetensors")}
        result = run_converter(self.gguf, self.out)
        self.assertEqual(result.returncode, 1)
        self.assertIn("already contains a converted container", result.stderr)
        self.assertIn("--overwrite", result.stderr)
        after = {p.name: p.read_bytes() for p in self.out.glob("model-*.safetensors")}
        self.assertEqual(before, after)

    def test_dense_storage_dtypes(self):
        self.assertEqual(
            convert_gguf_to_olmoe._dense_to_storage(np.zeros(HIDDEN, np.float32)).dtype,
            np.float32)
        self.assertEqual(
            convert_gguf_to_olmoe._dense_to_storage(np.zeros((HIDDEN, HIDDEN), np.float32)).dtype,
            np.float16)


if __name__ == "__main__":
    unittest.main()