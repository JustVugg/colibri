"""GGUF -> colibri directory: both modes give the same model, the planner agrees.

A small but complete qwen35moe GGUF is written with gguf-py (every tensor the
engine reads, mixed ggml types as a UD file has them, a byte tokenizer), then
imported with tools/gguf_import.py in link and in copy mode. The engine must
produce the same tokens from the bare file, the link directory and the copy
container, on the float and on the integer path; the planner must see the
same experts in both directories; and in the container every expert slot must
start on a 4 KiB boundary of the file.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import gguf
    from gguf.constants import GGMLQuantizationType as QT
    from test_ggml_quants import make_blocks, TYPES
except ImportError:  # pragma: no cover
    gguf = None

H, LAYERS, E, TOPK, I, VOCAB = 256, 4, 8, 2, 256, 300
QH, KVH, HD, KH, VH, KD = 4, 2, 64, 4, 8, 32


def blocks(name, shape, rng, std=0.02):
    """Synthetic blocks of `name` with realistic magnitude (centred)."""
    blck, size, f16 = TYPES[name]
    rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
    raw = make_blocks(name, rows * nb, rng)
    probe = gguf.quants.dequantize(raw.reshape(-1), QT[name])
    s = float(np.std(probe)) or 1.0
    for off in f16:
        v = np.frombuffer(raw[:, off:off + 2].tobytes(), dtype=np.float16).astype(np.float32) * (std / s)
        raw[:, off:off + 2] = v.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return raw.reshape(*shape[:-1], nb * size)


def write_model(path, rng, dense=False, same_gate_up=False, token=None, ollama=False):
    """dense: llama.cpp's qwen35 (a dense model of the family): an MLP per layer instead of the
    experts, and no output.weight (the head tied to token_embd), as the small checkpoints ship.
    same_gate_up: every layer's expert up in its gate's type (layer 0 IQ2_XXS, the others
    Q4_K), the pairing the Vulkan tier keeps as the file's blocks.
    ollama: the file as Ollama's converter writes it (the blob of qwen3.6:35b): head_count_kv
    an array per layer (0 on the DeltaNet layers), rope.dimension_sections of three, the flags
    ssm.v_head_reordered and rope.mrope_interleaved, the DeltaNet's dt bias named ssm_dt, the
    shared expert's gate as F16 of shape [H, 1], and the MTP head (mtp.*) and the vision tower
    (v.*) in the model's file. The same weights as without."""
    arch = "qwen35" if dense else "qwen35moe"
    w = gguf.GGUFWriter(str(path), arch)
    p = arch + "."
    moe = (("feed_forward_length", I),) if dense else \
        (("expert_count", E), ("expert_used_count", TOPK), ("expert_feed_forward_length", I), ("expert_shared_feed_forward_length", I))
    for k, v in (("block_count", LAYERS), ("embedding_length", H), ("attention.head_count", QH),
                 *(() if ollama else (("attention.head_count_kv", KVH),)), ("attention.key_length", HD), ("attention.value_length", HD),
                 *moe, ("full_attention_interval", 4),
                 ("rope.dimension_count", HD // 4), ("ssm.conv_kernel", 4), ("ssm.state_size", KD),
                 ("ssm.group_count", KH), ("ssm.time_step_rank", VH), ("ssm.inner_size", VH * KD),
                 ("context_length", 4096)):
        w.add_uint32(p + k, v)
    if ollama:
        w.add_array(p + "attention.head_count_kv", [KVH if (L + 1) % 4 == 0 else 0 for L in range(LAYERS)])
        w.add_array(p + "rope.dimension_sections", [HD // 16, HD // 16, HD // 8])
        w.add_bool(p + "ssm.v_head_reordered", True); w.add_bool(p + "rope.mrope_interleaved", True)
        w.add_uint32(p + "vision.block_count", 1); w.add_uint32(p + "image_token_id", VOCAB - 2)
    w.add_float32(p + "rope.freq_base", 1e7); w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-6)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    if token:   # token: one of the vocabulary's words in its place (a non-ASCII one, say)
        toks[VOCAB - 4] = token
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(VOCAB - 1); w.add_bos_token_id(VOCAB - 3)

    def f32(name, shape): w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, H), "Q8_0")
    if not dense:
        q("output.weight", (VOCAB, H), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(H, np.float32))
    conv = 2 * KH * KD + VH * KD
    for L in range(LAYERS):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", np.ones(H, np.float32))
        w.add_tensor(b + "post_attention_norm.weight", np.ones(H, np.float32))
        if dense:
            q(b + "ffn_gate.weight", (I, H), "Q4_K"); q(b + "ffn_up.weight", (I, H), "Q4_K")
            q(b + "ffn_down.weight", (H, I), "Q6_K" if L % 2 else "Q5_K")
        else:
            f32(b + "ffn_gate_inp.weight", (E, H))
            sg = (rng.standard_normal((H,)) * 0.02).astype(np.float32)
            if ollama: w.add_tensor(b + "ffn_gate_inp_shexp.weight", sg.astype(np.float16).reshape(1, H))
            else: w.add_tensor(b + "ffn_gate_inp_shexp.weight", sg.astype(np.float16).astype(np.float32))
            q(b + "ffn_gate_shexp.weight", (I, H), "Q8_0"); q(b + "ffn_up_shexp.weight", (I, H), "Q8_0")
            q(b + "ffn_down_shexp.weight", (H, I), "Q8_0")
            q(b + "ffn_gate_exps.weight", (E, I, H), "IQ2_XXS" if L == 0 else "Q4_K")
            q(b + "ffn_up_exps.weight", (E, I, H), "IQ2_XXS" if same_gate_up and L == 0 else "Q4_K")
            q(b + "ffn_down_exps.weight", (E, H, I), "Q6_K" if L % 2 else "Q5_K")
        if (L + 1) % 4 == 0:
            q(b + "attn_q.weight", (QH * HD * 2, H), "Q8_0"); q(b + "attn_k.weight", (KVH * HD, H), "Q8_0")
            q(b + "attn_v.weight", (KVH * HD, H), "Q8_0"); q(b + "attn_output.weight", (H, QH * HD), "Q8_0")
            w.add_tensor(b + "attn_q_norm.weight", np.ones(HD, np.float32)); w.add_tensor(b + "attn_k_norm.weight", np.ones(HD, np.float32))
        else:
            q(b + "attn_qkv.weight", (conv, H), "Q8_0"); q(b + "attn_gate.weight", (VH * KD, H), "Q8_0")
            w.add_tensor(b + "ssm_a", -np.exp(rng.standard_normal(VH).astype(np.float32) * 0.1))
            f32(b + "ssm_alpha.weight", (VH, H)); f32(b + "ssm_beta.weight", (VH, H))
            f32(b + "ssm_conv1d.weight", (conv, 4)); w.add_tensor(b + ("ssm_dt" if ollama else "ssm_dt.bias"), np.zeros(VH, np.float32))
            w.add_tensor(b + "ssm_norm.weight", np.ones(KD, np.float32))
            q(b + "ssm_out.weight", (H, VH * KD), "Q8_0")
    if ollama:   # the MTP head and the vision tower, in the model's file
        f32("mtp.fc.weight", (H, 2 * H)); f32("mtp.layers.0.attn_norm.weight", (H,)); f32("mtp.layers.0.ffn_gate_inp.weight", (E, H))
        f32("v.blk.0.attn_q.weight", (H, H)); f32("v.blk.0.attn_q.bias", (H,)); f32("v.patch_embd.weight", (H, 3, 2, 2))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_mimo2(path, rng, types=None):
    """llama.cpp's mimo2 (MiMo-V2.6), small: a full-attention layer with a dense MLP, then
    sliding-window layers with sinks and routed experts in a GGUF's block types as a K-quant
    file has them (Q4_K gate and up, Q6_K and Q5_K down, IQ2_XXS in one layer), qkv as one
    [Q|K|V], the head Q6_K and the embedding Q8_0 (the mimo engine's GGUF gates). types: a
    tensor's GGUF name -> its type instead."""
    types = types or {}
    w = gguf.GGUFWriter(str(path), "mimo2")
    p = "mimo2."
    layers, h, mi, e, k, heads, hd, vd = 4, 256, 256, 8, 2, 4, 48, 32
    pattern = [0, 1, 1, 1]
    kv = [2 if x == 0 else 4 for x in pattern]
    for key, v in (("block_count", layers), ("embedding_length", h), ("feed_forward_length", 256),
                   ("attention.head_count", heads), ("attention.key_length", hd), ("attention.value_length", vd),
                   ("attention.sliding_window", 8), ("expert_count", e), ("expert_used_count", k),
                   ("expert_feed_forward_length", mi), ("rope.dimension_count", 16), ("context_length", 256)):
        w.add_uint32(p + key, v)
    w.add_array(p + "attention.head_count_kv", kv)
    w.add_array(p + "attention.sliding_window_pattern", pattern)
    w.add_float32(p + "rope.freq_base", 1e7); w.add_float32(p + "rope.freq_base_swa", 1e4)
    w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-6); w.add_float32(p + "attention.value_scale", 0.707)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(VOCAB - 1); w.add_bos_token_id(VOCAB - 3)

    def f32(name, shape, std=0.02): w.add_tensor(name, (rng.standard_normal(shape) * std).astype(np.float32))
    def q(name, shape, t):
        t = types.get(name, t)
        if t == "F32": w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32)); return
        w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, h), "Q8_0"); q("output.weight", (VOCAB, h), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(h, np.float32))
    for L in range(layers):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", np.ones(h, np.float32)); w.add_tensor(b + "ffn_norm.weight", np.ones(h, np.float32))
        q(b + "attn_qkv.weight", (heads * hd + kv[L] * (hd + vd), h), "Q4_K" if L % 2 else "Q8_0")
        q(b + "attn_output.weight", (h, heads * vd), "Q8_0")
        if pattern[L]:
            f32(b + "attn_sinks.weight", (heads,), 0.5)
        if L == 0:
            q(b + "ffn_gate.weight", (256, h), "Q4_K"); q(b + "ffn_up.weight", (256, h), "Q4_K")
            q(b + "ffn_down.weight", (h, 256), "Q6_K")
            continue
        f32(b + "ffn_gate_inp.weight", (e, h), 0.5); w.add_tensor(b + "exp_probs_b.bias", np.zeros(e, np.float32))
        gu = "IQ2_XXS" if L == 1 else "Q4_K"
        q(b + "ffn_gate_exps.weight", (e, mi, h), gu); q(b + "ffn_up_exps.weight", (e, mi, h), gu)
        q(b + "ffn_down_exps.weight", (e, h, mi), "Q6_K" if L % 2 else "Q5_K")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_glmdsa(path, rng):
    """llama.cpp's glm-dsa (GLM-5.2), small: a dense first layer, then MoE layers with a shared
    expert; MLA with kv_b split as k_b and v_b in Q8_0 (the colibri engine keeps them as the
    file's blocks), the DSA indexer, experts in Q4_K and IQ2_XXS with Q5_K and Q6_K down, a
    Q4_K and Q8_0 trunk, the head Q6_K and the embedding Q8_0 (colibri's GGUF chain gates)."""
    w = gguf.GGUFWriter(str(path), "glm-dsa")
    p = "glm-dsa."
    layers, h, mi, e, k, heads, ql, kvl, nope, rope, vd, ih, idh = 4, 256, 256, 8, 2, 4, 64, 32, 32, 8, 32, 2, 16
    for key, v in (("block_count", layers), ("embedding_length", h), ("feed_forward_length", 256),
                   ("expert_feed_forward_length", mi), ("attention.head_count", heads), ("rope.dimension_count", rope),
                   ("expert_count", e), ("expert_used_count", k), ("expert_shared_count", 1), ("leading_dense_block_count", 1),
                   ("attention.q_lora_rank", ql), ("attention.kv_lora_rank", kvl), ("attention.key_length_mla", nope + rope),
                   ("attention.value_length_mla", vd), ("attention.indexer.top_k", 4096), ("attention.indexer.head_count", ih),
                   ("attention.indexer.key_length", idh), ("context_length", 512)):
        w.add_uint32(p + key, v)
    w.add_float32(p + "rope.freq_base", 1e4); w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-5)
    w.add_float32(p + "expert_weights_scale", 1.0); w.add_bool(p + "expert_weights_norm", True)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|endoftext|>", "<|user|>", "<|assistant|>"]
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(VOCAB - 3); w.add_bos_token_id(VOCAB - 2)

    def f32(name, shape, std=0.02): w.add_tensor(name, (rng.standard_normal(shape) * std).astype(np.float32))
    def ones(name, n): w.add_tensor(name, np.ones(n, np.float32))
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, h), "Q8_0"); q("output.weight", (VOCAB, h), "Q6_K"); ones("output_norm.weight", h)
    for L in range(layers):
        b = f"blk.{L}."
        ones(b + "attn_norm.weight", h); ones(b + "ffn_norm.weight", h)
        q(b + "attn_q_a.weight", (ql, h), "Q4_K"); ones(b + "attn_q_a_norm.weight", ql)
        q(b + "attn_q_b.weight", (heads * (nope + rope), ql), "Q8_0")
        q(b + "attn_kv_a_mqa.weight", (kvl + rope, h), "Q4_K"); ones(b + "attn_kv_a_norm.weight", kvl)
        q(b + "attn_k_b.weight", (heads * kvl, nope), "Q8_0"); q(b + "attn_v_b.weight", (heads * vd, kvl), "Q8_0")
        q(b + "attn_output.weight", (h, heads * vd), "Q8_0")
        q(b + "indexer.attn_q_b.weight", (ih * idh, ql), "Q8_0"); q(b + "indexer.attn_k.weight", (idh, h), "Q8_0")
        f32(b + "indexer.proj.weight", (ih, h)); ones(b + "indexer.k_norm.weight", idh)
        w.add_tensor(b + "indexer.k_norm.bias", np.zeros(idh, np.float32))
        if L == 0:
            q(b + "ffn_gate.weight", (256, h), "Q4_K"); q(b + "ffn_up.weight", (256, h), "Q4_K"); q(b + "ffn_down.weight", (h, 256), "Q6_K")
            continue
        f32(b + "ffn_gate_inp.weight", (e, h), 0.5); w.add_tensor(b + "exp_probs_b.bias", np.zeros(e, np.float32))
        q(b + "ffn_gate_shexp.weight", (mi, h), "Q8_0"); q(b + "ffn_up_shexp.weight", (mi, h), "Q8_0")
        q(b + "ffn_down_shexp.weight", (h, mi), "Q8_0")
        gu = "IQ2_XXS" if L == 1 else "Q4_K"
        q(b + "ffn_gate_exps.weight", (e, mi, h), gu); q(b + "ffn_up_exps.weight", (e, mi, h), gu)
        q(b + "ffn_down_exps.weight", (e, h, mi), "Q6_K" if L % 2 else "Q5_K")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_deepseek41(path, rng):
    """A published deepseek41 GGUF (DeepSeek V4.1 Flash), small: no deepseek41.config, the
    config as keys; the compressors (KV sources 1 and 3, a gate where the ratio is 2) and
    the indexer (1, 3 and 4) in BF16; the experts in IQ1_S as the UD-IQ1_S file has them,
    with IQ2_XXS, Q4_K and Q5_K in other layers; the trunk in Q5_K, Q6_K and Q8_0; every
    engram table in one IQ4_NL per_layer_token_embd under global head offsets, and each
    engram layer's gate as one engram_gate (deepseek_v41's GGUF gates)."""
    w = gguf.GGUFWriter(str(path), "deepseek41")
    p = "deepseek41."
    VT = gguf.GGUFValueType
    layers, h, mi, e, k, heads, hd, rope, ql, ol, og = 6, 256, 256, 8, 2, 4, 64, 16, 64, 32, 2
    hc, ih, ihd, ngram, eh, ehd, evocab = 4, 4, 32, 4, 2, 32, 97
    ratios, kv_src, idx_src, eng = [0, 2, 2, 1, 1, 0], [1, 3], [1, 3, 4], [1, 4]
    for key, v in (("block_count", layers), ("context_length", 4096), ("embedding_length", h),
                   ("attention.head_count", heads), ("attention.head_count_kv", 1), ("expert_count", e),
                   ("expert_used_count", k), ("expert_gating_func", 4), ("attention.key_length", hd),
                   ("attention.value_length", hd), ("rope.dimension_count", rope), ("attention.q_lora_rank", ql),
                   ("attention.sliding_window", 8), ("expert_feed_forward_length", mi), ("expert_shared_count", 1),
                   ("attention.indexer.head_count", ih), ("attention.indexer.key_length", ihd),
                   ("attention.indexer.top_k", 4), ("attention.output_group_count", og),
                   ("attention.output_lora_rank", ol), ("hyper_connection.count", hc),
                   ("hyper_connection.sinkhorn_iterations", 20), ("attention.indexer.candidate_source_layer", 3),
                   ("attention.indexer.candidate_top_k_blocks", 2), ("attention.indexer.candidate_block_size", 2),
                   ("rope.scaling.original_context_length", 32), ("ple.ngram_size", ngram),
                   ("ple.heads_per_ngram", eh), ("embedding_length_per_layer_input", ehd),
                   ("engram.vocab_size", evocab), ("engram.pad_token_id", 2), ("engram.compressed_vocab_size", 64)):
        w.add_uint32(p + key, v)
    for key, v in (("rope.freq_base", 10000.0), ("rope.scaling.factor", 16.0), ("attention.layer_norm_rms_epsilon", 1e-20),
                   ("expert_weights_scale", 1.5), ("attention.compress_rope_freq_base", 40000.0),
                   ("hyper_connection.epsilon", 1e-6), ("rope.scaling.yarn_beta_fast", 32.0),
                   ("rope.scaling.yarn_beta_slow", 1.0)):
        w.add_float32(p + key, v)
    w.add_string(p + "rope.scaling.type", "yarn"); w.add_bool(p + "expert_weights_norm", True)
    w.add_array(p + "swiglu_clamp_exp", [10.0] * layers); w.add_array(p + "swiglu_clamp_shexp", [10.0] * layers)
    w.add_array(p + "attention.compress_ratios", ratios); w.add_array(p + "attention.kv_source_layers", kv_src)
    w.add_array(p + "attention.indexer.source_layers", idx_src); w.add_array(p + "ple.layers", eng)
    for key, v in (("dspark.block_size", 3), ("dspark.noise_token_id", 7), ("dspark.markov_rank", 32),
                   ("dspark.expert_count", 4), ("dspark.expert_used_count", 2)):   # write_dflash's head
        w.add_uint32(p + key, v)
    w.add_array(p + "dspark.target_layers", [3, 4, 5])
    # The engram: per layer, (ngram - 1) * heads buckets of distinct primes above the
    # engram vocabulary, their offsets running over every layer's (one tensor holds all),
    # and a layer's ngram odd hash multipliers, past 32 bits as the file's are.
    primes, c = [], evocab - 1
    while len(primes) < len(eng) * (ngram - 1) * eh:
        c += 1
        if all(c % d for d in range(2, int(c ** 0.5) + 1)):
            primes.append(c)
    offsets = [sum(primes[:i]) for i in range(len(primes))]
    mult = [int(x) * 2 + 1 for x in rng.integers(1 << 40, 1 << 50, size=len(eng) * ngram)]
    w.add_key_value(p + "ple.head_vocab_sizes", primes, VT.ARRAY, sub_type=VT.UINT32)
    w.add_key_value(p + "ple.head_offsets", offsets, VT.ARRAY, sub_type=VT.UINT64)
    w.add_key_value(p + "ple.layer_multipliers", mult, VT.ARRAY, sub_type=VT.UINT64)
    w.add_array(p + "engram.token_map", [i % 64 for i in range(VOCAB)])
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|begin_of_sentence|>", "<|end_of_sentence|>", "<|pad|>"]
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_bos_token_id(VOCAB - 3); w.add_eos_token_id(VOCAB - 2)

    def f32(name, shape, std=0.02, base=0.0):
        w.add_tensor(name, (base + rng.standard_normal(shape) * std).astype(np.float32))
    def bf16(name, shape, std=0.02, base=0.0):
        v = (base + rng.standard_normal(shape) * std).astype(np.float32)
        raw = (v.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(*shape[:-1], shape[-1] * 2)
        w.add_tensor(name, raw, raw_dtype=QT.BF16)
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, h), "Q8_0"); q("output.weight", (VOCAB, h), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(h, np.float32))
    q("per_layer_token_embd.weight", (sum(primes), ehd), "IQ4_NL")
    mix = (2 + hc) * hc
    for L in range(layers):
        b = f"blk.{L}."
        f32(b + "attn_norm.weight", (h,), 0.01, 1.0); f32(b + "ffn_norm.weight", (h,), 0.01, 1.0)
        for which in ("attn", "ffn"):
            f32(b + f"hc_{which}_fn.weight", (mix, hc * h), 0.01); f32(b + f"hc_{which}_base.weight", (mix,), 0.1)
            w.add_tensor(b + f"hc_{which}_scale.weight", np.full(3, 0.5, np.float32))
        f32(b + "attn_sinks.weight", (heads,), 0.1)
        q(b + "attn_q_a.weight", (ql, h), "Q5_K" if L % 2 else "Q6_K")
        w.add_tensor(b + "attn_q_a_norm.weight", np.ones(ql, np.float32))
        q(b + "attn_q_b.weight", (heads * hd, ql), "Q8_0")
        q(b + "attn_kv.weight", (hd, h), "Q6_K")
        w.add_tensor(b + "attn_kv_a_norm.weight", np.ones(hd, np.float32))
        q(b + "attn_output_a.weight", (og * ol, heads * hd // og), "Q8_0")
        q(b + "attn_output_b.weight", (h, og * ol), "Q8_0")
        if L in kv_src:
            bf16(b + "attn_compressor_kv.weight", (hd, h))
            w.add_tensor(b + "attn_compressor_norm.weight", np.ones(hd, np.float32))
            if ratios[L] > 1:
                bf16(b + "attn_compressor_gate.weight", (hd, h))
        if L in idx_src:
            bf16(b + "indexer.attn_q_b.weight", (ih * ihd, ql)); bf16(b + "indexer.proj.weight", (ih, h))
            if L in kv_src:
                bf16(b + "indexer.attn_k.weight", (ihd, hd))
                w.add_tensor(b + "indexer.k_norm.weight", np.ones(ihd, np.float32))
        f32(b + "ffn_gate_inp.weight", (e, h), 0.05)
        f32(b + "exp_probs_b.bias", (e,), 0.1); f32(b + "exp_probs_b_vl.bias", (e,), 0.1)
        q(b + "ffn_gate_shexp.weight", (mi, h), "Q6_K" if L % 2 else "Q8_0")
        q(b + "ffn_up_shexp.weight", (mi, h), "Q6_K" if L % 2 else "Q8_0")
        q(b + "ffn_down_shexp.weight", (h, mi), "Q6_K")
        gu = {0: "IQ2_XXS", 5: "Q4_K"}.get(L, "IQ1_S")
        q(b + "ffn_gate_exps.weight", (e, mi, h), gu); q(b + "ffn_up_exps.weight", (e, mi, h), gu)
        q(b + "ffn_down_exps.weight", (e, h, mi), {0: "IQ2_XXS", 5: "Q5_K"}.get(L, "IQ1_S"))
        if L in eng:
            f32(b + "engram_gate.weight", (hc, h), 0.05, 1.0)
            q(b + "engram_kv.weight", (h * (hc + 1), (ngram - 1) * eh * ehd), "Q8_0")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_dflash(path, rng, targets=(3, 4, 5)):
    """write_deepseek41's DSpark head as the published files ship it, a GGUF of its own (arch
    dflash): two stages as blk.<s> under V4.1's layer names, MXFP4 routed experts (4, 2 a
    token) and Q8_0 projections, and beside them fc (the target layers 3, 4 and 5 into the
    width), enc.output_norm, output_norm, conf_proj and the BF16 Markov head (rank 32). The
    experts' e8m0 scales are 2^-7 to 2^-4 (make_blocks draws any byte, 2^127 among them)."""
    w = gguf.GGUFWriter(str(path), "dflash")
    p = "dflash."
    stages, h, mi, e, k, heads, hd, ql, ol, og, hc, rank = 2, 256, 256, 4, 2, 4, 64, 64, 32, 2, 4, 32
    for key, v in (("block_count", stages), ("context_length", 4096), ("embedding_length", h),
                   ("feed_forward_length", mi), ("attention.head_count", heads), ("attention.head_count_kv", 1),
                   ("attention.key_length", hd), ("attention.value_length", hd), ("rope.dimension_count", 16),
                   ("attention.q_lora_rank", ql), ("attention.sliding_window", 8), ("expert_count", e),
                   ("expert_used_count", k), ("expert_feed_forward_length", mi), ("expert_shared_count", 1),
                   ("expert_gating_func", 4), ("attention.output_group_count", og), ("attention.output_lora_rank", ol),
                   ("hyper_connection.count", hc), ("hyper_connection.sinkhorn_iterations", 20), ("block_size", 3)):
        w.add_uint32(p + key, v)
    w.add_array(p + "target_layers", list(targets)); w.add_array(p + "attention.compress_ratios", [0] * stages)

    def f32(name, shape, std=0.02, base=0.0):
        w.add_tensor(name, (base + rng.standard_normal(shape) * std).astype(np.float32))
    def bf16(name, shape, std=0.05):
        v = (rng.standard_normal(shape) * std).astype(np.float32)
        w.add_tensor(name, (v.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(*shape[:-1], shape[-1] * 2),
                     raw_dtype=QT.BF16)
    def q(name, shape, t):
        if t == "MXFP4":   # each block's e8m0 byte first, then its 16 bytes of codes
            rows, nb = int(np.prod(shape[:-1])), shape[-1] // 32
            raw = make_blocks("MXFP4", rows * nb, rng)
            raw[:, 0] = rng.integers(120, 124, size=rows * nb, dtype=np.uint8)
            raw = raw.reshape(*shape[:-1], nb * 17)
        else:
            raw = blocks(t, shape, rng)
        w.add_tensor(name, raw, raw_dtype=QT[t])
    mix = (2 + hc) * hc
    for S in range(stages):
        b = f"blk.{S}."
        f32(b + "attn_norm.weight", (h,), 0.01, 1.0); f32(b + "ffn_norm.weight", (h,), 0.01, 1.0)
        for which in ("attn", "ffn"):
            f32(b + f"hc_{which}_fn.weight", (mix, hc * h), 0.01); f32(b + f"hc_{which}_base.weight", (mix,), 0.1)
            w.add_tensor(b + f"hc_{which}_scale.weight", np.full(3, 0.5, np.float32))
        f32(b + "attn_sinks.weight", (heads,), 0.1)
        q(b + "attn_q_a.weight", (ql, h), "Q8_0"); w.add_tensor(b + "attn_q_a_norm.weight", np.ones(ql, np.float32))
        q(b + "attn_q_b.weight", (heads * hd, ql), "Q8_0")
        q(b + "attn_kv.weight", (hd, h), "Q8_0"); w.add_tensor(b + "attn_kv_a_norm.weight", np.ones(hd, np.float32))
        q(b + "attn_output_a.weight", (og * ol, heads * hd // og), "Q8_0"); q(b + "attn_output_b.weight", (h, og * ol), "Q8_0")
        f32(b + "ffn_gate_inp.weight", (e, h), 0.05); f32(b + "exp_probs_b.bias", (e,), 0.1)
        q(b + "ffn_gate_shexp.weight", (mi, h), "Q8_0"); q(b + "ffn_up_shexp.weight", (mi, h), "Q8_0")
        q(b + "ffn_down_shexp.weight", (h, mi), "Q8_0")
        q(b + "ffn_gate_exps.weight", (e, mi, h), "MXFP4"); q(b + "ffn_up_exps.weight", (e, mi, h), "MXFP4")
        q(b + "ffn_down_exps.weight", (e, h, mi), "MXFP4")
    q("fc.weight", (h, len(targets) * h), "Q8_0")
    w.add_tensor("enc.output_norm.weight", np.ones(h, np.float32)); w.add_tensor("output_norm.weight", np.ones(h, np.float32))
    f32("conf_proj.weight", (1, h + rank), 0.05)
    bf16("markov_w1.weight", (VOCAB, rank)); bf16("markov_w2.weight", (VOCAB, rank))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def write_kimik3(path, rng):
    """A published kimi-k3 GGUF (Kimi K3), small: five layers, the first dense, the fourth MLA
    and the rest KDA (head_count_kv 1 and 0); the routed experts in the latent between
    ffn_routed_up and ffn_routed_down, IQ1_S and IQ2_XXS with IQ1_S down as UD-IQ1_S has
    them, a Q6_K and Q8_0 trunk; res_score (res_norm * res_proj), ssm_a (-exp(A_log)) and
    MLA's kv_b split per head into k_b (transposed) and v_b, as the file has them."""
    w = gguf.GGUFWriter(str(path), "kimi-k3")
    p = "kimi-k3."
    layers, h, lat, mi, e, k, heads, kdh, dense = 5, 256, 256, 256, 8, 2, 2, 128, 512
    ql, kvl, nope, rope, vh = 64, 64, 32, 16, 32
    kv = [0, 0, 0, 1, 0]
    P = heads * kdh
    for key, v in (("block_count", layers), ("context_length", 4096), ("embedding_length", h),
                   ("feed_forward_length", dense), ("attention.head_count", heads), ("expert_count", e),
                   ("expert_used_count", k), ("expert_group_used_count", 1), ("expert_gating_func", 2),
                   ("attention.key_length", kvl + rope), ("attention.value_length", kvl), ("ssm.conv_kernel", 4),
                   ("kda.head_dim", kdh), ("attention.q_lora_rank", ql), ("attention.kv_lora_rank", kvl),
                   ("rope.dimension_count", rope), ("attention.key_length_mla", nope + rope),
                   ("attention.value_length_mla", vh), ("expert_feed_forward_length", mi), ("expert_shared_count", 1),
                   ("leading_dense_block_count", 1), ("expert_latent_length", lat), ("attn_res.block_size", 2)):
        w.add_uint32(p + key, v)
    w.add_array(p + "attention.head_count_kv", kv)
    for key, v in (("rope.freq_base", 10000.0), ("attention.layer_norm_rms_epsilon", 1e-5), ("expert_weights_scale", 1.0),
                   ("kda.gate_lower_bound", -5.0), ("activation.situ_beta", 4.0), ("activation.situ_linear_beta", 25.0)):
        w.add_float32(p + key, v)
    w.add_bool(p + "expert_weights_norm", True)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["[BOS]", "[EOS]", "[PAD]"]
    w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("kimi-k2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_bos_token_id(VOCAB - 3); w.add_eos_token_id(VOCAB - 2)

    def f32(name, shape, std=0.02, base=0.0):
        w.add_tensor(name, (base + rng.standard_normal(shape) * std).astype(np.float32))
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, h), "Q8_0"); q("output.weight", (VOCAB, h), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(h, np.float32)); f32("output_res_score.weight", (h,), 0.05)
    for L in range(layers):
        b = f"blk.{L}."
        f32(b + "attn_norm.weight", (h,), 0.01, 1.0); f32(b + "ffn_norm.weight", (h,), 0.01, 1.0)
        f32(b + "attn_res_score.weight", (h,), 0.05); f32(b + "ffn_res_score.weight", (h,), 0.05)
        if kv[L] == 0:   # KDA
            q(b + "attn_q.weight", (P, h), "Q6_K"); q(b + "attn_k.weight", (P, h), "Q6_K")
            q(b + "attn_v.weight", (P, h), "Q6_K"); q(b + "ssm_g.weight", (P, h), "Q6_K")
            q(b + "attn_output.weight", (h, P), "Q6_K")
            for c3 in "qkv":
                f32(b + f"ssm_conv1d_{c3}.weight", (P, 1, 4), 0.2)
            q(b + "ssm_f_a.weight", (kdh, h), "Q6_K"); q(b + "ssm_f_b.weight", (P, kdh), "Q8_0")
            f32(b + "ssm_beta.weight", (heads, h), 0.05); f32(b + "ssm_dt.bias", (P,), 0.1)
            w.add_tensor(b + "ssm_norm.weight", np.ones(kdh, np.float32))
            w.add_tensor(b + "ssm_a", -np.exp(rng.standard_normal(heads).astype(np.float32) * 0.3))
        else:            # MLA
            q(b + "attn_q_a.weight", (ql, h), "Q6_K"); w.add_tensor(b + "attn_q_a_norm.weight", np.ones(ql, np.float32))
            q(b + "attn_q_b.weight", (heads * (nope + rope), ql), "Q8_0")
            q(b + "attn_kv_a_mqa.weight", (kvl + rope, h), "Q6_K"); w.add_tensor(b + "attn_kv_a_norm.weight", np.ones(kvl, np.float32))
            q(b + "attn_k_b.weight", (heads, kvl, nope), "Q8_0"); q(b + "attn_v_b.weight", (heads, vh, kvl), "Q8_0")
            q(b + "attn_output.weight", (h, heads * vh), "Q8_0"); q(b + "attn_gate.weight", (heads * vh, h), "Q6_K")
        if L == 0:
            q(b + "ffn_gate.weight", (dense, h), "Q6_K"); q(b + "ffn_up.weight", (dense, h), "Q6_K")
            q(b + "ffn_down.weight", (h, dense), "Q6_K")
            continue
        f32(b + "ffn_gate_inp.weight", (e, h), 0.05); f32(b + "exp_probs_b.bias", (e,), 0.1)
        w.add_tensor(b + "ffn_routed_norm.weight", np.ones(lat, np.float32))
        q(b + "ffn_routed_down.weight", (lat, h), "Q6_K"); q(b + "ffn_routed_up.weight", (h, lat), "Q6_K")
        q(b + "ffn_gate_shexp.weight", (mi, h), "Q6_K"); q(b + "ffn_up_shexp.weight", (mi, h), "Q6_K")
        q(b + "ffn_down_shexp.weight", (h, mi), "Q6_K")
        gu = "IQ2_XXS" if L == 1 else "IQ1_S"
        q(b + "ffn_gate_exps.weight", (e, mi, lat), gu); q(b + "ffn_up_exps.weight", (e, mi, lat), gu)
        q(b + "ffn_down_exps.weight", (e, lat, mi), "IQ1_S")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def engine_tokens(snap, ref, env=None):
    p = subprocess.run([str(ENGINE), "2", "8", str(ref)], env=dict(os.environ, SNAP=str(snap), OMP_NUM_THREADS="2",
                       COLI_NO_OMP_TUNE="1", **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    line = [l for l in out.splitlines() if l.startswith("C engine")]
    if not line:
        raise AssertionError(f"engine produced no tokens (rc {p.returncode}):\n{out[-2000:]}")
    return line[-1].split(":", 1)[1].split("Text")[0].split()


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class GgufImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = d / "tiny-qwen35moe.gguf"
        write_model(cls.src, np.random.default_rng(11))
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        for mode in ("link", "copy"):
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src), str(d / mode), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_directories_are_complete(self):
        for mode in ("link", "copy"):
            d = self.dir / mode
            for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "gguf.json"):
                self.assertTrue((d / f).is_file(), f"{mode}: {f}")
            cfg = json.loads((d / "config.json").read_text())
            self.assertEqual(cfg["model_type"], "qwen3_5_moe_text")
            self.assertEqual((cfg["num_hidden_layers"], cfg["num_experts"], cfg["hidden_size"]), (LAYERS, E, H))
            self.assertEqual(cfg["layer_types"][3], "full_attention")
            tok = json.loads((d / "tokenizer.json").read_text())
            self.assertNotIn("<|im_end|>", tok["model"]["vocab"])          # added tokens live in added_tokens only
            self.assertTrue(any(a["content"] == "<|im_end|>" and a["special"] for a in tok["added_tokens"]))
        self.assertTrue((self.dir / "copy" / "model.colibri.gguf").is_file())
        self.assertFalse((self.dir / "link" / "model.colibri.gguf").exists())

    def test_import_writes_utf8_whatever_the_locale(self):
        """#2189: Path.write_text took the locale's code page (cp1252 on that Windows box) and
        the byte tokenizer's characters (U+0100 and up) stopped the import; a child Python whose
        locale is ASCII (no UTF-8 mode, no locale coercion) must write the same files."""
        src = self.dir / "tiny-non-ascii.gguf"
        write_model(src, np.random.default_rng(11), token="\u0120caf\u00e9\u0100")
        env = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0", "LC_ALL": "C", "LANG": "C"}
        env.pop("PYTHONIOENCODING", None)
        dirs = {}
        for label, e in (("utf8", {**os.environ, "PYTHONUTF8": "1"}), ("ascii", env)):
            dirs[label] = self.dir / f"link-{label}"
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(src), str(dirs[label]), "--mode", "link"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=e)
            self.assertEqual(p.returncode, 0, f"{label}: " + p.stdout.decode("utf-8", "replace")[-2000:])
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "gguf.json"):
            self.assertEqual((dirs["ascii"] / f).read_bytes(), (dirs["utf8"] / f).read_bytes(), f)
        tok = json.loads((dirs["ascii"] / "tokenizer.json").read_text(encoding="utf-8"))
        self.assertIn("\u0120caf\u00e9\u0100", tok["model"]["vocab"])

    def test_same_tokens_from_file_link_and_copy(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                a = engine_tokens(self.src, self.ref, env)
                self.assertEqual(engine_tokens(self.dir / "link", self.ref, env), a)
                self.assertEqual(engine_tokens(self.dir / "copy", self.ref, env), a)

    def test_header_check(self):
        """GGUF_CHECK=1 checks every tensor the loader wants without reading one: it used to
        crash on the final norm, whose 1 it took off a weight it had not read."""
        p = subprocess.run([str(ENGINE), "2", "8", str(self.ref)], env=dict(os.environ, SNAP=str(self.src), GGUF_CHECK="1",
                           OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", "replace")
        self.assertEqual(p.returncode, 0, out[-2000:])
        self.assertIn("GGUF CHECK OK", out)

    def test_container_slots_are_4k_aligned(self):
        r = gguf.GGUFReader(str(self.dir / "copy" / "model.colibri.gguf"))
        self.assertEqual(r.fields["colibri.layout"].contents(), "expert-packed-v1")
        packed = [t for t in r.tensors if t.name.endswith("ffn_exps_colibri")]
        self.assertEqual(len(packed), LAYERS)
        for t in packed:
            self.assertEqual(t.data_offset % 4096, 0, t.name)
            self.assertEqual(int(t.shape[0]) % 4096, 0, t.name)      # slot stride
        # each matrix's [cols, rows], which the engines reading through st_gguf.h need
        dims = [int(x) for x in r.fields["colibri.expert_dims"].contents()]
        self.assertEqual(dims, [H, I, H, I, I, H] * LAYERS)
        # the dense tensors are the source's, byte for byte
        src = gguf.GGUFReader(str(self.src))
        s = {t.name: t for t in src.tensors}
        for t in r.tensors:
            if t.name in s:
                self.assertTrue(np.array_equal(np.array(t.data).view(np.uint8), np.array(s[t.name].data).view(np.uint8)), t.name)

    def test_planner_sees_the_same_experts(self):
        import resource_plan as rp
        a, b = rp.analyze_model(str(self.dir / "link")), rp.analyze_model(str(self.dir / "copy"))
        for k in ("expert_count", "expert_layers", "expert_bytes", "per_cap_bytes", "dense_bytes"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a["expert_count"], LAYERS * E)


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class OllamaQwen35Moe(unittest.TestCase):
    """The blob Ollama pulls for qwen3.6:35b comes from Ollama's converter, not llama.cpp's:
    head_count_kv is an array per layer (0 on the DeltaNet layers), the DeltaNet's dt bias is
    blk.N.ssm_dt, and the MTP head and the vision tower sit in the model's file. The import
    used to stop on the array, and the engine's check refused the file for the tensors it
    would not use; now both give the llama.cpp file's config and tokens."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = {}
        for kind in ("llama", "ollama"):
            cls.src[kind] = d / f"tiny-{kind}.gguf"
            write_model(cls.src[kind], np.random.default_rng(11), ollama=kind == "ollama")
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src[kind]), str(d / kind), "--mode", "link"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert p.returncode == 0, p.stdout.decode()
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_same_config(self):
        a = json.loads((self.dir / "llama" / "config.json").read_text())
        b = json.loads((self.dir / "ollama" / "config.json").read_text())
        self.assertEqual(b["num_key_value_heads"], KVH)
        for cfg in (a, b):   # the import's record of its source names the file
            for k in [k for k, v in cfg.items() if isinstance(v, dict) and "source" in v]: cfg.pop(k)
        self.assertEqual(a, b)

    def test_the_check_passes(self):
        p = subprocess.run([str(ENGINE), "2", "8", str(self.ref)], env=dict(os.environ, SNAP=str(self.src["ollama"]), GGUF_CHECK="1",
                           OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", "replace")
        self.assertEqual(p.returncode, 0, out[-2000:])
        self.assertIn("GGUF CHECK OK", out)
        self.assertNotIn("not used by the qwen36 engine", out)

    def test_the_same_tokens(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                a = engine_tokens(self.src["llama"], self.ref, env)
                self.assertEqual(engine_tokens(self.src["ollama"], self.ref, env), a)
                self.assertEqual(engine_tokens(self.dir / "ollama", self.ref, env), a)


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class GgufImportDense(unittest.TestCase):
    """llama.cpp's qwen35 (the family's dense models): the importer writes a dense config with
    the head tied to the embedding, and the engine gives the same tokens from the bare file
    and from the link directory, on the float and on the integer path."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = d / "tiny-qwen35.gguf"
        write_model(cls.src, np.random.default_rng(12), dense=True)
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src), str(d / "link"), "--mode", "link"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dense_config(self):
        cfg = json.loads((self.dir / "link" / "config.json").read_text())
        self.assertEqual(cfg["model_type"], "qwen3_5_text")
        self.assertEqual((cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]), (LAYERS, H, I))
        self.assertNotIn("num_experts", cfg)
        self.assertTrue(cfg["tie_word_embeddings"])

    def test_same_tokens_from_file_and_link(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                self.assertEqual(engine_tokens(self.dir / "link", self.ref, env), engine_tokens(self.src, self.ref, env))


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class ContainerLayers(unittest.TestCase):
    """The container of a model whose first layers are dense (GLM-5.3-Flash, Inkling): those
    layers have no slot and zeros in the layout keys, the others their packed experts; and an
    engine that does not read containers (qwen4exp) is refused before anything is written."""
    def test_dense_layers_get_no_slot(self):
        import tools.gguf_import as gi
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m.gguf"
            w = gguf.GGUFWriter(str(src), "glm5-next")
            w.add_tensor("blk.0.ffn_gate.weight", blocks("Q8_0", (64, 64), rng), raw_dtype=QT["Q8_0"])
            for k, shape in (("gate", (4, 32, 64)), ("up", (4, 32, 64)), ("down", (4, 64, 32))):
                w.add_tensor(f"blk.1.ffn_{k}_exps.weight", blocks("Q8_0", shape, rng), raw_dtype=QT["Q8_0"])
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            parts, readers, meta, tensors = gi.open_model(src)
            out = Path(d) / "c.gguf"
            gi.write_packed(parts, readers, meta, tensors, 2, out)
            for rd in readers:
                rd.close()
            r = gguf.GGUFReader(str(out))
            self.assertEqual([t.name for t in r.tensors if t.name.endswith("ffn_exps_colibri")], ["blk.1.ffn_exps_colibri"])
            stride = [int(x) for x in r.fields["colibri.expert_stride"].contents()]
            self.assertEqual(stride[0], 0)
            self.assertEqual(stride[1] % 4096, 0)
            self.assertEqual([int(x) for x in r.fields["colibri.expert_dims"].contents()], [0] * 6 + [64, 32, 64, 32, 32, 64])
            self.assertIn("blk.0.ffn_gate.weight", [t.name for t in r.tensors])

    def test_no_container_for_qwen4exp(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "q.gguf"
            w = gguf.GGUFWriter(str(src), "qwen4exp")
            w.add_tensor("token_embd.weight", np.zeros((4, 32), np.float32))
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(src), str(Path(d) / "out"), "--mode", "copy"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--mode link", p.stdout.decode())
            self.assertFalse((Path(d) / "out").exists())


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class PlannerExperts(unittest.TestCase):
    """The planner's inventory of a GGUF: one entry per routed expert under a name every family
    counts (resource_plan._GGUF_EXPERT), and the model's own layers from its config, so that
    a file whose nextn_predict_layers does not count its blocks (antirez's V4) loses none."""
    def test_layers_from_the_config(self):
        import resource_plan as rp
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m.gguf"
            w = gguf.GGUFWriter(str(src), "deepseek4")
            w.add_uint32("deepseek4.block_count", 3); w.add_uint32("deepseek4.nextn_predict_layers", 1)
            for L in range(3):
                for k, shape in (("gate", (4, 32, 64)), ("up", (4, 32, 64)), ("down", (4, 64, 32))):
                    w.add_tensor(f"blk.{L}.ffn_{k}_exps.weight", blocks("Q8_0", shape, rng), raw_dtype=QT["Q8_0"])
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            def layers_seen(n):
                return sorted({int(m.group(1)) for name, *_ in rp._gguf_tensor_sizes([src], n)
                               for m in [rp._GGUF_EXPERT.match(name)] if m})
            self.assertEqual(layers_seen(3), [0, 1, 2])
            self.assertEqual(layers_seen(None), [0, 1])   # the file's own count: the last block taken for MTP
            per = [size for name, size, *_ in rp._gguf_tensor_sizes([src], 3) if name.startswith("model.layers.0.mlp.experts.1.")]
            self.assertEqual(sum(per), 3 * 32 * 2 * 34)   # gate, up, down: 32 rows of two Q8_0 blocks each


class Qwen3MoeConfig(unittest.TestCase):
    """llama.cpp's qwen3moe (Qwen3-30B-A3B, Qwen3-Coder) as the qwen36 engine's config, from the
    metadata of unsloth's Qwen3-Coder-30B-A3B-Instruct GGUF: every layer attention, no shared
    expert, the head untied when the file has output.weight."""
    def test_coder_metadata(self):
        import tools.gguf_import as gi
        p = "qwen3moe."
        meta = {p + k: v for k, v in {
            "block_count": 48, "context_length": 262144, "embedding_length": 2048, "feed_forward_length": 5472,
            "attention.head_count": 32, "attention.head_count_kv": 4, "rope.freq_base": 10000000.0,
            "attention.layer_norm_rms_epsilon": 1e-6, "expert_used_count": 8, "attention.key_length": 128,
            "attention.value_length": 128, "expert_count": 128, "expert_feed_forward_length": 768,
            "expert_shared_feed_forward_length": 0}.items()}
        tensors = {"token_embd.weight": {"dims": [2048, 151936]}, "output.weight": {"dims": [2048, 151936]}}
        cfg, layers = gi.qwen3moe_config(meta, tensors)
        self.assertEqual(layers, 48)
        self.assertEqual(cfg["model_type"], "qwen3_moe")
        self.assertEqual((cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]), (128, 8, 768))
        self.assertEqual((cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]), (32, 4, 128))
        self.assertEqual(cfg["layer_types"], ["full_attention"] * 48)
        self.assertFalse(cfg["tie_word_embeddings"])
        self.assertEqual(cfg["vocab_size"], 151936)


class Qwen3DenseConfig(unittest.TestCase):
    """llama.cpp's qwen3 (the dense Qwen3) as the qwen36 engine's config: Qwen3-0.6B's numbers,
    whose head (128) is not the width over the heads (64), and whose embedding is tied."""
    def test_small_tied(self):
        import tools.gguf_import as gi
        p = "qwen3."
        meta = {p + k: v for k, v in {
            "block_count": 28, "context_length": 40960, "embedding_length": 1024, "feed_forward_length": 3072,
            "attention.head_count": 16, "attention.head_count_kv": 8, "rope.freq_base": 1000000.0,
            "attention.layer_norm_rms_epsilon": 1e-6, "attention.key_length": 128,
            "attention.value_length": 128}.items()}
        tensors = {"token_embd.weight": {"dims": [1024, 151936]}}
        cfg, layers = gi.qwen3_config(meta, tensors)
        self.assertEqual(layers, 28)
        self.assertEqual((cfg["model_type"], cfg["architectures"]), ("qwen3", ["Qwen3ForCausalLM"]))
        self.assertEqual((cfg["intermediate_size"], cfg["head_dim"]), (3072, 128))
        for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
            self.assertNotIn(k, cfg)
        self.assertEqual(cfg["layer_types"], ["full_attention"] * 28)
        self.assertTrue(cfg["tie_word_embeddings"])
        # the family registry plans it as a dense qwen36 model: KV on every layer, no experts
        from family_registry import family_by_id
        fam = family_by_id("qwen36")
        self.assertIn("qwen3", fam.model_types)

    def test_named_by_the_file(self):
        """A GGUF no variant of the family names keeps its own name (general.name) in the
        banner and the API id, not the family's (it was served as qwen3.6-colibri)."""
        import tools.gguf_import as gi
        from family_registry import resolve_model, display_for, default_model_id
        meta = {"general.name": "Qwen3-0.6B", "general.basename": "Qwen3", "general.size_label": "0.6B"}
        self.assertEqual(gi.model_name(meta), {"name": "Qwen3-0.6B", "size": "0.6B"})
        self.assertEqual(gi.model_name({"general.basename": "Qwen3", "general.size_label": "8B"}), {"name": "Qwen3-8B", "size": "8B"})
        self.assertEqual(gi.model_name({}), {})
        cfg, _ = gi.qwen3_config({"qwen3." + k: v for k, v in {
            "block_count": 28, "embedding_length": 1024, "feed_forward_length": 3072,
            "attention.head_count": 16, "attention.head_count_kv": 8, "attention.key_length": 128}.items()},
            {"token_embd.weight": {"dims": [1024, 151936]}})
        with tempfile.TemporaryDirectory() as d:
            named = dict(cfg, colibri_gguf={"arch": "qwen3", "mode": "link", "source": "x.gguf", **gi.model_name(meta)})
            (Path(d) / "config.json").write_text(json.dumps(named))
            r = resolve_model(d)
            self.assertEqual(display_for(r), ("Qwen3-0.6B", "0.6B"))
            self.assertEqual(default_model_id(r), "qwen3-0.6b-colibri")
            # a GGUF of a model colibri names keeps that name and id (Qwen3-Coder-30B-A3B)
            coder, _ = gi.qwen3moe_config({"qwen3moe." + k: v for k, v in {
                "block_count": 48, "embedding_length": 2048, "attention.head_count": 32, "attention.head_count_kv": 4,
                "attention.key_length": 128, "expert_count": 128, "expert_used_count": 8,
                "expert_feed_forward_length": 768}.items()}, {"token_embd.weight": {"dims": [2048, 151936]}})
            coder["colibri_gguf"] = {"arch": "qwen3moe", "name": "Qwen3-Coder-30B-A3B-Instruct"}
            (Path(d) / "config.json").write_text(json.dumps(coder))
            r = resolve_model(d)
            self.assertEqual(display_for(r), ("Qwen3-Coder-30B-A3B", "30B"))
            self.assertEqual(default_model_id(r), "qwen3-coder-30b-a3b-colibri")


class DeepSeekV4Config(unittest.TestCase):
    """llama.cpp's deepseek4 as the deepseek_v4 engine's config.json, from unsloth's
    DeepSeek-V4-Flash-0731 header numbers; layers counted by the attention blocks (antirez's
    main files say nextn_predict_layers 1 and keep the MTP head apart)."""
    def test_flash_metadata(self):
        import tools.gguf_import as gi
        p = "deepseek4."
        meta = {p + k: v for k, v in {
            "block_count": 43, "context_length": 1048576, "embedding_length": 4096, "attention.head_count": 64,
            "attention.head_count_kv": 1, "attention.key_length": 512, "rope.dimension_count": 64,
            "attention.q_lora_rank": 1024, "attention.output_group_count": 8, "attention.output_lora_rank": 1024,
            "attention.sliding_window": 128, "attention.indexer.head_count": 64, "attention.indexer.key_length": 128,
            "attention.indexer.top_k": 512, "expert_count": 256, "expert_used_count": 6, "expert_shared_count": 1,
            "expert_feed_forward_length": 2048, "hash_layer_count": 3, "hyper_connection.count": 4,
            "hyper_connection.sinkhorn_iterations": 20, "expert_weights_scale": 1.5, "expert_gating_func": 4,
            "attention.compress_ratios": [0, 0] + [4, 128] * 21 + [0], "attention.compress_rope_freq_base": 160000.0,
            "swiglu_clamp_exp": [10.0] * 43, "nextn_predict_layers": 1, "rope.scaling.factor": 16.0,
            "rope.scaling.original_context_length": 65536}.items()}
        tensors = {"token_embd.weight": {"dims": [4096, 129280]}}
        tensors.update({f"blk.{b}.attn_q_a.weight": {} for b in range(43)})
        cfg, layers = gi.deepseek4_config(meta, tensors)
        self.assertEqual(layers, 43)
        self.assertEqual((cfg["model_type"], cfg["scoring_func"], cfg["num_hash_layers"]), ("deepseek_v4", "sqrtsoftplus", 3))
        self.assertEqual((cfg["o_groups"], cfg["q_lora_rank"], cfg["head_dim"]), (8, 1024, 512))
        self.assertEqual(cfg["compress_ratios"][:4], [0, 0, 4, 128])
        self.assertEqual(cfg["num_nextn_predict_layers"], 0)
        self.assertEqual(cfg["rope_scaling"]["original_max_position_embeddings"], 65536)

    def test_pre_tokenizers(self):
        import tools.gguf_import as gi
        ds = gi.pre_tokenizer("joyai-llm")
        self.assertEqual([x["type"] for x in ds["pretokenizers"]], ["Split", "Split", "Split", "ByteLevel"])
        self.assertIn("][A-Za-z]+|", ds["pretokenizers"][2]["pattern"]["Regex"])   # what tok.h keys DeepSeek on
        self.assertIn("\\p{Lu}", gi.pre_tokenizer("inkling")["pretokenizers"][0]["pattern"]["Regex"])   # o200k
        self.assertEqual(gi.pre_tokenizer("qwen2"), {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True})


class InklingConfig(unittest.TestCase):
    """llama.cpp's inkling as the inkling engine's config.json, from unsloth's Inkling-Small
    header: global layers every sixth, two dense blocks, KV heads from the per-layer array."""
    def test_small_metadata(self):
        import tools.gguf_import as gi
        p = "inkling."
        local = [i % 6 != 5 for i in range(42)]
        meta = {p + k: v for k, v in {
            "block_count": 42, "context_length": 1048576, "embedding_length": 4096, "feed_forward_length": 16384,
            "attention.head_count": 32, "attention.head_count_kv": [4 if l else 8 for l in local],
            "attention.key_length": 128, "vocab_size": 201024, "expert_count": 256, "expert_used_count": 6,
            "expert_feed_forward_length": 2048, "expert_shared_count": 2, "expert_weights_scale": 8.0,
            "attention.sliding_window": 512, "attention.sliding_window_pattern": local, "d_rel": 16,
            "rel_extent": 1024, "shortconv_kernel": 4, "dense_block_count": 2, "logit_scale_denom": 16.0,
            "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1, "unpadded_vocab_size": 200058}.items()}
        meta["tokenizer.ggml.eos_token_id"] = 200006
        cfg, layers = gi.inkling_config(meta, {"token_embd.weight": {"dims": [4096, 201024]}})
        self.assertEqual((layers, cfg["model_type"]), (42, "inkling"))
        self.assertEqual((cfg["num_key_value_heads"], cfg["swa_num_key_value_heads"]), (8, 4))
        self.assertEqual(cfg["local_layer_ids"], [i for i in range(42) if i % 6 != 5])
        self.assertEqual(cfg["mlp_layer_types"][:3], ["dense", "dense", "sparse"])
        self.assertEqual((cfg["intermediate_size"], cfg["moe_intermediate_size"], cfg["n_shared_experts"]), (16384, 2048, 2))
        self.assertEqual((cfg["unpadded_vocab_size"], cfg["eos_token_id"]), (200058, 200006))


class GlmDsaConfig(unittest.TestCase):
    """llama.cpp's glm-dsa as the colibri engine's config.json, from unsloth's GLM-5.3 header:
    MLA's head dims from the _mla lengths, the NextN block past the model's layers, every
    layer's indexer its own when its weights are in the file."""
    def test_glm53_metadata(self):
        import tools.gguf_import as gi
        p = "glm-dsa."
        meta = {p + k: v for k, v in {
            "block_count": 79, "context_length": 1048576, "embedding_length": 6144, "feed_forward_length": 12288,
            "attention.head_count": 64, "attention.head_count_kv": 1, "rope.freq_base": 8000000.0,
            "attention.layer_norm_rms_epsilon": 1e-5, "expert_count": 256, "expert_used_count": 8,
            "expert_group_count": 1, "expert_group_used_count": 1, "expert_gating_func": 2,
            "attention.key_length": 576, "attention.value_length": 512, "leading_dense_block_count": 3,
            "vocab_size": 154880, "attention.q_lora_rank": 2048, "attention.kv_lora_rank": 512,
            "attention.key_length_mla": 256, "attention.value_length_mla": 256, "expert_feed_forward_length": 2048,
            "expert_shared_count": 1, "expert_weights_scale": 2.5, "expert_weights_norm": True,
            "rope.dimension_count": 64, "nextn_predict_layers": 1, "attention.indexer.head_count": 32,
            "attention.indexer.key_length": 128, "attention.indexer.top_k": 2048}.items()}
        meta["tokenizer.ggml.eos_token_id"] = 154820
        tensors = {"token_embd.weight": {"dims": [6144, 154880]}, "output.weight": {"dims": [6144, 154880]},
                   "blk.78.nextn.eh_proj.weight": {"dims": [12288, 6144]},
                   **{f"blk.{i}.indexer.attn_q_b.weight": {"dims": [2048, 4096]} for i in range(79) if i != 7}}
        cfg, layers = gi.glmdsa_config(meta, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["num_nextn_predict_layers"]), (78, "glm_moe_dsa", 1))
        self.assertEqual((cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]), (192, 64, 256))
        self.assertEqual((cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["first_k_dense_replace"]), (2048, 512, 3))
        self.assertEqual((cfg["routed_scaling_factor"], cfg["rope_parameters"]["rope_theta"]), (2.5, 8000000.0))
        self.assertEqual(len(cfg["indexer_types"]), 78)
        self.assertEqual(cfg["indexer_types"][7], "shared")
        self.assertEqual(cfg["indexer_types"].count("full"), 77)
        self.assertFalse(cfg["tie_word_embeddings"])
        del tensors["blk.78.nextn.eh_proj.weight"]   # a file without its NextN block: every block a layer
        self.assertEqual(gi.glmdsa_config(meta, tensors)[1], 79)


class Mimo2Config(unittest.TestCase):
    """llama.cpp's mimo2 as the mimo engine's config.json, from MiMo-V2.6-Flash's header: the
    layer kinds and their KV heads from the pattern, the sinks from the sliding layers'
    tensors, qkv as one [Q|K|V], the rotary width the engine rounds to 64."""
    def test_flash_metadata(self):
        import tools.gguf_import as gi
        p = "mimo2."
        pattern = [0] + [1, 1, 1, 1, 0] * 9 + [1, 1] + [1, 1, 1]   # 48 layers + 3 NextN
        meta = {p + k: v for k, v in {
            "block_count": 51, "context_length": 1048576, "embedding_length": 4096, "feed_forward_length": 16384,
            "attention.head_count": 64, "attention.head_count_kv": [4 if x == 0 else 8 for x in pattern],
            "rope.freq_base": 10000000.0, "rope.freq_base_swa": 10000.0, "expert_count": 256, "expert_used_count": 8,
            "attention.key_length": 192, "attention.value_length": 128, "attention.sliding_window": 128,
            "attention.sliding_window_pattern": pattern, "expert_feed_forward_length": 2048,
            "rope.dimension_count": 64, "attention.layer_norm_rms_epsilon": 1e-6,
            "attention.value_scale": 0.707, "nextn_predict_layers": 3}.items()}
        tensors = {"token_embd.weight": {"dims": [4096, 152576]}, "output.weight": {"dims": [4096, 152576]},
                   **{f"blk.{i}.ffn_gate_exps.weight": {} for i in range(1, 48)},
                   **{f"blk.{i}.attn_sinks.weight": {} for i in range(48) if pattern[i] == 1}}
        cfg, layers = gi.mimo2_config(meta, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["qkv_shards"]), (48, "mimo_v2", 1))
        self.assertEqual((cfg["num_key_value_heads"], cfg["swa_num_key_value_heads"]), (4, 8))
        self.assertEqual(cfg["hybrid_layer_pattern"], pattern[:48])
        self.assertEqual(cfg["moe_layer_freq"], [0] + [1] * 47)
        self.assertEqual((cfg["add_full_attention_sink_bias"], cfg["add_swa_attention_sink_bias"]), (False, True))
        self.assertEqual(int(cfg["head_dim"] * cfg["partial_rotary_factor"]), 64)
        self.assertEqual((cfg["rope_theta"], cfg["swa_rope_theta"], cfg["sliding_window"]), (10000000.0, 10000.0, 128))


MIMO = HERE / ("mimo.exe" if sys.platform == "win32" else "mimo")


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(MIMO.is_file(), "mimo is not built")
class MimoGguf(unittest.TestCase):
    """A mimo2 GGUF of block types (write_mimo2: K-quant and IQ2_XXS experts, a Q8_0 and Q4_K
    trunk, a Q6_K head, a Q8_0 embedding) on the mimo engine's CPU: the link directory and the
    copy container give the same tokens, on the integer and on the float path."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        src = d / "tiny-mimo2.gguf"
        write_mimo2(src, np.random.default_rng(5))
        for mode in ("link", "copy"):
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(src), str(d / mode), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tokens(self, mode, env=None):
        p = subprocess.run([str(MIMO), str(self.dir / mode), "--ids", "5 6 7 8 20 21 22 23", "--ngen", "6"],
                           env=dict(os.environ, COLI_TEMP="0", OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1", **(env or {})),
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(p.returncode, 0, p.stderr.decode("utf-8", "replace")[-2000:])
        err = p.stderr.decode("utf-8", "replace")
        self.assertIn("GGUF experts: IQ2_XXS gate", err)
        lines = [l for l in p.stdout.decode().splitlines() if l.strip()]
        self.assertTrue(lines, err[-2000:])
        return lines[-1].split()

    def test_same_tokens_from_link_and_copy(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            link, copy = self.tokens("link", env), self.tokens("copy", env)
            self.assertEqual(len(link), 6, link)
            self.assertEqual(link, copy, env)


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class KimiK3Gguf(unittest.TestCase):
    """A published kimi-k3 GGUF: the checkpoint's text_config from the keys (KDA layers by
    head_count_kv, MLA's own head sizes) and the kimi-k2 split, as tools/k3_tokenizer.py's."""
    def test_the_config_and_the_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_kimik3(d / "k3.gguf", np.random.default_rng(6))
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "k3.gguf"), str(d / "out")],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(p.returncode, 0, p.stdout.decode())
            cfg = json.loads((d / "out" / "config.json").read_text())
            t = cfg["text_config"]
            self.assertEqual(cfg["model_type"], "kimi_k3")
            self.assertEqual({k: t[k] for k in ("model_type", "num_hidden_layers", "first_k_dense_replace", "qk_nope_head_dim",
                                                "qk_rope_head_dim", "v_head_dim", "routed_expert_hidden_size",
                                                "num_shared_experts", "attn_res_block_size")},
                             {"model_type": "kimi_linear", "num_hidden_layers": 5, "first_k_dense_replace": 1,
                              "qk_nope_head_dim": 32, "qk_rope_head_dim": 16, "v_head_dim": 32,
                              "routed_expert_hidden_size": 256, "num_shared_experts": 1, "attn_res_block_size": 2})
            self.assertEqual(t["linear_attn_config"]["kda_layers"], [1, 2, 3, 5])
            sys.path.insert(0, str(HERE / "tools"))
            import k3_tokenizer
            tok = json.loads((d / "out" / "tokenizer.json").read_text())
            self.assertEqual(tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"], k3_tokenizer.PAT)

    def test_a_gating_other_than_sigmoid_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_kimik3(d / "k3.gguf", np.random.default_rng(6))
            r = gguf.GGUFReader(str(d / "k3.gguf"))
            meta = {}
            for k, f in r.fields.items():
                if k.startswith("kimi-k3.") and len(f.data) == 1:
                    meta[k] = f.parts[f.data[0]].tolist()[0]
                elif k.startswith("kimi-k3."):
                    meta[k] = [f.parts[i].tolist()[0] for i in f.data]
            tensors = {t.name: {"dims": [int(x) for x in t.shape]} for t in r.tensors}
            sys.path.insert(0, str(HERE / "tools"))
            import gguf_import as gi
            cfg, layers = gi.kimik3_config(meta, tensors)
            self.assertEqual(layers, 5)
            with self.assertRaises(SystemExit):
                gi.kimik3_config(dict(meta, **{"kimi-k3.expert_gating_func": 1}), tensors)


class DeepSeek41Config(unittest.TestCase):
    """antirez's deepseek41: the checkpoint's config.json from deepseek41.config as it is, and
    the engram sidecar (dsv41_engram.json, which tools/prepare_dsv41.py builds from the
    checkpoint) from the file's engram metadata; the planner keeps the tables out of RAM."""
    TEXT = {"num_hidden_layers": 2, "engram_layer_ids": [1], "engram_max_ngram_size": 3,
            "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_token_id": 2}

    def meta(self, **over):
        p = "deepseek41.engram."
        m = {"deepseek41.config": json.dumps({"model_type": "deepseek_v41", "text_config": self.TEXT}),
             p + "encoding": "e4m3_e8m0_32_row264", p + "layer_ids": [1], p + "rows": [97 + 101 + 103 + 107],
             p + "primes": [97, 101, 103, 107], p + "multipliers": [76632096046245, 4839876093313, 35959672319349],
             p + "token_map": [0, 1, 5, 3]}
        m.update({p + k: v for k, v in over.items()})
        return m

    def test_config_and_sidecar(self):
        import tools.gguf_import as gi
        tensors = {f"blk.{b}.attn_q_a.weight": {} for b in range(2)}
        cfg, layers = gi.deepseek41_config(self.meta(), tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["text_config"]), (2, "deepseek_v41", self.TEXT))
        e = gi.deepseek41_engram(self.meta(), cfg)
        self.assertEqual(e["primes"], [[[97, 101], [103, 107]]])
        self.assertEqual(e["offsets"], [[0, 97, 198, 301]])
        self.assertEqual(e["multipliers"], [["76632096046245", "4839876093313", "35959672319349"]])   # past a double's 53 bits: strings
        self.assertEqual((e["pad_id"], e["head_dim"], e["num_embeddings"]), (5, 32, [408]))   # the pad token's compressed id

    def test_inconsistent_metadata_is_refused(self):
        import tools.gguf_import as gi
        cfg = json.loads(self.meta()["deepseek41.config"])
        for over in ({"layer_ids": [0]}, {"encoding": "e4m3_row256"}, {"rows": [400]}, {"primes": [97, 101, 103]}):
            with self.assertRaises(SystemExit, msg=over):
                gi.deepseek41_engram(self.meta(**over), cfg)
        with self.assertRaises(SystemExit):   # a block short of the config's layers
            gi.deepseek41_config(self.meta(), {"blk.0.attn_q_a.weight": {}})

    def test_planner_keeps_the_tables_on_disk(self):
        from family_registry import _dsv41_resident_inventory as resident
        self.assertEqual(resident("blk.1.engram_embd.weight", 10 ** 11, {}), 0)          # a GGUF's table
        self.assertEqual(resident("layers.1.engram.embed.weight", 10 ** 11, {}), 0)      # the checkpoint's
        self.assertEqual(resident("blk.1.engram_kv.weight", 1000, {}), 1000)             # its projection is resident


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class DeepSeek41DSparkHead(unittest.TestCase):
    """The published files' DSpark head, a dflash GGUF: --mtp links it (checked against the
    model's keys, named in gguf.json); one next to the model is only named, since its
    drafts measured slower on UD-IQ1_S."""
    def test_the_head_next_to_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_deepseek41(d / "ds41.gguf", np.random.default_rng(3))
            head = d / "dspark-ds41-Q8_0.gguf"
            write_dflash(head, np.random.default_rng(4))
            for flag, want in ((["--mtp", str(head)], True), ([], False)):
                out = d / ("dir" if want else "plain")
                p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "ds41.gguf"), str(out), *flag],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(p.returncode, 0, p.stdout.decode())
                link = json.loads((out / "gguf.json").read_text())
                text = json.loads((out / "config.json").read_text())["text_config"]
                if not want:
                    self.assertNotIn("dspark", link); self.assertNotIn("n_mtp_layers", text)
                    self.assertIn(f"none linked; {head} is this model's, --mtp links it", p.stdout.decode())
                    continue
                self.assertEqual(Path(link["dspark"]), (d / "dspark-ds41-Q8_0.gguf").resolve())
                self.assertEqual({k: v for k, v in text.items() if k == "n_mtp_layers" or k.startswith("dspark_")},
                                 {"n_mtp_layers": 2, "dspark_block_size": 3, "dspark_noise_token_id": 7,
                                  "dspark_target_layer_ids": [3, 4, 5], "dspark_markov_rank": 32,
                                  "dspark_n_routed_experts": 4, "dspark_n_activated_experts": 2})

    def test_a_head_of_another_model_is_passed_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            write_deepseek41(d / "ds41.gguf", np.random.default_rng(3))
            other = d / "dspark-other-Q8_0.gguf"
            write_dflash(other, np.random.default_rng(4), targets=(2, 4, 5))
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "ds41.gguf"), str(d / "out")],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertEqual(p.returncode, 0, p.stdout.decode())
            self.assertIn("DSpark head: none found", p.stdout.decode())   # not named: not this model's
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(d / "ds41.gguf"), str(d / "out2"),
                                "--mtp", str(other)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("its target layers are not the model's", p.stdout.decode())


class DeepSeek41PublishedConfig(unittest.TestCase):
    """The published deepseek41 GGUFs (K-quant and IQ types): no deepseek41.config, so the
    checkpoint's text_config from the keys, and the sidecar from the ple keys, whose head offsets
    run over every table in one tensor (per_layer_token_embd); the planner keeps it on disk."""
    P = "deepseek41."

    def meta(self, **over):
        p = self.P
        m = {p + k: v for k, v in {
            "block_count": 2, "context_length": 1048576, "embedding_length": 256, "attention.head_count": 4,
            "attention.head_count_kv": 1, "attention.key_length": 64, "rope.dimension_count": 16,
            "attention.q_lora_rank": 32, "attention.output_lora_rank": 32, "attention.output_group_count": 2,
            "attention.layer_norm_rms_epsilon": 1e-20, "expert_count": 8, "expert_used_count": 2,
            "expert_feed_forward_length": 64, "expert_shared_count": 1, "expert_weights_scale": 1.5,
            "expert_weights_norm": True, "expert_gating_func": 4, "swiglu_clamp_exp": [10.0, 10.0],
            "attention.sliding_window": 128, "attention.compress_ratios": [0, 2, 0], "attention.compress_rope_freq_base": 160000.0,
            "attention.kv_source_layers": [1], "attention.indexer.source_layers": [1],
            "attention.indexer.head_count": 2, "attention.indexer.key_length": 32, "attention.indexer.top_k": 16,
            "attention.indexer.candidate_source_layer": 1, "attention.indexer.candidate_top_k_blocks": 4,
            "attention.indexer.candidate_block_size": 8, "hyper_connection.count": 4,
            "hyper_connection.sinkhorn_iterations": 20, "hyper_connection.epsilon": 1e-6,
            "rope.scaling.type": "yarn", "rope.scaling.factor": 16.0, "rope.scaling.original_context_length": 65536,
            "ple.layers": [0, 1], "ple.ngram_size": 3, "ple.heads_per_ngram": 2, "embedding_length_per_layer_input": 32,
            "ple.head_vocab_sizes": [97, 101, 103, 107, 109, 113, 127, 131],
            "ple.head_offsets": [0, 97, 198, 301, 408, 517, 630, 757],
            "ple.layer_multipliers": [76632096046245, 4839876093313, 35959672319349, 73987337458391, 67716810739261, 51510806800915],
            "engram.pad_token_id": 2, "engram.token_map": [0, 1, 5, 3]}.items()}
        m.update({p + k: v for k, v in over.items()})
        return m

    TENSORS = {"token_embd.weight": {"dims": [256, 1000]}, "output.weight": {"dims": [256, 1000]},
               **{f"blk.{b}.attn_q_a.weight": {} for b in range(2)}}

    def test_config_and_sidecar(self):
        import tools.gguf_import as gi
        cfg, layers = gi.deepseek41_config(self.meta(), self.TENSORS)
        t = cfg["text_config"]
        self.assertEqual((layers, cfg["model_type"], t["num_hidden_layers"], t["vocab_size"]), (2, "deepseek_v41", 2, 1000))
        self.assertEqual((t["hidden_size"], t["head_dim"], t["qk_rope_head_dim"], t["o_groups"], t["o_lora_rank"]), (256, 64, 16, 2, 32))
        self.assertEqual((t["n_routed_experts"], t["num_experts_per_tok"], t["routed_scaling_factor"], t["swiglu_limit"]), (8, 2, 1.5, 10.0))
        self.assertEqual((t["kv_source_layer_ids"], t["index_source_layer_ids"], t["candidate_source_layer_id"]), ([1], [1], 1))
        self.assertEqual((t["rms_norm_eps"], t["hc_mult"], t["compress_ratios"]), (1e-20, 4, [0, 2, 0]))
        self.assertEqual(t["rope_scaling"]["factor"], 16.0)
        self.assertEqual((t["engram_layer_ids"], t["engram_max_ngram_size"], t["engram_n_heads"], t["engram_head_dim"]), ([0, 1], 3, 2, 32))
        e = gi.deepseek41_engram(self.meta(), cfg)
        self.assertEqual(e["primes"], [[[97, 101], [103, 107]], [[109, 113], [127, 131]]])
        self.assertEqual(e["offsets"], [[0, 97, 198, 301], [408, 517, 630, 757]])   # the one tensor's rows
        self.assertEqual(e["num_embeddings"], [408, 480])
        self.assertEqual(e["multipliers"], [["76632096046245", "4839876093313", "35959672319349"],
                                            ["73987337458391", "67716810739261", "51510806800915"]])
        self.assertEqual(e["pad_id"], 5)

    def test_inconsistent_metadata_is_refused(self):
        import tools.gguf_import as gi
        cfg, _ = gi.deepseek41_config(self.meta(), self.TENSORS)
        for over in ({"ple.head_offsets": [0, 97, 198, 301, 408, 517, 630, 758]}, {"ple.head_vocab_sizes": [97, 101, 103]},
                     {"ple.layer_multipliers": [1, 3, 5]}):
            with self.assertRaises(SystemExit, msg=over):
                gi.deepseek41_engram(self.meta(**over), cfg)
        with self.assertRaises(SystemExit):
            gi.deepseek41_config(self.meta(expert_gating_func=1), self.TENSORS)

    def test_planner_keeps_the_table_on_disk(self):
        from family_registry import _dsv41_resident_inventory as resident
        self.assertEqual(resident("per_layer_token_embd.weight", 10 ** 11, {}), 0)
        self.assertEqual(resident("engram.embed.weight", 10 ** 11, {}), 0)
        self.assertEqual(resident("blk.1.engram_gate.weight", 1000, {}), 1000)


class OlmoeConfig(unittest.TestCase):
    """llama.cpp's olmoe as the olmoe engine's config.json, from the metadata llama.cpp's
    converter writes for OLMoE-1B-7B: no norm_topk_prob in the file, so false, as the official
    checkpoints have it and as llama.cpp computes them."""
    META = {"olmoe.block_count": 16, "olmoe.context_length": 4096, "olmoe.embedding_length": 2048,
            "olmoe.feed_forward_length": 1024, "olmoe.attention.head_count": 16,
            "olmoe.attention.head_count_kv": 16, "olmoe.rope.freq_base": 10000.0,
            "olmoe.attention.layer_norm_rms_epsilon": 1e-5, "olmoe.expert_count": 64,
            "olmoe.expert_used_count": 8, "tokenizer.ggml.eos_token_id": 50279}

    def test_1b_7b_metadata(self):
        import tools.gguf_import as gi
        tensors = {"token_embd.weight": {"dims": [2048, 50304]}, "output.weight": {"dims": [2048, 50304]}}
        cfg, layers = gi.olmoe_config(self.META, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["vocab_size"], cfg["tie_word_embeddings"]), (16, "olmoe", 50304, False))
        self.assertEqual((cfg["num_experts"], cfg["num_experts_per_tok"], cfg["intermediate_size"]), (64, 8, 1024))
        self.assertIs(cfg["norm_topk_prob"], False)
        cfg, _ = gi.olmoe_config({**self.META, "olmoe.expert_weights_norm": True}, tensors)
        self.assertIs(cfg["norm_topk_prob"], True)
        with self.assertRaises(SystemExit):
            gi.olmoe_config({k: v for k, v in self.META.items() if k != "olmoe.expert_count"}, tensors)


class Qwen4ExpLayers(unittest.TestCase):
    """Which layers attend, from either way a qwen4exp GGUF says it: unsloth's Qwen3.8-Flash-Next
    files give full_attention_interval (no attention.recurrent_layers), newer llama.cpp
    writes recurrent_layers; compress_ratios, when not all 0, must agree."""
    def meta(self, **extra):
        p = "qwen4exp."
        m = {p + k: v for k, v in {
            "block_count": 8, "embedding_length": 2560, "attention.head_count": 24, "attention.head_count_kv": 2,
            "attention.key_length": 256, "rope.dimension_count": 64, "expert_count": 512, "expert_used_count": 10,
            "expert_feed_forward_length": 640, "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 16,
            "ssm.time_step_rank": 48, "ssm.inner_size": 6144, "hyper_connection.count": 4,
            "hyper_connection.low_rank": 320, "attention.indexer.head_count": 4, "attention.indexer.key_length": 128,
            "attention.indexer.top_k": 2048, "ple.layers": [1], "ple.ngram_size": 3, "ple.heads_per_ngram": 8,
            "ple.conv_kernel": 4, "embedding_length_per_layer_input": 160}.items()}
        m.update({p + k: v for k, v in extra.items()})
        return m

    def config(self, meta):
        import tools.gguf_import as gi
        tensors = {"token_embd.weight": {"dims": [2560, 248320]},
                   **{f"blk.{i}.indexer.k_proj.weight": {"dims": [2560, 128]} for i in range(8)}}
        return gi.qwen4exp_config(meta, tensors)[0]

    def test_interval_and_recurrent_layers_agree(self):
        ratios = [0, 0, 0, 4, 0, 0, 0, 4]
        old = self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": ratios}))
        new = self.config(self.meta(**{"attention.recurrent_layers": [1, 1, 1, 0, 1, 1, 1, 0],
                                       "attention.compress_ratios": ratios}))
        self.assertEqual(old["layer_types"], ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(old, new)
        self.assertEqual(old["indexer_compress_ratio"], 4)

    def test_contradicting_metadata_is_refused(self):
        with self.assertRaises(SystemExit):
            self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": [0, 0, 4, 0, 0, 0, 4, 0]}))


if __name__ == "__main__":
    unittest.main()
