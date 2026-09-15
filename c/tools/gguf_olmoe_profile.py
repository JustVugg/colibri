#!/usr/bin/env python3
"""OLMoE profile: GGUF tensor layout -> colibri container layout.

This module owns everything OLMoE-specific about the conversion and nothing
about file I/O or CLI:

  * name mapping from GGUF (llama.cpp) tensor names to the names c/olmoe.c
    loads: model.embed_tokens.weight, model.layers.N.self_attn.q_proj.weight,
    and per-expert model.layers.N.mlp.experts.E.{merged_weight,qs};
  * dequantized GGUF order -> logical HuggingFace order (GGUF stores dims
    reversed);
  * row-wise int8 quantization identical to olmoe.c's decoder contract
    W[o,i] ~= q[o,i]*scale[o] (same math as tools/convert_olmoe_merged.py);
  * per-expert gate|up|down merge into one merged_weight + one .qs;
  * config.json reconstruction from GGUF olmoe.* metadata.

It is imported by convert_gguf_to_olmoe.py (CLI) and unit-tested directly.
"""

import re

import numpy as np

EXPERT_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")

DENSE_SIMPLE = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

DENSE_LAYERED_RE = re.compile(
    r"^blk\.(\d+)\.(attn_norm|ffn_norm|attn_q|attn_k|attn_v|attn_output|"
    r"attn_q_norm|attn_k_norm|ffn_gate_inp)\.weight$")

LAYER_SUFFIX = {
    "attn_norm": "input_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight",
    "attn_q": "self_attn.q_proj.weight",
    "attn_k": "self_attn.k_proj.weight",
    "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight",
    "attn_q_norm": "self_attn.q_norm.weight",
    "attn_k_norm": "self_attn.k_norm.weight",
    "ffn_gate_inp": "mlp.gate.weight",
}

CONFIG_KEYS = {
    "olmoe.embedding_length": "hidden_size",
    "olmoe.block_count": "num_hidden_layers",
    "olmoe.attention.head_count": "num_attention_heads",
    "olmoe.attention.head_count_kv": "num_key_value_heads",
    "olmoe.expert_count": "num_experts",
    "olmoe.expert_used_count": "num_experts_per_tok",
    "olmoe.feed_forward_length": "intermediate_size",
    "olmoe.rope.freq_base": "rope_theta",
    "olmoe.attention.layer_norm_rms_epsilon": "rms_norm_eps",
}


def parse_expert(name):
    """(layer, kind) for an expert tensor name, or None."""
    match = EXPERT_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def dense_target(name):
    """colibri name for a GGUF dense tensor, or None if not a known dense tensor."""
    if name in DENSE_SIMPLE:
        return DENSE_SIMPLE[name]
    match = DENSE_LAYERED_RE.match(name)
    if match:
        layer, kind = int(match.group(1)), match.group(2)
        return "model.layers.%d.%s" % (layer, LAYER_SUFFIX[kind])
    return None


def to_logical(flat, gguf_dims):
    """Reshape GGUF-order flat values into the logical (HuggingFace) shape.

    GGUF stores the source tensor's bytes unchanged and only reverses the dims
    labels, so the flat data is already C-order of the original shape: reshape
    to reversed(gguf_dims) reproduces it. No transpose is involved.
    """
    return np.ascontiguousarray(flat.reshape(tuple(reversed(gguf_dims))))


def restore_rope_layout(weights, n_head, n_kv_head=None):
    """Undo llama.cpp's RoPE permutation of q_proj/k_proj weights.

    conversion/olmo.py permutes q_proj and k_proj exactly like Llama:
        forward: w.reshape(n, 2, out // n // 2, *rest).swapaxes(1, 2).reshape(w.shape)
    with n = n_kv_head when it differs from n_head (k_proj), else n_head. That
    is NOT an involution unless out // n // 2 == 2, so the inverse is the axis
    transpose of the (m, 2) decomposition:
        inverse: w.reshape(n, m, 2, *rest).swapaxes(1, 2).reshape(w.shape)
    with m = out // n // 2. c/olmoe.c uses HF-style RoPE (adjacent pairs per
    head), so the GGUF's permuted q/k must be restored to the HF layout.
    """
    n = n_kv_head if (n_kv_head and n_head != n_kv_head) else n_head
    w = np.asarray(weights)
    if w.ndim < 1 or n < 1 or w.shape[0] % n != 0 or (w.shape[0] // n) % 2 != 0:
        raise ValueError("cannot restore RoPE layout for shape %r with %d heads"
                         % (w.shape, n))
    m = w.shape[0] // n // 2
    rest = w.shape[1:]
    return np.ascontiguousarray(
        w.reshape((n, m, 2) + rest).swapaxes(1, 2).reshape(w.shape))


def quantize_row(weights):
    """Row-wise int8 quantization, identical math to c/olmoe.c's decoder.

    olmoe.c computes y[o] = scale[o] * sum_i x[i]*q[o,i], so the stored bytes
    must satisfy W[o,i] ~= q[o,i]*scale[o] with scale[o] = row_absmax/127.
    Rounding is numpy's rint (round-half-to-even), matching convert_olmoe_merged.py.
    """
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2:
        raise ValueError("quantize_row expects a 2-D weight, got shape %r" % (w.shape,))
    row_max = np.maximum(np.max(np.abs(w), axis=1, keepdims=True), np.float32(1e-12))
    scales = (row_max / np.float32(127.0)).astype(np.float32)
    quantized = np.clip(np.rint(w / scales), -128.0, 127.0).astype(np.int8)
    return quantized, scales.reshape(-1)


def merge_expert(gate, up, down):
    """Quantize and merge one expert's gate/up/down into colibri's layout.

    merged_weight = int8(gate | up | down) flattened;
    qs            = f32(gate rows | up rows | down rows).
    gate/up: [inter, hidden]; down: [hidden, inter].
    """
    q_gate, s_gate = quantize_row(gate)
    q_up, s_up = quantize_row(up)
    q_down, s_down = quantize_row(down)
    merged = np.concatenate((q_gate.reshape(-1), q_up.reshape(-1), q_down.reshape(-1)))
    scales = np.concatenate((s_gate, s_up, s_down)).astype(np.float32)
    return merged, scales


def expert_matrix_shape(gguf_dims):
    """HF shape of one expert's matrix inside a fused [d0, d1, E] expert tensor."""
    return (gguf_dims[1], gguf_dims[0])


def expert_bytes_per_slice(gguf_dims, ggml_type):
    """Bytes of one expert inside a fused expert tensor (contiguous slice)."""
    from gguf_reader import ggml_block_spec
    block_elems, block_bytes = ggml_block_spec(ggml_type)
    if gguf_dims[0] % block_elems != 0:
        raise ValueError(
            "expert tensor dims[0]=%d is not a multiple of the block %d"
            % (gguf_dims[0], block_elems))
    row_blocks = gguf_dims[0] // block_elems
    return row_blocks * block_bytes * gguf_dims[1]


def _token_type_of(meta, index):
    entries = meta.get("tokenizer.ggml.token_type")
    if not entries or index >= len(entries):
        return 1
    return int(entries[index])


def build_tokenizer(meta):
    """Reconstruct an HF-style tokenizer.json from GGUF tokenizer metadata.

    Feed shape matches what c/tok.h loads for GPT-2/byte-level BPE:
      model.vocab (token string -> id), model.merges (space-separated "left
      right" entries), added_tokens (id/content/special), pre_tokenizer that
      does not select the o200k/Kimi maybe-flags, plus bos/eos/pad and the chat
      template. token_type 3=CONTROL (special) and 4=USER_DEFINED become added
      tokens; 5=UNUSED and 6=BYTE stay plain vocab entries.
    """
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens:
        raise ValueError("GGUF metadata missing tokenizer.ggml.tokens")
    merges = meta.get("tokenizer.ggml.merges")

    vocab = {}
    added = []
    # ggml token_type: 1=NORMAL stays plain BPE vocab; 3=CONTROL and
    # 4=USER_DEFINED become added tokens (atomic, literal); 5=UNUSED and 6=BYTE
    # stay plain vocab entries (not atomic specials).
    for index, token in enumerate(tokens):
        vocab[token] = index
        kind = _token_type_of(meta, index)
        if kind in (3, 4):
            added.append({"id": index, "content": token, "special": kind == 3})

    model = {"type": "BPE", "vocab": vocab}
    if merges:
        model["merges"] = list(merges)

    tokenizer = {
        "add_bos_token": bool(meta.get("tokenizer.ggml.add_bos_token")),
        "add_eos_token": bool(meta.get("tokenizer.ggml.add_eos_token")),
        "model": model,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "added_tokens": added,
    }
    for gguf_key, hf_key in (
            ("tokenizer.ggml.bos_token_id", "bos_token"),
            ("tokenizer.ggml.eos_token_id", "eos_token"),
            ("tokenizer.ggml.padding_token_id", "pad_token")):
        token_id = meta.get(gguf_key)
        if token_id is not None and int(token_id) < len(tokens):
            tokenizer[hf_key] = tokens[int(token_id)]
    if "tokenizer.chat_template" in meta:
        tokenizer["chat_template"] = meta["tokenizer.chat_template"]
    return tokenizer


def build_config(meta, norm_topk_prob=False, extra=None):
    """Reconstruct the config.json c/olmoe.c loads from GGUF metadata.

    norm_topk_prob is NOT in GGUF metadata; the official OLMoE checkpoints use
    false (transformers' OlmoeConfig default), so that is the default here. Use
    the CLI flag to override for a checkpoint that trained with it on.
    """
    config = {}
    for gguf_key, cfg_key in CONFIG_KEYS.items():
        if gguf_key not in meta:
            raise ValueError("GGUF metadata missing %s" % gguf_key)
        config[cfg_key] = meta[gguf_key]
    tokens = meta.get("tokenizer.ggml.tokens")
    if not tokens:
        raise ValueError("GGUF metadata missing tokenizer.ggml.tokens (needed for vocab_size)")
    config["vocab_size"] = len(tokens)
    config["norm_topk_prob"] = bool(norm_topk_prob)
    config["model_type"] = "olmoe"
    config["architectures"] = ["OlmoeForCausalLM"]
    if extra:
        config.update(extra)
    return config
