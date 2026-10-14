#!/usr/bin/env python3
"""Generate the tiny Clef checkpoint used by tests/test_clef_tiny.py.

numpy-only and deterministic: CI regenerates it from SEED and gets byte-identical
files. The reference answers in clef_tiny/ref.json are NOT produced here: they
come from Clef's own joint_schema_model.py on transformers, run offline by
tools/make_clef_ref.py, so the engine and its oracle never share an author.

The fixture has the layout of the release (Cloudflare/clef) at toy widths:

    config.json                   Qwen3_5ForConditionalGeneration: the dense
                                  hybrid text model (3 Gated DeltaNet layers per
                                  attention layer, output-gated attention,
                                  partial interleaved M-RoPE) and a small ViT
    model.safetensors             F16, every value exactly representable, so the
                                  container (convert_qwen36.py writes F16) holds
                                  the very weights the f32 reference runs
    joint_head_config.json        the joint schema head: 2 routing layers, 4
    joint_head.safetensors        decoder layers, as the release, at width 32
    tokenizer.json                byte-level BPE trained here, with the release's
                                  normalizer (NFC), pre-tokenizer regex and added
                                  tokens (renumbered after the vocabulary)
    tokenizer_config.json         the release's (Qwen2Tokenizer)

Then `tools/convert_qwen36.py --model clef_tiny --out clef_tiny_c --ebits 8`
turns it into the container the engine reads, head files included.
"""

import argparse
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20261003

HIDDEN = 64
LAYERS = 8
Q_HEADS, KV_HEADS, HEAD_DIM = 4, 1, 32
DN_KEY_HEADS, DN_VALUE_HEADS, DN_HEAD_DIM = 2, 4, 16
INTER = 128
VOCAB = 1024
MERGES = 600
HEAD = {"hidden_size": HIDDEN, "width": 32, "routing_layers": 2, "layers": 4,
        "heads": 4, "feedforward": 64}
VISION = {"depth": 1, "hidden_size": 32, "num_heads": 2, "intermediate_size": 64,
          "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
          "in_channels": 3, "out_hidden_size": HIDDEN, "num_position_embeddings": 64}

# The release's added tokens, in its order (ids 248044..248076 there).
ADDED = [("<|endoftext|>", True), ("<|im_start|>", True), ("<|im_end|>", True),
         ("<|object_ref_start|>", True), ("<|object_ref_end|>", True), ("<|box_start|>", True),
         ("<|box_end|>", True), ("<|quad_start|>", True), ("<|quad_end|>", True),
         ("<|vision_start|>", True), ("<|vision_end|>", True), ("<|vision_pad|>", True),
         ("<|image_pad|>", True), ("<|video_pad|>", True), ("<tool_call>", False),
         ("</tool_call>", False), ("<|fim_prefix|>", False), ("<|fim_middle|>", False),
         ("<|fim_suffix|>", False), ("<|fim_pad|>", False), ("<|repo_name|>", False),
         ("<|file_sep|>", False), ("<tool_response>", False), ("</tool_response>", False),
         ("<think>", False), ("</think>", False), ("<|audio_start|>", True),
         ("<|audio_end|>", True), ("<tts_pad>", True), ("<tts_text_bos>", True),
         ("<tts_text_eod>", True), ("<tts_text_bos_single>", True), ("<|audio_pad|>", True)]

PRETOKENIZE = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+"
               r"[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

TOKENIZER_CONFIG = {
    "add_prefix_space": False, "audio_bos_token": "<|audio_start|>",
    "audio_eos_token": "<|audio_end|>", "audio_token": "<|audio_pad|>", "backend": "tokenizers",
    "bos_token": None, "clean_up_tokenization_spaces": False, "eos_token": "<|im_end|>",
    "errors": "replace", "image_token": "<|image_pad|>", "model_max_length": 262144,
    "pad_token": "<|endoftext|>", "split_special_tokens": False,
    "tokenizer_class": "Qwen2Tokenizer", "unk_token": None, "video_token": "<|video_pad|>",
    "vision_bos_token": "<|vision_start|>", "vision_eos_token": "<|vision_end|>"}

# What the merges are learned from: the prompt the reference renders, the JSON of
# options, and text in a few scripts, so a rendered record is a few hundred tokens.
CORPUS = """
Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.
STATE: SCHEMA FIELDS: FIELD ID: TYPE: INSTRUCTION: ALLOWED OPTIONS: OPTION END FIELD JOINT SCHEMA DECISIONS:
{"description":"The proposition is true or the answer is yes.","option_id":"true"}
{"description":"The proposition is false or the answer is no.","option_id":"false"}
choice score noul option_id description criteria instructions state
Which department should handle this request? How urgent is this request? Is the invoice overdue?
Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.
Our checkout started returning errors and orders are blocked. The payment provider is down.
billing technical sales other payments invoices refunds bugs outages errors pricing contracts
not urgent soon blocking critical deadline the user threatens to leave the customer is angry
{"invoice":{"currency":"USD","status":"overdue","total":1250.0,"vendor":"Acme"}}
{"role":"user","content":"I cannot log in to my account since yesterday."}
Buongiorno, la fattura è sbagliata. Grüße aus München. ευχαριστώ πολύ. 我们的账户被锁了。
"""


def bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) +
          list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def train_bpe(corpus, merges):
    """Plain BPE over word pieces, ties broken by the pair's text: the same merges
    on every platform. The pieces only have to be plausible; the tokenizer then
    splits with the release's own regex."""
    byte_map = bytes_to_unicode()
    words = {}
    for piece in re.findall(r" ?\w+| ?[^\s\w]+|\s+", corpus):
        symbols = tuple(byte_map[b] for b in piece.encode("utf-8"))
        words[symbols] = words.get(symbols, 0) + 1
    learned = []
    for _ in range(merges):
        counts = {}
        for symbols, freq in words.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] = counts.get(pair, 0) + freq
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        if counts[best] < 2:
            break
        learned.append(best)
        merged = {}
        for symbols, freq in words.items():
            out, i = [], 0
            while i < len(symbols):
                if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == best:
                    out.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + freq
        words = merged
    return byte_map, learned


def build_tokenizer():
    byte_map, merges = train_bpe(CORPUS, MERGES)
    vocab = {}
    for b in range(256):
        vocab[byte_map[b]] = len(vocab)
    for left, right in merges:
        if left + right not in vocab:
            vocab[left + right] = len(vocab)
    added = []
    for content, special in ADDED:
        added.append({"id": len(vocab) + len(added), "content": content, "single_word": False,
                      "lstrip": False, "rstrip": False, "normalized": False, "special": special})
    if added[-1]["id"] >= VOCAB:
        raise SystemExit(f"tokenizer needs {added[-1]['id'] + 1} ids, the model has {VOCAB}")
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": PRETOKENIZE}, "behavior": "Isolated",
             "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
             "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": False,
                           "use_regex": False},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                    "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None,
                  "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": False,
                  "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [[left, right] for left, right in merges]},
    }
    return tokenizer, {entry["content"]: entry["id"] for entry in added}


def model_config(ids):
    layer_types = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(LAYERS)]
    text = {
        "attention_bias": False, "attention_dropout": 0.0, "attn_output_gate": True,
        "bos_token_id": ids["<|endoftext|>"], "dtype": "float16",
        "eos_token_id": ids["<|endoftext|>"], "full_attention_interval": 4,
        "head_dim": HEAD_DIM, "hidden_act": "silu", "hidden_size": HIDDEN,
        "initializer_range": 0.02, "intermediate_size": INTER, "layer_types": layer_types,
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": DN_HEAD_DIM,
        "linear_num_key_heads": DN_KEY_HEADS, "linear_num_value_heads": DN_VALUE_HEADS,
        "linear_value_head_dim": DN_HEAD_DIM, "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 4096, "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 0, "mtp_use_dedicated_embeddings": False,
        "num_attention_heads": Q_HEADS, "num_hidden_layers": LAYERS,
        "num_key_value_heads": KV_HEADS, "output_gate_type": "swish", "pad_token_id": None,
        "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-06,
        # head_dim 32 x 0.25 = 8 rotary dims, four frequencies: [2, 1, 1] puts one on
        # every axis, as the release's [11, 11, 10] spreads its 32
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [2, 1, 1],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                            "rope_type": "default"},
        "tie_word_embeddings": False, "use_cache": True, "vocab_size": VOCAB,
    }
    vision = dict(VISION, deepstack_visual_indexes=[], dtype="float16",
                  hidden_act="gelu_pytorch_tanh", initializer_range=0.02,
                  model_type="qwen3_5_vision")
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "dtype": "float16",
            "image_token_id": ids["<|image_pad|>"], "language_model_only": False,
            "model_type": "qwen3_5", "text_config": text, "tie_word_embeddings": False,
            "transformers_version": "5.10.2", "video_token_id": ids["<|video_pad|>"],
            "vision_config": vision, "vision_end_token_id": ids["<|vision_end|>"],
            "vision_start_token_id": ids["<|vision_start|>"]}


def build_backbone(rng):
    d, tensors = HIDDEN, {}

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def put(name, array):
        tensors[name] = array.astype(np.float16)

    lm = "model.language_model."
    put(lm + "embed_tokens.weight", normal((VOCAB, d), 1.0))
    for i in range(LAYERS):
        p = f"{lm}layers.{i}."
        put(p + "input_layernorm.weight", normal(d, 0.1))           # applied as 1 + w
        put(p + "post_attention_layernorm.weight", normal(d, 0.1))
        if i % 4 == 3:
            q = Q_HEADS * HEAD_DIM
            put(p + "self_attn.q_proj.weight", normal((2 * q, d), 1.0 / np.sqrt(d)))   # + output gate
            put(p + "self_attn.k_proj.weight", normal((KV_HEADS * HEAD_DIM, d), 1.0 / np.sqrt(d)))
            put(p + "self_attn.v_proj.weight", normal((KV_HEADS * HEAD_DIM, d), 1.0 / np.sqrt(d)))
            put(p + "self_attn.o_proj.weight", normal((d, q), 0.5 / np.sqrt(q)))
            put(p + "self_attn.q_norm.weight", normal(HEAD_DIM, 0.1))
            put(p + "self_attn.k_norm.weight", normal(HEAD_DIM, 0.1))
        else:
            key, value = DN_KEY_HEADS * DN_HEAD_DIM, DN_VALUE_HEADS * DN_HEAD_DIM
            conv = 2 * key + value
            put(p + "linear_attn.in_proj_qkv.weight", normal((conv, d), 1.0 / np.sqrt(d)))
            put(p + "linear_attn.in_proj_z.weight", normal((value, d), 1.0 / np.sqrt(d)))
            put(p + "linear_attn.in_proj_b.weight", normal((DN_VALUE_HEADS, d), 1.0 / np.sqrt(d)))
            put(p + "linear_attn.in_proj_a.weight", normal((DN_VALUE_HEADS, d), 1.0 / np.sqrt(d)))
            put(p + "linear_attn.conv1d.weight", normal((conv, 1, 4), 0.5))
            put(p + "linear_attn.dt_bias", normal(DN_VALUE_HEADS, 0.5))
            put(p + "linear_attn.A_log", np.log(rng.uniform(1.0, 8.0, DN_VALUE_HEADS)).astype(np.float32))
            put(p + "linear_attn.norm.weight", 1.0 + normal(DN_HEAD_DIM, 0.1))
            put(p + "linear_attn.out_proj.weight", normal((d, value), 0.5 / np.sqrt(value)))
        put(p + "mlp.gate_proj.weight", normal((INTER, d), 1.0 / np.sqrt(d)))
        put(p + "mlp.up_proj.weight", normal((INTER, d), 1.0 / np.sqrt(d)))
        put(p + "mlp.down_proj.weight", normal((d, INTER), 0.5 / np.sqrt(INTER)))
    put(lm + "norm.weight", normal(d, 0.1))
    put("lm_head.weight", normal((VOCAB, d), 1.0 / np.sqrt(d)))
    v, vh, vi = "model.visual.", VISION["hidden_size"], VISION["intermediate_size"]
    patch = VISION["in_channels"] * VISION["temporal_patch_size"] * VISION["patch_size"] ** 2
    put(v + "patch_embed.proj.weight", normal((vh, VISION["in_channels"], VISION["temporal_patch_size"],
                                              VISION["patch_size"], VISION["patch_size"]),
                                             1.0 / np.sqrt(patch)))
    put(v + "patch_embed.proj.bias", normal(vh, 0.1))
    put(v + "pos_embed.weight", normal((VISION["num_position_embeddings"], vh), 0.5))
    for i in range(VISION["depth"]):
        b = f"{v}blocks.{i}."
        for norm in ("norm1", "norm2"):
            put(b + norm + ".weight", 1.0 + normal(vh, 0.1))
            put(b + norm + ".bias", normal(vh, 0.1))
        put(b + "attn.qkv.weight", normal((3 * vh, vh), 1.0 / np.sqrt(vh)))
        put(b + "attn.qkv.bias", normal(3 * vh, 0.1))
        put(b + "attn.proj.weight", normal((vh, vh), 1.0 / np.sqrt(vh)))
        put(b + "attn.proj.bias", normal(vh, 0.1))
        put(b + "mlp.linear_fc1.weight", normal((vi, vh), 1.0 / np.sqrt(vh)))
        put(b + "mlp.linear_fc1.bias", normal(vi, 0.1))
        put(b + "mlp.linear_fc2.weight", normal((vh, vi), 1.0 / np.sqrt(vi)))
        put(b + "mlp.linear_fc2.bias", normal(vh, 0.1))
    wide = vh * VISION["spatial_merge_size"] ** 2
    put(v + "merger.norm.weight", 1.0 + normal(vh, 0.1))
    put(v + "merger.norm.bias", normal(vh, 0.1))
    put(v + "merger.linear_fc1.weight", normal((wide, wide), 1.0 / np.sqrt(wide)))
    put(v + "merger.linear_fc1.bias", normal(wide, 0.1))
    put(v + "merger.linear_fc2.weight", normal((HIDDEN, wide), 1.0 / np.sqrt(wide)))
    put(v + "merger.linear_fc2.bias", normal(HIDDEN, 0.1))
    return tensors


def build_head(rng):
    """JointSchemaHead's state dict, at scales where the prior, the cosine and the
    residual scorer all move the logits by a few units."""
    H, W, F = HEAD["hidden_size"], HEAD["width"], HEAD["feedforward"]
    tensors = {}

    def normal(shape, scale):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def put(name, array):
        tensors[name] = np.asarray(array, dtype=np.float32).astype(np.float16)

    def norm(stem, n):
        put(stem + ".weight", 1.0 + normal(n, 0.1))
        put(stem + ".bias", normal(n, 0.1))

    def attention(stem):
        put(stem + ".in_proj_weight", normal((3 * W, W), 1.0 / np.sqrt(W)))
        put(stem + ".in_proj_bias", normal(3 * W, 0.1))
        put(stem + ".out_proj.weight", normal((W, W), 1.0 / np.sqrt(W)))
        put(stem + ".out_proj.bias", normal(W, 0.1))

    norm("hidden_norm", H)
    for name in ("memory_projection", "question_projection", "option_question_projection",
                 "global_projection", "option_context_projection", "option_lexical_projection"):
        put(name + ".weight", normal((W, H), 1.0 / np.sqrt(H)))
    put("type_embedding.weight", normal((3, W), 0.5))
    for i in range(HEAD["routing_layers"]):
        stem = f"evidence_layers.{i}."
        norm(stem + "query_norm", W)
        norm(stem + "memory_norm", W)
        attention(stem + "attention")
        norm(stem + "feedforward_norm", W)
        put(stem + "feedforward.0.weight", normal((F, W), 1.0 / np.sqrt(W)))
        put(stem + "feedforward.0.bias", normal(F, 0.1))
        put(stem + "feedforward.3.weight", normal((W, F), 1.0 / np.sqrt(F)))
        put(stem + "feedforward.3.bias", normal(W, 0.1))
    norm("option_summary_norm", W)
    for i in range(HEAD["layers"]):
        stem = f"layers.{i}."
        attention(stem + "self_attn")
        attention(stem + "multihead_attn")
        put(stem + "linear1.weight", normal((F, W), 1.0 / np.sqrt(W)))
        put(stem + "linear1.bias", normal(F, 0.1))
        put(stem + "linear2.weight", normal((W, F), 1.0 / np.sqrt(F)))
        put(stem + "linear2.bias", normal(W, 0.1))
        for n in ("norm1", "norm2", "norm3"):
            norm(stem + n, W)
    norm("field_norm", W)
    norm("option_norm", W)
    put("residual_scorer.0.weight", normal((W, 4 * W), 1.0 / np.sqrt(4 * W)))
    put("residual_scorer.0.bias", normal(W, 0.1))
    put("residual_scorer.3.weight", normal((1, W), 3.0 / np.sqrt(W)))
    put("residual_scorer.3.bias", normal(1, 0.1))
    put("prior_logit_scale", np.array(np.log(6.0)))
    put("joint_logit_scale", np.array(np.log(4.0)))
    put("residual_gate", np.array(0.5))
    return tensors


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, array in tensors.items():
        dtype = {"float16": "F16", "float32": "F32"}[array.dtype.name]
        header[name] = {"dtype": dtype, "shape": list(array.shape),
                        "data_offsets": [offset, offset + array.nbytes]}
        offset += array.nbytes
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        for array in tensors.values():
            handle.write(np.ascontiguousarray(array).view(np.uint8).tobytes())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="./clef_tiny")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer, ids = build_tokenizer()
    write_json(out / "tokenizer.json", tokenizer)
    write_json(out / "tokenizer_config.json", TOKENIZER_CONFIG)
    write_json(out / "config.json", model_config(ids))
    write_json(out / "joint_head_config.json", HEAD)
    rng = np.random.default_rng(SEED)
    write_safetensors(out / "model.safetensors", build_backbone(rng))
    write_safetensors(out / "joint_head.safetensors", build_head(rng))
    print(f"wrote {out}: {len(tokenizer['model']['vocab'])} vocab entries, "
          f"{len(tokenizer['model']['merges'])} merges, {len(ids)} added tokens", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
