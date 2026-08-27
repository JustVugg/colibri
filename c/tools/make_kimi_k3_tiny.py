#!/usr/bin/env python3
"""Generate the tiny Kimi K3 checkpoint used by tests/test_kimi_k3_tiny.py.

numpy-only and deterministic: CI regenerates model.safetensors from SEED and
byte-identical weights come out on every platform (NumPy's PCG64 stream and
elementwise float32 arithmetic are platform-stable).  The reference tokens in
ref.json are NOT produced here: they come from Moonshot's own modeling code
(tools/make_kimi_k3_ref.py, run offline once, provenance pinned by SHA-256),
so the engine and its oracle never share an author.

The fixture is deliberately the smallest checkpoint that still crosses every
distinct execution path of kimi_k3.c:

  layer 0: KDA (Gated DeltaNet)  + dense MLP
  layer 1: gated MLA             + dense MLP
  layer 2: KDA                   + sparse MoE (routed MXFP4 experts)
  layer 3-5: gated MLA           + sparse MoE

Two engine contracts the emission must honour byte-for-byte:
  - routed experts are native MXFP4 g32: e2m1 nibble pairs packed two per
    byte plus one e8m0 exponent byte per group of 32.  kimi_k3.c checks the
    six per-expert tensor sizes exactly and refuses a container that does
    not match (w1/w3 packed [moe_inter, latent/2], scale [moe_inter,
    latent/32]; w2 packed [latent, moe_inter/2], scale [latent, moe_inter/32]).
  - the loader preads expert bytes raw from recorded fd/offset and leans on
    the back-to-back layout of the six tensors, so their emission order is
    load-bearing (w1 packed, w1 scale, w2 packed, w2 scale, w3 packed,
    w3 scale) and the safetensors writer must preserve insertion order.

No tokenizer.json is written on purpose: without one the engine prints raw
generated token ids on stdout, which is exactly what the token-exact test
wants to parse.

Greedy determinism is engineered, not hoped for: lm_head is a scaled cyclic
shift of embed_tokens, giving every position a wide top-1 logit margin
instead of relying on accidental near-ties between random rows, and the EOS
row is zeroed so fixed-length regressions can detect early truncation.  The
margin is then verified, not assumed: make_kimi_k3_ref.py asserts a minimum
top1-top2 logit gap at every emitted position.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20260823

VOCAB = 320
HIDDEN = 128
LAYERS = 6
FIRST_DENSE = 2          # layers 0..1 use the dense MLP path
DENSE_INTER = 64
HEADS = 4
Q_RANK = 32
KV_RANK = 32
QK_NOPE = 16
QK_ROPE = 8
V_HEAD = 16
EXPERTS = 8
TOPK = 2
MOE_INTER = 32           # engine requires a multiple of 32
LATENT = 32              # engine requires a multiple of 32
SHARED = 1
RES_BLOCK = 2
KDA_HEADS = 2
KDA_HD = 16
CONV_K = 4               # engine requires <= 8
KDA_LAYERS = [1, 3]      # 1-indexed, per the engine's config schema
EOS_ID = 1

# One config for two readers.  kimi_k3.c consumes the numeric core and
# linear_attn_config.kda_layers; the extra semantic keys (router activation,
# topk method, situ, gated NoPE MLA, full_attn_layers, use_full_rank_gate)
# are ignored by the engine but consumed by Moonshot's KimiLinearConfig, so
# the vendor reference and the engine read the SAME file and cannot drift.
# Values mirror the real Kimi-K3 checkpoint's text_config (the behavior the
# engine hardcodes), with only the dimensions shrunk.
CONFIG = {
    "model_type": "kimi_linear",
    "architectures": ["KimiLinearForCausalLM"],
    "hidden_size": HIDDEN,
    "num_hidden_layers": LAYERS,
    "vocab_size": VOCAB,
    "first_k_dense_replace": FIRST_DENSE,
    "intermediate_size": DENSE_INTER,
    "num_attention_heads": HEADS,
    "num_key_value_heads": HEADS,
    "q_lora_rank": Q_RANK,
    "kv_lora_rank": KV_RANK,
    "qk_nope_head_dim": QK_NOPE,
    "qk_rope_head_dim": QK_ROPE,
    "v_head_dim": V_HEAD,
    "num_experts": EXPERTS,
    "num_experts_per_token": TOPK,
    "moe_intermediate_size": MOE_INTER,
    "routed_expert_hidden_size": LATENT,
    "num_shared_experts": SHARED,
    "attn_res_block_size": RES_BLOCK,
    "hidden_act": "situ",
    "activation_situ_beta": 1.0,
    "activation_situ_linear_beta": 1.0,
    "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc",
    "use_grouped_topk": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "moe_renormalize": True,
    "moe_layer_freq": 1,
    "routed_scaling_factor": 1.0,
    "latent_moe_use_norm": True,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
    "linear_attn_config": {
        "num_heads": KDA_HEADS,
        "head_dim": KDA_HD,
        "short_conv_kernel_size": CONV_K,
        "kda_layers": KDA_LAYERS,
        "full_attn_layers": [x for x in range(1, LAYERS + 1)
                             if x not in KDA_LAYERS],
        "use_full_rank_gate": True,
        "gate_lower_bound": -5.0,
    },
    "bos_token_id": 0,
    "eos_token_id": EOS_ID,
}

ST_DTYPE = {np.dtype("float32"): "F32", np.dtype("uint8"): "U8"}


def build_tensors(rng):
    """Every weight of the tiny model, in emission order."""
    proj = KDA_HEADS * KDA_HD
    qk_head = QK_NOPE + QK_ROPE
    shared_inter = MOE_INTER * SHARED
    tensors = {}

    def put(name, arr):
        assert name not in tensors, name
        tensors[name] = np.ascontiguousarray(arr)

    def normal(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def ones(n):
        return np.ones(n, dtype=np.float32)

    def mxfp4(rows, cols):
        """One packed MXFP4 matrix.  All 16 e2m1 codes are finite, so any
        nibble pattern is legal; exponents stay near 127 (2^0) to keep the
        dequantized magnitudes tame and never hit the 0xFF NaN code."""
        packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        scale = rng.integers(124, 131, size=(rows, cols // 32), dtype=np.uint8)
        return packed, scale

    embed = normal(VOCAB, HIDDEN, s=0.13)
    put("model.embed_tokens.weight", embed)
    put("model.norm.weight", ones(HIDDEN))
    put("model.output_attn_res_norm.weight", ones(HIDDEN))
    put("model.output_attn_res_proj.weight", normal(1, HIDDEN, s=0.1))
    # A scaled cyclic shift of the embedding: token t points firmly at
    # (t+1) % VOCAB, so greedy argmax never rides a near-tie between random
    # rows.  EOS is deliberately unattractive (zero row) so every
    # fixed-length regression can detect early truncation.
    lm_head = np.roll(embed, -1, axis=0) * 4.0
    lm_head[EOS_ID] = 0.0
    put("lm_head.weight", lm_head)

    for i in range(LAYERS):
        p = f"model.layers.{i}."
        put(p + "input_layernorm.weight", ones(HIDDEN))
        put(p + "post_attention_layernorm.weight", ones(HIDDEN))
        put(p + "self_attention_res_norm.weight", ones(HIDDEN))
        put(p + "self_attention_res_proj.weight", normal(1, HIDDEN, s=0.1))
        put(p + "mlp_res_norm.weight", ones(HIDDEN))
        put(p + "mlp_res_proj.weight", normal(1, HIDDEN, s=0.1))

        a = p + "self_attn."
        if (i + 1) in KDA_LAYERS:            # kda_layers is 1-indexed
            for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
                put(a + n + ".weight", normal(proj, HIDDEN, s=0.05))
            put(a + "o_proj.weight", normal(HIDDEN, proj, s=0.05))
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                put(a + n + ".weight", normal(proj, 1, CONV_K, s=0.3))
            put(a + "f_a_proj.weight", normal(KDA_HD, HIDDEN, s=0.05))
            put(a + "f_b_proj.weight", normal(proj, KDA_HD, s=0.05))
            put(a + "b_proj.weight", normal(KDA_HEADS, HIDDEN, s=0.05))
            put(a + "dt_bias", normal(proj, s=0.1))
            put(a + "o_norm.weight", ones(KDA_HD))
            # stored zero-padded to kda_hd; the engine reads the first
            # kda_heads entries and exponentiates them
            put(a + "A_log", normal(KDA_HD, s=0.1))
        else:
            put(a + "q_a_proj.weight", normal(Q_RANK, HIDDEN, s=0.05))
            put(a + "q_b_proj.weight", normal(HEADS * qk_head, Q_RANK, s=0.05))
            put(a + "kv_a_proj_with_mqa.weight",
                normal(KV_RANK + QK_ROPE, HIDDEN, s=0.05))
            put(a + "kv_b_proj.weight",
                normal(HEADS * (QK_NOPE + V_HEAD), KV_RANK, s=0.05))
            put(a + "o_proj.weight", normal(HIDDEN, HEADS * V_HEAD, s=0.05))
            put(a + "g_proj.weight", normal(HEADS * V_HEAD, HIDDEN, s=0.05))
            put(a + "q_a_layernorm.weight", ones(Q_RANK))
            put(a + "kv_a_layernorm.weight", ones(KV_RANK))

        if i < FIRST_DENSE:
            put(p + "mlp.gate_proj.weight", normal(DENSE_INTER, HIDDEN, s=0.05))
            put(p + "mlp.up_proj.weight", normal(DENSE_INTER, HIDDEN, s=0.05))
            put(p + "mlp.down_proj.weight", normal(HIDDEN, DENSE_INTER, s=0.05))
        else:
            b = p + "block_sparse_moe."
            # closed-form router weights: wide, stable expert preferences per
            # hidden direction, so top-k selection cannot ride a near-tie
            gate = 0.05 * np.sin(
                np.arange(EXPERTS * HIDDEN, dtype=np.float32) * 0.017 + i
            ).reshape(EXPERTS, HIDDEN).astype(np.float32)
            put(b + "gate.weight", gate)
            put(b + "gate.e_score_correction_bias",
                np.linspace(-0.03, 0.03, EXPERTS, dtype=np.float32))
            put(b + "routed_expert_norm.weight", ones(LATENT))
            put(b + "routed_expert_down_proj.weight",
                normal(LATENT, HIDDEN, s=0.05))
            put(b + "routed_expert_up_proj.weight",
                normal(HIDDEN, LATENT, s=0.05))
            put(b + "shared_experts.gate_proj.weight",
                normal(shared_inter, HIDDEN, s=0.05))
            put(b + "shared_experts.up_proj.weight",
                normal(shared_inter, HIDDEN, s=0.05))
            put(b + "shared_experts.down_proj.weight",
                normal(HIDDEN, shared_inter, s=0.05))
            for e in range(EXPERTS):
                q = b + f"experts.{e}."
                w1p, w1s = mxfp4(MOE_INTER, LATENT)
                w2p, w2s = mxfp4(LATENT, MOE_INTER)
                w3p, w3s = mxfp4(MOE_INTER, LATENT)
                # order is load-bearing: the engine preads these six raw and
                # back-to-back from recorded offsets
                put(q + "w1.weight_packed", w1p)
                put(q + "w1.weight_scale", w1s)
                put(q + "w2.weight_packed", w2p)
                put(q + "w2.weight_scale", w2s)
                put(q + "w3.weight_packed", w3p)
                put(q + "w3.weight_scale", w3s)
    return tensors


def write_safetensors(path, tensors):
    """Hand-rolled writer: preserves dict insertion order, which the expert
    contiguity contract depends on (the safetensors library reorders)."""
    header, offset = {}, 0
    for name, arr in tensors.items():
        header[name] = {
            "dtype": ST_DTYPE[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + arr.nbytes],
        }
        offset += arr.nbytes
    blob = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for arr in tensors.values():
            fh.write(arr.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="./kimi_k3_tiny")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing model.safetensors")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "model.safetensors"
    if model_path.exists() and not args.force:
        print(f"{model_path} exists; use --force to regenerate",
              file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    tensors = build_tensors(rng)
    write_safetensors(model_path, tensors)
    (out / "config.json").write_text(json.dumps(CONFIG, indent=2) + "\n")

    total = sum(a.nbytes for a in tensors.values())
    print(f"{out}: {len(tensors)} tensors, {total / 1e6:.1f} MB, "
          f"{LAYERS} layers ({LAYERS - FIRST_DENSE} sparse, "
          f"KDA at {KDA_LAYERS}) x {EXPERTS} experts, latent {LATENT}, "
          f"seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
