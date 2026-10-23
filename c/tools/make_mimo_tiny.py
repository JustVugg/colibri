#!/usr/bin/env python3
"""Generate the tiny MiMo-V2.6 checkpoint used by tests/mimo_tiny_harness.py.

numpy-only and deterministic: CI regenerates the shards from SEED and
byte-identical weights come out on every platform.  The reference tokens in
ref.json are NOT produced here: they come from Xiaomi's own modeling code
(tools/make_mimo_ref.py, run offline once, vendor files pinned by SHA-256),
so the engine and its oracle never share an author.

The fixture stores every tensor the way the released checkpoint does, because
the storage is where a reading of MiMo goes wrong while a float reference
still agrees:

  - routed experts are MXFP4 g32 (e2m1 nibble pairs, low nibble = even column,
    one e8m0 byte per 32 columns), the six tensors of an expert back to back
    in the file order of the real shards: down, down scale, gate, gate scale,
    up, up scale;
  - qkv_proj and the layer-0 MLP are FP8 e4m3 with a F32 weight_scale_inv per
    128x128 block;
  - the fused qkv_proj is PRE-SHARDED in num_key_value_heads chunks
    [Q_0|K_0|V_0|Q_1|K_1|V_1] with the scale grid tiled per chunk.  The full
    layers' chunk (176 rows) is not a whole number of blocks, so the per-chunk
    grid has 4 rows where a contiguous one would have 3: a reader that assumes
    [Q|K|V] with one continuous grid fails here, as it would on the real
    checkpoint (108 scale rows, not 106, on a Flash full-attention layer);
  - o_proj, router, norms, sinks, embed and lm_head are BF16.

The geometry crosses every path of mimo.c: full-attention and sliding-window
layers with different KV head counts (2 and 4 heads over 2 chunks), a window
of 8 that every case outgrows, the SWA sink logit, partial RoPE with the two
thetas, the value scale, the dense layer 0 and MoE layers.

No tokenizer.json on purpose: without one the engine prints raw generated
token ids on stdout, which is what the token-exact test parses.  Greedy
determinism is engineered, then verified by make_mimo_ref.py: the head is
random at a large scale (wide top-2 gaps), the router's top-k boundary is
checked for near-ties at every token and layer, and the layers are checked to
matter (the argmax must differ from the embedding-only one at most positions).
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20260930

VOCAB = 320
HIDDEN = 256
LAYERS = 6
PATTERN = [0, 1, 1, 0, 1, 1]        # 1 = sliding window, 0 = full attention
MOE_FREQ = [0, 1, 1, 1, 1, 1]       # layer 0 dense, like the release
DENSE_INTER = 256
HEADS = 4                           # full attention
KV_HEADS = 2                        # also the qkv pre-shard count (ckpt_tp)
SWA_HEADS = 4
SWA_KV_HEADS = 4                    # two per chunk, as on the release
HEAD_DIM = 48                       # rope_dim = int(48 * 0.334) = 16
V_HEAD_DIM = 32
EXPERTS = 16
TOPK = 4
MOE_INTER = 64                      # MXFP4 needs multiples of 32
WINDOW = 8
EOS_ID = 1
BLOCK = 128
EMBED_S = 0.05                      # embedding scale
LM_S = 4.0                          # head scale
EXP_LO, EXP_HI = 121, 126           # MXFP4 e8m0 exponent range

CONFIG = {
    "architectures": ["MiMoV2ForCausalLM"],
    "model_type": "mimo_v2",
    "vocab_size": VOCAB,
    "hidden_size": HIDDEN,
    "intermediate_size": DENSE_INTER,
    "num_hidden_layers": LAYERS,
    "hybrid_layer_pattern": PATTERN,
    "moe_layer_freq": MOE_FREQ,
    "num_attention_heads": HEADS,
    "num_key_value_heads": KV_HEADS,
    "head_dim": HEAD_DIM,
    "v_head_dim": V_HEAD_DIM,
    "swa_num_attention_heads": SWA_HEADS,
    "swa_num_key_value_heads": SWA_KV_HEADS,
    "swa_head_dim": HEAD_DIM,
    "swa_v_head_dim": V_HEAD_DIM,
    "sliding_window": WINDOW,
    "sliding_window_size": WINDOW,
    "attention_chunk_size": WINDOW,
    "add_full_attention_sink_bias": False,
    "add_swa_attention_sink_bias": True,
    "attention_value_scale": 0.707,
    "attention_projection_layout": "fused_qkv",
    "attention_bias": False,
    "partial_rotary_factor": 0.334,
    "rope_theta": 10000000.0,
    "swa_rope_theta": 10000.0,
    "rope_parameters": {"partial_rotary_factor": 0.334, "rope_theta": 10000000.0,
                        "rope_type": "default", "type": "default"},
    "n_routed_experts": EXPERTS,
    "num_experts_per_tok": TOPK,
    "moe_intermediate_size": MOE_INTER,
    "n_shared_experts": None,
    "n_group": 1,
    "topk_group": 1,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": None,
    "moe_router_dtype": "bfloat16",
    "hidden_act": "silu",
    "layernorm_epsilon": 1e-6,
    "max_position_embeddings": 256,
    "tie_word_embeddings": False,
    "bos_token_id": None,
    "eos_token_id": EOS_ID,
    "pad_token_id": 0,
    "dtype": "bfloat16",
    "quantization_config": {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "store_dtype": "mxfp4",
        "mxfp4_block_size": 32,
        "weight_block_size": [BLOCK, BLOCK],
        "ignored_layers": [f"model.layers.{i}.self_attn.o_proj"
                           for i in range(LAYERS)],
    },
}


# --vision: a tiny MiMo ViT on the same language model. Six blocks cross every
# path of the tower: full attention at both ends (no sink), row-order and
# column-order windows, and the permutations entering and leaving a column run
# (types -1, 0, 1, 1, 0, -1; the release's pattern is the same alphabet).
VISION = {
    "depth": 6,
    "hidden_size": 64,
    "intermediate_size": 96,
    "num_heads": 4,
    "num_key_value_heads": 2,
    "qk_channels": 16,
    "out_hidden_size": HIDDEN,
    "patch_size": 16,
    "spatial_patch_size": 16,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "in_chans": 3,
    "hidden_act": "silu",
    "fullatt_block_indexes": [0, 5],
    "vit_window_attn_types": [-1, 0, 1, 1, 0, -1],
    "visual_token_window_size": 5,
    "use_sink": True,
    "window_size": 128,
    "tokens_per_second": 2,
}
IMAGE_TOKEN, VISION_START, VISION_END = 300, 301, 302
IMAGE_W, IMAGE_H = 192, 128          # already a multiple of 32: no resize
PREPROCESSOR = {
    "min_pixels": 3136,
    "max_pixels": 12845056,
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
    "image_processor_type": "Qwen2VLImageProcessor",
    "processor_class": "Qwen2_5_VLProcessor",
}


def bf16_bits(x):
    """f32 -> bf16 bit patterns, round to nearest even."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)
    return ((bits + rounding) >> 16).astype(np.uint16)


class BF16(np.ndarray):
    """uint16 bit patterns tagged for the writer as BF16."""


class F8(np.ndarray):
    """uint8 e4m3fn codes tagged for the writer as F8_E4M3."""


def tag(arr, cls):
    return np.ascontiguousarray(arr).view(cls)


def st_dtype(arr):
    if isinstance(arr, BF16):
        return "BF16"
    if isinstance(arr, F8):
        return "F8_E4M3"
    return {np.dtype("float32"): "F32", np.dtype("uint8"): "U8"}[arr.dtype]


def build_tensors(rng):
    """Every weight of the tiny model, in emission order, as {name: array}."""
    tensors = {}

    def put(name, arr):
        assert name not in tensors, name
        tensors[name] = arr

    def normal(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def bf16(x):
        return tag(bf16_bits(x), BF16)

    def norm_weight(n):
        return bf16(1.0 + normal(n, s=0.1))

    def fp8_codes(rows, cols):
        """Random finite e4m3fn codes with exponent fields 4..9 (|w| roughly
        1/8 .. 7.5 before the block scale); never the 0x7F/0xFF NaN codes."""
        sign = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8) << 7
        exp = rng.integers(4, 10, size=(rows, cols), dtype=np.uint8) << 3
        man = rng.integers(0, 8, size=(rows, cols), dtype=np.uint8)
        return sign | exp | man

    def fp8_scales(rows, cols):
        return (rng.uniform(0.004, 0.012, size=(rows, cols))).astype(np.float32)

    def fp8_matrix(name, out, inp):
        put(name + ".weight", tag(fp8_codes(out, inp), F8))
        put(name + ".weight_scale_inv",
            fp8_scales(-(-out // BLOCK), -(-inp // BLOCK)))

    def fp8_qkv(name, heads, kv_heads):
        """Pre-sharded fused qkv: KV_HEADS chunks of [Q_c|K_c|V_c] rows, the
        scale grid tiled per chunk (ceil(rows_per_chunk / 128) rows each)."""
        chunks = KV_HEADS
        rows_per_chunk = ((heads // chunks) * HEAD_DIM
                          + (kv_heads // chunks) * HEAD_DIM
                          + (kv_heads // chunks) * V_HEAD_DIM)
        put(name + ".weight",
            tag(fp8_codes(chunks * rows_per_chunk, HIDDEN), F8))
        put(name + ".weight_scale_inv",
            fp8_scales(chunks * -(-rows_per_chunk // BLOCK),
                       -(-HIDDEN // BLOCK)))

    def mxfp4(rows, cols):
        """One packed MXFP4 matrix.  All 16 e2m1 codes are finite; exponents
        stay near 127 so dequantized magnitudes stay tame, never 0xFF."""
        packed = rng.integers(0, 256, size=(rows, cols // 2), dtype=np.uint8)
        scale = rng.integers(EXP_LO, EXP_HI, size=(rows, cols // 32), dtype=np.uint8)
        return packed, scale

    embed = normal(VOCAB, HIDDEN, s=EMBED_S)
    put("model.embed_tokens.weight", bf16(embed))
    # A random head at a large scale: logits spread over ~LM_S * 16, so the
    # top-2 gap clears float noise by orders of magnitude, while the choice
    # still depends on the whole residual stream (make_mimo_ref.py checks that
    # the layers change the argmax, not just the embedding).  EOS is a zero
    # row: never the argmax, so a fixed-length run detects early truncation.
    lm_head = normal(VOCAB, HIDDEN, s=LM_S)
    lm_head[EOS_ID] = 0.0
    put("model.norm.weight", norm_weight(HIDDEN))
    put("lm_head.weight", bf16(lm_head))

    for i in range(LAYERS):
        p = f"model.layers.{i}."
        swa = PATTERN[i] == 1
        heads = SWA_HEADS if swa else HEADS
        kv_heads = SWA_KV_HEADS if swa else KV_HEADS
        put(p + "input_layernorm.weight", norm_weight(HIDDEN))
        put(p + "post_attention_layernorm.weight", norm_weight(HIDDEN))
        fp8_qkv(p + "self_attn.qkv_proj", heads, kv_heads)
        put(p + "self_attn.o_proj.weight",
            bf16(normal(HIDDEN, heads * V_HEAD_DIM, s=0.05)))
        if swa:
            put(p + "self_attn.attention_sink_bias", bf16(normal(heads, s=1.0)))
        if not MOE_FREQ[i]:
            fp8_matrix(p + "mlp.gate_proj", DENSE_INTER, HIDDEN)
            fp8_matrix(p + "mlp.up_proj", DENSE_INTER, HIDDEN)
            fp8_matrix(p + "mlp.down_proj", HIDDEN, DENSE_INTER)
            continue
        # closed-form router: wide, stable expert preferences per hidden
        # direction, so top-k selection cannot ride a near-tie
        gate = 0.08 * np.sin(np.arange(EXPERTS * HIDDEN, dtype=np.float32)
                             * 0.013 + 0.7 * i).reshape(EXPERTS, HIDDEN)
        put(p + "mlp.gate.weight", bf16(gate))
        put(p + "mlp.gate.e_score_correction_bias",
            np.linspace(-0.02, 0.02, EXPERTS, dtype=np.float32))
        for e in range(EXPERTS):
            q = p + f"mlp.experts.{e}."
            down, down_s = mxfp4(HIDDEN, MOE_INTER)
            gate_w, gate_s = mxfp4(MOE_INTER, HIDDEN)
            up, up_s = mxfp4(MOE_INTER, HIDDEN)
            # the file order of the release, back to back
            put(q + "down_proj.weight", down)
            put(q + "down_proj.weight_scale", down_s)
            put(q + "gate_proj.weight", gate_w)
            put(q + "gate_proj.weight_scale", gate_s)
            put(q + "up_proj.weight", up)
            put(q + "up_proj.weight_scale", up_s)
    return tensors


def build_vision(rng):
    """The tower's tensors, named and shaped as in the release (BF16; the merger
    carries no biases and its LayerNorm no bias, as there)."""
    v = VISION
    d, f, heads, kvh = v["hidden_size"], v["intermediate_size"], v["num_heads"], v["num_key_value_heads"]
    hd = v["qk_channels"]
    merged = d * v["spatial_merge_size"] ** 2
    tensors = {}

    def bf16(x):
        return tag(bf16_bits(x), BF16)

    def normal(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    tensors["visual.patch_embed.proj.weight"] = bf16(normal(d, 3, 2, 16, 16, s=0.03))
    for i in range(v["depth"]):
        p = f"visual.blocks.{i}."
        tensors[p + "norm1.weight"] = bf16(1.0 + normal(d, s=0.1))
        tensors[p + "norm2.weight"] = bf16(1.0 + normal(d, s=0.1))
        tensors[p + "attn.qkv.weight"] = bf16(normal((heads + 2 * kvh) * hd, d, s=0.15))
        tensors[p + "attn.qkv.bias"] = bf16(normal((heads + 2 * kvh) * hd, s=0.1))
        tensors[p + "attn.proj.weight"] = bf16(normal(d, heads * hd, s=0.1))
        tensors[p + "attn.proj.bias"] = bf16(normal(d, s=0.05))
        if i not in v["fullatt_block_indexes"]:
            tensors[p + "attn.sinks"] = bf16(normal(heads, s=1.0))
        for n in ("gate_proj", "up_proj"):
            tensors[p + f"mlp.{n}.weight"] = bf16(normal(f, d, s=0.1))
            tensors[p + f"mlp.{n}.bias"] = bf16(normal(f, s=0.05))
        tensors[p + "mlp.down_proj.weight"] = bf16(normal(d, f, s=0.1))
        tensors[p + "mlp.down_proj.bias"] = bf16(normal(d, s=0.05))
    tensors["visual.merger.ln_q.weight"] = bf16(1.0 + normal(d, s=0.1))
    tensors["visual.merger.mlp.0.weight"] = bf16(normal(merged, merged, s=0.05))
    tensors["visual.merger.mlp.2.weight"] = bf16(normal(HIDDEN, merged, s=0.2))
    return tensors


def test_image(rng):
    """A deterministic picture with structure at several scales, so the
    windows and the merge see different content everywhere."""
    y, x = np.mgrid[0:IMAGE_H, 0:IMAGE_W].astype(np.float32)
    r = 127 + 120 * np.sin(x / 9.0) * np.cos(y / 13.0)
    g = 127 + 120 * np.sin((x + y) / 17.0)
    b = (x * 3 + y * 5) % 256
    img = np.stack([r, g, b], axis=-1) + rng.integers(0, 24, size=(IMAGE_H, IMAGE_W, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def write_safetensors(path, tensors):
    """Hand-rolled writer: keeps insertion order, which the expert contiguity
    the engine exploits depends on (the safetensors library reorders)."""
    header, offset = {}, 0
    for name, arr in tensors.items():
        header[name] = {"dtype": st_dtype(arr), "shape": list(arr.shape),
                        "data_offsets": [offset, offset + arr.nbytes]}
        offset += arr.nbytes
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for arr in tensors.values():
            fh.write(np.ascontiguousarray(arr).view(np.uint8).tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="./mimo_tiny")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing shards")
    parser.add_argument("--vision", action="store_true",
                        help="add a tiny ViT, the image tokens, a test picture "
                             "and its patches (the vision fixture)")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    shard_names = ["model-00001-of-00002.safetensors",
                   "model-00002-of-00002.safetensors"]
    if any((out / s).exists() for s in shard_names) and not args.force:
        print(f"{out} already has shards; use --force to regenerate",
              file=sys.stderr)
        return 1

    rng = np.random.default_rng(SEED)
    tensors = build_tensors(rng)
    config = dict(CONFIG)
    if args.vision:
        tensors.update(build_vision(np.random.default_rng(SEED + 1)))
        config.update({"vision_config": VISION, "image_token_id": IMAGE_TOKEN,
                       "vision_start_token_id": VISION_START,
                       "vision_end_token_id": VISION_END,
                       "vision_model_type": "mimovl"})
        (out / "preprocessor_config.json").write_text(
            json.dumps(PREPROCESSOR, indent=2) + "\n")
        from PIL import Image
        Image.fromarray(test_image(np.random.default_rng(SEED + 2))).save(
            out / "image.png")
        # the patches exactly as the gateway builds them (tools/qwen38_image.py,
        # the Qwen2-VL processor); make_mimo_ref.py checks them bit for bit
        # against the official processor
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import qwen38_image
        patches, grid_h, grid_w = qwen38_image.preprocess(
            (out / "image.png").read_bytes(), model_dir=str(out))
        patches.tofile(out / "patches.f32")
        (out / "image.json").write_text(json.dumps(
            {"grid_h": grid_h, "grid_w": grid_w,
             "tokens": grid_h * grid_w // 4}) + "\n")
    # two shards like the release: the experts of layers 3.. in the second,
    # so the loader has to follow the index across files
    second = {n: a for n, a in tensors.items()
              if ".mlp.experts." in n and int(n.split(".")[2]) >= 3}
    first = {n: a for n, a in tensors.items() if n not in second}
    weight_map = {}
    for shard, part in zip(shard_names, (first, second)):
        write_safetensors(out / shard, part)
        weight_map.update({n: shard for n in part})
    total = sum(a.nbytes for a in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map},
        indent=1) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"{out}: {len(tensors)} tensors, {total / 1e6:.2f} MB in 2 shards, "
          f"{LAYERS} layers (pattern {PATTERN}), {EXPERTS} experts top-{TOPK}, "
          f"seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
