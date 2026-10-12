#!/usr/bin/env python3
"""Generate the tiny Kolibri-1 checkpoints used by the kolibri1 engine tests.

numpy-only and deterministic: the weights come from SEED and are byte-identical
on every platform, and do not depend on the layout or on the layer-type list
(the same tensors are only placed differently).  No oracle output is produced
here; reference tokens belong to a separate maintainer-run tool.

The fixture stores every tensor the way the released checkpoint does, because
the storage is where a reading goes wrong while a float reference still agrees:

  - the 31 name patterns of the release, with its dtypes: embed, lm_head, the
    norms, the router gate and expert_bias in BF16; q/k/v/o, the shared expert
    and every routed expert in F8_E4M3 with a F32 weight_scale_inv per 128x128
    block, grid [ceil(out/128), ceil(in/128)];
  - q is heads*head_dim wide (1024 here), not hidden (256): attention is not
    square, and so the q and o scale grids are not square either ([8,2], [2,8]);
  - the MoE width is 192, so the expert blocks end in a PARTIAL 128-block;
  - separate q_proj / k_proj / v_proj and per-expert gate / up / down tensors;
  - the tensor named post_attention_layernorm is the pre-FFN norm;
    there is no shared_expert_gate tensor.

Layouts (--layout):

  simple    two shards, tensors in construction order (an expert's down, gate and
            up back to back, each weight followed by its scale), the way a
            converter that knows nothing about the release would write them.
  release   four shards that reproduce the quirks measured on the real release:
            inside a shard the F32 scales come first, then the BF16
            tensors, then the F8 tensors, each group sorted by NAME (so experts appear in the order 0, 1, 2, ...,
            exactly as the real shards list 0, 1, 10, 100, ...); and about 60%
            of the experts keep down|gate|up together while 20% have their gate
            in the previous shard and 20% their up in the next one (scales go
            with their weights).  The mix is exact and seeded, not random per
            run.  The real placement rule is not known;
            this reproduces its classes.

--layer-types release|alt: `release` is the checkpoint's pattern (a layer is
full attention iff l % 5 == 4: {s,s,s,s,F,s} for 6 layers); `alt` is
{s,s,F,s,s,F}, valid for the generic loader but not for --validate --release.

Real geometry reduced: hidden 256, 8 query / 2 KV heads (group 4), head_dim 128,
moe and shared intermediate 192, 8 experts top 4 (--topk), 6 layers, window 17,
vocab 320.
"""

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

SEED = 20261006

VOCAB = 320
HIDDEN = 256
LAYERS = 6
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 128
EXPERTS = 8
TOPK_DEFAULT = 4
MOE_INTER = 192
SHARED_INTER = 192
WINDOW = 17
EOS_ID = 1
PAD_ID = 0
BLOCK = 128
EMBED_S = 0.05
LM_S = 4.0

RELEASE_SHARDS = 4

LAYER_TYPES = {
    "release": ["sliding_attention"] * 4 + ["full_attention", "sliding_attention"],
    "alt": ["sliding_attention", "sliding_attention", "full_attention",
            "sliding_attention", "sliding_attention", "full_attention"],
}


def make_config(layer_types, topk):
    return {
        "architectures": ["Kolibri1ForCausalLM"],
        "model_type": "kolibri1",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-06,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "num_experts": EXPERTS,
        "num_experts_per_tok": topk,
        "moe_intermediate_size": MOE_INTER,
        "shared_expert_intermediate_size": SHARED_INTER,
        "norm_topk_prob": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
        "sliding_window": WINDOW,
        "layer_types": list(layer_types),
        "bos_token_id": None,
        "eos_token_id": EOS_ID,
        "pad_token_id": PAD_ID,
        "dtype": "bfloat16",
        "head_dtype": "float32",
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [BLOCK, BLOCK],
            "modules_to_not_convert": [f"model.layers.{i}.mlp.gate"
                                       for i in range(LAYERS)],
        },
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
    return {np.dtype("float32"): "F32"}[arr.dtype]


def dtype_group(arr):
    """Order of the dtype groups inside a release shard: F32, BF16, F8."""
    return {"F32": 0, "BF16": 1, "F8_E4M3": 2}[st_dtype(arr)]


def build_tensors(rng):
    """Every weight of the tiny model, in construction order, {name: array}."""
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
        """Distinct, non-power-of-two F32 scales per block."""
        return rng.uniform(0.004, 0.012, size=(rows, cols)).astype(np.float32)

    def fp8_matrix(name, out, inp):
        put(name + ".weight", tag(fp8_codes(out, inp), F8))
        put(name + ".weight_scale_inv",
            fp8_scales(-(-out // BLOCK), -(-inp // BLOCK)))

    embed = normal(VOCAB, HIDDEN, s=EMBED_S)
    put("model.embed_tokens.weight", bf16(embed))
    lm_head = normal(VOCAB, HIDDEN, s=LM_S)
    lm_head[EOS_ID] = 0.0
    put("model.norm.weight", norm_weight(HIDDEN))
    put("lm_head.weight", bf16(lm_head))

    q_width = HEADS * HEAD_DIM
    kv_width = KV_HEADS * HEAD_DIM
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        put(p + "input_layernorm.weight", norm_weight(HIDDEN))
        put(p + "post_attn_norm.weight", norm_weight(HIDDEN))
        put(p + "post_attention_layernorm.weight", norm_weight(HIDDEN))
        put(p + "post_ffn_norm.weight", norm_weight(HIDDEN))
        put(p + "self_attn.q_norm.weight", norm_weight(HEAD_DIM))
        put(p + "self_attn.k_norm.weight", norm_weight(HEAD_DIM))
        fp8_matrix(p + "self_attn.q_proj", q_width, HIDDEN)
        fp8_matrix(p + "self_attn.k_proj", kv_width, HIDDEN)
        fp8_matrix(p + "self_attn.v_proj", kv_width, HIDDEN)
        fp8_matrix(p + "self_attn.o_proj", HIDDEN, q_width)
        # closed-form router: wide, stable expert preferences per hidden
        # direction, so top-k selection cannot ride a near-tie
        gate = 0.08 * np.sin(np.arange(EXPERTS * HIDDEN, dtype=np.float32)
                             * 0.013 + 0.7 * i).reshape(EXPERTS, HIDDEN)
        put(p + "mlp.gate.weight", bf16(gate))
        put(p + "moe.router.expert_bias",
            bf16(np.linspace(-0.02, 0.02, EXPERTS, dtype=np.float32)))
        fp8_matrix(p + "mlp.shared_experts.gate_proj", SHARED_INTER, HIDDEN)
        fp8_matrix(p + "mlp.shared_experts.up_proj", SHARED_INTER, HIDDEN)
        fp8_matrix(p + "mlp.shared_experts.down_proj", HIDDEN, SHARED_INTER)
        for e in range(EXPERTS):
            q = p + f"mlp.experts.{e}."
            fp8_matrix(q + "down_proj", HIDDEN, MOE_INTER)
            fp8_matrix(q + "gate_proj", MOE_INTER, HIDDEN)
            fp8_matrix(q + "up_proj", MOE_INTER, HIDDEN)
    return tensors


def write_safetensors(path, tensors):
    """Hand-rolled writer: keeps the insertion order (the file order the
    engine's placement table has to cope with; the safetensors library
    reorders)."""
    header, offset = {"__metadata__": {"format": "pt"}}, 0
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


def tensor_layer(name):
    parts = name.split(".")
    return int(parts[2]) if parts[:2] == ["model", "layers"] else None


def expert_key(name):
    """(layer, expert) when `name` belongs to a routed expert, else None."""
    parts = name.split(".")
    if parts[:2] == ["model", "layers"] and parts[3:5] == ["mlp", "experts"]:
        return int(parts[2]), int(parts[5])
    return None


def release_placement(tensors):
    """Shard index per tensor name for --layout release, and the class of every
    expert (C = down|gate|up in one shard, G = gate in the previous shard,
    U = up in the next one)."""
    names = sorted(tensors)
    total = sum(tensors[n].nbytes for n in names)
    home, acc = {}, 0
    for n in names:
        home[n] = min(RELEASE_SHARDS - 1, acc * RELEASE_SHARDS // total)
        acc += tensors[n].nbytes
    experts = sorted({expert_key(n) for n in names if expert_key(n)})
    # an expert lives with its down_proj weight; G and U move one projection
    ehome = {k: home[f"model.layers.{k[0]}.mlp.experts.{k[1]}.down_proj.weight"]
             for k in experts}
    crng = np.random.default_rng(SEED + 7)
    n_g = round(0.2 * len(experts))
    n_u = round(0.2 * len(experts))
    g_ok = [k for k in experts if ehome[k] >= 1]
    g = set(map(tuple, (g_ok[i] for i in crng.permutation(len(g_ok))[:n_g])))
    u_ok = [k for k in experts if ehome[k] <= RELEASE_SHARDS - 2 and k not in g]
    u = set(map(tuple, (u_ok[i] for i in crng.permutation(len(u_ok))[:n_u])))
    cls = {k: "G" if k in g else "U" if k in u else "C" for k in experts}
    shard = dict(home)
    for k in experts:
        base = f"model.layers.{k[0]}.mlp.experts.{k[1]}."
        for proj in ("down_proj", "gate_proj", "up_proj"):
            for suffix in (".weight", ".weight_scale_inv"):
                shard[base + proj + suffix] = ehome[k]
        if cls[k] == "G":
            for suffix in (".weight", ".weight_scale_inv"):
                shard[base + "gate_proj" + suffix] = ehome[k] - 1
        elif cls[k] == "U":
            for suffix in (".weight", ".weight_scale_inv"):
                shard[base + "up_proj" + suffix] = ehome[k] + 1
    return shard, cls


def layout_simple(tensors):
    """Two shards, experts of layers 3.. in the second, construction order."""
    names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    second = {n: a for n, a in tensors.items()
              if expert_key(n) and expert_key(n)[0] >= 3}
    first = {n: a for n, a in tensors.items() if n not in second}
    return list(zip(names, (first, second))), {}


def layout_release(tensors):
    shard, cls = release_placement(tensors)
    parts = [dict() for _ in range(RELEASE_SHARDS)]
    for n in sorted(tensors, key=lambda n: (dtype_group(tensors[n]), n)):
        parts[shard[n]][n] = tensors[n]
    names = [f"model-{i + 1:05d}-of-{RELEASE_SHARDS:05d}.safetensors"
             for i in range(RELEASE_SHARDS)]
    return list(zip(names, parts)), cls


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", "--output", dest="out", default="./kolibri1_tiny")
    parser.add_argument("--layout", choices=("simple", "release"), default="simple")
    parser.add_argument("--layer-types", choices=sorted(LAYER_TYPES), default="release")
    parser.add_argument("--topk", type=int, default=TOPK_DEFAULT)
    parser.add_argument("--force", action="store_true", help="overwrite existing shards")
    args = parser.parse_args()
    if not 1 <= args.topk <= EXPERTS:
        parser.error(f"--topk must be in 1..{EXPERTS}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("*.safetensors"))
    if existing and not args.force:
        print(f"{out} already has shards; use --force to regenerate", file=sys.stderr)
        return 1
    for old in existing + list(out.glob("model.safetensors.index.json")):
        old.unlink()

    tensors = build_tensors(np.random.default_rng(SEED))
    shards, cls = (layout_release if args.layout == "release" else layout_simple)(tensors)
    weight_map = {}
    for name, part in shards:
        write_safetensors(out / name, part)
        weight_map.update({n: name for n in part})
    total = sum(a.nbytes for a in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total},
         "weight_map": {n: weight_map[n] for n in sorted(weight_map)}},
        indent=1) + "\n")
    config = make_config(LAYER_TYPES[args.layer_types], args.topk)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    counts = {c: sum(1 for v in cls.values() if v == c) for c in "CGU"} if cls else {}
    manifest = {
        "seed": SEED, "layout": args.layout, "layer_types": args.layer_types,
        "tensors": len(tensors), "total_bytes": total,
        "expert_classes": counts,
        "shard_sha256": {name: sha256_file(out / name) for name, _ in shards},
    }
    (out / "fixture.json").write_text(json.dumps(manifest, indent=1) + "\n")
    mix = (f", experts C/G/U = {counts['C']}/{counts['G']}/{counts['U']}" if counts else "")
    print(f"{out}: {len(tensors)} tensors, {total / 1e6:.2f} MB in {len(shards)} shards "
          f"({args.layout}), {LAYERS} layers ({args.layer_types}), {EXPERTS} experts "
          f"top-{args.topk}{mix}, seed {SEED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
