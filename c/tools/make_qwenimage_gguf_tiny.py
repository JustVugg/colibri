#!/usr/bin/env python3
"""A tiny Qwen-Image-2.1 in GGUF form, for the engine's GGUF path (numpy only, no torch).

    make_qwenimage_gguf_tiny.py TINY OUTDIR [--turbo]

TINY is a fixture written by make_qwenimage_tiny.py: its tokenizer (processor/) and its VAE
(vae/, z_dim 16) are reused. Written to OUTDIR:

  dit.gguf   a random denoiser as unsloth ships Qwen-Image-2.1's: no metadata at all, every
             name under "model.diffusion_model.", each block's gate_layer and proj fused in
             one img_mlp.gate_up, the types of a K-quant file (Q2_K to Q6_K, Q8_0, F16, BF16,
             F32). 256 wide, so every block type fills whole 256-wide blocks.
  dit_io16.gguf another one whose txt_in and modulation are BF16 (int8 rows at load, kept or
             not): there only the blocks' matrices differ between the file's blocks and int8
             rows, and the int8 x int8 product must give the same bytes both ways.
  te.gguf    a random text encoder as llama.cpp writes Qwen3-VL's language model (arch
             qwen3vl, blk.N names, the tiny's tokenizer in the metadata), K-quant types.
  ref/       the same weights decoded to f32 (tools/gguf_dequant.py) in the diffusers layout
             of the checkpoint: the pipeline the GGUFs must reproduce.

tests/test_qwenimage_gguf_tiny.py imports dit.gguf with tools/gguf_import.py and runs both.
The sizes are the real model's ratios at toy scale: head dim 128 (RoPE axes 16/56/56),
mlp_ratio 3, the text encoder's width the denoiser's context width.
"""
import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_dequant import dequantize  # noqa: E402

DIM, HEADS, HD, LAYERS, MLP_RATIO = 256, 2, 128, 2, 3
TE_H, TE_LAYERS, TE_HEADS, TE_KV, TE_INTER = 256, 2, 2, 1, 512
IN_CH = 16   # the tiny VAE's z_dim
PREFIX = "model.diffusion_model."
# ggml type ids and block (elements, bytes)
T = {"F32": 0, "F16": 1, "Q8_0": 8, "Q2_K": 10, "Q3_K": 11, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "BF16": 30}
BLK = {"Q8_0": (32, 34), "Q2_K": (256, 84), "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210)}
# where each block type keeps its f16 scale(s): the bytes set so the weights have a sane spread
F16_AT = {"Q8_0": (0,), "Q2_K": (80, 82), "Q3_K": (108,), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208,)}
TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]


def bf16_round(x):
    u = x.astype(np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def tensor(rng, shape, kind, std=0.02):
    """(raw bytes, ggml type, the f32 values the bytes decode to)."""
    shape = tuple(shape)
    n = int(np.prod(shape))
    if kind == "F32":
        v = (rng.standard_normal(shape) * std).astype(np.float32)
        return v.tobytes(), T[kind], v
    if kind == "F16":
        h = (rng.standard_normal(shape) * std).astype(np.float16)
        return h.tobytes(), T[kind], h.astype(np.float32)
    if kind == "BF16":
        b = bf16_round(rng.standard_normal(shape) * std)
        return b.tobytes(), T[kind], (b.astype(np.uint32) << 16).view(np.float32).reshape(shape)
    blck, size = BLK[kind]
    assert shape[-1] % blck == 0, (shape, kind)
    nb = n // blck
    raw = rng.integers(0, 256, size=(nb, size), dtype=np.uint8)
    for off in F16_AT[kind]:   # positive scales of 1/64 first, then measured and set to std
        raw[:, off:off + 2] = np.frombuffer(np.full(nb, 1 / 64, np.float16).tobytes(), np.uint8).reshape(nb, 2)
    s = float(np.std(dequantize(raw.tobytes(), T[kind], n))) or 1.0
    for off in F16_AT[kind]:
        d = (rng.uniform(0.5, 1.5, nb) * (std / s) / 64).astype(np.float16)
        raw[:, off:off + 2] = np.frombuffer(d.tobytes(), np.uint8).reshape(nb, 2)
    v = dequantize(raw.tobytes(), T[kind], n).astype(np.float32).reshape(shape)
    return raw.tobytes(), T[kind], v


def write_gguf(path, kv, tensors):
    """GGUF v3: kv = [(key, type, value)], tensors = [(name, shape rows-first, ggml type, raw)]."""
    def s(x):
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def val(t, v):
        if t == 8:
            return s(v)
        if t == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(val(et, i) for i in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f", 7: "<?"}[t], v)
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))
    for k, t, v in kv:
        out += s(k) + struct.pack("<I", t) + val(t, v)
    off = 0
    offs = []
    for name, shape, ty, raw in tensors:
        dims = list(reversed(shape))
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", ty, off)
        offs.append(off)
        off = (off + len(raw) + 31) // 32 * 32
    out += b"\0" * ((-len(out)) % 32)
    with open(path, "wb") as f:
        f.write(out)
        for (name, shape, ty, raw), o in zip(tensors, offs):
            f.write(raw)
            f.write(b"\0" * ((-len(raw)) % 32))


def write_safetensors(path, named):
    header, blobs, off = {}, [], 0
    for name, v in named.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header).encode()
    h += b" " * ((-len(h)) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def denoiser(rng, io16=False):
    """(GGUF tensors, f32 checkpoint tensors). io16: txt_in and modulation as BF16."""
    D, M = DIM, DIM * MLP_RATIO
    gg, ref = [], {}

    def add(name, shape, kind, std=0.02, ones=False):
        if ones:
            v = (1 + rng.standard_normal(shape) * 0.05).astype(np.float32)
            raw, ty = (v.tobytes(), T["F32"]) if kind == "F32" else (bf16_round(v).tobytes(), T["BF16"])
            v = v if kind == "F32" else (bf16_round(v).astype(np.uint32) << 16).view(np.float32)
        else:
            raw, ty, v = tensor(rng, shape, kind, std)
        gg.append((PREFIX + name, list(shape), ty, raw))
        return v
    ref["img_in.weight"] = add("img_in.weight", (D, IN_CH), "BF16", 0.2)
    ref["time_text_embed.timestep_embedder.linear_1.weight"] = add("time_text_embed.timestep_embedder.linear_1.weight", (D, 256), "Q8_0", 0.05)
    ref["time_text_embed.timestep_embedder.linear_2.weight"] = add("time_text_embed.timestep_embedder.linear_2.weight", (D, D), "Q8_0", 0.05)
    ref["modulation.1.weight"] = add("modulation.1.weight", (4 * D, D), "BF16" if io16 else "Q5_K", 0.05)
    ref["norm_out.linear.weight"] = add("norm_out.linear.weight", (D, D), "F16", 0.05)
    ref["proj_out.weight"] = add("proj_out.weight", (IN_CH, D), "Q8_0", 0.05)
    ref["txt_in.in_layer.weight"] = add("txt_in.in_layer.weight", (D, TE_H), "BF16" if io16 else "Q6_K", 0.05)
    ref["txt_in.out_layer.weight"] = add("txt_in.out_layer.weight", (D, D), "BF16" if io16 else "Q8_0", 0.05)
    ref["txt_in.text_norm.weight"] = add("txt_in.text_norm.weight", (TE_H,), "BF16", ones=True)
    kinds = [("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q4_K"), ("Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q2_K", "Q5_K")]
    for l in range(LAYERS):
        b = f"transformer_blocks.{l}."
        kq, kk, kv, ko, kgu, kout = kinds[l % 2]
        ref[b + "attn.to_q.weight"] = add(b + "attn.to_q.weight", (D, D), kq)
        ref[b + "attn.to_k.weight"] = add(b + "attn.to_k.weight", (D, D), kk)
        ref[b + "attn.to_v.weight"] = add(b + "attn.to_v.weight", (D, D), kv)
        ref[b + "attn.to_out.0.weight"] = add(b + "attn.to_out.0.weight", (D, D), ko)
        gu = add(b + "img_mlp.gate_up.weight", (2 * M, D), kgu)
        ref[b + "img_mlp.gate_layer.weight"], ref[b + "img_mlp.proj.weight"] = gu[:M], gu[M:]
        ref[b + "img_mlp.out.weight"] = add(b + "img_mlp.out.weight", (D, M), kout)
        ref[b + "attn.norm_q.weight"] = add(b + "attn.norm_q.weight", (HD,), "F32", ones=True)
        ref[b + "attn.norm_k.weight"] = add(b + "attn.norm_k.weight", (HD,), "F32", ones=True)
    return gg, ref


def text_encoder(rng, vocab):
    H, I, qd, kd = TE_H, TE_INTER, TE_HEADS * HD, TE_KV * HD
    gg, ref = [], {}
    p = "model.language_model."

    def add(gname, hname, shape, kind, std=0.02, ones=False):
        if ones:
            v = (1 + rng.standard_normal(shape) * 0.05).astype(np.float32)
            raw, ty = v.tobytes(), T["F32"]
        else:
            raw, ty, v = tensor(rng, shape, kind, std)
        gg.append((gname, list(shape), ty, raw))
        if hname:
            ref[hname] = v
    add("token_embd.weight", p + "embed_tokens.weight", (vocab, H), "Q4_K", 0.5)
    add("output_norm.weight", p + "norm.weight", (H,), "F32", ones=True)
    add("output.weight", None, (vocab, H), "Q6_K")
    kinds = [("Q4_K", "Q5_K", "Q6_K", "Q3_K", "Q2_K", "Q4_K", "Q5_K"), ("Q5_K", "Q6_K", "Q8_0", "Q2_K", "Q3_K", "Q6_K", "Q4_K")]
    for l in range(TE_LAYERS):
        g, h = f"blk.{l}.", f"{p}layers.{l}."
        a, b, c, d, e, f, gd = kinds[l % 2]
        add(g + "attn_norm.weight", h + "input_layernorm.weight", (H,), "F32", ones=True)
        add(g + "ffn_norm.weight", h + "post_attention_layernorm.weight", (H,), "F32", ones=True)
        add(g + "attn_q.weight", h + "self_attn.q_proj.weight", (qd, H), a)
        add(g + "attn_k.weight", h + "self_attn.k_proj.weight", (kd, H), b)
        add(g + "attn_v.weight", h + "self_attn.v_proj.weight", (kd, H), c)
        add(g + "attn_output.weight", h + "self_attn.o_proj.weight", (H, qd), d)
        add(g + "attn_q_norm.weight", h + "self_attn.q_norm.weight", (HD,), "F32", ones=True)
        add(g + "attn_k_norm.weight", h + "self_attn.k_norm.weight", (HD,), "F32", ones=True)
        add(g + "ffn_gate.weight", h + "mlp.gate_proj.weight", (I, H), e)
        add(g + "ffn_up.weight", h + "mlp.up_proj.weight", (I, H), f)
        add(g + "ffn_down.weight", h + "mlp.down_proj.weight", (H, I), gd)
    return gg, ref


def tokenizer_kv(tok, vocab):
    """The tiny's tokenizer.json as llama.cpp's GGUF metadata: the tokens by id (the ids past the
    last one padded as [PADn]), CONTROL (3) for the special ones, the merges."""
    toks = [f"[PAD{i}]" for i in range(vocab)]
    types = [1] * vocab
    for t, i in tok["model"]["vocab"].items():
        toks[i] = t
    for a in tok["added_tokens"]:
        toks[a["id"]] = a["content"]
        types[a["id"]] = 3 if a.get("special") else 4
    merges = [" ".join(m) if isinstance(m, list) else m for m in tok["model"]["merges"]]
    ids = {a["content"]: a["id"] for a in tok["added_tokens"]}
    return [("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "qwen2"),
            ("tokenizer.ggml.tokens", 9, (8, toks)), ("tokenizer.ggml.token_type", 9, (5, types)),
            ("tokenizer.ggml.merges", 9, (8, merges)),
            ("tokenizer.ggml.eos_token_id", 4, ids["<|im_end|>"]), ("tokenizer.ggml.padding_token_id", 4, ids["<|endoftext|>"]),
            ("tokenizer.ggml.bos_token_id", 4, ids["<|endoftext|>"]), ("tokenizer.ggml.add_bos_token", 7, False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("tiny", type=Path)
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--turbo", action="store_true", help="the reference folder takes Turbo's scheduler and sigmas")
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()
    tiny, out = args.tiny, args.outdir
    tok = json.loads((tiny / "processor" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = int(json.loads((tiny / "text_encoder" / "config.json").read_text())["text_config"]["vocab_size"])
    rng = np.random.default_rng(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    gg, dit_ref = denoiser(rng)
    write_gguf(out / "dit.gguf", [], gg)
    write_gguf(out / "dit_io16.gguf", [], denoiser(np.random.default_rng(args.seed + 1), io16=True)[0])
    tg, te_ref = text_encoder(rng, vocab)
    p = "qwen3vl."
    kv = [("general.architecture", 8, "qwen3vl"), ("general.name", 8, "tiny Qwen3-VL"),
          (p + "block_count", 4, TE_LAYERS), (p + "context_length", 4, 262144), (p + "embedding_length", 4, TE_H),
          (p + "feed_forward_length", 4, TE_INTER), (p + "attention.head_count", 4, TE_HEADS),
          (p + "attention.head_count_kv", 4, TE_KV), (p + "rope.freq_base", 6, 5000000.0),
          (p + "attention.layer_norm_rms_epsilon", 6, 1e-6), (p + "attention.key_length", 4, HD),
          (p + "attention.value_length", 4, HD)] + tokenizer_kv(tok, vocab)
    write_gguf(out / "te.gguf", kv, tg)
    # the f32 checkpoint the two GGUFs decode to
    ref = out / "ref"
    if ref.exists():
        shutil.rmtree(ref)
    for sub in ("transformer", "text_encoder", "scheduler"):
        (ref / sub).mkdir(parents=True)
    write_safetensors(ref / "transformer" / "diffusion_pytorch_model.safetensors", dit_ref)
    write_safetensors(ref / "text_encoder" / "model.safetensors", te_ref)
    (ref / "transformer" / "config.json").write_text(json.dumps({
        "_class_name": "QwenImage21Transformer2DModel", "attention_head_dim": HD, "axes_dims_rope": [16, 56, 56],
        "causal_condition": True, "context_in_dim": TE_H, "eps": 1e-6, "in_channels": IN_CH, "mlp_ratio": MLP_RATIO,
        "num_attention_heads": HEADS, "num_layers": LAYERS, "out_channels": IN_CH, "patch_size": 1}, indent=1))
    (ref / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "text_config": {
        "hidden_size": TE_H, "num_hidden_layers": TE_LAYERS, "num_attention_heads": TE_HEADS,
        "num_key_value_heads": TE_KV, "head_dim": HD, "intermediate_size": TE_INTER, "vocab_size": vocab,
        "rms_norm_eps": 1e-6, "rope_theta": 5000000}}, indent=1))
    sched = json.loads((tiny / "scheduler" / "scheduler_config.json").read_text())
    index = json.loads((tiny / "model_index.json").read_text())
    if args.turbo:
        sched.update(use_dynamic_shifting=False, shift_terminal=None)
        index["sample_sigmas"] = TURBO_SIGMAS
    else:
        index.pop("sample_sigmas", None)
    (ref / "scheduler" / "scheduler_config.json").write_text(json.dumps(sched, indent=1))
    (ref / "model_index.json").write_text(json.dumps(index, indent=1))
    shutil.copytree(tiny / "processor", ref / "processor")
    shutil.copytree(tiny / "vae", ref / "vae")
    print(f"wrote {out}: dit.gguf ({(out / 'dit.gguf').stat().st_size} bytes), te.gguf "
          f"({(out / 'te.gguf').stat().st_size} bytes), ref/ (f32)")


if __name__ == "__main__":
    main()
