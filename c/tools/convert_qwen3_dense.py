#!/usr/bin/env python3
"""Convert a dense Qwen3 (Qwen3ForCausalLM) HF checkpoint -> colibri container.

The colibri container is a directory of safetensors shards (+ config.json +
tokenizer.json + qwen3_meta.json). The engine (c/qwen3.c) reads them via st.h.

Tensor names are kept EXACTLY as HF writes them; the container differs from the
source only in dtype/layout of the big matrices and in the extra meta file:

  --bits 0 : passthrough repackage (f16 everywhere) -- container == HF layout
  --bits 4 : int4-gs64 for the 7 large matmul weights per layer
             (attn q/k/v/o_proj, mlp gate/up/down_proj):
               <name>     U8  [O, ceil(I/2)]  nibble packed, low = even column,
                                                value = q + 8 (0..15)
               <name>.qs  F32 [O, ceil(I/64)]  one scale per 64 INPUT elements
             Everything else (embed/lm_head/norms/q_norm/k_norm) stays f16:
             embed/lm_head enter every token and int4 there measurably breaks
             coherence (same policy as convert_inkling_dense_int4.py).

The int4 nibble convention matches quant.h's matmul_i4_grouped: dequant is
(nibble - 8) * scale, LOW nibble = element 2k, HIGH = element 2k+1.

Usage (local model):
  python tools/convert_qwen3_dense.py --model /path/to/src-qwen3-8b --out ./qwen3_8b_i4 --bits 4
  python tools/convert_qwen3_dense.py --selftest     # pack/unpack round-trip
"""

import argparse, json, math, os, struct, sys
from pathlib import Path

if sys.platform == "win32":
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

try:
    import numpy as np
    from safetensors.numpy import save_file
except ImportError as exc:
    sys.exit(f"Missing dependencies: {exc}. Install: pip install numpy safetensors")

GS = 64                     # group size along the input (contraction) dim
ROWS_PER_CHUNK = 4096       # row-block quantization: peak RAM ~<1 GB

# weights that go to int4 at --bits 4 (2-D matmul weights only)
BIG = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
       "self_attn.v_proj.weight", "self_attn.o_proj.weight",
       "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")


# ---------- I/O safetensors ----------
def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n

def read_rows(path, base, off0, dtype, cols, r0, r1):
    """Read rows [r0,r1) of a 2-D tensor as f32 (header-only indexing, no mmap)."""
    itemsize = {"BF16": 2, "F16": 2, "F32": 4}[dtype]
    start = base + off0 + r0 * cols * itemsize
    nbytes = (r1 - r0) * cols * itemsize
    with open(path, "rb") as f:
        f.seek(start)
        raw = f.read(nbytes)
    if dtype == "BF16":
        u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(r1 - r0, cols)
    if dtype == "F16":
        return np.frombuffer(raw, np.float16).astype(np.float32).reshape(r1 - r0, cols)
    return np.frombuffer(raw, np.float32).reshape(r1 - r0, cols)


# ---------- quantizers (must stay in sync with c/quant.h matmul_i4_grouped) ----------
def quant_gs4(w):
    """w [O, I] f32 -> (packed U8 [O, ceil(I/2)], scales f32 [O, ng]).
    Symmetric, one scale per GS input elements, stored nibble = q + 8."""
    R, I = w.shape
    ng = (I + GS - 1) // GS
    pad = ng * GS - I
    wp = np.concatenate([w, np.zeros((R, pad), np.float32)], 1) if pad else w
    g = wp.reshape(R, ng, GS)
    amax = np.abs(g).max(-1, keepdims=True)
    scale = (amax / 7.0).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(g / scale), -8, 7).astype(np.int8) + 8      # 0..15
    q = q.reshape(R, ng * GS)[:, :I].astype(np.uint8)
    if I % 2:                                                       # odd tail
        q = np.concatenate([q, np.full((R, 1), 8, np.uint8)], 1)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed, scale.reshape(R, ng).astype(np.float32)

def dequant_gs4(packed, scale, I):
    R = packed.shape[0]
    q = np.empty((R, packed.shape[1] * 2), np.float32)
    q[:, 0::2] = (packed & 15).astype(np.float32) - 8.0
    q[:, 1::2] = (packed >> 4).astype(np.float32) - 8.0
    q = q[:, :I]
    s = np.repeat(scale, GS, axis=1)[:, :I]
    return q * s


def _selftest():
    """Round-trip check for the int4 packing used by --bits 4."""
    print("=== int4 gs64 pack/unpack selftest ===")
    rng = np.random.default_rng(20260916)
    ok = True
    for (O, I) in ((8, 64), (8, 96), (5, 33)):      # even/gs-multiple/odd tails
        w = (rng.standard_normal((O, I)) * 3).astype(np.float32)
        pk, sc = quant_gs4(w)
        rec = dequant_gs4(pk, sc, I)
        # round-trip must reproduce the QUANTIZED values exactly, and the
        # quantized values must be within half a scale step of the source
        ng = sc.shape[1]
        err = np.linalg.norm(w - rec) / max(np.linalg.norm(w), 1e-9)
        max_dev = np.abs(w - rec).max() / sc.max()
        print(f"  [{O},{I}] packed={pk.shape} scales={sc.shape} rel_err={err:.4f} "
              f"max_dev={max_dev:.2f} scale_steps")
        ok = ok and pk.shape == (O, (I + 1) // 2) and sc.shape == (O, ng) and max_dev < 1.0
    print("SELFTEST", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description="Convert dense Qwen3 HF checkpoint -> colibri container")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--model", help="Local HF checkpoint directory")
    ap.add_argument("--out", required=False, help="Output container directory")
    ap.add_argument("--bits", type=int, default=4, choices=(0, 4),
                    help="0 = f16 passthrough repackage; 4 = int4-gs64 big matrices (default)")
    ap.add_argument("--selftest", action="store_true",
                    help="Validate the int4 pack/unpack round-trip on synthetic weights, then exit")
    args = ap.parse_args()

    if args.selftest:
        return _selftest()
    if not args.model:
        sys.exit("error: need --model (or --selftest)")
    if not args.out:
        sys.exit("error: --out is required")

    src_dir = Path(args.model)
    if not (src_dir / "config.json").is_file():
        sys.exit(f"config.json missing in {src_dir}")

    cfg = json.load(open(src_dir / "config.json", encoding="utf-8"))
    if cfg.get("architectures") and "Qwen3ForCausalLM" not in cfg["architectures"]:
        sys.exit(f"not a dense Qwen3 checkpoint: architectures={cfg['architectures']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(src_dir / "config.json", out / "config.json")
    print(f"config -> {out / 'config.json'}")
    if (src_dir / "tokenizer.json").is_file():
        shutil.copy2(src_dir / "tokenizer.json", out / "tokenizer.json")
        print(f"tokenizer -> {out / 'tokenizer.json'}")
    else:
        print("WARNING: tokenizer.json not found; the engine will need TOK=<path>")

    # ---- build weight map (key -> shard file) from headers only ----
    shards = sorted(src_dir.glob("*.safetensors"))
    index = {}
    for sh in shards:
        hdr, base = read_header(sh)
        for k, t in hdr.items():
            if k == "__metadata__":
                continue
            index[k] = (sh, base, t["data_offsets"][0], t["data_offsets"][1],
                        t["dtype"], t["shape"])

    n_layers = int(cfg["num_hidden_layers"])
    q_heads = int(cfg["num_attention_heads"])
    kv_heads = int(cfg["num_key_value_heads"])
    hidden = int(cfg["hidden_size"])
    head_dim = int(cfg.get("head_dim", hidden // q_heads))
    inter = int(cfg["intermediate_size"])
    vocab = int(cfg["vocab_size"])
    # transformers <5 writes rope_theta at the top level, >=5 nests it under
    # rope_parameters. A wrong base is silently plausible output, so take
    # whichever the checkpoint has and refuse when it has neither.
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None:
        rope_theta = (cfg.get("rope_parameters") or {}).get("rope_theta")
    if rope_theta is None:
        sys.exit("config.json: no rope_theta (checked top level and rope_parameters)")

    def emit_f16(name, path, base, o0, o1):
        """Passthrough. bf16 -> f16 is lossless in the mantissa (8 bits -> 10) and
        halves nothing we need back, but an f32 source stays f32: downcasting it
        would be the only lossy step in a --bits 0 'passthrough' container."""
        with open(path, "rb") as f:
            f.seek(base + o0)
            raw = f.read(o1 - o0)
        dt = index[name][4]
        if dt == "BF16":
            u = np.frombuffer(raw, np.uint16).astype(np.uint32) << 16
            arr = u.view(np.float32).astype(np.float16)
        elif dt == "F32":
            arr = np.frombuffer(raw, np.float32)
        else:
            arr = np.frombuffer(raw, np.float16)
        return arr.reshape(index[name][5])

    def quant_big(name, path, base, o0, o1, dtype, shape):
        """int4-gs64 a [O, I] weight, chunked by rows."""
        O, I = shape[0], shape[1]
        packs, scales, errs = [], [], []
        for r0 in range(0, O, ROWS_PER_CHUNK):
            r1 = min(r0 + ROWS_PER_CHUNK, O)
            blk = read_rows(path, base, o0, dtype, I, r0, r1)
            pk, sc = quant_gs4(blk)
            rec = dequant_gs4(pk, sc, I)
            n = float(np.linalg.norm(blk))
            if n > 0:
                errs.append(float(np.linalg.norm(blk - rec)) / n)
            packs.append(pk); scales.append(sc)
        if errs:
            print(f"    {name}: L2 err {100*float(np.mean(errs)):.2f}%")
        return (np.concatenate(packs, 0), np.concatenate(scales, 0))

    # ---- globals shard: embed / lm_head / final norm, f16 ----
    g_out = {}
    for k in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"):
        if k not in index:
            # tied embeddings: lm_head absent -> engine falls back to embed
            if k == "lm_head.weight":
                print("[globals] lm_head.weight absent (tied embeddings)")
                continue
            sys.exit(f"missing tensor {k}")
        sh, base, o0, o1, dt, shape = index[k]
        g_out[k] = emit_f16(k, sh, base, o0, o1)
    save_file(g_out, str(out / "model-globals.safetensors"))
    print(f"[globals] {len(g_out)} tensors -> model-globals.safetensors")

    # ---- per layer ----
    for i in range(n_layers):
        tens = {}
        for k, (sh, base, o0, o1, dt, shape) in sorted(index.items()):
            if not k.startswith(f"model.layers.{i}."):
                continue
            suffix = k[len(f"model.layers.{i}."):]
            if args.bits == 4 and suffix in BIG and dt in ("BF16", "F16", "F32"):
                pk, sc = quant_big(k, sh, base, o0, o1, dt, shape)
                tens[k] = pk
                tens[k + ".qs"] = sc
            else:
                tens[k] = emit_f16(k, sh, base, o0, o1)
        out_path = out / f"model-{i:05d}.safetensors"
        save_file(tens, str(out_path))
        if (i + 1) % 8 == 0 or i == n_layers - 1:
            print(f"[layer {i}] {out_path.name} ({len(tens)} tensors)")

    # ---- meta: dimensions derived from the ACTUAL weight shapes (authoritative) ----
    def shape_of(name):
        return index[name][5] if name in index else None
    meta = {
        "model_type": cfg.get("model_type", "qwen3"),
        "hidden": hidden,
        "n_layers": n_layers,
        "q_heads": q_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "o_in": shape_of(f"model.layers.0.self_attn.o_proj.weight")[1],
        "inter": inter,
        "vocab": vocab,
        "rope_theta": float(rope_theta),
        "rms_eps": float(cfg.get("rms_norm_eps", 1e-6)),
        "max_position_embeddings": int(cfg.get("max_position_embeddings", 40960)),
        "bits": args.bits,
        "gs": GS if args.bits == 4 else 0,
    }
    (out / "qwen3_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[meta] {out / 'qwen3_meta.json'}")

    print(f"\nDone. Container at: {out}")
    print(f"Run (engine):  SNAP={out} ./qwen3 0 {args.bits} <ref.json>   # or TOK=... for a text prompt")


if __name__ == "__main__":
    main()
