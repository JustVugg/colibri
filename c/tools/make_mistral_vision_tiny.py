#!/usr/bin/env python3
"""A tiny Mistral-Large-4 vision tower (Pixtral) as an mmproj GGUF, in the published file's layout,
and the rows tools/mistral_vision_ref.py gives for random patches: mistral_vision.h is checked
against them (make mistral-vision-check).

The layout is unsloth's mmproj-*.gguf: architecture clip, projector pixtral, the big matrices BF16,
norms, the patch kernel and the merger F32, and q and k of every head in interleaved rotary order
(a GGUF converter's permutation), which the engine rotates by pairs and the reference undoes.

--release DIR also writes the tower as Mistral's release has it (consolidated-vision.safetensors in
its names, BF16, the same q and k order; the vision_config added to DIR's config.json, a
processor_config.json): what tools/ml4_mmproj.py and the FP8 converter read. The F32 tensors are
BF16 values, as the published file's are, so the file they write is this one tensor for tensor,
which --same-as MMPROJ checks.

  python3 tools/make_mistral_vision_tiny.py --out DIR [--release DIR] [--check BINARY]
  python3 tools/make_mistral_vision_tiny.py --out DIR --same-as MMPROJ
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_flint_tiny import _gguf_bytes   # noqa: E402

U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9
GG_F32, GG_BF16 = 0, 30


def bf16(a):
    """f32 -> bf16 bits, round to nearest even (what a converter writes)"""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_values(a):
    """f32 rounded to the nearest bf16 value, kept as f32 (the release's norms widened)"""
    return (bf16(a).astype(np.uint32) << 16).view(np.float32).reshape(np.shape(a))


# the mmproj's names -> the release's (tools/ml4_mmproj.py reads them back)
RELEASE = {"v.patch_embd.weight": "vision_encoder.patch_conv.weight", "v.pre_ln.weight": "vision_encoder.ln_pre.weight",
           "mm.input_norm.weight": "pre_mm_projector_norm.weight",
           "mm.patch_merger.weight": "patch_merger.merging_layer.weight",
           "mm.1.weight": "vision_language_adapter.w_in.weight", "mm.2.weight": "vision_language_adapter.w_out.weight"}
RELEASE_LAYER = {"attn_q": "attention.wq", "attn_k": "attention.wk", "attn_v": "attention.wv", "attn_out": "attention.wo",
                 "ln1": "attention_norm", "ln2": "ffn_norm", "ffn_gate": "feed_forward.w1", "ffn_up": "feed_forward.w3",
                 "ffn_down": "feed_forward.w2"}


def release_name(name):
    if name in RELEASE:
        return RELEASE[name]
    if name.startswith("v.blk."):
        _v, _blk, l, kind, _w = name.split(".")
        return f"vision_encoder.transformer.layers.{l}.{RELEASE_LAYER[kind]}.weight"
    return None   # v.token_embd.img_break: the text model's [IMG_BREAK] row in the release


def write_release(path, tensors, a):
    """the tower in the release's names and dtype, beside (or as) a checkpoint"""
    sys.path.insert(0, str(HERE))
    import ml4_mmproj
    path.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, kind, dims, payload in tensors:
        rel = release_name(name)
        if rel is None:
            continue
        shape = list(reversed(dims))
        data = payload if kind == GG_BF16 else bf16(np.frombuffer(payload, dtype=np.float32)).tobytes()
        out[rel] = ("BF16", shape, data)
    ml4_mmproj.st_write(path / "consolidated-vision.safetensors", out)
    cfg_path = path / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"model_type": "mistral_large4"}
    cfg["vision_config"] = {"model_type": "pixtral", "hidden_size": a.hidden, "intermediate_size": a.inter,
                            "num_hidden_layers": a.depth, "num_attention_heads": a.heads, "head_dim": a.hidden // a.heads,
                            "patch_size": a.patch, "image_size": 1540, "num_channels": 3, "hidden_act": "silu",
                            "rope_parameters": {"rope_theta": 10000.0, "rope_type": "axial"}}
    cfg["spatial_merge_size"] = a.merge
    cfg_path.write_text(json.dumps(cfg, indent=1))
    (path / "processor_config.json").write_text(json.dumps({
        "image_processor": {"image_mean": [0.48145467, 0.4578275, 0.40821072], "image_std": [0.26862954, 0.26130258, 0.27577711],
                            "patch_size": a.patch, "size": {"longest_edge": 1540}},
        "patch_size": a.patch, "spatial_merge_size": a.merge}, indent=1))


def same_as(ours, theirs):
    """every tensor of theirs is ours: name, type, dims and bytes; ours has only img_break more"""
    sys.path.insert(0, str(HERE))
    from gguf_reader import GGUFReader
    with GGUFReader(str(ours)) as a, GGUFReader(str(theirs)) as b:
        mine = {t["name"]: t for t in a.tensors}
        names = [t["name"] for t in b.tensors]
        extra = sorted(set(mine) - set(names) - {"v.token_embd.img_break"})
        if extra:
            sys.exit(f"{theirs}: {len(extra)} tensor(s) missing, {extra[0]} first")
        for t in b.tensors:
            m = mine.get(t["name"])
            if m is None:
                sys.exit(f"{theirs}: {t['name']} is not a tensor of the tower")
            if (m["ggml_type"], list(m["dims"])) != (t["ggml_type"], list(t["dims"])):
                sys.exit(f"{t['name']}: type {t['type_name']} dims {list(t['dims'])}, the tower's {m['type_name']} {list(m['dims'])}")
            if a.read_tensor(m) != b.read_tensor(t):
                sys.exit(f"{t['name']}: other values than the tower's")
        for key in ("clip.projector_type", "clip.vision.block_count", "clip.vision.embedding_length",
                    "clip.vision.attention.head_count", "clip.vision.projection_dim", "clip.vision.patch_size",
                    "clip.vision.spatial_merge_size", "clip.vision.feed_forward_length"):
            if a.metadata.get(key) != b.metadata.get(key):
                sys.exit(f"{theirs}: {key} {b.metadata.get(key)!r}, the tower's {a.metadata.get(key)!r}")
    print(f"mistral vision: {theirs} is the tower, {len(names)} tensors equal")


def permute(w, heads):
    """the release's q or k rows to a GGUF converter's interleaved order (row 2j <- j, 2j + 1 <- j + hd / 2)"""
    out, i = w.shape
    return w.reshape(heads, 2, out // heads // 2, i).swapaxes(1, 2).reshape(out, i)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--inter", type=int, default=96)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--merge", type=int, default=2)
    ap.add_argument("--out-hidden", type=int, default=48)
    ap.add_argument("--grid", type=int, nargs=2, default=[6, 8], help="patch rows, patch columns")
    ap.add_argument("--eps", type=float, default=1e-5, help="the text model's RMSNorm eps")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--check", default=None, help="tests/mistral_vision_check: run it and compare within 1e-4")
    ap.add_argument("--release", type=Path, default=None, help="also write the tower in the release's layout here")
    ap.add_argument("--same-as", type=Path, default=None, help="check this mmproj is DIR's tower (nothing written)")
    a = ap.parse_args()
    if a.same_as:
        return same_as(a.out / "mmproj.gguf", a.same_as)
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    D, H, F, P, m, OH = a.hidden, a.heads, a.inter, a.patch, a.merge, a.out_hidden
    mat = lambda o, i: (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)   # noqa: E731
    norm = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)          # noqa: E731
    tensors = []

    def put(name, a32, kind):
        dims = list(reversed(a32.shape))
        if kind == GG_F32 and release_name(name):
            a32 = bf16_values(a32)   # the release keeps them BF16, the published file widens them
        payload = bf16(a32).tobytes() if kind == GG_BF16 else np.ascontiguousarray(a32, dtype=np.float32).tobytes()
        tensors.append((name, kind, dims, payload))

    put("v.patch_embd.weight", (rng.standard_normal((D, 3, P, P)) / np.sqrt(3 * P * P)).astype(np.float32), GG_F32)
    put("v.pre_ln.weight", norm(D), GG_F32)
    for l in range(a.depth):
        b = f"v.blk.{l}."
        put(b + "attn_q.weight", permute(mat(D, D), H), GG_BF16)
        put(b + "attn_k.weight", permute(mat(D, D), H), GG_BF16)
        put(b + "attn_v.weight", mat(D, D), GG_BF16)
        put(b + "attn_out.weight", mat(D, D), GG_BF16)
        put(b + "ln1.weight", norm(D), GG_F32)
        put(b + "ln2.weight", norm(D), GG_F32)
        put(b + "ffn_gate.weight", mat(F, D), GG_BF16)
        put(b + "ffn_up.weight", mat(F, D), GG_BF16)
        put(b + "ffn_down.weight", mat(D, F), GG_BF16)
    put("mm.input_norm.weight", norm(D), GG_F32)
    put("mm.patch_merger.weight", mat(D, D * m * m), GG_F32)
    put("mm.1.weight", mat(OH, D), GG_BF16)
    put("mm.2.weight", mat(OH, OH), GG_BF16)
    put("v.token_embd.img_break", rng.standard_normal(OH).astype(np.float32), GG_F32)
    kv = [("general.architecture", STR, "clip"), ("general.type", STR, "mmproj"),
          ("general.name", STR, "mistral-large4 vision tiny"),
          ("clip.has_vision_encoder", BOOL, True), ("clip.projector_type", STR, "pixtral"),
          ("clip.vision.projection_dim", U32, OH), ("clip.vision.image_size", U32, 1540),
          ("clip.vision.patch_size", U32, P), ("clip.vision.embedding_length", U32, D),
          ("clip.vision.feed_forward_length", U32, F), ("clip.vision.block_count", U32, a.depth),
          ("clip.vision.attention.head_count", U32, H),
          ("clip.vision.image_mean", ARR, (F32, [0.48145467, 0.4578275, 0.40821072])),
          ("clip.vision.image_std", ARR, (F32, [0.26862954, 0.26130258, 0.27577711])),
          ("clip.vision.attention.layer_norm_epsilon", F32, 1e-5), ("clip.rope.freq_base", F32, 10000.0),
          ("clip.use_silu", BOOL, True), ("clip.vision.spatial_merge_size", U32, m)]
    (a.out / "mmproj.gguf").write_bytes(_gguf_bytes(kv, tensors))
    if a.release:
        write_release(a.release, tensors, a)
    gh, gw = a.grid
    (rng.standard_normal(gh * gw * 3 * P * P).astype(np.float32) * 0.8).tofile(a.out / "patches.f32")
    subprocess.run([sys.executable, str(HERE / "mistral_vision_ref.py"), str(a.out / "mmproj.gguf"),
                    str(a.out / "patches.f32"), str(gh), str(gw), str(a.out / "ref.f32"), "--eps", str(a.eps)], check=True)
    (a.out / "ref.json").write_text(json.dumps({"grid": [gh, gw], "out_hidden": OH, "eps": a.eps}), encoding="utf-8")
    if a.check:
        subprocess.run([a.check, str(a.out / "mmproj.gguf"), str(a.out / "patches.f32"), str(gh), str(gw),
                        str(a.out / "engine.f32"), str(OH), str(a.eps)], check=True)
        got = np.fromfile(a.out / "engine.f32", dtype=np.float32)
        ref = np.fromfile(a.out / "ref.f32", dtype=np.float32)
        if got.shape != ref.shape:
            sys.exit(f"mistral vision: {got.size} values, {ref.size} expected")
        err = float(np.abs(got - ref).max() / np.sqrt((ref ** 2).mean()))
        print(f"mistral vision: {ref.size // OH} rows, max |engine - reference| {err:.2e} of the rows' rms")
        if not err < 1e-4:
            sys.exit("mistral vision: the engine's rows are not the reference's")


if __name__ == "__main__":
    main()
