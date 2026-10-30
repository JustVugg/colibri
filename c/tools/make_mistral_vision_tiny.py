#!/usr/bin/env python3
"""A tiny Mistral-Large-4 vision tower (Pixtral) as an mmproj GGUF, in the published file's layout,
and the rows tools/mistral_vision_ref.py gives for random patches: mistral_vision.h is checked
against them (make mistral-vision-check).

The layout is unsloth's mmproj-*.gguf: architecture clip, projector pixtral, the big matrices BF16,
norms, the patch kernel and the merger F32, and q and k of every head in interleaved rotary order
(a GGUF converter's permutation), which the engine rotates by pairs and the reference undoes.

  python3 tools/make_mistral_vision_tiny.py --out DIR [--check BINARY]
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
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    D, H, F, P, m, OH = a.hidden, a.heads, a.inter, a.patch, a.merge, a.out_hidden
    mat = lambda o, i: (rng.standard_normal((o, i)) / np.sqrt(i)).astype(np.float32)   # noqa: E731
    norm = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)          # noqa: E731
    tensors = []

    def put(name, a32, kind):
        dims = list(reversed(a32.shape))
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
