#!/usr/bin/env python3
"""Float32 numpy reference of Mistral-Large-4's vision tower and projector (Pixtral, Mistral3),
written operation by operation from transformers' modeling_pixtral.py and modeling_mistral3.py,
for mistral_vision.h to be checked against: patches in (tools/mistral_image.py's layout), the
rows the model reads out.

  python3 tools/mistral_vision_ref.py MMPROJ PATCHES.f32 GRID_H GRID_W OUT.f32 [--eps E]

GRID_H x GRID_W are the patch grid (twice the merged tokens a side). numpy only, layer by layer
(a BF16 tower of 1.7B is 6.8 GB as float32 at once).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_reader import GGUFReader   # noqa: E402


def tensor(r, name):
    t = r.tensor(name)
    raw = r.read_tensor(t)
    dims = list(reversed(t["dims"]))   # ggml ne0 first: the numpy shape is the reverse
    if t["type_name"] == "F32":
        a = np.frombuffer(raw, dtype=np.float32)
    elif t["type_name"] == "F16":
        a = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif t["type_name"] == "BF16":
        a = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        raise SystemExit(f"{name}: type {t['type_name']} not read by the reference")
    return a.reshape(dims).astype(np.float32)


def unpermute(w, heads):
    """An mmproj's q or k rows back to the release's order: GGUF converters store each head's rows
    interleaved (row 2j the release's j, 2j + 1 its j + head_dim / 2) for an interleaved rotation;
    the reference then rotates halves, as modeling_pixtral.py does."""
    out, i = w.shape
    return w.reshape(heads, out // heads // 2, 2, i).swapaxes(1, 2).reshape(out, i)


def rms(x, w, eps):
    var = (x.astype(np.float32) ** 2).mean(-1, keepdims=True)
    return w * (x * (1.0 / np.sqrt(var + eps)))


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def rope_tables(gh, gw, hd, theta):
    """PixtralVisionRotaryEmbedding: compute_axial_rope_parameters, then forward with the
    (row, column) position ids and recomposition_frequencies."""
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
    inv2d = np.concatenate([inv[0::2], inv[1::2]])
    hpos, wpos = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    pos = np.stack([hpos.ravel(), wpos.ravel()], axis=-1).astype(np.float32)   # [N, 2]
    freqs = pos[..., None] * inv2d                                              # [N, 2, hd / 2]
    out = []
    for f in (np.cos(freqs), np.sin(freqs)):
        ch = np.split(f, 2, axis=-1)            # freq.chunk(2, dim=-1)
        fh, fw = ch[0][:, 0], ch[1][:, 1]       # m[:, i % 2]
        fhw = np.concatenate([fh, fw], axis=-1)
        out.append(np.concatenate([fhw, fhw], axis=-1))
    return out


def gelu(x):
    from math import erf
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mmproj"); ap.add_argument("patches"); ap.add_argument("grid_h", type=int)
    ap.add_argument("grid_w", type=int); ap.add_argument("out")
    ap.add_argument("--eps", type=float, default=1e-5, help="the text model's RMSNorm eps (the projector's norm)")
    a = ap.parse_args()
    r = GGUFReader(a.mmproj)
    md = r.metadata
    D = int(md["clip.vision.embedding_length"]); H = int(md["clip.vision.attention.head_count"])
    L = int(md["clip.vision.block_count"]); P = int(md["clip.vision.patch_size"])
    m = int(md.get("clip.vision.spatial_merge_size", 2)); theta = float(md.get("clip.rope.freq_base", 10000.0))
    veps = float(md.get("clip.vision.attention.layer_norm_epsilon", 1e-5))
    hd = D // H; gh, gw = a.grid_h, a.grid_w; N = gh * gw
    x = np.fromfile(a.patches, dtype=np.float32).reshape(N, 3 * P * P)
    conv = tensor(r, "v.patch_embd.weight").reshape(D, 3 * P * P)   # Conv2d [D, 3, P, P], stride P
    x = x @ conv.T
    x = rms(x, tensor(r, "v.pre_ln.weight"), veps)
    cos, sin = rope_tables(gh, gw, hd, theta)
    for l in range(L):
        w = lambda n: tensor(r, f"v.blk.{l}.{n}.weight")   # noqa: E731
        h = rms(x, w("ln1"), veps)
        q = (h @ unpermute(w("attn_q"), H).T).reshape(N, H, hd).transpose(1, 0, 2)
        k = (h @ unpermute(w("attn_k"), H).T).reshape(N, H, hd).transpose(1, 0, 2)
        v = (h @ w("attn_v").T).reshape(N, H, hd).transpose(1, 0, 2)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        s = (q @ k.transpose(0, 2, 1)) * (hd ** -0.5)
        s = np.exp(s - s.max(-1, keepdims=True)); s = s / s.sum(-1, keepdims=True)
        att = (s @ v).transpose(1, 0, 2).reshape(N, D)
        x = x + att @ w("attn_out").T
        h = rms(x, w("ln2"), veps)
        g = h @ w("ffn_gate").T
        x = x + ((g / (1.0 + np.exp(-g))) * (h @ w("ffn_up").T)) @ w("ffn_down").T
    # Mistral3MultiModalProjector: norm, patch merger (unfold kernel m stride m), linear, GELU, linear
    x = rms(x, tensor(r, "mm.input_norm.weight"), a.eps)
    grid = x.reshape(gh, gw, D).transpose(2, 0, 1)                      # [D, gh, gw]
    blocks = grid.reshape(D, gh // m, m, gw // m, m).transpose(1, 3, 0, 2, 4)   # [bh, bw, D, m, m]
    blocks = blocks.reshape((gh // m) * (gw // m), D * m * m)           # unfold's (channel, ki, kj)
    y = blocks @ tensor(r, "mm.patch_merger.weight").T
    y = gelu(y @ tensor(r, "mm.1.weight").T)
    y = y @ tensor(r, "mm.2.weight").T
    y.astype(np.float32).tofile(a.out)
    print(f"{y.shape[0]} rows of {y.shape[1]}: rms {float(np.sqrt((y ** 2).mean())):.4f}")


if __name__ == "__main__":
    main()
