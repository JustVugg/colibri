#!/usr/bin/env python3
"""A tiny LFM3 (LiquidAI/LFM3-VL's text model, model_type "lfm3") with random weights, as an
lfm3 GGUF with the tensor names and keys of the published files (LiquidAI/LFM3-VL-2B-GGUF),
and the greedy continuation of a prompt as a ref.json.

The text model, from the release's modeling_lfm3.py:

1. a layer is a short gated convolution ("conv") or attention ("full_attention",
   "sliding_attention"), its input normalised (operator_norm) and its output too
   (post_operator_norm) before the residual; the feed-forward branch the same way
   (ffn_norm, then post_ffn_norm): RMSNorms with a plain weight;
2. the conv: in_proj to B, C and x (in that order), C * conv(B * x), out_proj; the conv is
   causal and by channel over conv_L_cache taps, conv[t] = sum_k w[k] * bx[t + k - (K-1)];
3. attention: q, k and v projections, an RMSNorm on every q and k head, RoPE (rotate_half)
   at rope_theta on the full layers for the first partial_rotary_factor * head_dim / 2
   frequencies of the whole head (the rest unrotated: the published files say so in
   rope_freqs, 1 then 1e30), at the sliding layers' own theta on every frequency, a window
   of sliding_window rows (its own included) on the sliding layers, then every head's
   output times sigmoid(gate_proj(the attention's input)) (one gate per head), o_proj;
4. the MLP: w2(silu(w1 x) * w3 x);
5. the final RMSNorm (embedding_norm) and the tied embedding as the head.

The reference is written from that definition, in float32, and was checked against the
release's own code on a model of this shape (logits up to 43 within 3e-5). --dense Q8_0 stores the
matrices in Q8_0 blocks; the reference then reads the blocks' values and quantizes the
matrices' inputs as the engine's int8 kernels do (--act f32: as the Vulkan chain reads them, in float).

    python3 tools/make_lfm3_tiny.py --out lfm3_tiny --emit-ref lfm3_tiny/ref.json

--vision adds LFM3-VL's tower and projector at toy size (modeling_lfm3_vl.py, the engine's
lfm3_vision.h), as an mmproj GGUF beside the model (mmproj.gguf, the published file's names and
keys: the patch embedding as one conv kernel a frame), and a picture in the ref.json: its patches,
and its rows over the prompt's <|image|> tokens (token 100 is <|image|>); the rows themselves in
image_rows.f32, for a check closer than the tokens (a 5% error in them keeps the same continuation).
"""
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import torch
except ImportError as exc:   # pragma: no cover
    sys.exit(f"Missing deps: {exc}. Run: pip install torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_flint_tiny import _gguf_bytes, _q8_0  # noqa: E402
from make_muse_tiny import _act_q8  # noqa: E402


def tiny_config(vocab=128, hidden=64, heads=8, kv_heads=2, head_dim=16, inter=128, window=5, taps=4):
    # the release's pattern at toy depth: convs between the attention layers, a sliding one
    # after each full one
    types = ["conv", "full_attention", "conv", "sliding_attention", "conv", "full_attention", "conv"]
    return {
        "model_type": "lfm3", "vocab_size": vocab, "hidden_size": hidden,
        "num_hidden_layers": len(types), "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "head_dim": head_dim, "intermediate_size": inter, "layer_types": types,
        "sliding_window": window, "conv_L_cache": taps, "norm_eps": 1e-5,
        "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 1000000.0,
                                               "partial_rotary_factor": 0.5},
                            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
        "max_position_embeddings": 4096, "bos_token_id": 1, "eos_token_id": vocab - 1,
        "tie_word_embeddings": True,
    }


def random_weights(cfg, seed):
    """The weights under the GGUF's names, in the checkpoint's (and the GGUF's) layout."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.15):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n, around=1.0):
        return (around + 0.2 * torch.randn(n, generator=g)).float()

    D, V, H, KV, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"], cfg["head_dim"]
    F, K = cfg["intermediate_size"], cfg["conv_L_cache"]
    w = {"token_embd.weight": r(V, D, std=1.0), "output_norm.weight": norm(D)}
    for i, kind in enumerate(cfg["layer_types"]):
        p = f"blk.{i}."
        w[p + "attn_norm.weight"] = norm(D)
        w[p + "post_attention_norm.weight"] = norm(D, 0.5)
        w[p + "ffn_norm.weight"] = norm(D)
        w[p + "post_ffw_norm.weight"] = norm(D, 0.5)
        w[p + "ffn_gate.weight"] = r(F, D)
        w[p + "ffn_up.weight"] = r(F, D)
        w[p + "ffn_down.weight"] = r(D, F)
        if kind == "conv":
            w[p + "shortconv.in_proj.weight"] = r(3 * D, D, std=0.3)
            w[p + "shortconv.conv.weight"] = r(D, K, std=0.5)
            w[p + "shortconv.out_proj.weight"] = r(D, D)
        else:
            w[p + "attn_q.weight"] = r(H * hd, D)
            w[p + "attn_k.weight"] = r(KV * hd, D)
            w[p + "attn_v.weight"] = r(KV * hd, D)
            w[p + "attn_q_norm.weight"] = norm(hd)
            w[p + "attn_k_norm.weight"] = norm(hd)
            w[p + "attn_gate.weight"] = r(H, D, std=0.5)
            w[p + "attn_output.weight"] = r(D, H * hd)
    return w


IMAGE_TOKEN = 100


def tiny_vision_config():
    return {"hidden": 32, "heads": 4, "kv_heads": 2, "inter": 48, "patch": 4, "merge": 2, "temporal": 2,
            "in_ch": 3, "proj": 40, "eps": 1e-5, "theta": 1e6,
            "types": ["conv", "full_attention", "conv", "full_attention"]}


def random_vision(vc, out_hidden, seed):
    """The tower and projector under the mmproj's names; the patch embedding as the
    checkpoint's linear map [D, C x T x P x P] (split into frames when written)."""
    g = torch.Generator().manual_seed(seed + 7)

    def r(*shape, std=0.2):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n, around=1.0):
        return (around + 0.2 * torch.randn(n, generator=g)).float()

    D, F, H, KV = vc["hidden"], vc["inter"], vc["heads"], vc["kv_heads"]
    hd, P = D // H, vc["proj"]
    w = {"v.patch_embd": r(D, vc["in_ch"] * vc["temporal"] * vc["patch"] ** 2, std=0.1),
         "v.embedding_norm.weight": norm(D),
         "mm.input_norm.weight": norm(D), "mm.input_norm.bias": r(D, std=0.1),
         "mm.0.weight": r(P, 4 * D, std=0.15), "mm.0.bias": r(P, std=0.1),
         "mm.2.weight": r(out_hidden, P, std=0.3), "mm.2.bias": r(out_hidden, std=0.1)}
    for i, kind in enumerate(vc["types"]):
        p = f"v.blk.{i}."
        w[p + "ln1.weight"], w[p + "ln2.weight"] = norm(D), norm(D)
        w[p + "ffn_gate.weight"], w[p + "ffn_up.weight"], w[p + "ffn_down.weight"] = r(F, D), r(F, D), r(D, F)
        if kind == "conv":
            w[p + "conv2d_in_proj.weight"] = r(3 * D, D, std=0.3)
            w[p + "conv2d_spatial.weight"] = r(D, 1, 3, 3, std=0.4)
            w[p + "conv2d_out_proj.weight"] = r(D, D)
        else:
            w[p + "attn_q.weight"], w[p + "attn_k.weight"], w[p + "attn_v.weight"] = r(H * hd, D), r(KV * hd, D), r(KV * hd, D)
            w[p + "attn_q_norm.weight"], w[p + "attn_k_norm.weight"] = norm(hd), norm(hd)
            w[p + "attn_out.weight"] = r(D, H * hd)
    return w


def vision_ref(vc, w, patches, gh, gw):
    """The tower and projector, float32: patches in merge-block order -> one row a 2 x 2 block."""
    F_ = torch.nn.functional
    eps, m, D, H, KV = vc["eps"], vc["merge"], vc["hidden"], vc["heads"], vc["kv_heads"]
    hd, N = D // H, gh * gw

    def rms(x, weight):
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    x = rms(patches @ w["v.patch_embd"].T, w["v.embedding_norm.weight"])
    k = torch.arange(N)
    blk, cell = k // (m * m), k % (m * m)
    row, col = (blk // (gw // m)) * m + cell // m, (blk % (gw // m)) * m + cell % m
    sd = hd // 2
    inv = 1.0 / (vc["theta"] ** (torch.arange(0, sd, 2, dtype=torch.float32) / sd))
    ang = torch.cat([row[:, None].float() * inv, col[:, None].float() * inv], -1)
    ang = torch.cat([ang, ang], -1)
    cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]

    def rot(t):
        return t * cos + torch.cat([-t[..., hd // 2:], t[..., :hd // 2]], -1) * sin
    for i, kind in enumerate(vc["types"]):
        p = f"v.blk.{i}."
        h = rms(x, w[p + "ln1.weight"])
        if kind == "full_attention":
            q = rot(rms((h @ w[p + "attn_q.weight"].T).view(N, H, hd), w[p + "attn_q_norm.weight"]))
            kk = rot(rms((h @ w[p + "attn_k.weight"].T).view(N, KV, hd), w[p + "attn_k_norm.weight"]))
            v = (h @ w[p + "attn_v.weight"].T).view(N, KV, hd)
            kk, v = kk.repeat_interleave(H // KV, 1), v.repeat_interleave(H // KV, 1)
            a = (torch.einsum("nhd,mhd->hnm", q, kk) / math.sqrt(hd)).softmax(-1)
            out = torch.einsum("hnm,mhd->nhd", a, v).reshape(N, D) @ w[p + "attn_out.weight"].T
        else:
            b, c, xx = (h @ w[p + "conv2d_in_proj.weight"].T).chunk(3, -1)
            fr = (b * xx).view(gh // m, gw // m, m, m, D).permute(4, 0, 2, 1, 3).reshape(1, D, gh, gw)
            fr = F_.conv2d(fr, w[p + "conv2d_spatial.weight"], padding=1, groups=D)
            fr = fr.view(D, gh // m, m, gw // m, m).permute(1, 3, 2, 4, 0).reshape(N, D)
            out = (c * fr) @ w[p + "conv2d_out_proj.weight"].T
        x = x + out
        h = rms(x, w[p + "ln2.weight"])
        x = x + (F_.silu(h @ w[p + "ffn_gate.weight"].T) * (h @ w[p + "ffn_up.weight"].T)) @ w[p + "ffn_down.weight"].T
    y = F_.layer_norm(x, (D,), w["mm.input_norm.weight"], w["mm.input_norm.bias"], 1e-6).reshape(-1, 4 * D)
    y = F_.gelu(y @ w["mm.0.weight"].T + w["mm.0.bias"])
    return y @ w["mm.2.weight"].T + w["mm.2.bias"]


def write_mmproj(path, vc, w, out_hidden):
    """The tower as LiquidAI's mmproj GGUF has it: arch clip, projector lfm3vl, F32 here."""
    import numpy as np
    U32, F32, BOOL, STR = 4, 6, 7, 8
    kv = [("general.architecture", STR, "clip"), ("general.type", STR, "mmproj"),
          ("clip.has_vision_encoder", BOOL, True), ("clip.vision.projector_type", STR, "lfm3vl")]
    for k, v in (("block_count", len(vc["types"])), ("embedding_length", vc["hidden"]),
                 ("feed_forward_length", vc["inter"]), ("attention.head_count", vc["heads"]),
                 ("attention.head_count_kv", vc["kv_heads"]), ("patch_size", vc["patch"]),
                 ("spatial_merge_size", vc["merge"]), ("projection_dim", out_hidden)):
        kv.append(("clip.vision." + k, U32, int(v)))
    kv.append(("clip.vision.attention.layer_norm_epsilon", F32, float(vc["eps"])))
    tensors = []
    pe = w["v.patch_embd"].view(vc["hidden"], vc["in_ch"], vc["temporal"], vc["patch"], vc["patch"])
    for t in range(vc["temporal"]):   # one [D, C, P, P] kernel a frame
        a = pe[:, :, t].contiguous().numpy().astype(np.float32)
        tensors.append(("v.patch_embd.weight" + (f".{t}" if t else ""), 0, list(reversed(a.shape)), a.tobytes()))
    for name, t in w.items():
        if name == "v.patch_embd":
            continue
        a = t.float().contiguous().numpy().astype(np.float32)
        tensors.append((name, 0, list(reversed(a.shape)), a.tobytes()))
    Path(path).write_bytes(_gguf_bytes(kv, tensors))


class Lfm3Ref:
    """LFM3's text forward over a whole sequence, float32, no cache. dense_q8: the matrices'
    inputs quantized in blocks of 32 first, as the engine's int8 kernels read a Q8_0 matrix."""

    def __init__(self, cfg, w, dense_q8=False, image_rows=None):
        self.c, self.w, self.dense_q8, self.image_rows = cfg, w, dense_q8, image_rows

    def mm(self, x, name):
        return (_act_q8(x) if self.dense_q8 else x) @ self.w[name].T

    def rms(self, x, weight):
        eps = self.c["norm_eps"]
        return x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5) * weight

    def rope(self, x, kind):   # x [T, heads, hd]: rotate_half, the zero-tailed frequencies
        hd = x.shape[-1]
        rp = self.c["rope_parameters"][kind]
        angles = int(rp.get("partial_rotary_factor", 1.0) * hd) // 2
        inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, 2 * angles, 2, dtype=torch.int64).float() / hd))
        inv = torch.cat([inv, torch.zeros(hd // 2 - angles)])
        ang = torch.arange(x.shape[0])[:, None].float() * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def conv(self, i, h):
        p, D, K = f"blk.{i}.", self.c["hidden_size"], self.c["conv_L_cache"]
        bcx = self.mm(h, p + "shortconv.in_proj.weight")
        b, c, x = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
        bx = torch.cat([torch.zeros(K - 1, D), b * x])
        taps = self.w[p + "shortconv.conv.weight"]   # [D, K]
        out = sum(taps[:, k] * bx[k:k + h.shape[0]] for k in range(K))
        return self.mm(c * out, p + "shortconv.out_proj.weight")

    def attention(self, i, h):
        c, w, p = self.c, self.w, f"blk.{i}."
        T, kind = h.shape[0], c["layer_types"][i]
        H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = self.rms(self.mm(h, p + "attn_q.weight").view(T, H, hd), w[p + "attn_q_norm.weight"])
        k = self.rms(self.mm(h, p + "attn_k.weight").view(T, KV, hd), w[p + "attn_k_norm.weight"])
        v = self.mm(h, p + "attn_v.weight").view(T, KV, hd)
        q, k = self.rope(q, kind), self.rope(k, kind)
        t = torch.arange(T)
        allowed = t[None, :] <= t[:, None]
        if kind == "sliding_attention":
            allowed &= t[None, :] > t[:, None] - c["sliding_window"]
        out = torch.empty(T, H, hd)
        rep = H // KV
        for hh in range(H):
            s = (q[:, hh, :] @ k[:, hh // rep, :].T) / math.sqrt(hd)
            s = s.masked_fill(~allowed, float("-inf"))
            out[:, hh, :] = torch.softmax(s, -1) @ v[:, hh // rep, :]
        out = out * torch.sigmoid(self.mm(h, p + "attn_gate.weight"))[:, :, None]
        return self.mm(out.reshape(T, H * hd), p + "attn_output.weight")

    def logits(self, ids):
        w, c = self.w, self.c
        x = w["token_embd.weight"][torch.tensor(ids)]
        if self.image_rows is not None:   # the picture's rows over its tokens, in order
            at = [i for i, t in enumerate(ids) if t == IMAGE_TOKEN][:self.image_rows.shape[0]]   # the prompt's run
            x = x.clone()
            x[at] = self.image_rows
        for i, kind in enumerate(c["layer_types"]):
            p = f"blk.{i}."
            h = self.rms(x, w[p + "attn_norm.weight"])
            a = self.conv(i, h) if kind == "conv" else self.attention(i, h)
            x = x + self.rms(a, w[p + "post_attention_norm.weight"])
            f = self.rms(x, w[p + "ffn_norm.weight"])
            m = self.mm(torch.nn.functional.silu(self.mm(f, p + "ffn_gate.weight")) * self.mm(f, p + "ffn_up.weight"),
                        p + "ffn_down.weight")
            x = x + self.rms(m, w[p + "post_ffw_norm.weight"])
        return self.mm(self.rms(x, w["output_norm.weight"]), "token_embd.weight")

    def generate(self, prompt, n):
        """The greedy continuation and the smallest margin between the first and the
        second logit over its steps."""
        ids, gap = list(prompt), float("inf")
        for _ in range(n):
            top = self.logits(ids)[-1].topk(2)
            gap = min(gap, float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
        return ids, gap


def _kind(name, t, dense_type):
    """The GGUF type a tensor goes in: the matrices' (the conv's taps and the vectors F32,
    as published), F32 for a row that is not whole blocks."""
    kind = dense_type if t.ndim == 2 and "shortconv.conv" not in name else "F32"
    return kind if kind == "F32" or t.shape[-1] % 32 == 0 else "F32"


def as_stored(w, dense_type):
    """The weights through the GGUF's blocks and back (the reference computes with what the
    engine reads)."""
    import numpy as np
    out = {}
    for name, t in w.items():
        if _kind(name, t, dense_type) != "Q8_0":
            out[name] = t
            continue
        raw = np.frombuffer(_q8_0(t.float().numpy().astype(np.float32)), dtype=np.uint8).reshape(-1, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32)
        q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
        out[name] = torch.from_numpy((q * d).reshape(t.shape)).float()
    return out


def write_gguf(path, cfg, w, dense_type="F32", vision=False):
    """The weights as LiquidAI's lfm3 GGUFs have them: names, keys (head_count_kv and the
    window's pattern by layer), the full layers' partial rotation as rope_freqs."""
    import numpy as np
    U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9
    V, hd = cfg["vocab_size"], cfg["head_dim"]
    types = cfg["layer_types"]
    p = "lfm3."
    kv = [("general.architecture", STR, "lfm3"), ("general.name", STR, "lfm3 tiny")]
    for k, v in (("block_count", len(types)), ("context_length", cfg["max_position_embeddings"]),
                 ("embedding_length", cfg["hidden_size"]), ("feed_forward_length", cfg["intermediate_size"]),
                 ("attention.head_count", cfg["num_attention_heads"]),
                 ("attention.key_length", hd), ("attention.value_length", hd), ("rope.dimension_count", hd),
                 ("attention.sliding_window", cfg["sliding_window"]), ("shortconv.l_cache", cfg["conv_L_cache"])):
        kv.append((p + k, U32, int(v)))
    rp = cfg["rope_parameters"]
    kv += [(p + "attention.head_count_kv", ARR, (I32, [0 if t == "conv" else cfg["num_key_value_heads"] for t in types])),
           (p + "rope.freq_base", F32, float(rp["full_attention"]["rope_theta"])),
           (p + "rope.freq_base_swa", F32, float(rp["sliding_attention"]["rope_theta"])),
           (p + "attention.layer_norm_rms_epsilon", F32, float(cfg["norm_eps"])),
           (p + "attention.sliding_window_pattern", ARR, (BOOL, [t == "sliding_attention" for t in types])),
           ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "lfm2"),
           ("tokenizer.ggml.tokens", ARR, (STR, ["<|image|>" if vision and b == IMAGE_TOKEN else f"<0x{b:02X}>" for b in range(V)])),
           ("tokenizer.ggml.token_type", ARR, (I32, [1] * V)),
           ("tokenizer.ggml.merges", ARR, (STR, [])),
           ("tokenizer.ggml.bos_token_id", U32, cfg["bos_token_id"]),
           ("tokenizer.ggml.eos_token_id", U32, cfg["eos_token_id"])]
    angles = int(rp["full_attention"]["partial_rotary_factor"] * hd) // 2
    factors = np.array([1.0] * angles + [1e30] * (hd // 2 - angles), dtype=np.float32)
    tensors = [("rope_freqs.weight", 0, [hd // 2], factors.tobytes())]
    for name, t in w.items():
        a = t.float().contiguous().numpy().astype(np.float32)
        dims = list(reversed(a.shape))
        if _kind(name, t, dense_type) == "Q8_0":
            tensors.append((name, 8, dims, _q8_0(a)))
        else:
            tensors.append((name, 0, dims, a.tobytes()))
    Path(path).write_bytes(_gguf_bytes(kv, tensors))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="a directory: model.gguf goes here")
    ap.add_argument("--emit-ref", default=None, help="write the greedy continuation here")
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dense", default="F32", choices=("F32", "Q8_0"), help="the GGUF's matrices: F32 or Q8_0")
    ap.add_argument("--act", default=None, choices=("q8", "f32"),
                    help="a Q8_0 matrix's inputs in the reference: q8 as the CPU's integer kernels quantize them "
                         "(the default with Q8_0), f32 as the Vulkan chain reads them")
    ap.add_argument("--vision", action="store_true",
                    help="LFM3-VL's tower too: mmproj.gguf, and a 4 x 6-patch picture in the prompt")
    ap.add_argument("--min-gap", type=float, default=0.1,
                    help="draw seeds from --seed on until every greedy step's first logit beats the "
                         "second by this much, so a quantized matrix cannot flip a near tie")
    args = ap.parse_args()
    cfg = tiny_config(window=args.window)
    prompt = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    vc, gh, gw = tiny_vision_config(), 4, 6
    if args.vision:   # a picture of 4 x 6 patches: six tokens after the third
        prompt = prompt[:3] + [IMAGE_TOKEN] * ((gh // 2) * (gw // 2)) + prompt[3:]
    for seed in range(args.seed, args.seed + 200):
        w = as_stored(random_weights(cfg, seed), args.dense)
        rows = None
        if args.vision:
            vw = random_vision(vc, cfg["hidden_size"], seed)
            patches = (torch.randn(gh * gw, vc["in_ch"] * vc["temporal"] * vc["patch"] ** 2,
                                   generator=torch.Generator().manual_seed(seed + 11)) * 0.5).float()
            with torch.no_grad():
                rows = vision_ref(vc, vw, patches, gh, gw)
        if not args.emit_ref:
            break
        with torch.no_grad():
            q8 = args.dense == "Q8_0" and (args.act or "q8") == "q8"
            full, gap = Lfm3Ref(cfg, w, dense_q8=q8, image_rows=rows).generate(prompt, args.max_new)
        if gap >= args.min_gap:
            break
    else:
        sys.exit(f"no seed in {args.seed}..{args.seed + 199} gives a margin of {args.min_gap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_gguf(out / "model.gguf", cfg, w, args.dense, vision=args.vision)
    if args.vision:   # the tower, and the rows it gives for the picture (the engine's Q36_VIS_DUMP checks against them)
        write_mmproj(out / "mmproj.gguf", vc, vw, cfg["hidden_size"])
        (out / "image_rows.f32").write_bytes(rows.numpy().astype("float32").tobytes())
        print(f"tiny LFM3-VL tower -> {out / 'mmproj.gguf'}, its rows for the picture -> {out / 'image_rows.f32'}")
    print(f"tiny LFM3 -> {out / 'model.gguf'} ({sum(v.numel() for v in w.values()) / 1e6:.2f}M parameters, "
          f"matrices {args.dense})")
    if args.emit_ref:
        Path(args.emit_ref).parent.mkdir(parents=True, exist_ok=True)
        ref = {"prompt_ids": prompt, "full_ids": full, "mode": "full", "model": "lfm3_tiny"}
        if args.vision:
            ref["image"] = {"grid_h": gh, "grid_w": gw, "patches": [round(float(v), 7) for v in patches.flatten()]}
        Path(args.emit_ref).write_text(json.dumps(ref, indent=2))
        print(f"ref.json -> {args.emit_ref} (seed {seed}, smallest margin {gap:.3f})\n"
              f"  prompt_ids={prompt}\n  full_ids ={full}")


if __name__ == "__main__":
    main()
