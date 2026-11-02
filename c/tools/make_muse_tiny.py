#!/usr/bin/env python3
"""A tiny Muse Spark (meta-models/Muse-Spark-Open-816B-A42B, model_type "muse_spark") with
random weights, as a muse-spark GGUF with the tensor names and keys of the published files
(unsloth/Muse-Spark-Open-816B-A42B-GGUF), and the greedy continuation of a prompt as a
ref.json.

The text model, from transformers' MuseSpark definition (modular_muse_spark.py, on top of
modular_muse_glimmer.py):

1. the token embedding goes through an RMSNorm without a weight;
2. attention: q, k and v projections, an RMSNorm without a weight on every q and k head
   (q times qk_scale_factor after it), RoPE except on the full-attention layers
   (layer_rope_theta 0), a window of sliding_window rows (its own included) on the
   sliding layers, then an RMSNorm without a weight on every head's output, times
   sigmoid(gate_proj(the attention's input)), then o_proj;
3. after attention an RMSNorm without a weight (post_norm_eps), and the residual mixed by a
   learned gate: alpha * residual + beta * branch, beta = sigmoid(g / T),
   alpha = sqrt(max(sigmoid(-g / T) * (1 + beta), 1e-3));
4. the feed-forward branch: an RMSNorm with (1 + w), the router on its output (float32,
   times output_multiplier, sigmoid; the top-k chosen on sigmoid + e_score_correction_bias,
   weighted by the sigmoids renormalised), a projection to the experts' width
   (pre_expert_proj, moe_hidden_size), an RMSNorm with (1 + w), the experts (SwiGLU of
   moe_intermediate_size), on every expert's output x * (1 + w) and then an RMSNorm without
   a weight (post_norm_eps), the weighted sum, a projection back (post_expert_proj), the
   same "scale, then normalise" norm on the branch, and the residual gate again;
5. the final RMSNorm with a plain weight, lm_head, the logits times output_multiplier and
   capped at final_logit_softcapping * tanh(x / final_logit_softcapping).

The GGUF folds what the published files fold: every (1 + w) norm is stored with the 1
added, the residual gates as their alpha and beta vectors, q_norm as qk_scale_factor and
k_norm as ones; and it holds q's and k's rows (and their norms) in the published files'
rotary order, the two halves of every head interleaved (file row 2i + j = row j * hd/2 + i),
which the engine puts back at load. The reference below is written on the stored values,
in the checkpoint's order, in float32.

    python3 tools/make_muse_tiny.py --out muse_tiny --emit-ref muse_tiny/ref.json
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


def tiny_config(vocab=128, hidden=64, layers=6, heads=8, kv_heads=2, head_dim=16, experts=16,
                topk=4, latent=96, moe_inter=32, window=5):
    # the release's pattern at toy depth: a full-attention layer without RoPE at 1, 5, 9...
    types = ["full_attention" if i % 4 == 1 else "sliding_attention" for i in range(layers)]
    return {
        "model_type": "muse_spark_text", "vocab_size": vocab, "hidden_size": hidden,
        "num_hidden_layers": layers, "num_attention_heads": heads, "num_key_value_heads": kv_heads,
        "head_dim": head_dim, "num_local_experts": experts, "num_experts_per_tok": topk,
        "moe_hidden_size": latent, "moe_intermediate_size": moe_inter, "intermediate_size": 4 * hidden,
        "layer_types": types, "layer_rope_theta": [0 if t == "full_attention" else 10000.0 for t in types],
        "sliding_window": window, "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
        "qk_scale_factor": 2.5, "output_multiplier": 1.0 / math.sqrt(hidden / 16),
        "final_logit_softcapping": 20.0, "residual_gate_temperature": 0.3,
        "routed_scaling_factor": 1.0, "max_position_embeddings": 4096,
        "bos_token_id": 1, "eos_token_id": vocab - 1,
    }


def random_weights(cfg, seed):
    """Stored values, as the GGUF holds them (see the module docstring)."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.15):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n):   # 1 + w, around 1
        return (1.0 + 0.2 * torch.randn(n, generator=g)).float()

    def gate(n):   # alpha and beta of a residual gate from its parameter
        w = r(n, std=1.0) / cfg["residual_gate_temperature"]
        beta = torch.sigmoid(w)
        alpha = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + beta), min=1e-3))
        return alpha.float(), beta.float()

    D, V, H, KV, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"], cfg["head_dim"]
    E, Lt, F = cfg["num_local_experts"], cfg["moe_hidden_size"], cfg["moe_intermediate_size"]
    w = {"token_embd.weight": r(V, D, std=1.0), "output_norm.weight": r(D, std=1.0),
         "output.weight": r(V, D, std=0.3)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"blk.{i}."
        w[p + "attn_norm.weight"] = norm(D)
        w[p + "attn_q.weight"] = r(H * hd, D)
        w[p + "attn_k.weight"] = r(KV * hd, D)
        w[p + "attn_v.weight"] = r(KV * hd, D)
        w[p + "attn_q_norm.weight"] = torch.full((hd,), cfg["qk_scale_factor"])
        w[p + "attn_k_norm.weight"] = torch.ones(hd)
        w[p + "attn_gate.weight"] = r(H * hd, D, std=0.3)
        w[p + "attn_output.weight"] = r(D, H * hd)
        w[p + "attn_resid_alpha.weight"], w[p + "attn_resid_beta.weight"] = gate(D)
        w[p + "ffn_norm.weight"] = norm(D)
        w[p + "ffn_gate_inp.weight"] = r(E, D, std=0.5)
        w[p + "exp_probs_b.bias"] = r(E, std=0.1)
        w[p + "ffn_latent_down.weight"] = r(Lt, D)
        w[p + "ffn_pre_exp_norm.weight"] = norm(Lt)
        w[p + "ffn_gate_exps.weight"] = r(E, F, Lt)
        w[p + "ffn_up_exps.weight"] = r(E, F, Lt)
        w[p + "ffn_down_exps.weight"] = r(E, Lt, F)
        w[p + "ffn_post_exp_norm.weight"] = norm(Lt)
        w[p + "ffn_latent_up.weight"] = r(D, Lt)
        w[p + "post_ffw_norm.weight"] = norm(D)
        w[p + "ffn_resid_alpha.weight"], w[p + "ffn_resid_beta.weight"] = gate(D)
    return w


def _act_q8(x):
    """An activation row as the engine's quantized kernels take it (gq_quant_act): blocks
    of 32, a float scale max|x| / 127, codes rounded to nearest even and clamped."""
    b = x.reshape(-1, 32)
    amax = b.abs().max(-1, keepdim=True).values
    d = amax / 127.0
    inv = torch.where(amax > 0, 127.0 / torch.where(amax > 0, amax, torch.ones_like(amax)), torch.zeros_like(amax))
    q = torch.clamp(torch.round(b * inv), -127, 127)
    return (q * d).reshape(x.shape)


class MuseRef:
    """Muse Spark's text forward over a whole sequence, float32, no cache. act_q8: the
    experts read their inputs as the engine's int8 kernels do, for a GGUF whose experts
    are in blocks (the weights are the blocks' own values, as_stored); dense_q8: the 2-D
    matrices too (every one but the router)."""

    def __init__(self, cfg, w, act_q8=False, dense_q8=False):
        self.c, self.w, self.act_q8, self.dense_q8 = cfg, w, act_q8, dense_q8

    def mm(self, x, name):
        """x @ W^T for a dense matrix, its rows quantized first when the engine does"""
        return (_act_q8(x) if self.dense_q8 else x) @ self.w[name].T

    @staticmethod
    def rms(x, eps, weight=None):
        y = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + eps, -0.5)
        return y if weight is None else y * weight

    def rope(self, x, pos, theta):   # x [T, heads, hd], rotate_half over the whole head
        hd = x.shape[-1]
        inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.int64).float() / hd))
        ang = pos[:, None].float() * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1)[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1)[:, None, :]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def attention(self, i, h):
        c, w, p = self.c, self.w, f"blk.{i}."
        T = h.shape[0]
        H, KV, hd, eps = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["rms_norm_eps"]
        q = self.mm(h, p + "attn_q.weight").view(T, H, hd)
        k = self.mm(h, p + "attn_k.weight").view(T, KV, hd)
        v = self.mm(h, p + "attn_v.weight").view(T, KV, hd)
        q = self.rms(q, eps, w[p + "attn_q_norm.weight"])
        k = self.rms(k, eps, w[p + "attn_k_norm.weight"])
        theta = c["layer_rope_theta"][i]
        if theta:
            pos = torch.arange(T)
            q, k = self.rope(q, pos, theta), self.rope(k, pos, theta)
        t = torch.arange(T)
        allowed = t[None, :] <= t[:, None]
        if c["layer_types"][i] == "sliding_attention":
            allowed &= t[None, :] > t[:, None] - c["sliding_window"]
        out = torch.empty(T, H, hd)
        rep = H // KV
        for hh in range(H):
            s = (q[:, hh, :] @ k[:, hh // rep, :].T) / math.sqrt(hd)
            s = s.masked_fill(~allowed, float("-inf"))
            out[:, hh, :] = torch.softmax(s, -1) @ v[:, hh // rep, :]
        out = self.rms(out, eps).reshape(T, H * hd)
        out = out * torch.sigmoid(self.mm(h, p + "attn_gate.weight"))
        return self.mm(out, p + "attn_output.weight")

    def moe(self, i, h):
        c, w, p = self.c, self.w, f"blk.{i}."
        logits = (h @ w[p + "ffn_gate_inp.weight"].T) * c["output_multiplier"]
        scores = torch.sigmoid(logits)
        K = c["num_experts_per_tok"]
        top = torch.topk(scores + w[p + "exp_probs_b.bias"], K, dim=-1).indices
        wts = scores.gather(1, top)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-15) * c["routed_scaling_factor"]
        lat = self.rms(self.mm(h, p + "ffn_latent_down.weight"), c["rms_norm_eps"], w[p + "ffn_pre_exp_norm.weight"])
        out = torch.zeros_like(lat)
        for r in range(h.shape[0]):
            for j in range(K):
                e = int(top[r, j])
                xin = _act_q8(lat[r]) if self.act_q8 else lat[r]
                g = xin @ w[p + "ffn_gate_exps.weight"][e].T
                u = xin @ w[p + "ffn_up_exps.weight"][e].T
                a = torch.nn.functional.silu(g) * u
                y = (_act_q8(a) if self.act_q8 else a) @ w[p + "ffn_down_exps.weight"][e].T
                y = self.rms(y * w[p + "ffn_post_exp_norm.weight"], c["post_norm_eps"])
                out[r] += wts[r, j] * y
        return self.mm(out, p + "ffn_latent_up.weight")

    def logits(self, ids):
        w, c = self.w, self.c
        x = self.rms(w["token_embd.weight"][torch.tensor(ids)], c["rms_norm_eps"])
        for i in range(c["num_hidden_layers"]):
            p = f"blk.{i}."
            a = self.attention(i, self.rms(x, c["rms_norm_eps"], w[p + "attn_norm.weight"]))
            a = self.rms(a, c["post_norm_eps"])
            x = w[p + "attn_resid_alpha.weight"] * x + w[p + "attn_resid_beta.weight"] * a
            f = self.moe(i, self.rms(x, c["rms_norm_eps"], w[p + "ffn_norm.weight"]))
            f = self.rms(f * w[p + "post_ffw_norm.weight"], c["post_norm_eps"])
            x = w[p + "ffn_resid_alpha.weight"] * x + w[p + "ffn_resid_beta.weight"] * f
        lo = self.mm(self.rms(x, c["rms_norm_eps"], w["output_norm.weight"]), "output.weight")
        cap = c["final_logit_softcapping"]
        return cap * torch.tanh(lo * c["output_multiplier"] / cap)

    def generate(self, prompt, n):
        """The greedy continuation and the smallest margin between the first and the
        second logit over its steps."""
        ids, gap = list(prompt), float("inf")
        for _ in range(n):
            top = self.logits(ids)[-1].topk(2)
            gap = min(gap, float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
        return ids, gap


def _kind(name, t, expert_type, dense_type):
    """The GGUF type a tensor goes in: the experts', the 2-D matrices' (the router stays
    F32, as published), F32 for the vectors and for a row that is not whole blocks."""
    kind = expert_type if "_exps." in name else dense_type if t.ndim == 2 and "gate_inp" not in name else "F32"
    return kind if kind == "F32" or t.shape[-1] % 32 == 0 else "F32"


def as_stored(w, expert_type, dense_type):
    """The weights through the GGUF's blocks and back: the reference then computes with
    what the engine reads, and only the engine's own arithmetic (its int8 activations)
    stands between the two."""
    import numpy as np
    out = {}
    for name, t in w.items():
        if _kind(name, t, expert_type, dense_type) != "Q8_0":
            out[name] = t
            continue
        raw = np.frombuffer(_q8_0(t.float().numpy().astype(np.float32)), dtype=np.uint8).reshape(-1, 34)
        d = raw[:, :2].copy().view(np.float16).astype(np.float32)
        q = raw[:, 2:].copy().view(np.int8).astype(np.float32)
        out[name] = torch.from_numpy((q * d).reshape(t.shape)).float()
    return out


def write_gguf(path, cfg, w, expert_type="Q8_0", dense_type="F32"):
    """The weights as unsloth's muse-spark GGUFs have them: names, keys, the norms and the
    router in F32, the matrices in dense_type, the experts in expert_type."""
    import numpy as np
    U32, F32, BOOL, STR, ARR = 4, 6, 7, 8, 9
    L, V = cfg["num_hidden_layers"], cfg["vocab_size"]
    p = "muse-spark."
    kv = [("general.architecture", STR, "muse-spark"), ("general.name", STR, "muse spark tiny")]
    for k, v in (("block_count", L), ("context_length", cfg["max_position_embeddings"]),
                 ("embedding_length", cfg["hidden_size"]), ("feed_forward_length", cfg["intermediate_size"]),
                 ("attention.head_count", cfg["num_attention_heads"]),
                 ("attention.head_count_kv", cfg["num_key_value_heads"]),
                 ("expert_count", cfg["num_local_experts"]), ("expert_used_count", cfg["num_experts_per_tok"]),
                 ("attention.key_length", cfg["head_dim"]), ("attention.value_length", cfg["head_dim"]),
                 ("attention.sliding_window", cfg["sliding_window"]),
                 ("moe_latent_size", cfg["moe_hidden_size"]),
                 ("expert_feed_forward_length", cfg["moe_intermediate_size"]), ("expert_gating_func", 2)):
        kv.append((p + k, U32, int(v)))
    theta = max(cfg["layer_rope_theta"])
    kv += [(p + "rope.freq_base", F32, float(theta)), (p + "rope.freq_base_swa", F32, float(theta)),
           (p + "attention.layer_norm_rms_epsilon", F32, float(cfg["rms_norm_eps"])),
           (p + "post_norm_epsilon", F32, float(cfg["post_norm_eps"])),
           (p + "logit_scale", F32, float(cfg["output_multiplier"])),
           (p + "final_logit_softcapping", F32, float(cfg["final_logit_softcapping"])),
           (p + "expert_weights_scale", F32, float(cfg["routed_scaling_factor"])),
           (p + "expert_weights_norm", BOOL, True),
           (p + "attention.sliding_window_pattern", ARR,
            (BOOL, [t == "sliding_attention" for t in cfg["layer_types"]])),
           ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "gpt-4o"),
           ("tokenizer.ggml.tokens", ARR, (STR, [f"<0x{b:02X}>" for b in range(V)])),
           ("tokenizer.ggml.token_type", ARR, (5, [1] * V)),
           ("tokenizer.ggml.merges", ARR, (STR, [])),
           ("tokenizer.ggml.bos_token_id", U32, cfg["bos_token_id"]),
           ("tokenizer.ggml.eos_token_id", U32, cfg["eos_token_id"])]
    def rotary_order(name, t):
        """the published files' q/k rows: each head's two halves interleaved"""
        heads = {"attn_q.weight": cfg["num_attention_heads"], "attn_k.weight": cfg["num_key_value_heads"],
                 "attn_q_norm.weight": 1, "attn_k_norm.weight": 1}.get(name.split(".", 2)[-1])
        if not heads:
            return t
        hd = cfg["head_dim"]
        return t.reshape(heads, 2, hd // 2, *t.shape[1:]).transpose(1, 2).reshape(t.shape)

    tensors = []
    for name, t in w.items():
        t = rotary_order(name, t)
        a = t.float().contiguous().numpy().astype(np.float32)
        dims = list(reversed(a.shape))
        if _kind(name, t, expert_type, dense_type) == "Q8_0":
            tensors.append((name, 8, dims, _q8_0(a)))
        else:
            tensors.append((name, 0, dims, a.tobytes()))
    Path(path).write_bytes(_gguf_bytes(kv, tensors))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="a directory: model.gguf goes here")
    ap.add_argument("--emit-ref", default=None, help="write the greedy continuation here")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dense", default="F32", choices=("F32", "Q8_0"),
                    help="the GGUF's 2-D matrices (attention, projections, head): F32, or Q8_0")
    ap.add_argument("--experts", default="Q8_0", choices=("F32", "Q8_0"), help="the routed experts' GGUF type")
    ap.add_argument("--act", default=None, choices=("q8", "f32"),
                    help="the inputs of the matrices in blocks, in the reference: q8 as the integer kernels "
                         "quantize them (the CPU's, CUDA's; the default with Q8_0), f32 as Vulkan reads them")
    ap.add_argument("--min-gap", type=float, default=0.1,
                    help="draw seeds from --seed on until every greedy step's first logit beats the "
                         "second by this much, so a quantized matrix cannot flip a near tie")
    args = ap.parse_args()
    cfg = tiny_config(window=args.window)
    prompt = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    for seed in range(args.seed, args.seed + 200):
        w = as_stored(random_weights(cfg, seed), args.experts, args.dense)
        if not args.emit_ref:
            break
        with torch.no_grad():
            q8 = (args.act or ("q8" if "Q8_0" in (args.experts, args.dense) else "f32")) == "q8"
            full, gap = MuseRef(cfg, w, act_q8=q8 and args.experts == "Q8_0",
                                dense_q8=q8 and args.dense == "Q8_0").generate(prompt, args.max_new)
        if gap >= args.min_gap:
            break
    else:
        sys.exit(f"no seed in {args.seed}..{args.seed + 199} gives a margin of {args.min_gap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_gguf(out / "model.gguf", cfg, w, args.experts, args.dense)
    print(f"tiny Muse Spark -> {out / 'model.gguf'} ({sum(v.numel() for v in w.values()) / 1e6:.2f}M "
          f"parameters, experts {args.experts}, matrices {args.dense})")
    if args.emit_ref:
        Path(args.emit_ref).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_ref).write_text(json.dumps(
            {"prompt_ids": prompt, "full_ids": full, "mode": "full", "model": "muse_tiny"}, indent=2))
        print(f"ref.json -> {args.emit_ref} (seed {seed}, smallest margin {gap:.3f})\n"
              f"  prompt_ids={prompt}\n  full_ids ={full}")


if __name__ == "__main__":
    main()
