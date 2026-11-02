#!/usr/bin/env python3
"""A tiny Flint (RecursiveSI/flint, model_type "flint") with random weights, in the
checkpoint's own layout, and the greedy continuation of a prompt as a ref.json.

Flint is a Qwen3-MoE decoder with five changes (the model card and the release's
model definition, sglang/flint_sglang/model.py):

1. a learned sink per attention head (self_attn.softmax_offset): one more logit in
   the softmax's denominator, with no value;
2. sliding-window attention on the layers config.layer_types names
   "sliding_attention": a row sees itself and the sliding_window - 1 rows before it;
3. no RoPE on the layers in config.no_rope_layer_indices;
4. a dense SwiGLU MLP (intermediate_size) on config.mlp_only_layers, the routed
   experts everywhere else;
5. sigmoid routing: the top-k of sigmoid(router logits), their sigmoids renormalised
   to sum to one, plus an ungated shared expert (mlp.shared_expert).

transformers ships no Flint model, so the reference here is written from that
definition, in float32: FlintRef below. The fixture has no tokenizer; --vocab 128
keeps every id an ASCII byte for the serve harnesses, as make_qwen36_tiny.py does.

    python3 tools/make_flint_tiny.py --out flint_tiny --emit-ref flint_tiny/ref_full.json
"""
import argparse
import json
import math
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError as exc:   # pragma: no cover
    sys.exit(f"Missing deps: {exc}. Run: pip install torch safetensors")


def tiny_config(vocab=128, hidden=64, layers=7, heads=8, kv_heads=2, head_dim=16, experts=16,
                topk=4, moe_inter=32, shared_inter=32, dense_inter=96, window=5, dense=2):
    # the real pattern at toy depth: two dense full-attention layers, then sliding
    # layers with a full one every fourth, the full ones without RoPE
    types = ["full_attention", "full_attention"]
    for i in range(2, layers):
        types.append("full_attention" if (i - 1) % 4 == 0 or i == layers - 1 else "sliding_attention")
    nope = [i for i, t in enumerate(types) if t == "full_attention"]
    return {
        "architectures": ["FlintForCausalLM"], "model_type": "flint",
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 1,
        "decoder_sparse_step": 1, "eos_token_id": [vocab - 1], "head_dim": head_dim,
        "hidden_act": "silu", "hidden_size": hidden, "intermediate_size": dense_inter,
        "layer_types": types, "max_position_embeddings": 4096, "mlp_only_layers": list(range(dense)),
        "moe_intermediate_size": moe_inter, "norm_topk_prob": True,
        "num_attention_heads": heads, "num_experts": experts, "num_experts_per_tok": topk,
        "num_hidden_layers": layers, "num_key_value_heads": kv_heads,
        "no_rope_layer_indices": nope, "pad_token_id": 0, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "router_score_function": "sigmoid",
        "shared_expert_intermediate_size": shared_inter, "sliding_window": window,
        "softmax_type": "learnable", "tie_word_embeddings": False, "torch_dtype": "float32",
        "use_sliding_window": True, "vocab_size": vocab,
    }


def random_weights(cfg, seed):
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.15):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n):   # around 1, so a container that read them zero-centred could not pass
        return (1.0 + 0.2 * torch.randn(n, generator=g)).float()

    D, V, H, KV, hd = cfg["hidden_size"], cfg["vocab_size"], cfg["num_attention_heads"], \
        cfg["num_key_value_heads"], cfg["head_dim"]
    w = {"model.embed_tokens.weight": r(V, D, std=1.0), "model.norm.weight": norm(D),
         "lm_head.weight": r(V, D, std=0.3)}
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = norm(D)
        w[p + "post_attention_layernorm.weight"] = norm(D)
        w[p + "self_attn.q_proj.weight"] = r(H * hd, D)
        w[p + "self_attn.k_proj.weight"] = r(KV * hd, D)
        w[p + "self_attn.v_proj.weight"] = r(KV * hd, D)
        w[p + "self_attn.o_proj.weight"] = r(D, H * hd)
        w[p + "self_attn.q_norm.weight"] = norm(hd)
        w[p + "self_attn.k_norm.weight"] = norm(hd)
        w[p + "self_attn.softmax_offset"] = r(H, std=1.0)
        if i in cfg["mlp_only_layers"]:
            F = cfg["intermediate_size"]
            w[p + "mlp.gate_proj.weight"] = r(F, D)
            w[p + "mlp.up_proj.weight"] = r(F, D)
            w[p + "mlp.down_proj.weight"] = r(D, F)
            continue
        E, F, SF = cfg["num_experts"], cfg["moe_intermediate_size"], cfg["shared_expert_intermediate_size"]
        w[p + "mlp.gate.weight"] = r(E, D, std=0.5)
        for e in range(E):
            w[p + f"mlp.experts.{e}.gate_proj.weight"] = r(F, D)
            w[p + f"mlp.experts.{e}.up_proj.weight"] = r(F, D)
            w[p + f"mlp.experts.{e}.down_proj.weight"] = r(D, F)
        w[p + "mlp.shared_expert.gate_proj.weight"] = r(SF, D)
        w[p + "mlp.shared_expert.up_proj.weight"] = r(SF, D)
        w[p + "mlp.shared_expert.down_proj.weight"] = r(D, SF)
    return w


class FlintRef:
    """Flint's forward over a whole sequence, float32, no cache (the fixture is tiny)."""

    def __init__(self, cfg, w):
        self.c, self.w = cfg, w

    def rms(self, x, weight):
        eps = self.c["rms_norm_eps"]
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

    def rope(self, x, pos):   # x [T, heads, hd], rotate_half over the whole head
        hd = x.shape[-1]
        inv = 1.0 / (self.c["rope_theta"] ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
        ang = pos[:, None].double() * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def attention(self, i, h):
        c, w, p = self.c, self.w, f"model.layers.{i}.self_attn."
        T = h.shape[0]
        H, KV, hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        q = (h @ w[p + "q_proj.weight"].T).view(T, H, hd)
        k = (h @ w[p + "k_proj.weight"].T).view(T, KV, hd)
        v = (h @ w[p + "v_proj.weight"].T).view(T, KV, hd)
        q, k = self.rms(q, w[p + "q_norm.weight"]), self.rms(k, w[p + "k_norm.weight"])
        if i not in c["no_rope_layer_indices"]:
            pos = torch.arange(T)
            q, k = self.rope(q, pos), self.rope(k, pos)
        sliding = c["layer_types"][i] == "sliding_attention"
        t = torch.arange(T)
        allowed = t[None, :] <= t[:, None]
        if sliding:
            allowed &= t[None, :] > t[:, None] - c["sliding_window"]
        sink = w[p + "softmax_offset"].float()
        out = torch.empty(T, H, hd)
        rep = H // KV
        for hh in range(H):
            s = (q[:, hh, :] @ k[:, hh // rep, :].T) / math.sqrt(hd)
            s = s.masked_fill(~allowed, float("-inf"))
            m = torch.maximum(s.max(-1, keepdim=True).values, sink[hh].expand(T, 1))
            e = torch.exp(s - m)
            den = e.sum(-1, keepdim=True) + torch.exp(sink[hh] - m)
            out[:, hh, :] = (e / den) @ v[:, hh // rep, :]
        return out.reshape(T, H * hd) @ w[p + "o_proj.weight"].T

    def swiglu(self, x, pre):
        w = self.w
        g, u = x @ w[pre + "gate_proj.weight"].T, x @ w[pre + "up_proj.weight"].T
        return (torch.nn.functional.silu(g) * u) @ w[pre + "down_proj.weight"].T

    def mlp(self, i, h):
        c, w, p = self.c, self.w, f"model.layers.{i}.mlp."
        if i in c["mlp_only_layers"]:
            return self.swiglu(h, p)
        scores = torch.sigmoid(h @ w[p + "gate.weight"].T)
        K = c["num_experts_per_tok"]
        top = torch.topk(scores, K, dim=-1).indices
        wts = scores.gather(1, top)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20)
        out = torch.zeros_like(h)
        for r in range(h.shape[0]):
            for j in range(K):
                e = int(top[r, j])
                out[r] += wts[r, j] * self.swiglu(h[r:r + 1], p + f"experts.{e}.")[0]
        return out + self.swiglu(h, p + "shared_expert.")

    def logits(self, ids):
        w, c = self.w, self.c
        x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        for i in range(c["num_hidden_layers"]):
            x = x + self.attention(i, self.rms(x, w[f"model.layers.{i}.input_layernorm.weight"]))
            x = x + self.mlp(i, self.rms(x, w[f"model.layers.{i}.post_attention_layernorm.weight"]))
        return self.rms(x, w["model.norm.weight"]) @ w["lm_head.weight"].T

    def generate(self, prompt, n):
        """The greedy continuation and the smallest margin between the first and the
        second logit over its steps."""
        ids, gap = list(prompt), float("inf")
        for _ in range(n):
            top = self.logits(ids)[-1].topk(2)
            gap = min(gap, float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
        return ids, gap


def _gguf_bytes(kv, tensors, alignment=32):
    """A GGUF v3 file: kv is [(key, type, value)], tensors [(name, ggml_type, dims, payload)];
    dims in ggml order (the row length first). Written here so the fixture needs no package."""
    import struct

    def string(text):
        data = text.encode("utf-8")
        return struct.pack("<Q", len(data)) + data

    def value(vt, v):
        if vt == 8:
            return string(v)
        if vt == 7:
            return b"\x01" if v else b"\x00"
        if vt == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(value(et, item) for item in items)
        return struct.pack({4: "<I", 5: "<i", 6: "<f"}[vt], v)

    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for key, vt, v in kv:
        out += string(key) + struct.pack("<I", vt) + value(vt, v)
    align = lambda n: (n + alignment - 1) // alignment * alignment
    info, offset = b"", 0
    for name, gtype, dims, payload in tensors:
        info += string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        info += struct.pack("<IQ", gtype, offset)
        offset = align(offset + len(payload))
    out += info
    out += b"\x00" * (align(len(out)) - len(out))
    for _name, _gtype, _dims, payload in tensors:
        out += payload + b"\x00" * (align(len(payload)) - len(payload))
    return out


def _q8_0(a):
    """ggml's Q8_0: blocks of 32, an f16 scale max|x|/127 and 32 int8 (round half away)."""
    import numpy as np
    blocks = a.reshape(-1, 32).astype(np.float32)
    d = np.abs(blocks).max(axis=1) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0)
    q = np.clip(np.trunc(blocks * inv[:, None] + np.copysign(0.5, blocks)), -127, 127).astype(np.int8)
    out = np.empty((blocks.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def write_gguf(path, cfg, w, expert_type="Q8_0", dense_type="F32"):
    """The same weights as llama.cpp's flint GGUF has them (the names, keys and per-layer
    patterns of unsloth/flint-GGUF): F32 everywhere, the routed experts in expert_type
    (F32 or Q8_0)."""
    import numpy as np
    U32, F32, BOOL, STR, ARR = 4, 6, 7, 8, 9
    L, D, E, V = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_experts"], cfg["vocab_size"]
    p = "flint."
    kv = [("general.architecture", STR, "flint"), ("general.name", STR, "flint tiny")]
    for k, v in (("block_count", L), ("context_length", cfg["max_position_embeddings"]),
                 ("embedding_length", D), ("feed_forward_length", cfg["intermediate_size"]),
                 ("attention.head_count", cfg["num_attention_heads"]),
                 ("attention.head_count_kv", cfg["num_key_value_heads"]),
                 ("attention.key_length", cfg["head_dim"]), ("attention.value_length", cfg["head_dim"]),
                 ("expert_count", E), ("expert_used_count", cfg["num_experts_per_tok"]),
                 ("expert_gating_func", 2), ("expert_feed_forward_length", cfg["moe_intermediate_size"]),
                 ("expert_shared_feed_forward_length", cfg["shared_expert_intermediate_size"]),
                 ("leading_dense_block_count", len(cfg["mlp_only_layers"])), ("expert_shared_count", 1),
                 ("attention.sliding_window", cfg["sliding_window"])):
        kv.append((p + k, U32, int(v)))
    kv += [(p + "rope.freq_base", F32, float(cfg["rope_theta"])),
           (p + "attention.layer_norm_rms_epsilon", F32, float(cfg["rms_norm_eps"])),
           (p + "expert_weights_norm", BOOL, True), (p + "expert_weights_scale", F32, 1.0),
           (p + "attention.sliding_window_pattern", ARR,
            (BOOL, [t == "sliding_attention" for t in cfg["layer_types"]])),
           (p + "attention.rope_pattern", ARR, (BOOL, [i not in cfg["no_rope_layer_indices"] for i in range(L)])),
           ("tokenizer.ggml.model", STR, "gpt2"),
           ("tokenizer.ggml.tokens", ARR, (STR, [f"<0x{b:02X}>" for b in range(V)])),
           ("tokenizer.ggml.token_type", ARR, (5, [1] * V)),
           ("tokenizer.ggml.merges", ARR, (STR, [])),
           ("tokenizer.ggml.bos_token_id", U32, cfg["bos_token_id"]),
           ("tokenizer.ggml.eos_token_id", U32, cfg["eos_token_id"][0])]
    tensors = []

    def put(name, t, quant=False):
        a = t.float().contiguous().numpy().astype(np.float32)
        dims = list(reversed(a.shape))
        if not quant and dense_type == "Q8_0" and a.ndim == 2 and a.shape[-1] % 32 == 0:
            quant = True   # the dense matrices in blocks too, as a published file has them
        if quant and (expert_type == "Q8_0" or a.ndim == 2):
            tensors.append((name, 8, dims, _q8_0(a)))
        else:
            tensors.append((name, 0, dims, a.tobytes()))

    put("token_embd.weight", w["model.embed_tokens.weight"])
    put("output.weight", w["lm_head.weight"])
    put("output_norm.weight", w["model.norm.weight"])
    names = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight",
             "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
             "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
             "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
             "self_attn.softmax_offset": "attn_sinks.weight", "mlp.gate_proj.weight": "ffn_gate.weight",
             "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
             "mlp.gate.weight": "ffn_gate_inp.weight",
             "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
             "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
             "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight"}
    for i in range(L):
        for hf, gg in names.items():
            key = f"model.layers.{i}.{hf}"
            if key in w:
                put(f"blk.{i}.{gg}", w[key])
        if i in cfg["mlp_only_layers"]:
            continue
        for proj, gg in (("gate_proj", "ffn_gate_exps"), ("up_proj", "ffn_up_exps"), ("down_proj", "ffn_down_exps")):
            stack = torch.stack([w[f"model.layers.{i}.mlp.experts.{e}.{proj}.weight"] for e in range(E)])
            put(f"blk.{i}.{gg}.weight", stack, quant=True)
    Path(path).write_bytes(_gguf_bytes(kv, tensors))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--emit-ref", default=None, help="write the greedy continuation here")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--vocab", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--dense", type=int, default=2, help="dense layers in front (the release: 2)")
    ap.add_argument("--gguf", default=None, help="also write the model as a flint GGUF here")
    ap.add_argument("--gguf-dense", default="F32", choices=("F32", "Q8_0"),
                    help="the GGUF's 2-D dense matrices (attention, MLPs, router, head): F32, or Q8_0 as the published files")
    ap.add_argument("--gguf-experts", default="Q8_0", choices=("F32", "Q8_0"),
                    help="the routed experts' GGUF type")
    ap.add_argument("--min-gap", type=float, default=0.1,
                    help="draw seeds from --seed on until every greedy step's first logit beats the "
                         "second by this much: the container's int8 experts move the logits by a few "
                         "thousandths, and a near tie would fail the gate for no fault")
    args = ap.parse_args()
    cfg = tiny_config(vocab=args.vocab, hidden=args.hidden, window=args.window, dense=args.dense)
    prompt = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    for seed in range(args.seed, args.seed + 200):
        w = random_weights(cfg, seed)
        if not args.emit_ref:
            break
        with torch.no_grad():
            full, gap = FlintRef(cfg, w).generate(prompt, args.max_new)
        if gap >= args.min_gap:
            break
    else:
        sys.exit(f"no seed in {args.seed}..{args.seed + 199} gives a margin of {args.min_gap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    save_file({k: v.contiguous() for k, v in w.items()}, str(out / "model.safetensors"),
              metadata={"format": "pt"})
    print(f"tiny Flint -> {out} ({sum(v.numel() for v in w.values()) / 1e6:.2f}M parameters)")
    if args.gguf:
        write_gguf(args.gguf, cfg, w, args.gguf_experts, args.gguf_dense)
        print(f"GGUF -> {args.gguf} (experts {args.gguf_experts})")
    if args.emit_ref:
        Path(args.emit_ref).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_ref).write_text(json.dumps(
            {"prompt_ids": prompt, "full_ids": full, "mode": "full", "model": "flint_tiny"}, indent=2))
        print(f"ref.json -> {args.emit_ref} (seed {seed}, smallest margin {gap:.3f})\n"
              f"  prompt_ids={prompt}\n  full_ids ={full}")


if __name__ == "__main__":
    main()
