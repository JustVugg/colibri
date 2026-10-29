#!/usr/bin/env python3
"""A tiny Mistral-Large-4 (mistralai/Mistral-Large-4-1T-A52B's text model, model_type
"mistral_large4_text") with random weights, as a mistral-large4 GGUF with the tensor names and
keys of the published files (unsloth/Mistral-Large-4-1T-A52B-GGUF), and the greedy
continuation of a prompt as a ref.json.

The text model, from the release's modeling_mistral_large4.py:

1. a layer: RMSNorm, attention, residual; RMSNorm, MLP (the first layer dense) or MoE
   (softmax router, top-k renormalised, times routed_scaling_factor, plus a shared expert),
   residual; then the final RMSNorm, the head and a softcap: cap * tanh(logits / cap);
2. attention is GLM-5.2's DSA without the query LoRA: q_proj straight from the input, the KV
   latent from kv_a (an RMSNorm on its kv_lora part, the rest the shared rotary key), expanded
   by kv_b into each head's key and value. A layer whose kv_source is another reads that
   layer's latent (its keys already rotated) and has no kv_a of its own. NoPE layers
   (rope_pattern false) have no rotary part: the query head is nope + rope wide, all of it
   unrotated, the latent has no rotary key;
3. RoPE on interleaved pairs, YaRN's frequencies (factor, original context, beta fast and
   slow) times exp(alpha), alpha learned: the GGUF's rope_freqs.weight is exp(-alpha), and the
   frequency is YaRN's divided by it. No attention scaling: attention_factor 1, softmax at
   (nope + rope)^-0.5;
4. SSMax: every head's query times s * log(position + 1) + b, its own s and b (attn_ssmax);
5. the indexer (a layer whose indexer type is "full"): q = wq_b(LayerNorm(x)) times its own
   SSMax, k = LayerNorm(wk(x)), both rotated on their first rope dims on RoPE layers, the
   scores relu(q . k / sqrt(index_head_dim)) summed over the index heads with the weights
   weights_proj(x) / sqrt(index heads), causal; its top index_topk keys are the ones the
   attention reads. A "shared" layer reads the last full layer's selection.

    python3 tools/make_mistral_large4_tiny.py --out ml4_tiny --emit-ref ml4_tiny/ref.json
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


def tiny_config():
    # the release's patterns at toy depth: a dense first layer, every fourth layer NoPE, the
    # last NoPE layer reading the first NoPE layer's latent, full and shared indexers
    return {
        "vocab": 128, "hidden": 64, "heads": 4, "kv_lora": 32, "nope": 16, "rope": 16, "v_head": 32,
        "dense_inter": 96, "moe_inter": 32, "experts": 8, "topk": 2, "shared": 1, "dense_layers": 1,
        "index_heads": 2, "index_head_dim": 32, "index_topk": 6,
        "rope_pattern": [True, True, True, False, True, True, True, False],
        "kv_source": [0, 1, 2, 3, 4, 5, 6, 3],
        "indexer_full": [True, True, True, True, False, False, True, False],
        "theta": 10000.0, "yarn_factor": 4.0, "yarn_orig": 16, "beta_fast": 32.0, "beta_slow": 1.0,
        "eps": 1e-5, "ln_eps": 1e-6, "softcap": 8.0, "routed_scale": 1.0,
        "bos": 1, "eos": 127, "context": 4096,
    }


def yarn_inv_freq(c):
    """transformers' _compute_yarn_parameters (truncate on), dim = the rotary head dim."""
    dim, base, factor = c["rope"], c["theta"], c["yarn_factor"]

    def corr(rot):
        return (dim * math.log(c["yarn_orig"] / (rot * 2 * math.pi))) / (2 * math.log(base))
    low, high = max(math.floor(corr(c["beta_fast"])), 0), min(math.ceil(corr(c["beta_slow"])), dim - 1)
    if low == high:
        high += 0.001
    ramp = torch.clamp((torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
    pos = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    extra = 1 - ramp
    return (1.0 / (factor * pos)) * (1 - extra) + (1.0 / pos) * extra


def random_weights(c, seed):
    """The weights under the GGUF's names; kv_b as the checkpoint has it ([H, nope' + v, L],
    nope' the layer's key width without rotary part) beside the GGUF's k_b and v_b."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, std=0.15):
        return (torch.randn(*shape, generator=g) * std).float()

    def norm(n, around=1.0):
        return (around + 0.2 * torch.randn(n, generator=g)).float()

    D, V, H, L = c["hidden"], c["vocab"], c["heads"], c["kv_lora"]
    NI, IH, ID = c["index_heads"], c["index_head_dim"], c["moe_inter"]
    w = {"token_embd.weight": r(V, D, std=1.0), "output_norm.weight": norm(D), "output.weight": r(V, D, std=0.3),
         "alpha": r(c["rope"] // 2, std=0.3)}
    for i in range(len(c["rope_pattern"])):
        p, rope = f"blk.{i}.", c["rope_pattern"][i]
        nope = c["nope"] + (0 if rope else c["rope"])
        w[p + "attn_norm.weight"], w[p + "ffn_norm.weight"] = norm(D), norm(D)
        w[p + "attn_q.weight"] = r(H * (c["nope"] + c["rope"]), D)
        if c["kv_source"][i] == i:
            w[p + "attn_kv_a_mqa.weight"] = r(L + (c["rope"] if rope else 0), D, std=0.3)
            w[p + "attn_kv_a_norm.weight"] = norm(L)
        kvb = r(H, nope + c["v_head"], L, std=0.3)
        w[p + "kv_b"] = kvb
        w[p + "attn_k_b.weight"] = kvb[:, :nope, :].transpose(1, 2).contiguous()   # [H][L][nope]
        w[p + "attn_v_b.weight"] = kvb[:, nope:, :].contiguous()                     # [H][V][L]
        w[p + "attn_output.weight"] = r(D, H * c["v_head"])
        w[p + "attn_ssmax.weight"] = r(H, std=0.2)
        w[p + "attn_ssmax.bias"] = norm(H)
        if c["indexer_full"][i]:
            w[p + "indexer.q_norm.weight"], w[p + "indexer.q_norm.bias"] = norm(D), r(D, std=0.1)
            w[p + "indexer.attn_q_b.weight"] = r(NI * IH, D)
            w[p + "indexer.attn_k.weight"] = r(IH, D)
            w[p + "indexer.k_norm.weight"], w[p + "indexer.k_norm.bias"] = norm(IH), r(IH, std=0.1)
            w[p + "indexer.proj.weight"] = r(NI, D, std=0.5)
            w[p + "indexer.ssmax.weight"] = r(NI, std=0.2)
            w[p + "indexer.ssmax.bias"] = norm(NI)
        if i < c["dense_layers"]:
            F = c["dense_inter"]
            w[p + "ffn_gate.weight"], w[p + "ffn_up.weight"], w[p + "ffn_down.weight"] = r(F, D), r(F, D), r(D, F)
        else:
            E, S = c["experts"], ID * c["shared"]
            w[p + "ffn_gate_inp.weight"] = r(E, D, std=0.5)
            w[p + "ffn_gate_exps.weight"], w[p + "ffn_up_exps.weight"] = r(E, ID, D), r(E, ID, D)
            w[p + "ffn_down_exps.weight"] = r(E, D, ID)
            w[p + "ffn_gate_shexp.weight"], w[p + "ffn_up_shexp.weight"] = r(S, D), r(S, D)
            w[p + "ffn_down_shexp.weight"] = r(D, S)
    return w


class Ref:
    """The text forward over a whole sequence, float32, no cache."""

    def __init__(self, c, w):
        self.c, self.w = c, w
        self.inv = yarn_inv_freq(c) * torch.exp(w["alpha"])

    def rms(self, x, weight):
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.c["eps"]))

    def ln(self, x, weight, bias):
        return torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, self.c["ln_eps"])

    def rot(self, x, T):   # x [T, heads, d]: interleaved pairs, the angle of pair j at position t
        ang = torch.arange(T, dtype=torch.float32)[:, None] * self.inv[None, :]
        cos, sin = ang.cos()[:, None, :], ang.sin()[:, None, :]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)

    def ssmax(self, q, s, b):   # q [T, heads, d]
        lp = torch.log(torch.arange(q.shape[0], dtype=torch.float32) + 1)
        return q * (s[None, :] * lp[:, None] + b[None, :])[:, :, None]

    def indexer(self, i, x, T):
        c, w, p = self.c, self.w, f"blk.{i}.indexer."
        NI, IH, R = c["index_heads"], c["index_head_dim"], c["rope"]
        q = (self.ln(x, w[p + "q_norm.weight"], w[p + "q_norm.bias"]) @ w[p + "attn_q_b.weight"].T).view(T, NI, IH)
        q = self.ssmax(q, w[p + "ssmax.weight"], w[p + "ssmax.bias"])
        k = self.ln(x @ w[p + "attn_k.weight"].T, w[p + "k_norm.weight"], w[p + "k_norm.bias"]).view(T, 1, IH)
        if c["rope_pattern"][i]:
            q = torch.cat([self.rot(q[..., :R], T), q[..., R:]], -1)
            k = torch.cat([self.rot(k[..., :R], T), k[..., R:]], -1)
        scores = torch.relu(torch.einsum("thd,sd->ths", q, k[:, 0, :]) / math.sqrt(IH))
        weights = (x @ w[p + "proj.weight"].T) / math.sqrt(NI)
        idx = torch.einsum("th,ths->ts", weights, scores)
        t = torch.arange(T)
        idx = idx.masked_fill(t[None, :] > t[:, None], float("-inf"))
        # ties (scores of exactly 0 when every head's relu is 0, frequent at this size) to the
        # earlier position, as the engine breaks them; torch.topk leaves their order open
        return torch.sort(-idx, dim=-1, stable=True).indices[:, :min(c["index_topk"], T)]

    def attention(self, i, x, latents, prev_sel):
        c, w, p = self.c, self.w, f"blk.{i}."
        T, H, R, L, V = x.shape[0], c["heads"], c["rope"], c["kv_lora"], c["v_head"]
        rope = c["rope_pattern"][i]
        nope = c["nope"] + (0 if rope else R)
        q = (x @ w[p + "attn_q.weight"].T).view(T, H, c["nope"] + R)
        if c["kv_source"][i] == i:
            kv = x @ w[p + "attn_kv_a_mqa.weight"].T
            lat = self.rms(kv[:, :L], w[p + "attn_kv_a_norm.weight"])
            kr = self.rot(kv[:, L:].view(T, 1, R), T) if rope else None
            latents[i] = (lat, kr)
        else:
            lat, kr = latents[c["kv_source"][i]]
        if rope:
            q = torch.cat([q[..., :nope], self.rot(q[..., nope:], T)], -1)
        q = self.ssmax(q, w[p + "attn_ssmax.weight"], w[p + "attn_ssmax.bias"])
        kvb = w[p + "kv_b"]                                   # [H, nope + V, L]
        kn = torch.einsum("hdl,tl->thd", kvb[:, :nope, :], lat)
        v = torch.einsum("hvl,tl->thv", kvb[:, nope:, :], lat)
        k = torch.cat([kn, kr.expand(T, H, R)], -1) if rope else kn
        sel = self.indexer(i, x, T) if c["indexer_full"][i] else prev_sel
        allowed = torch.zeros(T, T, dtype=torch.bool)
        allowed.scatter_(1, sel, True)
        t = torch.arange(T)
        allowed &= t[None, :] <= t[:, None]
        s = torch.einsum("thd,shd->hts", q, k) / math.sqrt(c["nope"] + R)
        s = s.masked_fill(~allowed[None], float("-inf"))
        o = torch.einsum("hts,shv->thv", torch.softmax(s, -1), v).reshape(T, H * V)
        return o @ w[p + "attn_output.weight"].T, sel

    def mlp(self, i, x):
        c, w, p = self.c, self.w, f"blk.{i}."
        silu = torch.nn.functional.silu
        if i < c["dense_layers"]:
            return (silu(x @ w[p + "ffn_gate.weight"].T) * (x @ w[p + "ffn_up.weight"].T)) @ w[p + "ffn_down.weight"].T
        probs = torch.softmax(x @ w[p + "ffn_gate_inp.weight"].T, -1)
        top = probs.topk(c["topk"], -1)
        wts = top.values / (top.values.sum(-1, keepdim=True) + 1e-20) * c["routed_scale"]
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(c["topk"]):
                e = int(top.indices[t, j])
                h = silu(w[p + "ffn_gate_exps.weight"][e] @ x[t]) * (w[p + "ffn_up_exps.weight"][e] @ x[t])
                out[t] += wts[t, j] * (w[p + "ffn_down_exps.weight"][e] @ h)
        sh = (silu(x @ w[p + "ffn_gate_shexp.weight"].T) * (x @ w[p + "ffn_up_shexp.weight"].T)) @ w[p + "ffn_down_shexp.weight"].T
        return out + sh

    def logits(self, ids):
        c, w = self.c, self.w
        x = w["token_embd.weight"][torch.tensor(ids)]
        latents, sel = {}, None
        for i in range(len(c["rope_pattern"])):
            a, sel = self.attention(i, self.rms(x, w[f"blk.{i}.attn_norm.weight"]), latents, sel)
            x = x + a
            x = x + self.mlp(i, self.rms(x, w[f"blk.{i}.ffn_norm.weight"]))
        z = self.rms(x, w["output_norm.weight"]) @ w["output.weight"].T
        return c["softcap"] * torch.tanh(z / c["softcap"])

    def generate(self, prompt, n):
        ids, gap = list(prompt), float("inf")
        for _ in range(n):
            top = self.logits(ids)[-1].topk(2)
            gap = min(gap, float(top.values[0] - top.values[1]))
            ids.append(int(top.indices[0]))
        return ids, gap


def write_gguf(path, c, w, dense_type="F32"):
    """The weights as unsloth's mistral-large4 GGUFs have them: the names, the keys (the rotary
    pattern, the KV sources and the indexer types by layer), the rope's exp(-alpha)."""
    import numpy as np
    U32, I32, F32, BOOL, STR, ARR = 4, 5, 6, 7, 8, 9
    p, n = "mistral-large4.", len(c["rope_pattern"])
    kv = [("general.architecture", STR, "mistral-large4"), ("general.name", STR, "mistral-large4 tiny")]
    for k, v in (("block_count", n), ("context_length", c["context"]), ("embedding_length", c["hidden"]),
                 ("feed_forward_length", c["dense_inter"]), ("attention.head_count", c["heads"]),
                 ("attention.head_count_kv", 1), ("expert_count", c["experts"]), ("expert_used_count", c["topk"]),
                 ("expert_group_count", 1), ("expert_group_used_count", 1), ("expert_gating_func", 1),
                 ("attention.key_length", c["kv_lora"] + c["rope"]), ("attention.value_length", c["kv_lora"]),
                 ("leading_dense_block_count", c["dense_layers"]), ("vocab_size", c["vocab"]),
                 ("attention.kv_lora_rank", c["kv_lora"]), ("attention.key_length_mla", c["nope"] + c["rope"]),
                 ("attention.value_length_mla", c["v_head"]), ("expert_feed_forward_length", c["moe_inter"]),
                 ("expert_shared_count", c["shared"]), ("rope.dimension_count", c["rope"]),
                 ("rope.scaling.original_context_length", c["yarn_orig"]),
                 ("attention.indexer.head_count", c["index_heads"]), ("attention.indexer.key_length", c["index_head_dim"]),
                 ("attention.indexer.top_k", c["index_topk"])):
        kv.append((p + k, U32, int(v)))
    kv += [(p + "rope.freq_base", F32, c["theta"]), (p + "attention.layer_norm_rms_epsilon", F32, c["eps"]),
           (p + "attention.layer_norm_epsilon", F32, c["ln_eps"]),
           (p + "expert_weights_scale", F32, c["routed_scale"]), (p + "expert_weights_norm", BOOL, True),
           (p + "rope.scaling.type", STR, "yarn"), (p + "rope.scaling.factor", F32, c["yarn_factor"]),
           (p + "rope.scaling.yarn_beta_fast", F32, c["beta_fast"]), (p + "rope.scaling.yarn_beta_slow", F32, c["beta_slow"]),
           (p + "rope.scaling.yarn_log_multiplier", F32, 1.0),
           (p + "attention.indexer.types", ARR, (BOOL, list(c["indexer_full"]))),
           (p + "attention.rope_pattern", ARR, (BOOL, list(c["rope_pattern"]))),
           (p + "attention.kv_source_map", ARR, (I32, list(c["kv_source"]))),
           (p + "final_logit_softcapping", F32, c["softcap"]),
           ("tokenizer.ggml.model", STR, "gpt2"), ("tokenizer.ggml.pre", STR, "tekken-num3"),
           ("tokenizer.ggml.tokens", ARR, (STR, [f"<0x{b:02X}>" for b in range(c["vocab"])])),
           ("tokenizer.ggml.token_type", ARR, (I32, [1] * c["vocab"])),
           ("tokenizer.ggml.merges", ARR, (STR, [])),
           ("tokenizer.ggml.bos_token_id", U32, c["bos"]), ("tokenizer.ggml.eos_token_id", U32, c["eos"])]
    tensors = [("rope_freqs.weight", 0, [c["rope"] // 2], torch.exp(-w["alpha"]).numpy().astype(np.float32).tobytes())]
    for name, t in w.items():
        if name in ("alpha",) or name.endswith("kv_b"):
            continue
        a = t.float().contiguous().numpy().astype(np.float32)
        dims = list(reversed(a.shape))
        if dense_type == "Q8_0" and a.ndim >= 2 and a.shape[-1] % 32 == 0 and "norm" not in name and "ssmax" not in name \
                and "gate_inp" not in name and "indexer.proj" not in name:
            tensors.append((name, 8, dims, _q8_0(a)))
        else:
            tensors.append((name, 0, dims, a.tobytes()))
    Path(path).write_bytes(_gguf_bytes(kv, tensors))


# the tensors the release ships in FP8 (128 x 128 blocks, qscale_weight multiplied); the rest
# (kv_a, kv_b, the indexer's wk and weights_proj, the embeddings, the head) as the release
# keeps them, here in float32
FP8_KINDS = ("attn_q.weight", "attn_output.weight", "indexer.attn_q_b.weight", "ffn_gate.weight", "ffn_up.weight",
             "ffn_down.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
             "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")


def fp8_roundtrip(t):
    """t as its FP8 e4m3 codes times a scale per 128 x 128 block (amax / 448): (codes, scales, values)"""
    O, I = t.shape[-2], t.shape[-1]
    bo, bi = (O + 127) // 128, (I + 127) // 128
    flat = t.reshape(-1, O, I)
    codes, scales, vals = [], [], []
    for m in flat:
        sc = torch.zeros(bo, bi)
        for a in range(bo):
            for b in range(bi):
                blk = m[a * 128:(a + 1) * 128, b * 128:(b + 1) * 128]
                sc[a, b] = max(float(blk.abs().max()), 1e-12) / 448.0
        S = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:O, :I]
        q = (m / S).to(torch.float8_e4m3fn)
        codes.append(q); scales.append(sc); vals.append(q.to(torch.float32) * S)
    shape = t.shape
    return torch.stack(codes).reshape(shape), torch.stack(scales).reshape(*shape[:-2], bo, bi), torch.stack(vals).reshape(shape)


def as_fp8(w):
    """The weights the FP8 release would give back: those tensors through their FP8 round trip"""
    out = dict(w)
    for name, t in w.items():
        if any(name.endswith(k) for k in FP8_KINDS):
            out[name] = fp8_roundtrip(t)[2]
    return out


def write_fp8_checkpoint(path, c, w):
    """The model as Mistral's FP8 release has it: consolidated.safetensors in Mistral's names
    (layers.N.attention.wq.weight + .qscale_weight, experts.E.w1/w2/w3, ...), config.json with a
    text_config. tools/convert_fp8_to_int4.py reads it into a container."""
    from safetensors.torch import save_file
    path = Path(path); path.mkdir(parents=True, exist_ok=True)
    t, n = {}, len(c["rope_pattern"])

    def put(name, tensor, fp8=False):
        if fp8:
            q, sc, _ = fp8_roundtrip(tensor)
            t[name] = q.contiguous()
            t[name[:-len("weight")] + "qscale_weight"] = sc.contiguous().float()
        else:
            t[name] = tensor.contiguous().float()
    put("tok_embeddings.weight", w["token_embd.weight"]); put("norm.weight", w["output_norm.weight"])
    put("output.weight", w["output.weight"]); put("lerope.alpha", w["alpha"])
    for i in range(n):
        g, m = f"blk.{i}.", f"layers.{i}."
        put(m + "attention_norm.weight", w[g + "attn_norm.weight"]); put(m + "ffn_norm.weight", w[g + "ffn_norm.weight"])
        put(m + "attention.wq.weight", w[g + "attn_q.weight"], True)
        if g + "attn_kv_a_mqa.weight" in w:
            put(m + "attention.wkv_a_with_mqa.weight", w[g + "attn_kv_a_mqa.weight"])
            put(m + "attention.kv_a_norm.weight", w[g + "attn_kv_a_norm.weight"])
        kvb = w[g + "kv_b"]
        put(m + "attention.wkv_b.weight", kvb.reshape(-1, kvb.shape[-1]))
        put(m + "attention.wo.weight", w[g + "attn_output.weight"], True)
        put(m + "attention.ssmax_scaling.s", w[g + "attn_ssmax.weight"]); put(m + "attention.ssmax_scaling.b", w[g + "attn_ssmax.bias"])
        if g + "indexer.attn_q_b.weight" in w:
            x = m + "attention.indexer."
            put(x + "q_input_norm.weight", w[g + "indexer.q_norm.weight"]); put(x + "q_input_norm.bias", w[g + "indexer.q_norm.bias"])
            put(x + "wq_b.weight", w[g + "indexer.attn_q_b.weight"], True)
            put(x + "wk.weight", w[g + "indexer.attn_k.weight"])
            put(x + "k_norm.weight", w[g + "indexer.k_norm.weight"]); put(x + "k_norm.bias", w[g + "indexer.k_norm.bias"])
            put(x + "weights_proj.weight", w[g + "indexer.proj.weight"])
            put(x + "ssmax_scaling.s", w[g + "indexer.ssmax.weight"]); put(x + "ssmax_scaling.b", w[g + "indexer.ssmax.bias"])
        if g + "ffn_gate.weight" in w:
            for ours, theirs in (("ffn_gate", "w1"), ("ffn_down", "w2"), ("ffn_up", "w3")):
                put(m + f"feed_forward.{theirs}.weight", w[g + ours + ".weight"], True)
        else:
            put(m + "gate.weight", w[g + "ffn_gate_inp.weight"])
            for ours, theirs in (("ffn_gate", "w1"), ("ffn_down", "w2"), ("ffn_up", "w3")):
                for e in range(c["experts"]):
                    put(m + f"experts.{e}.{theirs}.weight", w[g + ours + "_exps.weight"][e], True)
                put(m + f"shared_experts.{theirs}.weight", w[g + ours + "_shexp.weight"], True)
    save_file(t, str(path / "consolidated-00001-of-00001.safetensors"))
    text = {"model_type": "mistral_large4_text", "num_hidden_layers": n, "vocab_size": c["vocab"], "hidden_size": c["hidden"],
            "intermediate_size": c["dense_inter"], "moe_intermediate_size": c["moe_inter"], "num_attention_heads": c["heads"],
            "num_key_value_heads": c["heads"], "n_routed_experts": c["experts"], "num_experts_per_tok": c["topk"],
            "n_shared_experts": c["shared"], "first_k_dense_replace": c["dense_layers"], "kv_lora_rank": c["kv_lora"],
            "qk_nope_head_dim": c["nope"], "qk_rope_head_dim": c["rope"], "v_head_dim": c["v_head"], "head_dim": c["rope"],
            "index_topk": c["index_topk"], "index_n_heads": c["index_heads"], "index_head_dim": c["index_head_dim"],
            "indexer_types": ["full" if f else "shared" for f in c["indexer_full"]],
            "no_rope_layers": [1 if r else 0 for r in c["rope_pattern"]], "kv_source_layers": list(c["kv_source"]),
            "final_logit_softcapping": c["softcap"], "rms_norm_eps": c["eps"], "routed_scaling_factor": c["routed_scale"],
            "rope_parameters": {"rope_type": "yarn", "rope_theta": c["theta"], "factor": c["yarn_factor"],
                                "original_max_position_embeddings": c["yarn_orig"], "beta_fast": c["beta_fast"],
                                "beta_slow": c["beta_slow"], "attention_factor": 1.0},
            "bos_token_id": c["bos"], "eos_token_id": c["eos"], "n_group": 1, "topk_group": 1}
    (path / "config.json").write_text(json.dumps({"model_type": "mistral_large4", "text_config": text,
                                                  "tie_word_embeddings": False}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="a directory: model.gguf goes here")
    ap.add_argument("--emit-ref", default=None, help="write the greedy continuation here")
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--prompt-ids", default="1,7,12,30,44,5,9,61,18,22,3,40")
    ap.add_argument("--fp8-checkpoint", default=None,
                    help="also write the model as Mistral's FP8 release (consolidated safetensors in its names, "
                         "config.json) to this directory; the reference is then the FP8 weights'")
    ap.add_argument("--override", default=None,
                    help="JSON object over the tiny config's keys (a feature off, to find where two forwards part)")
    ap.add_argument("--min-gap", type=float, default=0.1,
                    help="draw seeds from --seed on until every greedy step's first logit beats the second by this much")
    args = ap.parse_args()
    c = tiny_config()
    if args.override:
        c.update(json.loads(args.override))
    prompt = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
    for seed in range(args.seed, args.seed + 200):
        w = random_weights(c, seed)
        if args.fp8_checkpoint:   # the reference reads what the FP8 release gives back
            w = as_fp8(w)
            for i in range(len(c["rope_pattern"])):   # kv_b's split halves stay the FP8-free kv_b's
                p = f"blk.{i}."
                nope = c["nope"] + (0 if c["rope_pattern"][i] else c["rope"])
                w[p + "attn_k_b.weight"] = w[p + "kv_b"][:, :nope, :].transpose(1, 2).contiguous()
                w[p + "attn_v_b.weight"] = w[p + "kv_b"][:, nope:, :].contiguous()
        if not args.emit_ref:
            break
        with torch.no_grad():
            full, gap = Ref(c, w).generate(prompt, args.max_new)
        if gap >= args.min_gap:
            break
    else:
        sys.exit(f"no seed in {args.seed}..{args.seed + 199} gives a margin of {args.min_gap}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_gguf(out / "model.gguf", c, w)
    print(f"tiny Mistral-Large-4 -> {out / 'model.gguf'}")
    if args.fp8_checkpoint:
        write_fp8_checkpoint(args.fp8_checkpoint, c, w)
        print(f"tiny Mistral-Large-4, FP8 release layout -> {args.fp8_checkpoint}")
    if args.emit_ref:
        Path(args.emit_ref).parent.mkdir(parents=True, exist_ok=True)
        # teacher forcing: one forward over the whole sequence, the argmax at every position
        # (TF=1 checks the engine's prompt reading apart from its decode)
        with torch.no_grad():
            tf_pred = Ref(c, w).logits(full).argmax(-1).tolist()
        Path(args.emit_ref).write_text(json.dumps({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred,
                                                   "mode": "full", "model": "mistral_large4_tiny"}, indent=2))
        print(f"ref.json -> {args.emit_ref} (seed {seed}, smallest margin {gap:.3f})\n"
              f"  prompt_ids={prompt}\n  full_ids ={full}")


if __name__ == "__main__":
    main()
