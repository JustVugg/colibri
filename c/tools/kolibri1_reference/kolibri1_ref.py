# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference implementation of Kolibri-1, used only to write the tiny-fixture oracle
(tools/make_kolibri1_ref.py). docs/kolibri1.md describes the contract it implements.

Written from the checkpoint's published semantics, the vendor plugin `kolibri1.py`
(Aleph Alpha, aleph-alpha-inference 049a6a7, Apache-2.0) and the vLLM 0.29.0 sources it
runs on.  Not derived from llama.cpp or from any colibri engine code.

Numerical contract: FP8 block weights are dequantised exactly (fp8 -> fp32, times the
fp32 block scale), activations / residual stream / router / head are fp32 (fp64 with
dtype=torch.float64).  The head uses fp32 operands (deliberate).

Execution is layer-major over all positions: attention is chunked over query blocks, routed
experts are dequantised in bounded batches, each tensor is read once per run.
"""
from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
from safetensors import safe_open

FP8 = torch.float8_e4m3fn
BLOCK = 128


# ----------------------------------------------------------------------------------------------
# checkpoint access and exact FP8 block dequantisation
# ----------------------------------------------------------------------------------------------
def expand_scale(scale: torch.Tensor, out_features: int, in_features: int, block=(BLOCK, BLOCK)):
    """[ceil(out/bo), ceil(in/bi)] grid -> [out, in] (the scale grid is [out_blocks, in_blocks])."""
    bo, bi = block
    assert scale.shape == (math.ceil(out_features / bo), math.ceil(in_features / bi)), (
        scale.shape,
        out_features,
        in_features,
    )
    return scale.repeat_interleave(bo, 0)[:out_features].repeat_interleave(bi, 1)[:, :in_features]


def dequant_fp8(w: torch.Tensor, scale: torch.Tensor, dtype=torch.float32, block=(BLOCK, BLOCK), divide: bool = False):
    """W = fp8_value * weight_scale_inv[row_block, col_block]  (multiply).

    fp8 -> `dtype` is exact; the scale is fp32 (exactly representable in fp64).  In fp32 the product is
    one IEEE round-to-nearest-even of the exact product; in fp64 the product is exact (4-bit x 24-bit).
    """
    assert w.dtype == FP8 and scale.dtype == torch.float32
    out_f, in_f = w.shape
    se = expand_scale(scale, out_f, in_f, block).to(dtype)
    if divide:  # MUTATION ONLY (wrong scale direction); never used by the reference
        return w.to(dtype) / se
    return w.to(dtype) * se


class Checkpoint:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "config.json")) as f:
            self.cfg = json.load(f)
        idx = os.path.join(path, "model.safetensors.index.json")
        self._handles: dict[str, object] = {}
        if os.path.exists(idx):
            with open(idx) as f:
                self.map = json.load(f)["weight_map"]
        else:
            fn = "model.safetensors"
            h = safe_open(os.path.join(path, fn), framework="pt", device="cpu")
            self._handles[fn] = h
            self.map = {k: fn for k in h.keys()}

    def _h(self, fn):
        if fn not in self._handles:
            self._handles[fn] = safe_open(os.path.join(self.path, fn), framework="pt", device="cpu")
        return self._handles[fn]

    def has(self, name: str) -> bool:
        return name in self.map

    def raw(self, name: str) -> torch.Tensor:
        return self._h(self.map[name]).get_tensor(name)

    def tensor_dtype_str(self, name: str) -> str:
        return str(self._h(self.map[name]).get_slice(name).get_dtype())

    def weight(self, prefix: str, dtype=torch.float32, round_bf16: bool = False, bf16_scales: bool = False, divide: bool = False) -> torch.Tensor:
        """Dense [out, in] weight for `prefix` (without '.weight').  FP8 tensors are dequantised exactly.
        Non-FP8 storage (bf16 / fp32) is widened to `dtype`; `round_bf16` first rounds through bf16, which
        is what the vendor loader does to a fp32-stored tensor (fixture F0)."""
        w = self.raw(prefix + ".weight")
        if w.dtype == FP8:
            sc = self.raw(prefix + ".weight_scale_inv")
            if bf16_scales:  # vendor-emulation only: Marlin stores the block scales as bf16
                sc = sc.to(torch.bfloat16).to(torch.float32)
            return dequant_fp8(w, sc, dtype, divide=divide)
        if round_bf16:
            w = w.to(torch.bfloat16)
        return w.to(dtype)

    def vec(self, name: str, dtype=torch.float32, round_bf16: bool = False) -> torch.Tensor:
        t = self.raw(name)
        if round_bf16:
            t = t.to(torch.bfloat16)
        return t.to(dtype)


# ----------------------------------------------------------------------------------------------
# operations
# ----------------------------------------------------------------------------------------------
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm(x; w) = w * x * rsqrt(mean(x^2) + eps), mean over the last dim."""
    var = x.pow(2).mean(-1, keepdim=True)
    return w * (x * torch.rsqrt(var + eps))


def route(z: torch.Tensor, bias: torch.Tensor, k: int):
    """sigmoid_logit_add routing (op 10): select top-k of (z + bias); weight = sigmoid(z) of the
    selected (unbiased); no renormalisation.  Returns (weights [T,k], ids [T,k] int32, top-(k+1) biased
    scores [T,k+1] sorted desc)."""
    biased = z + bias
    top = torch.topk(biased, k=min(k + 1, z.shape[-1]), dim=-1)
    ids = top.indices[:, :k]
    w = torch.sigmoid(z.gather(1, ids))
    return w, ids.to(torch.int32), top.values


def rope_tables(positions: torch.Tensor, rotary_dim: int, theta: float, dtype):
    """neox-style tables; inv_freq[i] = 1/theta^(2i/rotary_dim), angle = pos * inv_freq.
    Computed in fp64 and cast (exact fp32 tables)."""
    i = torch.arange(0, rotary_dim, 2, dtype=torch.float64)
    inv = 1.0 / (theta ** (i / rotary_dim))
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    return torch.cos(ang).to(dtype), torch.sin(ang).to(dtype)


def apply_rope_neox(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T,H,D]; rotary over the full D (rotary_dim = D).  x1 = x[..., :D/2], x2 = x[..., D/2:]."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


def attention_chunked(q, k, v, window: Optional[int], chunk: int, gqa_interleaved: bool = False):
    """q [T,Hq,D], k/v [T,Hkv,D].  Causal; if `window` is set, key j is visible to query i iff
    i - j <= window - 1 (so `window` keys incl. self; W=513 -> self + 512 previous).
    GQA: query head h uses kv head h // (Hq/Hkv).  Scale D^-0.5.  Returns [T, Hq*D] head-major."""
    T, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    assert Hq % Hkv == 0
    scale = D ** -0.5
    out = torch.empty(T, Hq, D, dtype=q.dtype)
    for q0 in range(0, T, chunk):
        q1 = min(T, q0 + chunk)
        k0 = max(0, q0 - (window - 1)) if window is not None else 0
        if gqa_interleaved:  # MUTATION ONLY: q head h uses kv head h % Hkv
            K = k[k0:q1].repeat(1, G, 1)
            V = v[k0:q1].repeat(1, G, 1)
        else:
            K = k[k0:q1].repeat_interleave(G, dim=1)  # [S,Hq,D]: kv head j serves q heads G*j .. G*j+G-1
            V = v[k0:q1].repeat_interleave(G, dim=1)
        s = torch.einsum("qhd,shd->hqs", q[q0:q1], K) * scale
        qp = torch.arange(q0, q1)[:, None]
        kp = torch.arange(k0, q1)[None, :]
        vis = kp <= qp
        if window is not None:
            vis &= (qp - kp) <= (window - 1)
        s = s.masked_fill(~vis[None], float("-inf"))
        p = torch.softmax(s, dim=-1)
        out[q0:q1] = torch.einsum("hqs,shd->qhd", p, V)
    return out.reshape(T, Hq * D)


def _round_bf16(x):
    return x.to(torch.bfloat16).to(x.dtype)


# ----------------------------------------------------------------------------------------------
MUTATIONS = ("scale_div", "window_m1", "window_p1", "gqa_interleaved", "rope_full", "post_ffn_misplaced")


class Kolibri1Ref:
    def __init__(
        self,
        ckpt_dir: str,
        dtype=torch.float32,
        attn_chunk: int = 256,
        expert_batch: int = 8,
        round_nonfp8_bf16: bool = False,
        window_override: Optional[int] = None,
        marlin_emulation: bool = False,
        bf16_emulation: bool = False,
        threads: Optional[int] = None,
        mutate: Optional[str] = None,
    ):
        self.ck = Checkpoint(ckpt_dir)
        self.cfg = self.ck.cfg
        self.dtype = dtype
        self.attn_chunk = attn_chunk
        self.expert_batch = expert_batch
        self.round_bf16 = round_nonfp8_bf16
        # MUTATION switches (stage B3 mutation rejection only; the reference itself has mutate=None)
        assert mutate is None or mutate in MUTATIONS, mutate
        self.mutate = mutate
        self.window = window_override if window_override is not None else self.cfg["sliding_window"]
        if mutate == "window_m1":
            self.window = self.cfg["sliding_window"] - 1
        elif mutate == "window_p1":
            self.window = self.cfg["sliding_window"] + 1
        self.marlin_emulation = marlin_emulation
        # DIAGNOSTIC ONLY (not the contract): approximate the vendor bf16 pipeline: bf16 scales, activations,
        # residual stream and per-expert Marlin rounding sequence.  Used to separate numerics-driven divergence from structure.
        self.bf16_emulation = bf16_emulation
        self.R = _round_bf16 if bf16_emulation else (lambda t: t)
        if bf16_emulation:
            self.marlin_emulation = True
        if threads:
            torch.set_num_threads(threads)
        self.L = self.cfg["num_hidden_layers"]
        self.H = self.cfg["hidden_size"]
        self.Hq = self.cfg["num_attention_heads"]
        self.Hkv = self.cfg["num_key_value_heads"]
        self.D = self.cfg["head_dim"]
        self.E = self.cfg["num_experts"]
        self.K = self.cfg["num_experts_per_tok"]
        self.eps = self.cfg["rms_norm_eps"]
        self.theta = float(self.cfg.get("rope_theta", 10000.0))
        self.layer_types = self.cfg["layer_types"]
        assert not self.cfg.get("norm_topk_prob", False), "norm_topk_prob is false for Kolibri-1"
        assert not self.cfg.get("tie_word_embeddings", False)

    # -- helpers
    def _w(self, prefix):
        return self.ck.weight(prefix, self.dtype, self.round_bf16, self.bf16_emulation, divide=(self.mutate == "scale_div"))

    def _v(self, name):
        return self.ck.vec(name, self.dtype, self.round_bf16)

    def is_sliding(self, l: int) -> bool:
        return self.layer_types[l] == "sliding_attention"

    def router_from_x2(self, layer: int, x2: torch.Tensor):
        """Router GEMM from a supplied x2 (fp32 operands): z = x2 @ gate^T, [T,E]; also returns bias."""
        p = f"model.layers.{layer}"
        gate = self._w(f"{p}.mlp.gate")
        bias = self._v(f"{p}.moe.router.expert_bias")
        return x2.to(self.dtype) @ gate.T, bias

    def _expert_ffn(self, x, wg, wu, wd):
        R = self.R
        g = R(x @ wg.T)
        u = R(x @ wu.T)
        a = R(torch.nn.functional.silu(g) * u)
        return a @ wd.T

    # ------------------------------------------------------------------------------------------
    # individual operators: each takes explicit inputs (the common-input operator harness feeds them the vendor runtime's tensors);
    # layer() composes exactly these functions, so a mutation applies to both views.
    # ------------------------------------------------------------------------------------------
    def op_input_norm(self, l: int, f_prev: torch.Tensor, r_prev: Optional[torch.Tensor]):
        """op 1: layer 0 (r_prev None): x = RMSNorm(h), r = h.  Else fused add s = f_prev + r_prev, x = RMSNorm(s), r = s."""
        R = self.R
        s = f_prev if r_prev is None else R(f_prev + r_prev)
        x = R(rmsnorm(R(s), self._v(f"model.layers.{l}.input_layernorm.weight"), self.eps))
        return x, s

    def op_qkv(self, l: int, xin: torch.Tensor) -> torch.Tensor:
        """op 2: fused qkv = xin @ [Wq;Wk;Wv]^T, [T, Hq*D + 2*Hkv*D]."""
        p = f"model.layers.{l}.self_attn"
        w = torch.cat([self._w(f"{p}.q_proj"), self._w(f"{p}.k_proj"), self._w(f"{p}.v_proj")], 0)
        return self.R(xin @ w.T)

    def split_qkv(self, qkv: torch.Tensor):
        T = qkv.shape[0]
        qs, ks = self.Hq * self.D, self.Hkv * self.D
        return (qkv[:, :qs].reshape(T, self.Hq, self.D), qkv[:, qs:qs + ks].reshape(T, self.Hkv, self.D),
                qkv[:, qs + ks:].reshape(T, self.Hkv, self.D))

    def op_qk_norm(self, l: int, q: torch.Tensor, k: torch.Tensor):
        """op 3: per-head RMSNorm of q and k (v un-normed)."""
        p = f"model.layers.{l}.self_attn"
        R = self.R
        return (R(rmsnorm(q, self._v(f"{p}.q_norm.weight"), self.eps)), R(rmsnorm(k, self._v(f"{p}.k_norm.weight"), self.eps)))

    def uses_rope(self, l: int) -> bool:
        return self.is_sliding(l) or self.mutate == "rope_full"

    def op_rope(self, l: int, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """op 4: neox RoPE on q and k over the full head dim (applied only where uses_rope(l))."""
        cos, sin = rope_tables(positions, self.D, self.theta, self.dtype)
        R = self.R
        return R(apply_rope_neox(q, cos, sin)), R(apply_rope_neox(k, cos, sin))

    def op_attention(self, l: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """op 5: scaled causal (sliding on sliding layers) GQA attention; q [T,Hq,D], k/v [T,Hkv,D] -> [T,Hq*D]."""
        return self.R(attention_chunked(q, k, v, self.window if self.is_sliding(l) else None, self.attn_chunk,
                                        gqa_interleaved=(self.mutate == "gqa_interleaved")))

    def op_o_proj(self, l: int, ao: torch.Tensor) -> torch.Tensor:
        return self.R(ao @ self._w(f"model.layers.{l}.self_attn.o_proj").T)

    def op_post_attn_norm(self, l: int, o: torch.Tensor) -> torch.Tensor:
        return self.R(rmsnorm(o, self._v(f"model.layers.{l}.post_attn_norm.weight"), self.eps))

    def op_resid_norm(self, l: int, a: torch.Tensor, r: torch.Tensor):
        """op 8: ffn_inp = r + a (the new residual); x2 = RMSNorm(ffn_inp; post_attention_layernorm) (the pre-FFN norm)."""
        R = self.R
        ffn_inp = R(R(r) + a)
        x2 = R(rmsnorm(ffn_inp, self._v(f"model.layers.{l}.post_attention_layernorm.weight"), self.eps))
        return x2, ffn_inp

    def op_route(self, z: torch.Tensor, bias: torch.Tensor):
        """op 10: (weights [T,K], ids [T,K] int32, top-(K+1) biased scores)."""
        return route(z, bias, self.K)

    def op_routed(self, l: int, x2: torch.Tensor, ids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """op 11 with SUPPLIED expert ids and weights: weight applied after down_proj; bounded dequant batches."""
        p = f"model.layers.{l}"
        T = x2.shape[0]
        routed = torch.zeros(T, self.H, dtype=self.dtype)
        ids_l = ids.long()
        w = w.to(self.dtype)
        used = torch.unique(ids_l).tolist()
        for b0 in range(0, len(used), self.expert_batch):
            batch = used[b0 : b0 + self.expert_batch]
            wts = {e: (self._w(f"{p}.mlp.experts.{e}.gate_proj"), self._w(f"{p}.mlp.experts.{e}.up_proj"),
                       self._w(f"{p}.mlp.experts.{e}.down_proj")) for e in batch}
            for e in batch:
                tok, slot = (ids_l == e).nonzero(as_tuple=True)
                d = self._expert_ffn(x2[tok], *wts[e])
                we = w[tok, slot][:, None]
                if self.marlin_emulation:  # vendor emulation, optional
                    y = _round_bf16(_round_bf16(d) * _round_bf16(we))
                else:
                    y = we * d
                routed.index_add_(0, tok, y)
            del wts
        return self.R(routed)

    def op_shared(self, l: int, x2: torch.Tensor) -> torch.Tensor:
        p = f"model.layers.{l}"
        return self.R(self._expert_ffn(x2, self._w(f"{p}.mlp.shared_experts.gate_proj"), self._w(f"{p}.mlp.shared_experts.up_proj"),
                                       self._w(f"{p}.mlp.shared_experts.down_proj")))

    def op_combine(self, sh: torch.Tensor, routed: torch.Tensor) -> torch.Tensor:
        """op 13: moe_sum = shared + routed (before post_ffn_norm)."""
        return self.R(sh + routed)

    def op_post_ffn(self, l: int, m: torch.Tensor) -> torch.Tensor:
        """op 14: post_ffn_norm on the combined sum."""
        return self.R(rmsnorm(m, self._v(f"model.layers.{l}.post_ffn_norm.weight"), self.eps))

    def moe_out(self, l: int, sh: torch.Tensor, routed: torch.Tensor):
        """ops 13-14 composed from supplied shared and routed outputs: (moe_sum, post_ffn).  Honours the post_ffn_misplaced mutation."""
        if self.mutate == "post_ffn_misplaced":
            return self.R(sh + routed), self.R(self.op_post_ffn(l, routed) + sh)
        m = self.op_combine(sh, routed)
        return m, self.op_post_ffn(l, m)

    def head_weight(self) -> torch.Tensor:
        return self._w("lm_head")

    def op_final_norm(self, f: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        R = self.R
        return R(rmsnorm(R(f + r), self._v("model.norm.weight"), self.eps))

    def op_head(self, hn: torch.Tensor) -> torch.Tensor:
        head = self._w("lm_head")  # BF16 storage; fp32 operands
        return hn.to(self.dtype) @ head.T

    # -- composed pieces
    def attn_pre_o(self, l: int, x_prev: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Ops 1-5 from the completed stream x_prev: input norm, QKV, q/k norm, RoPE (sliding layers only), attention."""
        xin, _ = self.op_input_norm(l, x_prev, None)
        q, k, v = self.split_qkv(self.op_qkv(l, xin))
        q, k = self.op_qk_norm(l, q, k)
        if self.uses_rope(l):
            q, k = self.op_rope(l, q, k, positions)
        return self.op_attention(l, q, k, v)

    # -- one layer
    def layer(self, l: int, x_prev: torch.Tensor, positions: torch.Tensor, cap: Optional[dict], replay: Optional[tuple] = None):
        """replay = (ids [T,K], w [T,K]) -> routing replay: both implementations use these expert ids/weights."""
        T = x_prev.shape[0]
        cap_ = (lambda n, t: cap.__setitem__(f"L{l}.{n}", t.detach().clone())) if cap is not None else (lambda n, t: None)

        ao = self.attn_pre_o(l, x_prev, positions)
        cap_("attn_pre_o", ao)
        o = self.op_o_proj(l, ao)
        cap_("attn_o", o)
        a = self.op_post_attn_norm(l, o)
        cap_("post_attn_norm", a)
        x2, ffn_inp = self.op_resid_norm(l, a, x_prev)
        cap_("ffn_inp", ffn_inp)
        cap_("x2", x2)

        z, bias = self.router_from_x2(l, x2)
        cap_("router_logits", z)
        w, ids, top_biased = self.op_route(z, bias)
        cap_("top_biased", top_biased)
        if replay is not None:
            ids, w = replay[0].to(torch.int32), replay[1].to(torch.float32)
        cap_("sel_ids", ids)
        cap_("sel_w", w)

        routed = self.op_routed(l, x2, ids, w)
        cap_("routed", routed)
        sh = self.op_shared(l, x2)
        cap_("shared", sh)
        m, f = self.moe_out(l, sh, routed)  # post_ffn_misplaced mutation lives in moe_out
        cap_("moe_sum", m)
        cap_("post_ffn", f)
        x_out = self.R(ffn_inp + f)  # completed layer-l stream x_l
        cap_("x_out", x_out)
        return x_out

    def run(self, ids, capture: bool = True, logits: bool = True, positions_logits: Optional[slice] = None, replay: Optional[dict] = None):
        """replay: {layer: (ids [T,K], w [T,K])} (routing replay control)."""
        ids = torch.as_tensor(ids, dtype=torch.long)
        T = ids.shape[0]
        cap: Optional[dict] = {} if capture else None
        positions = torch.arange(T)
        emb = self.ck.raw("model.embed_tokens.weight")
        if self.round_bf16:
            emb = emb.to(torch.bfloat16)
        x = emb[ids].to(self.dtype)  # op 0: no scaling
        if cap is not None:
            cap["embed"] = x.clone()
        for l in range(self.L):
            x = self.layer(l, x, positions, cap, None if replay is None else replay[l])
        hn = self.R(rmsnorm(self.R(x), self._v("model.norm.weight"), self.eps))  # op 16 (f_49 + r_49 == x_49)
        res = {"cap": cap, "final_norm": hn}
        if cap is not None:
            cap["final_norm"] = hn
        if logits:
            hh = hn if positions_logits is None else hn[positions_logits]
            res["logits"] = self.op_head(hh)
        return res
