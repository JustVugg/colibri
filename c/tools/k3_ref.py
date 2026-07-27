#!/usr/bin/env python3
"""Numpy reference for the first N layers of Kimi K3 — validates c/kimi_k3.c.

Reads tensors straight from the HF shards (BF16 -> f32, MXFP4 dequant), runs T
injected input rows [T, hidden] (the engine's K3_X0 hook, bypassing the
embedding, which lives in one of the last shards) through layers 0..N-1
exactly as modeling_kimi_linear.py does — KDA recurrence, gated NoPE MLA,
AttnRes, LatentMoE with SiTU-GLU — and writes the same trace layout the C
engine emits under K3_TRACE: per token, per layer, hidden[hidden] f32.

  gen   : k3_ref.py gen <x0.bin> --tokens 6 --dim 7168 [--seed 7]
  run   : k3_ref.py run <hf_dir> --x0 x0.bin --layers 4 --out ref.bin
  cmp   : k3_ref.py cmp <c_trace.bin> <ref.bin> --dim 7168 --layers 4

Pure numpy, no torch. Everything f32 (f64 only where the C engine also
accumulates in double: AttnRes scores, RMS means).
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6],
                dtype=np.float32)


class Shards:
    def __init__(self, d):
        self.dir = d
        idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        self.wmap = idx["weight_map"]
        self.hdrs = {}

    def _hdr(self, fn):
        if fn not in self.hdrs:
            with open(os.path.join(self.dir, fn), "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.hdrs[fn] = (json.loads(f.read(n)), 8 + n)
        return self.hdrs[fn]

    def get(self, name):
        fn = self.wmap[name]
        h, base = self._hdr(fn)
        v = h[name]
        o0, o1 = v["data_offsets"]
        with open(os.path.join(self.dir, fn), "rb") as f:
            f.seek(base + o0)
            raw = f.read(o1 - o0)
        if v["dtype"] == "BF16":
            u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
            a = u.view(np.float32)
        elif v["dtype"] == "F32":
            a = np.frombuffer(raw, dtype=np.float32)
        elif v["dtype"] == "U8":
            a = np.frombuffer(raw, dtype=np.uint8)
        else:
            raise ValueError(v["dtype"])
        return a.reshape(v["shape"]).copy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def rmsnorm(x, w, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(ms + eps))).astype(np.float32) * w


def situ(g, u, b1, b2):
    return (b1 * np.tanh(g / b1) * sigmoid(g)) * (b2 * np.tanh(u / b2))


def deq_mx4(packed, scale):
    O, half = packed.shape
    nib = np.empty((O, half * 2), dtype=np.uint8)
    nib[:, 0::2] = packed & 0xF
    nib[:, 1::2] = packed >> 4
    w = E2M1[nib]
    sc = np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127).astype(np.float32)
    I = half * 2
    return (w.reshape(O, I // 32, 32) * sc[:, :, None]).reshape(O, I)


def res_mix(prefix, bres, sw, eps):
    """entries = bres rows + prefix (last); softmax over RMS-normed dots; mix RAW."""
    vs = list(bres) + [prefix]
    sc = np.empty(len(vs), dtype=np.float32)
    sw64 = sw.astype(np.float64)
    for e, v in enumerate(vs):
        v64 = v.astype(np.float64)
        ms = np.mean(v64 ** 2)
        sc[e] = np.dot(v64, sw64) / np.sqrt(ms + eps)
    p = np.exp(sc - sc.max())
    p /= p.sum()
    out = np.zeros_like(prefix)
    for e, v in enumerate(vs):
        out += p[e] * v
    return out


class K3Ref:
    def __init__(self, hf_dir, n_layers):
        self.S = Shards(hf_dir)
        cfg = json.load(open(os.path.join(hf_dir, "config.json")))
        tc = cfg.get("text_config", cfg)
        self.tc = tc
        self.D = tc["hidden_size"]
        self.eps = tc["rms_norm_eps"]
        self.b1 = tc["activation_situ_beta"]
        self.b2 = tc["activation_situ_linear_beta"]
        self.res_bs = tc["attn_res_block_size"]
        self.first_dense = tc["first_k_dense_replace"]
        la = tc["linear_attn_config"]
        self.kda_layers = set(x - 1 for x in la["kda_layers"])   # 0-indexed
        self.kh, self.hd = la["num_heads"], la["head_dim"]
        self.lb = la["gate_lower_bound"]
        self.conv_k = la["short_conv_kernel_size"]
        self.n_layers = n_layers
        self.pfx = "language_model." if any(
            k.startswith("language_model.") for k in list(self.S.wmap)[:5]) else ""

    def t(self, name):
        return self.S.get(self.pfx + name)

    def kda(self, li, x):
        """x [T,D] -> [T,D]; sequential recurrence, T small."""
        p = f"model.layers.{li}.self_attn."
        T = x.shape[0]
        H, hd = self.kh, self.hd
        qp = x @ self.t(p + "q_proj.weight").T
        kp = x @ self.t(p + "k_proj.weight").T
        vp = x @ self.t(p + "v_proj.weight").T
        outs = []
        for nm, arr in (("q_conv1d", qp), ("k_conv1d", kp), ("v_conv1d", vp)):
            w = self.t(p + nm + ".weight").reshape(-1, self.conv_k)  # [P,K]
            P = arr.shape[1]
            pad = np.zeros((self.conv_k - 1, P), dtype=np.float32)
            xp = np.concatenate([pad, arr], axis=0)                  # [T+K-1, P]
            y = np.zeros_like(arr)
            for j in range(self.conv_k):
                y += xp[j:j + T] * w[:, j]
            outs.append(silu(y))
        q, k, v = outs
        g_low = (x @ self.t(p + "f_a_proj.weight").T) @ self.t(p + "f_b_proj.weight").T
        beta = sigmoid(x @ self.t(p + "b_proj.weight").T)            # [T,H]
        dt = self.t(p + "dt_bias")                                    # [H*hd]
        A = np.exp(self.t(p + "A_log")[:H])                           # padded ckpt
        onw = self.t(p + "o_norm.weight")
        gfull = x @ self.t(p + "g_proj.weight").T                     # [T,H*hd]
        q = q.reshape(T, H, hd)
        k = k.reshape(T, H, hd)
        v = v.reshape(T, H, hd)
        z = g_low.reshape(T, H, hd) + dt.reshape(H, hd)
        alpha = np.exp(self.lb * sigmoid(A[None, :, None] * z))       # [T,H,hd]
        qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6) * (hd ** -0.5)
        kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
        on = np.zeros((T, H, hd), dtype=np.float32)
        for h in range(H):
            S = np.zeros((hd, hd), dtype=np.float32)                  # S[k,v]
            for t in range(T):
                S = S * alpha[t, h][:, None]
                kS = kn[t, h] @ S
                vt = (v[t, h] - kS) * beta[t, h]
                S = S + np.outer(kn[t, h], vt)
                on[t, h] = qn[t, h] @ S
        ms = np.mean(on.astype(np.float64) ** 2, axis=-1, keepdims=True)
        on = (on * (1.0 / np.sqrt(ms + self.eps))).astype(np.float32) * onw
        on = on * sigmoid(gfull.reshape(T, H, hd))
        return on.reshape(T, H * hd) @ self.t(p + "o_proj.weight").T

    def mla(self, li, x):
        p = f"model.layers.{li}.self_attn."
        tc = self.tc
        T = x.shape[0]
        H = tc["num_attention_heads"]
        nope, rope, vh = tc["qk_nope_head_dim"], tc["qk_rope_head_dim"], tc["v_head_dim"]
        kvl = tc["kv_lora_rank"]
        qh = nope + rope
        qa = rmsnorm(x @ self.t(p + "q_a_proj.weight").T,
                     self.t(p + "q_a_layernorm.weight"), self.eps)
        q = (qa @ self.t(p + "q_b_proj.weight").T).reshape(T, H, qh)
        ckv = x @ self.t(p + "kv_a_proj_with_mqa.weight").T
        c = rmsnorm(ckv[:, :kvl], self.t(p + "kv_a_layernorm.weight"), self.eps)
        k_rot = ckv[:, kvl:]                                          # [T,rope] raw (NoPE)
        kv = (c @ self.t(p + "kv_b_proj.weight").T).reshape(T, H, nope + vh)
        k_pass, v = kv[..., :nope], kv[..., nope:]
        scale = qh ** -0.5
        sc = (np.einsum("thd,shd->hts", q[..., :nope], k_pass) +
              np.einsum("thd,sd->hts", q[..., nope:], k_rot)) * scale
        mask = np.tril(np.ones((T, T), dtype=bool))
        sc = np.where(mask[None], sc, -np.inf)
        sc = sc - sc.max(-1, keepdims=True)
        w = np.exp(sc)
        w /= w.sum(-1, keepdims=True)
        ctx = np.einsum("hts,shd->thd", w.astype(np.float32), v)
        g = sigmoid(x @ self.t(p + "g_proj.weight").T)
        ctx = ctx.reshape(T, H * vh) * g
        return ctx @ self.t(p + "o_proj.weight").T

    def moe(self, li, x):
        p = f"model.layers.{li}.block_sparse_moe."
        tc = self.tc
        T = x.shape[0]
        K = tc["num_experts_per_token"]
        router = self.t(p + "gate.weight")
        bias = self.t(p + "gate.e_score_correction_bias")
        lat_dn = self.t(p + "routed_expert_down_proj.weight")
        lat_up = self.t(p + "routed_expert_up_proj.weight")
        lat_nw = self.t(p + "routed_expert_norm.weight")
        sg = self.t(p + "shared_experts.gate_proj.weight")
        su = self.t(p + "shared_experts.up_proj.weight")
        sd = self.t(p + "shared_experts.down_proj.weight")
        out = np.zeros_like(x)
        experts = {}
        for t in range(T):
            s = sigmoid(x[t] @ router.T)
            sel = np.argsort(-(s + bias), kind="stable")[:K]
            w = s[sel]
            w = w / (w.sum() + 1e-20)
            z = x[t] @ lat_dn.T
            u = np.zeros(z.shape[0], dtype=np.float32)
            for e, we in zip(sel, w):
                if e not in experts:
                    ep = f"{p}experts.{e}."
                    W1 = deq_mx4(self.t(ep + "w1.weight_packed"), self.t(ep + "w1.weight_scale"))
                    W2 = deq_mx4(self.t(ep + "w2.weight_packed"), self.t(ep + "w2.weight_scale"))
                    W3 = deq_mx4(self.t(ep + "w3.weight_packed"), self.t(ep + "w3.weight_scale"))
                    experts[e] = (W1, W2, W3)
                W1, W2, W3 = experts[e]
                act = situ(z @ W1.T, z @ W3.T, self.b1, self.b2)
                u += we * (act @ W2.T)
            u = rmsnorm(u, lat_nw, self.eps)
            y = u @ lat_up.T
            act = situ(x[t] @ sg.T, x[t] @ su.T, self.b1, self.b2)
            out[t] = y + act @ sd.T
        return out

    def dense(self, li, x):
        p = f"model.layers.{li}.mlp."
        act = situ(x @ self.t(p + "gate_proj.weight").T,
                   x @ self.t(p + "up_proj.weight").T, self.b1, self.b2)
        return act @ self.t(p + "down_proj.weight").T

    def run(self, x0, out_path):
        """x0 [T,D]; per-token trace rows in the SAME order as the C engine
        (token-major: the C engine steps one token at a time through all
        layers). Recurrent/cache state must therefore persist across tokens —
        we instead run all T at once per layer and emit token-major at the
        end, which is equivalent because layer l for token t only depends on
        layers <= l of tokens <= t."""
        T = x0.shape[0]
        trace = np.zeros((T, self.n_layers, self.D), dtype=np.float32)
        hidden = x0.copy()
        bres = []
        prefix = None
        for li in range(self.n_layers):
            lp = f"model.layers.{li}."
            in_ln = self.t(lp + "input_layernorm.weight")
            post_ln = self.t(lp + "post_attention_layernorm.weight")
            asw = self.t(lp + "self_attention_res_norm.weight") * \
                self.t(lp + "self_attention_res_proj.weight").reshape(-1)
            msw = self.t(lp + "mlp_res_norm.weight") * \
                self.t(lp + "mlp_res_proj.weight").reshape(-1)
            prefix = hidden.copy()
            have_prefix = True
            if bres:
                hidden = np.stack([res_mix(prefix[t], [b[t] for b in bres], asw, self.eps)
                                   for t in range(T)])
            if li % self.res_bs == 0:
                bres.append(prefix.copy())
                have_prefix = False
            nrm = rmsnorm(hidden, in_ln, self.eps)
            att = self.kda(li, nrm) if li in self.kda_layers else self.mla(li, nrm)
            prefix = (prefix + att) if have_prefix else att
            mix = np.stack([res_mix(prefix[t], [b[t] for b in bres], msw, self.eps)
                            for t in range(T)])
            nrm = rmsnorm(mix, post_ln, self.eps)
            mlp = self.moe(li, nrm) if li >= self.first_dense else self.dense(li, nrm)
            prefix = prefix + mlp
            hidden = prefix.copy()
            trace[:, li] = hidden
            print(f"[ref] layer {li} done ({'KDA' if li in self.kda_layers else 'MLA'},"
                  f" {'moe' if li >= self.first_dense else 'dense'})", flush=True)
        trace.tofile(out_path)
        print(f"[ref] wrote {out_path}: {T} tokens x {self.n_layers} layers x {self.D}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("out")
    g.add_argument("--tokens", type=int, default=6)
    g.add_argument("--dim", type=int, default=7168)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--scale", type=float, default=0.02)
    r = sub.add_parser("run")
    r.add_argument("hf_dir")
    r.add_argument("--x0", required=True)
    r.add_argument("--layers", type=int, default=4)
    r.add_argument("--out", required=True)
    c = sub.add_parser("cmp")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--dim", type=int, default=7168)
    c.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "gen":
        rng = np.random.default_rng(a.seed)
        x = (rng.standard_normal((a.tokens, a.dim)) * a.scale).astype(np.float32)
        x.tofile(a.out)
        print(f"wrote {a.out}: {a.tokens}x{a.dim}")
    elif a.cmd == "run":
        x0 = np.fromfile(a.x0, dtype=np.float32).reshape(-1, 7168)
        K3Ref(a.hf_dir, a.layers).run(x0, a.out)
    else:
        A = np.fromfile(a.a, dtype=np.float32).reshape(-1, a.layers, a.dim)
        B = np.fromfile(a.b, dtype=np.float32).reshape(-1, a.layers, a.dim)
        assert A.shape == B.shape, (A.shape, B.shape)
        worst = 0.0
        for li in range(a.layers):
            x, y = A[:, li], B[:, li]
            rel = np.linalg.norm(x - y) / (np.linalg.norm(y) + 1e-30)
            worst = max(worst, rel)
            print(f"layer {li}: rel_l2 {rel:.3e}  (|c|={np.linalg.norm(x):.4g} |ref|={np.linalg.norm(y):.4g})")
        print("VERDICT:", "OK" if worst < 3e-4 else "MISMATCH", f"(worst {worst:.3e})")


if __name__ == "__main__":
    main()
