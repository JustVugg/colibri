#!/usr/bin/env python3
"""Full Colibri-style Hy3 forward in Python using safetensors weights."""
import json
import math
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import HYV3ForCausalLM

snap = "hy3_tiny"
ref = json.load(open("ref_hy3.json"))
ids = ref["full_ids"]
S = len(ids)

W = {}
with safe_open(f"{snap}/model.safetensors", framework="pt") as f:
    for k in f.keys():
        W[k] = f.get_tensor(k).float()

cfg = json.load(open(f"{snap}/config.json"))
D = cfg["hidden_size"]
H = cfg["num_attention_heads"]
Hkv = cfg["num_key_value_heads"]
hd = cfg["head_dim"]
E = cfg["num_experts"]
K = cfg["num_experts_per_tok"]
I_dense = cfg["intermediate_size"]
I_moe = cfg["moe_intermediate_size"]
n_shared = cfg["num_shared_experts"]
first_dense = cfg["first_k_dense_replace"]
n_layers = cfg["num_hidden_layers"]
eps = cfg["rms_norm_eps"]
theta = cfg["rope_parameters"]["rope_theta"]
route_norm = cfg.get("route_norm", True)
router_scale = cfg["router_scaling_factor"]
V = cfg["vocab_size"]


def rmsnorm(x, w):
    r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x * r * w


def rmsnorm_head(x, w):
    r = torch.rsqrt(x.pow(2).mean(-1) + eps)
    return x * r * w


def rope_head(v, pos):
    h2 = hd // 2
    out = v.clone()
    for j in range(h2):
        inv = theta ** (-2.0 * j / hd)
        ang = pos * inv
        c, s = math.cos(ang), math.sin(ang)
        a, b = v[j], v[j + h2]
        out[j] = a * c - b * s
        out[j + h2] = b * c + a * s
    return out


def mm(x, w):  # x [S,D], w [O,I]
    return x @ w.T


def silu(x):
    return x / (1 + torch.exp(-x))


def attn(layer, x, Kc, Vc):
    nrep = H // Hkv
    scale = hd ** -0.5
    qo, kvo = H * hd, Hkv * hd
    Q = mm(x, W[f"model.layers.{layer}.self_attn.q_proj.weight"]).view(S, H, hd)
    Kp = mm(x, W[f"model.layers.{layer}.self_attn.k_proj.weight"]).view(S, Hkv, hd)
    Vp = mm(x, W[f"model.layers.{layer}.self_attn.v_proj.weight"]).view(S, Hkv, hd)
    qn = W[f"model.layers.{layer}.self_attn.q_norm.weight"]
    kn = W[f"model.layers.{layer}.self_attn.k_norm.weight"]
    for s in range(S):
        pos = s
        for h in range(H):
            Q[s, h] = rmsnorm_head(Q[s, h], qn)
            Q[s, h] = rope_head(Q[s, h], pos)
        for kh in range(Hkv):
            Kp[s, kh] = rmsnorm_head(Kp[s, kh], kn)
            Kp[s, kh] = rope_head(Kp[s, kh], pos)
            Kc[layer][s, kh] = Kp[s, kh]
            Vc[layer][s, kh] = Vp[s, kh]
    ctx = torch.zeros(S, H, hd)
    for s in range(S):
        pos = s
        for h in range(H):
            kvh = h // nrep
            qv = Q[s, h]
            sc = torch.tensor([(qv @ Kc[layer][t, kvh]) * scale for t in range(pos + 1)])
            w = F.softmax(sc, dim=0)
            acc = torch.zeros(hd)
            for t in range(pos + 1):
                acc += w[t] * Vc[layer][t, kvh]
            ctx[s, h] = acc
    out = mm(ctx.view(S, H * hd), W[f"model.layers.{layer}.self_attn.o_proj.weight"])
    return out


def dense_mlp(layer, x):
    g = mm(x, W[f"model.layers.{layer}.mlp.gate_proj.weight"])
    u = mm(x, W[f"model.layers.{layer}.mlp.up_proj.weight"])
    h = silu(g) * u
    return mm(h, W[f"model.layers.{layer}.mlp.down_proj.weight"])


def moe(layer, x):
    bias = W[f"model.layers.{layer}.mlp.expert_bias"]
    router = W[f"model.layers.{layer}.mlp.router.gate.weight"]
    out = torch.zeros_like(x)
    sI = I_moe * n_shared
    for s in range(S):
        xs = x[s]
        logit = mm(xs.unsqueeze(0), router)[0]
        routing = torch.sigmoid(logit)
        choice = routing + bias
        idxs, ws = [], []
        for kk in range(K):
            mask = torch.ones(E, dtype=torch.bool)
            for j in idxs:
                mask[j] = False
            e = torch.argmax(torch.where(mask, choice, torch.tensor(-1e30)), dim=0).item()
            idxs.append(e)
            ws.append(routing[e].item())
        if route_norm:
            sm = sum(ws) + 1e-20
            ws = [w / sm for w in ws]
        ws = [w * router_scale for w in ws]
        acc = torch.zeros(D)
        for kk in range(K):
            e = idxs[kk]
            wgt = ws[kk]
            g = mm(xs.unsqueeze(0), W[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"])[0]
            u = mm(xs.unsqueeze(0), W[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"])[0]
            h = silu(g) * u
            acc += wgt * mm(h.unsqueeze(0), W[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"])[0]
        out[s] = acc
    sg = mm(x, W[f"model.layers.{layer}.mlp.shared_mlp.gate_proj.weight"])
    su = mm(x, W[f"model.layers.{layer}.mlp.shared_mlp.up_proj.weight"])
    sh = mm(silu(sg) * su, W[f"model.layers.{layer}.mlp.shared_mlp.down_proj.weight"])
    if False and cfg.get("enable_moe_fp32_combine", False):
        return (out.float() + sh.float()).to(x.dtype)
    return out + sh


Kc = {i: torch.zeros(S, Hkv, hd) for i in range(n_layers)}
Vc = {i: torch.zeros(S, Hkv, hd) for i in range(n_layers)}

x = W["model.embed_tokens.weight"][ids]  # [S,D]
for layer in range(n_layers):
    n = rmsnorm(x, W[f"model.layers.{layer}.input_layernorm.weight"])
    tmp = attn(layer, n, Kc, Vc)
    x = x + tmp
    n = rmsnorm(x, W[f"model.layers.{layer}.post_attention_layernorm.weight"])
    tmp = dense_mlp(layer, n) if layer < first_dense else moe(layer, n)
    x = x + tmp

pred = []
for s in range(S):
    row = rmsnorm(x[s], W["model.norm.weight"])
    lo = mm(row.unsqueeze(0), W["lm_head.weight"])[0]
    pred.append(int(lo.argmax()))

hf = HYV3ForCausalLM.from_pretrained(snap).eval()
with torch.no_grad():
    hf_pred = hf(torch.tensor([ids])).logits[0].argmax(-1).tolist()

print("py_c vs ref:", sum(a == b for a, b in zip(pred, ref["tf_pred"])), "/", S)
print("hf  vs ref:", sum(a == b for a, b in zip(hf_pred, ref["tf_pred"])), "/", S)
print("py_c vs hf :", sum(a == b for a, b in zip(pred, hf_pred)), "/", S)
for i, (a, b, c) in enumerate(zip(pred, ref["tf_pred"], hf_pred)):
    if a != b:
        print(f" mismatch pos={i} py_c={a} ref={b} hf={c}")
