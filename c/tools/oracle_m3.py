"""Numpy oracle for the MiniMax-M3 engine port. Reads a CONVERTED tiny container
(so the reference runs on the exact same dequantized weights as the C engine —
differences are implementation bugs, not quantization), teacher-forces a fixed
token sequence with the pinned M3 conventions, and writes the engine's REF gate:
  ref_m3.json  {prompt_ids, full_ids, tf_pred}   +   oracle_logits.npy [T,V]
Validate:  REF=ref_m3.json TF=1 COLI_MODEL=<container> ./colibri 8   (expect T/T)

Conventions (transformers modeling_minimax_m3_vl, pinned 2026-07-22):
  Gemma RMSNorm x/rms*(1+w) in f32; per-head QK-norm BEFORE partial NEOX RoPE
  (first `rotary` dims, split-half within them, inv_freq theta^(-2j/rot));
  GQA repeat_kv h -> h // (H/NK); scale 1/sqrt(head_dim); swigluoai
  (gate=min(g,lim), up=clamp(±lim), (up+1)*gate*sigmoid(alpha*gate));
  router sigmoid -> +bias for CHOICE only, raw-sigmoid weights renormalized,
  routed_out * routed_scaling + shared_out; pre-norm residual layers."""
import json, sys, glob
import numpy as np
from safetensors import safe_open

IND = sys.argv[1] if len(sys.argv) > 1 else "m3tiny_i8"

T = {}
for p in sorted(glob.glob(f"{IND}/out-*.safetensors")):
    with safe_open(p, framework="np") as f:
        for k in f.keys(): T[k] = f.get_tensor(k)
cfg = json.load(open(f"{IND}/config.json"))
D, L, H = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"]
NK, HD, ROT = cfg["num_key_value_heads"], cfg["head_dim"], cfg["rotary_dim"]
E, TOPK, V = cfg["num_local_experts"], cfg["num_experts_per_tok"], cfg["vocab_size"]
EPS, TH = cfg["rms_norm_eps"], cfg["rope_theta"]
A, LIM, RS = cfg["swiglu_alpha"], cfg["swiglu_limit"], cfg["routed_scaling_factor"]
FD = sum(1 for x in (cfg.get("moe_layer_freq") or []) if x == 0) if cfg.get("moe_layer_freq") else 0

def deq(name):
    """Dequant a container tensor exactly as the engine reads it (int8 per-row, f32)."""
    wq = T[name]
    if name + ".qs" not in T: return wq.astype(np.float32)
    s = T[name + ".qs"].astype(np.float32)
    q = wq.view(np.int8).astype(np.float32)
    O = s.shape[0]; q = q.reshape(O, -1)
    return q * s[:, None]

def rms(x, w, eps=EPS):                            # gemma: (1+w)
    r = 1.0 / np.sqrt((x.astype(np.float64)**2).mean(-1, keepdims=True) + eps)
    return (x * r * (1.0 + w)).astype(np.float32)

def rope(vh, pos):                                 # partial split-half NEOX on first ROT dims
    h2 = ROT // 2
    j = np.arange(h2, dtype=np.float32)
    fr = TH ** (-2.0 * j / ROT)
    ang = pos * fr; cs, sn = np.cos(ang), np.sin(ang)
    x1, x2 = vh[:h2].copy(), vh[h2:ROT].copy()
    vh[:h2] = x1 * cs - x2 * sn
    vh[h2:ROT] = x2 * cs + x1 * sn
    return vh

def act(g, u):                                     # swigluoai
    g = np.minimum(g, LIM)
    u = np.clip(u, -LIM, LIM)
    return (u + 1.0) * (g / (1.0 + np.exp(-A * g)))

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

ids = [(7 * i + 3) % V for i in range(24)]
S = len(ids)
emb = deq("model.embed_tokens.weight")
x = emb[ids].astype(np.float32)                    # [S,D]

for li in range(L):
    P = f"model.layers.{li}."
    nrm = rms(x, T[P + "input_layernorm.weight"].astype(np.float32))
    q = nrm @ deq(P + "self_attn.q_proj.weight").T   # [S,H*HD]
    k = nrm @ deq(P + "self_attn.k_proj.weight").T   # [S,NK*HD]
    v = nrm @ deq(P + "self_attn.v_proj.weight").T
    qn = T[P + "self_attn.q_norm.weight"].astype(np.float32)
    kn = T[P + "self_attn.k_norm.weight"].astype(np.float32)
    q = q.reshape(S, H, HD); k = k.reshape(S, NK, HD); v = v.reshape(S, NK, HD)
    for s in range(S):
        for h in range(H):  q[s, h] = rope(rms(q[s, h], qn), s)
        for h in range(NK): k[s, h] = rope(rms(k[s, h], kn), s)
    G = H // NK
    ctx = np.zeros((S, H, HD), np.float32)
    for s in range(S):
        for h in range(H):
            g = h // G
            sc = (q[s, h] @ k[:s+1, g].T) / np.sqrt(HD)
            sc = sc - sc.max(); e = np.exp(sc); a = e / e.sum()
            ctx[s, h] = a @ v[:s+1, g]
    x = x + ctx.reshape(S, H * HD) @ deq(P + "self_attn.o_proj.weight").T
    nrm = rms(x, T[P + "post_attention_layernorm.weight"].astype(np.float32))
    if li < FD:                                    # dense
        g = nrm @ deq(P + "mlp.gate_proj.weight").T
        u = nrm @ deq(P + "mlp.up_proj.weight").T
        x = x + act(g, u) @ deq(P + "mlp.down_proj.weight").T
    else:                                          # MoE + shared
        sg = nrm @ deq(P + "mlp.shared_experts.gate_proj.weight").T
        su = nrm @ deq(P + "mlp.shared_experts.up_proj.weight").T
        sh = act(sg, su) @ deq(P + "mlp.shared_experts.down_proj.weight").T
        logits = nrm @ T[P + "mlp.gate.weight"].astype(np.float32).T
        rw = sigmoid(logits.astype(np.float64)).astype(np.float32)
        bias = T[P + "mlp.gate.e_score_correction_bias"].astype(np.float32)
        routed = np.zeros_like(nrm)
        for s in range(S):
            choice = rw[s] + bias
            top = np.argsort(-choice)[:TOPK]
            wts = rw[s][top]; wts = wts / wts.sum()
            for t_i, e_i in enumerate(top):
                EP = P + f"mlp.experts.{e_i}."
                g = nrm[s] @ deq(EP + "gate_proj.weight").T
                u = nrm[s] @ deq(EP + "up_proj.weight").T
                routed[s] += wts[t_i] * (act(g, u) @ deq(EP + "down_proj.weight").T)
        x = x + routed * RS + sh

fn = T["model.norm.weight"].astype(np.float32)
lo = rms(x, fn) @ deq("lm_head.weight").T          # [S,V]
pred = lo.argmax(-1).tolist()
json.dump({"prompt_ids": ids[:4], "full_ids": ids, "tf_pred": pred},
          open("ref_m3.json", "w"))
np.save("oracle_logits.npy", lo)
print(f"oracle: {S} positions, logits [{S},{V}] -> ref_m3.json + oracle_logits.npy")
print("tf_pred:", pred)
