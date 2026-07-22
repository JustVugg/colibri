"""Tiny random MiniMax-M3-shaped checkpoint (VL layout, language_model.* prefix) for
engine-vs-oracle validation — the M3 counterpart of glm_tiny. Layer 0 dense + layer 1
MoE, GQA 4/2 heads with head_dim 8 (deliberately != hidden/heads to exercise the
explicit head_dim path), partial rotary 4 of 8, all 4 experts present, nonzero router
bias (exercises the choice-vs-weight distinction). Convert with:
  python3 tools/convert_fp8_to_int4.py --indir m3tiny --outdir m3tiny_i8 --ebits 8 --io-bits 8
then validate with tools/oracle_m3.py + the engine's REF/TF gate."""
import json, sys, os
import torch
from safetensors.torch import save_file

OUT = sys.argv[1] if len(sys.argv) > 1 else "m3tiny"
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(42)

D, L, H, NK, HD = 16, 2, 4, 2, 8
ROT, E, TOPK, MI, DI, SI, V = 4, 4, 2, 8, 12, 8, 32

t = {}
def w(n, o, i): t[n] = (torch.randn(o, i) * 0.3).to(torch.bfloat16)
def vv(n, o):   t[n] = (torch.randn(o) * 0.3).to(torch.bfloat16)

P = "language_model."
w(P+"model.embed_tokens.weight", V, D)
w(P+"lm_head.weight", V, D)
vv(P+"model.norm.weight", D)
for i in range(L):
    Lp = P + f"model.layers.{i}."
    vv(Lp+"input_layernorm.weight", D)
    vv(Lp+"post_attention_layernorm.weight", D)
    w(Lp+"self_attn.q_proj.weight", H*HD, D)
    w(Lp+"self_attn.k_proj.weight", NK*HD, D)
    w(Lp+"self_attn.v_proj.weight", NK*HD, D)
    w(Lp+"self_attn.o_proj.weight", D, H*HD)
    vv(Lp+"self_attn.q_norm.weight", HD)
    vv(Lp+"self_attn.k_norm.weight", HD)
    if i == 0:                                    # dense layer
        w(Lp+"mlp.gate_proj.weight", DI, D)
        w(Lp+"mlp.up_proj.weight", DI, D)
        w(Lp+"mlp.down_proj.weight", D, DI)
    else:                                         # MoE layer
        M = Lp + "block_sparse_moe."
        w(M+"gate.weight", E, D)
        t[M+"e_score_correction_bias"] = (torch.randn(E) * 0.1).to(torch.float32)
        for e in range(E):
            w(M+f"experts.{e}.w1.weight", MI, D)   # gate
            w(M+f"experts.{e}.w2.weight", D, MI)   # down
            w(M+f"experts.{e}.w3.weight", MI, D)   # up
        w(M+"shared_experts.gate_proj.weight", SI, D)
        w(M+"shared_experts.up_proj.weight", SI, D)
        w(M+"shared_experts.down_proj.weight", D, SI)
w("vision_tower.vision_model.embeddings.patch_embedding.weight", 8, 8)   # must be dropped

save_file(t, os.path.join(OUT, "model-00001-of-00001.safetensors"))
cfg = {"model_type": "minimax_m3_vl",
       "text_config": {
           "hidden_size": D, "num_hidden_layers": L, "num_attention_heads": H,
           "num_key_value_heads": NK, "head_dim": HD, "rotary_dim": ROT,
           "partial_rotary_factor": ROT/HD, "intermediate_size": MI,
           "dense_intermediate_size": DI, "shared_intermediate_size": SI,
           "num_local_experts": E, "num_experts_per_tok": TOPK, "n_shared_experts": 1,
           "scoring_func": "sigmoid", "use_routing_bias": True,
           "routed_scaling_factor": 2.0, "moe_layer_freq": [0, 1],
           "rms_norm_eps": 1e-6, "rope_theta": 5000000, "vocab_size": V,
           "use_qk_norm": True, "qk_norm_type": "per_head", "use_gemma_norm": True,
           "hidden_act": "swigluoai", "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
           "eos_token_id": V-1},
       "vision_config": {"hidden_size": 8}}
json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print(f"{OUT}: {len(t)} tensors, D={D} L={L} H={H}/{NK} hd={HD} rot={ROT} E={E} top{TOPK} V={V}")
