#!/usr/bin/env python3
"""Tiny random Hy3 (hy_v3) oracle for Colibri hy3.c validation."""
import json
import torch

try:
    from transformers import HYV3Config, HYV3ForCausalLM
except ImportError as e:
    raise SystemExit("pip install 'transformers>=5.6.0' torch safetensors") from e

torch.manual_seed(5678)

cfg = HYV3Config(
    vocab_size=256,
    hidden_size=128,
    intermediate_size=64,
    moe_intermediate_size=32,
    num_hidden_layers=5,
    first_k_dense_replace=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    num_experts=8,
    num_experts_per_tok=2,
    num_shared_experts=1,
    route_norm=True,
    router_scaling_factor=2.826,
    moe_router_use_sigmoid=True,
    moe_router_enable_expert_bias=True,
    qk_norm=True,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    tie_word_embeddings=False,
    rms_norm_eps=1e-5,
    max_position_embeddings=4096,
)
cfg._attn_implementation = "eager"

model = HYV3ForCausalLM(cfg).eval()
with torch.no_grad():
    for _, p in model.named_parameters():
        if p.dim() >= 2:
            p.normal_(0, 0.05)
    for layer in model.model.layers:
        mlp = layer.mlp
        if hasattr(mlp, "gate") and hasattr(mlp.gate, "e_score_correction_bias"):
            mlp.gate.e_score_correction_bias.copy_(
                torch.linspace(-0.1, 0.1, cfg.num_experts))

print("=== Hy3 tiny state_dict ===")
for n, p in model.state_dict().items():
    print(f"  {n:60s} {tuple(p.shape)}")

prompt = [3, 14, 159, 26, 53, 58, 200, 11, 77, 240, 5, 99]
ids = torch.tensor([prompt])
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=False, use_cache=True)
full = out[0].tolist()
print("\nprompt:", prompt)
print("full  :", full)

with torch.no_grad():
    lg = model(torch.tensor([full]), use_cache=False).logits[0]
tf_pred = lg.argmax(-1).tolist()
print("tf_pred:", tf_pred)

out_dir = "hy3_tiny"
model.save_pretrained(out_dir, safe_serialization=True)
json.dump(cfg.to_dict(), open(f"{out_dir}/config.json", "w"))
json.dump({"prompt_ids": prompt, "full_ids": full, "tf_pred": tf_pred}, open("ref_hy3.json", "w"))
print(f"\nsaved: {out_dir}/ and ref_hy3.json")
