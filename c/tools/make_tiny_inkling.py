#!/usr/bin/env python3
"""Build a tiny random-weight Inkling text model + oracle fixture for inkling.c.

Stage-A validation, mirroring the OLMoE ref.json flow: saves a small
InklingForCausalLM snapshot (safetensors + config.json) and a ref_inkling.json
with {prompt_ids, full_ids, tf_pred} from HF transformers, which the C engine
must reproduce token-for-token (run with bits=0 for f32-exact experts).

The tiny config exercises every architectural branch of the big model:
sliding + global layers, log-length tau scaling (n_floor set below the
sequence length), dense + sparse MLP layers, 2 shared experts, unpadded vocab.

Usage: python3 make_tiny_inkling.py <outdir> [--wide]

--wide keeps the real model's hidden size, 6144, and head dim, 128, in two layers (a
sliding one with a dense MLP, a global one with the experts): the size the Vulkan
expert batch stages its input at, and the dense chain's matrices, norms and
convolutions at the width they run at.
"""
import json
import sys

import torch

try:
    from transformers import InklingForCausalLM, InklingTextConfig
except ImportError:
    sys.exit("transformers has no Inkling support: pip install -U transformers")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    wide = "--wide" in sys.argv[1:]
    out = args[0] if args else "tiny_inkling"
    torch.manual_seed(0)

    if wide:
        cfg = InklingTextConfig(
            vocab_size=256,
            unpadded_vocab_size=250,
            hidden_size=6144,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=128,
            swa_num_attention_heads=8,
            swa_num_key_value_heads=4,
            swa_head_dim=128,
            sliding_window_size=8,
            d_rel=16,
            rel_extent=32,
            log_scaling_n_floor=8,
            log_scaling_alpha=0.1,
            local_layer_ids=[0],                 # global layer at 1
            dense_mlp_idx=1,
            dense_intermediate_size=128,
            moe_intermediate_size=64,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=2,
            route_scale=2.0,
            logits_mup_width_multiplier=4.0,
            max_position_embeddings=4096,
            eos_token_id=None,
        )
    else:
        cfg = InklingTextConfig(
            vocab_size=256,
            unpadded_vocab_size=250,
            hidden_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            swa_num_attention_heads=4,
            swa_num_key_value_heads=4,
            swa_head_dim=16,
            sliding_window_size=8,
            d_rel=8,
            rel_extent=32,
            log_scaling_n_floor=8,      # small, so tau != 1 is exercised
            log_scaling_alpha=0.1,
            local_layer_ids=[0, 1, 2, 3, 4, 6, 7],   # global layer at 5
            dense_mlp_idx=2,
            dense_intermediate_size=96,
            moe_intermediate_size=32,
            n_routed_experts=8,
            num_experts_per_tok=2,
            n_shared_experts=2,
            route_scale=2.0,
            logits_mup_width_multiplier=4.0,
            max_position_embeddings=4096,
            eos_token_id=None,
        )

    model = InklingForCausalLM(cfg).eval().float()

    prompt = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 24

    with torch.no_grad():
        gen = model.generate(
            ids, max_new_tokens=n_new, do_sample=False, use_cache=True
        )
        full = gen[0].tolist()
        tf = model(torch.tensor([full], dtype=torch.long)).logits[0].argmax(-1).tolist()

    model.save_pretrained(out, safe_serialization=True)
    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf}
    with open(f"{out}/ref_inkling.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny model + ref_inkling.json to {out}/")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
