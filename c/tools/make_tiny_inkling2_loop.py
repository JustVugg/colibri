#!/usr/bin/env python3
"""Build a tiny random-weight Inkling-2-Mini (inkling2_loop) text model and its oracle for inkling.c.

The reference is transformers' Inkling2LoopForCausalLM (models/inkling2_loop). It saves a small
snapshot (safetensors + config.json) and ref_inkling2.json with {prompt_ids, full_ids, tf_pred} and
the prompt's last-position logits, which the C engine must reproduce (bits=0: f32 experts).

Inkling-2-Mini is Inkling-2 with a loop: its middle layers run several passes with shared weights,
each pass with its own key/value and convolution caches, and from the second pass every loop layer
takes four messages (query, key, value, MLP) from the previous pass's block states and final
stream, mixed into its inputs by low-rank gates. The tiny config keeps that shape: 12 layers in
blocks of 3, the loop over layers 3-8 (two block boundaries inside it, at 3 and 6) for 3 passes,
a global layer inside the loop (5) and one after it (11), gates of rank 8. Every parameter is drawn
at random, the message queries and gates included, so that none of them is a no-op.

--g64 makes the hidden size and the expert width multiples of 64, as int4 g64 needs. --int4-ref
takes the container tools/convert_inkling2_int4.py wrote from this snapshot and writes the oracle
of the same model with the container's experts, as make_tiny_inkling2.py --int4-ref does.

Usage: python3 make_tiny_inkling2_loop.py <outdir> [--g64]
       python3 make_tiny_inkling2_loop.py --int4-ref <container> <snapshot> <outdir>
"""
import json
import os
import sys

import torch

try:
    from transformers import Inkling2LoopForCausalLM, Inkling2LoopTextConfig
except ImportError:
    sys.exit("transformers has no Inkling-2-Mini support: it needs transformers with models/inkling2_loop")

PROMPT = [7, 42, 199, 3, 88, 154, 21, 60, 9, 133, 77, 245, 17, 4, 230, 101, 66, 12]
N_NEW = 24


def oracle(model, out):
    ids = torch.tensor([PROMPT], dtype=torch.long)
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, use_cache=True)
        full = gen[0].tolist()
        logits = model(torch.tensor([full], dtype=torch.long)).logits[0]
        tf = logits.argmax(-1).tolist()
        last = model(ids).logits[0, -1].tolist()
    ref = {"prompt_ids": PROMPT, "full_ids": full, "tf_pred": tf, "prompt_last_logits": last}
    with open(f"{out}/ref_inkling2.json", "w") as f:
        json.dump(ref, f)
    print("continuation:", full[len(PROMPT):])


def text_config(g64):
    return Inkling2LoopTextConfig(
        vocab_size=256,
        unpadded_vocab_size=250,
        hidden_size=128 if g64 else 64,
        num_hidden_layers=12,           # global layers at 5 (inside the loop) and 11
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=16,
        sliding_window_size=8,
        d_rel=8,
        rel_extent=32,
        log_scaling_n_floor=8,          # small, so tau != 1 is exercised
        log_scaling_alpha=0.1,
        qk_l2_clip_threshold=2.0,       # low, so the clips bind
        rel_l2_clip_threshold=1.0,
        n_layers_per_attn_res_block=3,  # blocks at 0, 3, 6, 9: two of them inside the loop
        moe_intermediate_size=64 if g64 else 32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        route_scale=2.0,
        mlp_gate_softcap=1.0,           # low, so the softcaps bind
        mlp_up_softcap=2.0,
        final_logit_softcapping=5.0,
        max_position_embeddings=4096,
        eos_token_id=None,
        loop_layer_index_start=3,
        loop_layer_index_end=9,
        loop_num_passes=3,
        loop_gate_rank=8,
    )


def randomize(model):
    """Every parameter drawn at random, gains near one (the ones transformers starts at zero or
    one included: none of them is a no-op)."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1 and ("norm" in name or name.endswith(".scale")):
                p.copy_(1.0 + 0.2 * torch.randn(p.shape, generator=g))     # gains near one
            else:
                p.copy_(0.2 * torch.randn(p.shape, generator=g))


def int4_ref(container, snap, out):
    """make_tiny_inkling2.py's int4_ref on the loop's model class."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_tiny_inkling2 as base
    base.Inkling2ForCausalLM = Inkling2LoopForCausalLM
    base.oracle = oracle
    base.int4_ref(container, snap, out)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--int4-ref":
        if len(sys.argv) != 5:
            sys.exit(__doc__)
        int4_ref(*sys.argv[2:])
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit(__doc__)
    out = args[0]
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(0)
    model = Inkling2LoopForCausalLM(text_config("--g64" in sys.argv)).eval().float()
    randomize(model)
    model.save_pretrained(out, safe_serialization=True)
    oracle(model, out)
    print(f"saved tiny model + ref_inkling2.json to {out}/")


if __name__ == "__main__":
    main()
