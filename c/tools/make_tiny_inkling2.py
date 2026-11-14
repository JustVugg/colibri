#!/usr/bin/env python3
"""Build a tiny random-weight Inkling-2 text model and its oracle for inkling.c.

The reference is transformers' Inkling2ForCausalLM (models/inkling2). It saves a small snapshot
(safetensors + config.json) and ref_inkling2.json with {prompt_ids, full_ids, tf_pred} and the
prompt's last-position logits, which the C engine must reproduce (bits=0: f32 experts).

The tiny config exercises every branch of the big model: sliding and global layers (a global one
every sixth, as layer_types builds it), two blocks of the attention residuals, the global layers'
dynamic short convolutions and q/k/v input norms, the log-length tau (n_floor below the sequence),
the L2 clips, the softcapped SwiGLU, the logit softcap and an unpadded vocabulary. Every parameter
is drawn at random, the ones transformers starts at zero or one included (the residuals' queries,
the static convolution taps, the clip scales, the routing bias), so that none of them is a no-op.

--g64 makes the hidden size and the expert width multiples of 64, as int4 g64 needs
(tools/convert_inkling2_int4.py). --int4-ref takes the container that converter wrote from this
snapshot and writes the oracle of the same model with the container's experts (its int4 g64
weights decoded to f32): the engine reading the container must reproduce it.

--vision adds a tiny vision tower and one image among the prompt's tokens
(Inkling2ForConditionalGeneration); the oracle carries the image's pixels.

Usage: python3 make_tiny_inkling2.py <outdir> [--g64 | --vision]
       python3 make_tiny_inkling2.py --int4-ref <container> <snapshot> <outdir>
"""
import glob
import json
import os
import shutil
import sys

import numpy as np
import torch

try:
    from transformers import Inkling2ForCausalLM, Inkling2TextConfig
except ImportError:
    sys.exit("transformers has no Inkling-2 support: it needs transformers with models/inkling2")

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
    return Inkling2TextConfig(
        vocab_size=256,
        unpadded_vocab_size=250,
        hidden_size=128 if g64 else 64,
        num_hidden_layers=12,           # global layers at 5 and 11
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
        n_layers_per_attn_res_block=6,
        moe_intermediate_size=64 if g64 else 32,
        n_routed_experts=8,
        num_experts_per_tok=2,
        route_scale=2.0,
        mlp_gate_softcap=1.0,           # low, so the softcaps bind
        mlp_up_softcap=2.0,
        final_logit_softcapping=5.0,
        max_position_embeddings=4096,
        eos_token_id=None,
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


def build(out, g64):
    torch.manual_seed(0)
    model = Inkling2ForCausalLM(text_config(g64)).eval().float()
    randomize(model)
    model.save_pretrained(out, safe_serialization=True)
    oracle(model, out)
    print(f"saved tiny model + ref_inkling2.json to {out}/")


IMG = 252             # the image placeholder (Inkling-2-Small's is 200054), past the unpadded vocab
IMG_ROWS, IMG_COLS = 2, 3


def build_vision(out):
    """The same text model with a tiny vision tower (Inkling2ForConditionalGeneration): two
    layers, 4-pixel patches merged 2 x 2 (an image token is an 8 x 8 tile), the position table a
    4 x 4 grid; one random 2 x 3-token image among the prompt's tokens. The oracle adds the
    image's pixels ([tokens][tile][tile][3], in [-1, 1], as Inkling2ImageProcessor gives them)
    and its grid."""
    from transformers import Inkling2Config, Inkling2ForConditionalGeneration, Inkling2VisionConfig
    torch.manual_seed(0)
    vision = Inkling2VisionConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, patch_size=4,
                                  spatial_merge_size=2, posemb_grid_size=4, projection_hidden_dim=48,
                                  text_hidden_size=64, mlp_ratio=2.0, layer_norm_eps=1e-5)
    cfg = Inkling2Config(text_config=text_config(False).to_dict(), vision_config=vision.to_dict(), image_token_id=IMG)
    model = Inkling2ForConditionalGeneration(cfg).eval().float()
    randomize(model)
    g = torch.Generator().manual_seed(2)
    tile = vision.patch_size * vision.spatial_merge_size
    pixels = torch.rand(IMG_ROWS * IMG_COLS, tile, tile, 3, generator=g) * 2 - 1
    grid = torch.tensor([[1, IMG_ROWS, IMG_COLS]])
    prompt = PROMPT[:4] + [IMG] * (IMG_ROWS * IMG_COLS) + PROMPT[4:]
    ids = torch.tensor([prompt], dtype=torch.long)
    with torch.no_grad():
        gen = model.generate(ids, pixel_values=pixels, image_grid_thw=grid, max_new_tokens=N_NEW, do_sample=False)
        full = gen[0].tolist()
        tf = model(torch.tensor([full]), pixel_values=pixels, image_grid_thw=grid).logits[0].argmax(-1).tolist()
        last = model(ids, pixel_values=pixels, image_grid_thw=grid).logits[0, -1].tolist()
    model.save_pretrained(out, safe_serialization=True)
    ref = {"prompt_ids": prompt, "full_ids": full, "tf_pred": tf, "prompt_last_logits": last,
           "image_token_id": IMG, "image_grid_thw": [1, IMG_ROWS, IMG_COLS],
           "pixel_values": pixels.flatten().tolist()}
    with open(f"{out}/ref_inkling2.json", "w") as f:
        json.dump(ref, f)
    print("continuation:", full[len(prompt):])
    print(f"saved tiny model with vision + ref_inkling2.json to {out}/")


def int4_ref(container, snap, out):
    """The snapshot with its experts replaced by the container's int4 g64 weights, and its oracle."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import convert_inkling2_int4 as cv
    from safetensors.numpy import load_file, save_file
    tens = load_file(os.path.join(snap, "model.safetensors"))
    idx = {}
    for p in glob.glob(os.path.join(container, "*.safetensors")):
        for k, (dt, shape, off, nb) in cv.read_header(p)[0].items():
            idx[k] = (p, dt, shape, off, nb)

    def experts(name):
        p, dt, (E, R, half), off, nb = idx[name]
        q = np.frombuffer(cv.read_bytes(p, off, nb), np.uint8).reshape(E, R, half)
        p, dt, shape, off, nb = idx[name + ".qs"]
        s = np.frombuffer(cv.read_bytes(p, off, nb), "<f4").reshape(shape)
        return np.stack([cv.unpack_planar(q[e], s[e]) for e in range(E)])

    n = 0
    for k in list(tens):
        m = cv.EXPERT_RE.match(k)
        if not m:
            continue
        if m.group(3) == "w13":
            w = experts(m.group(1) + "gate_up_proj")          # [E, 2I, D], gate rows then up rows
            i = w.shape[1] // 2
            inter = np.empty_like(w)
            inter[:, 0::2], inter[:, 1::2] = w[:, :i], w[:, i:]
            tens[k] = inter
        else:
            tens[k] = experts(m.group(1) + "down_proj")
        n += 1
    os.makedirs(out, exist_ok=True)
    save_file(tens, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    for fn in ("config.json", "generation_config.json"):
        if os.path.exists(os.path.join(snap, fn)):
            shutil.copy(os.path.join(snap, fn), out)
    model = Inkling2ForCausalLM.from_pretrained(out, torch_dtype=torch.float32).eval()
    oracle(model, out)
    print(f"{n} expert tensors from {container}: oracle in {out}/ref_inkling2.json")


def main():
    a = sys.argv[1:]
    if a and a[0] == "--int4-ref":
        if len(a) != 4:
            sys.exit(__doc__)
        int4_ref(*a[1:])
        return
    if "--vision" in a:
        a = [x for x in a if x != "--vision"]
        build_vision(a[0] if a else "tiny_inkling2_vision")
        return
    g64 = "--g64" in a
    a = [x for x in a if x != "--g64"]
    build(a[0] if a else "tiny_inkling2", g64)


if __name__ == "__main__":
    main()
