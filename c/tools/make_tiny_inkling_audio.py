#!/usr/bin/env python3
"""Build a tiny random-weight MULTIMODAL Inkling fixture for inkling.c audio.

Same idea as make_tiny_inkling.py, but through InklingForConditionalGeneration
with an audio tower, so the DMel path is validated end to end: placeholder
tokens in the prompt, per-frame embedding-table lookup, the audio RMSNorm, and
the masked_scatter splice. ref_inkling.json gains "dmel" (flattened
[n_frames, n_mel_bins] levels); the C engine must reproduce tf_pred and the
greedy continuation token-for-token (run with bits=0 for f32-exact experts).

Usage: python3 make_tiny_inkling_audio.py <outdir>
"""
import json
import sys

import torch

try:
    from transformers import InklingForConditionalGeneration
    from transformers.models.inkling.configuration_inkling import (
        InklingAudioConfig,
        InklingConfig,
        InklingTextConfig,
        InklingVisionConfig,
    )
except ImportError:
    sys.exit("transformers has no Inkling support: pip install -U transformers")

AUDIO_TOK = 251        # inside the padded vocab (256), outside unpadded (250)
AUDIO_BOS = 252
N_FRAMES = 5
N_MEL = 8
MEL_VOCAB = 16


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "tiny_inkling_audio"
    torch.manual_seed(0)

    text_cfg = InklingTextConfig(
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
        log_scaling_n_floor=8,
        log_scaling_alpha=0.1,
        local_layer_ids=[0, 1, 2, 3, 4, 6, 7],
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
    audio_cfg = InklingAudioConfig(
        n_mel_bins=N_MEL,
        mel_vocab_size=MEL_VOCAB,
        text_hidden_size=64,
    )
    vision_cfg = InklingVisionConfig(
        decoder_dmodel=64,
        patch_size=4,
        temporal_patch_size=1,
        n_channels=3,
        n_layers=2,
    )
    cfg = InklingConfig(
        text_config=text_cfg,
        audio_config=audio_cfg,
        vision_config=vision_cfg,
        audio_token_id=AUDIO_TOK,
        audio_bos_token_id=AUDIO_BOS,
    )

    # The vision tower is irrelevant to the audio oracle (and its constructor
    # trips on tensor-typed dims with tiny configs) — stub it out. The C engine
    # skips model.visual.* regardless.
    import transformers.models.inkling.modeling_inkling as _mi

    class _NoVision(torch.nn.Module):
        def __init__(self, config):
            super().__init__()

    _mi.InklingVisionModel = _NoVision

    model = InklingForConditionalGeneration(cfg).eval().float()

    g = torch.Generator().manual_seed(7)
    dmel = torch.randint(0, MEL_VOCAB, (1, N_FRAMES, N_MEL), generator=g)

    prompt = [7, 42, 199, AUDIO_BOS] + [AUDIO_TOK] * N_FRAMES + [3, 88, 154, 21, 60]
    ids = torch.tensor([prompt], dtype=torch.long)
    n_new = 24

    with torch.no_grad():
        gen = model.generate(
            ids,
            audio_input_ids=dmel,
            max_new_tokens=n_new,
            do_sample=False,
            use_cache=True,
        )
        full = gen[0].tolist()
        tf = (
            model(torch.tensor([full], dtype=torch.long), audio_input_ids=dmel)
            .logits[0]
            .argmax(-1)
            .tolist()
        )

    model.save_pretrained(out, safe_serialization=True)
    ref = {
        "prompt_ids": prompt,
        "full_ids": full,
        "tf_pred": tf,
        "dmel": dmel[0].flatten().tolist(),
    }
    with open(f"{out}/ref_inkling.json", "w") as f:
        json.dump(ref, f)
    print(f"saved tiny multimodal model + ref_inkling.json to {out}/")
    print("continuation:", full[len(prompt):])


if __name__ == "__main__":
    main()
