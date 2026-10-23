#!/usr/bin/env python3
"""Tiny Qwen3.8-27B-shaped vision-language fixture and its token-exact reference.

Qwen3_5ForConditionalGeneration at toy widths: the dense text model of the
qwen38-27b-dense geometry (make_qwen36_tiny.py) plus a small ViT of the tower
the whole family shares. transformers generates greedily from a prompt that
holds ONE image, and ref.json records the prompt ids, the generated ids and the
image patches, so `./qwen36 <cap> 8 ref.json` can be held to the same tokens
(#1757). A random model answers anything; token equality is the test.

The image grid is 4 x 8 patches on purpose: 2 x 4 tokens after the merge, so
the text after the image resumes at start + max(2, 4) = start + 4 while 8
positions went by. Every position after the image then depends on getting
Qwen3.5's 3D rope positions right, not just the first one.

  python3 tools/make_qwen36_vl_tiny.py --out qwen36_vl_tiny
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
import make_qwen36_tiny  # noqa: E402

SEED = 20260927
IMAGE, VIDEO, START, END = 300, 301, 302, 303     # special ids inside the 320-token toy vocabulary


SPECIALS = {"<|image_pad|>": IMAGE, "<|video_pad|>": VIDEO, "<|vision_start|>": START,
            "<|vision_end|>": END, "<|im_start|>": 304, "<|im_end|>": 305, "<|endoftext|>": 319}


def write_tokenizer(out: Path):
    """One token per ASCII byte (the GPT-2 byte map tok.h builds) plus Qwen's
    special tokens as added tokens at their fixture ids, so a serve prompt written
    as text reaches the engine as exactly the oracle's ids."""
    from prefix_serve_harness import byte_level_vocab
    # ASCII bytes for ids 0..127 (the prompts only use a few of them); every
    # other id decodes to printable filler. A lone byte >= 0x80 is not valid UTF-8,
    # and the engine holds such bytes back from the stream, so ids that decode to
    # one would hide from a test that compares what the client receives.
    vocab = {k: v for k, v in byte_level_vocab(256).items() if v < 128}
    for i in range(128, 320):
        if i not in SPECIALS.values():
            vocab[f"t{i}"] = i
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t, i in sorted(SPECIALS.items(), key=lambda kv: kv[1])]
    (out / "tokenizer.json").write_text(json.dumps({
        "version": "1.0", "added_tokens": added,
        "model": {"type": "BPE", "vocab": vocab, "merges": []}}), encoding="utf-8")


def build(out: Path, grid_h=4, grid_w=8, max_new=16, seed=SEED):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    geo = make_qwen36_tiny.GEOMETRIES["qwen38-27b-dense"]
    hidden, layers = geo["hidden"], geo["n_layers"]
    text = dict(
        vocab_size=320, hidden_size=hidden, intermediate_size=geo["inter"],
        num_hidden_layers=layers, num_attention_heads=geo["q_heads"],
        num_key_value_heads=geo["kv_heads"], head_dim=32,
        max_position_embeddings=512, rms_norm_eps=1e-6, tie_word_embeddings=False,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=geo["dn_key_heads"], linear_num_value_heads=geo["dn_value_heads"],
        layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(layers)],
        hidden_act="silu", attention_bias=False, attention_dropout=0.0, use_cache=True,
        # the real checkpoint's rope: interleaved M-RoPE over a quarter of head_dim.
        # head_dim 32 gives four frequencies, so [2, 1, 1] puts at least one on
        # every axis (the real [11, 11, 10] spreads 32 the same way)
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                         "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                         "mrope_section": [2, 1, 1]},
        partial_rotary_factor=0.25, pad_token_id=0, bos_token_id=1, eos_token_id=319,
    )
    vision = dict(depth=2, hidden_size=32, num_heads=4, intermediate_size=64, patch_size=16,
                  spatial_merge_size=2, temporal_patch_size=2, in_channels=3,
                  out_hidden_size=hidden, num_position_embeddings=64,
                  hidden_act="gelu_pytorch_tanh", deepstack_visual_indexes=[])
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=IMAGE,
                           video_token_id=VIDEO, vision_start_token_id=START,
                           vision_end_token_id=END, tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = Qwen3_5ForConditionalGeneration(config).eval()
    with torch.no_grad():                       # a tower whose output is not noise-small
        for name, parameter in model.named_parameters():
            if "visual" in name:
                parameter.copy_(torch.randn_like(parameter) * 0.5)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    write_tokenizer(out)

    tokens = (grid_h // 2) * (grid_w // 2)
    prompt = [1, 2, 3, START] + [IMAGE] * tokens + [END, 4, 5, 6]
    generator = torch.Generator().manual_seed(seed + 1)
    features = 3 * 2 * 16 * 16
    patches = torch.randn(grid_h * grid_w, features, generator=generator)
    input_ids = torch.tensor([prompt])
    with torch.no_grad():
        full = model.generate(
            input_ids, pixel_values=patches, image_grid_thw=torch.tensor([[1, grid_h, grid_w]]),
            mm_token_type_ids=(input_ids == IMAGE).int(), max_new_tokens=max_new,
            do_sample=False, use_cache=True)[0].tolist()
    reference = {"prompt_ids": prompt, "full_ids": full, "mode": "full", "model": "qwen36_vl_tiny",
                 "image": {"grid_h": grid_h, "grid_w": grid_w,
                           "patches": [round(float(v), 7) for v in patches.reshape(-1)]}}
    (out / "ref.json").write_text(json.dumps(reference))
    print(f"wrote {out}: {tokens} image tokens, prompt {len(prompt)}, full {full[len(prompt):]}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=16)
    args = parser.parse_args()
    build(args.out, max_new=args.max_new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
