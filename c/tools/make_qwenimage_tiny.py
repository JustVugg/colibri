#!/usr/bin/env python3
"""Tiny random Qwen-Image-2.1 pipeline and its float32 reference.

Writes OUTDIR in the diffusers layout of the real checkpoint (model_index.json,
processor/, text_encoder/, transformer/, vae/, scheduler/), with the same
classes at toy widths, then runs it and writes OUTDIR/ref/ref.json and
OUTDIR/ref/ref.safetensors. The C engine loads OUTDIR exactly like the real
model and is held to those tensors.

  python3 tools/make_qwenimage_tiny.py OUTDIR [--width 64 --height 96 --steps 3]

What is tiny and what is not:
  text_encoder  Qwen3VLForConditionalGeneration, text model hidden 64, 2 layers,
                GQA 4/2, head_dim 16, MLP 128, interleaved M-RoPE [4, 2, 2],
                rope_theta 5e6; a one-block vision tower that t2i never runs.
                config.json uses the real checkpoint's keys (rope_scaling,
                rope_theta), bf16 weights in two shards plus an index.
  transformer   QwenImage21Transformer2DModel, 2 blocks, 2 heads x 32,
                RoPE axes [8, 12, 12], context 64, causal_condition, bf16 in
                two shards plus an index.
  vae           AutoencoderKLQwenImage21 with base_dim 8 / decoder_base_dim 12
                but the real depth (dim_mult of 5, 16x spatial), RGBA, f32.
                z_dim 16 (not 64) so a hard-coded 64 anywhere shows up; the
                latents_mean/std are the real first 16 values.
  processor     byte-level BPE over the 256 bytes with a few merges, the Qwen
                pre-tokenizer and specials, and a chat template trimmed to the
                branches the pipeline reaches. No downloads.
  scheduler     the real FlowMatchEulerDiscreteScheduler config.

Every weight is random (seeded), including the norm scales, so a C port that
forgets a norm weight or the zero-centred (1 + w) fails. The weights are saved
in their storage dtype, loaded back from disk and only then run in float32,
so the reference is what a C engine reading these files can reproduce. The
reference goes through qwenimage_ref.py (the same staged code the real model
runs), and is then cross-checked against QwenImage21Pipeline.from_pretrained
run end to end on the same directory.

Reference stack (pin this in CI; the ds venv this was generated with):
  diffusers 0.41.0.dev0, git commit 80c7ed262aeffbeb43ef13ae04baeb9b84515a69
  (pip install "git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69")
  transformers 5.17.0, torch 2.14.0+cpu, tokenizers 0.23.2, safetensors 0.8.0,
  numpy 2.5.3, pillow 12.3.0, python 3.14.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwenimage_ref as qref  # noqa: E402

SEED = 20260927
PROMPT = "a red fox sitting in the snow, photograph"

# ---------------------------------------------------------------- tokenizer

# BPE merges in rank order. Enough to exercise rank priority: " red" stops at
# " r" + "ed" because (e, d) outranks (r, e), and " snow" needs four merges.
MERGES = [("Ġ", "t"), ("h", "e"), ("i", "n"), ("Ġt", "he"), ("o", "w"), ("Ġ", "s"), ("Ġs", "n"),
          ("Ġsn", "ow"), ("e", "d"), ("Ġ", "a"), ("r", "e")]
# Qwen's special tokens in the real checkpoint's order (151643..151656 there).
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
            "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
            "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
VOCAB_SIZE = 288        # embedding rows; ids past the last special are never produced

QWEN_SPLIT_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+|\s+(?!\S)|\s+")

# The real processor/chat_template.jinja reduced to the branches the pipeline can reach (no tools, no tool
# role). For the system message the pipeline tokenizes to find drop_idx it renders the same text:
# "<|im_start|>system\n" + content + "<|im_end|>\n".
CHAT_TEMPLATE = r"""{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].content is string %}
        {{- messages[0].content }}
    {%- else %}
        {%- for content in messages[0].content %}
            {%- if 'text' in content %}
                {{- content.text }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- endif %}
{%- for message in messages %}
    {%- if message.role == "user" or message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n' }}
        {%- if message.content is string %}
            {{- message.content }}
        {%- else %}
            {%- for content in message.content %}
                {%- if content.type == 'image' or 'image' in content or 'image_url' in content %}
                    {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
                {%- elif content.type == 'video' or 'video' in content %}
                    {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
                {%- elif 'text' in content %}
                    {{- content.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
"""

# Image and video processor configs as in the real checkpoint (t2i never runs them, the processor loads them).
PREPROCESSOR_CONFIG = {
    "crop_size": None, "data_format": "channels_first", "default_to_square": True, "device": None,
    "disable_grouping": None, "do_center_crop": None, "do_convert_rgb": True, "do_normalize": True,
    "do_pad": None, "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "Qwen2VLImageProcessorFast", "image_std": [0.5, 0.5, 0.5],
    "input_data_format": None, "max_pixels": None, "merge_size": 2, "min_pixels": None, "pad_size": None,
    "patch_size": 16, "processor_class": "Qwen3VLProcessor", "resample": 3,
    "rescale_factor": 0.00392156862745098, "return_tensors": None,
    "size": {"longest_edge": 16777216, "shortest_edge": 65536}, "temporal_patch_size": 2,
}
VIDEO_PREPROCESSOR_CONFIG = {
    "crop_size": None, "data_format": "channels_first", "default_to_square": True, "device": None,
    "do_center_crop": None, "do_convert_rgb": True, "do_normalize": True, "do_rescale": True, "do_resize": True,
    "do_sample_frames": True, "fps": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    "input_data_format": None, "max_frames": 768, "merge_size": 2, "min_frames": 4, "num_frames": None,
    "pad_size": None, "patch_size": 16, "processor_class": "Qwen3VLProcessor", "resample": 3,
    "rescale_factor": 0.00392156862745098, "return_metadata": False,
    "size": {"longest_edge": 25165824, "shortest_edge": 4096}, "temporal_patch_size": 2,
    "video_metadata": None, "video_processor_type": "Qwen3VLVideoProcessor",
}


def bytes_to_unicode() -> list[tuple[int, str]]:
    """GPT-2's byte map in its own order: printable bytes first, so ' ' lands on id 220 as in Qwen."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [(b, chr(c)) for b, c in zip(bs, cs)]


def special_ids() -> dict[str, int]:
    base = 256 + len(MERGES)
    return {tok: base + i for i, tok in enumerate(SPECIALS)}


def write_processor(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    vocab = {ch: i for i, (_, ch) in enumerate(bytes_to_unicode())}
    for a, b in MERGES:
        vocab[a + b] = len(vocab)
    specials = special_ids()
    assert max(specials.values()) < VOCAB_SIZE
    added = [{"id": i, "content": t, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for t, i in specials.items()]
    byte_level = {"add_prefix_space": False, "trim_offsets": False, "use_regex": False}
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": QWEN_SPLIT_REGEX}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", **byte_level}]},
        "post_processor": {"type": "ByteLevel", **byte_level},
        "decoder": {"type": "ByteLevel", **byte_level},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "",
                  "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": [[a, b] for a, b in MERGES]},
    }
    (out / "tokenizer.json").write_text(json.dumps(tokenizer, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in MERGES), encoding="utf-8")
    (out / "added_tokens.json").write_text(json.dumps(specials, indent=2), encoding="utf-8")
    additional = SPECIALS[1:]
    tok_entry = {"lstrip": False, "normalized": False, "rstrip": False, "single_word": False}
    (out / "special_tokens_map.json").write_text(json.dumps({
        "additional_special_tokens": additional,
        "eos_token": {"content": "<|im_end|>", **tok_entry},
        "pad_token": {"content": "<|endoftext|>", **tok_entry}}, indent=2), encoding="utf-8")
    (out / "tokenizer_config.json").write_text(json.dumps({
        "add_bos_token": False, "add_prefix_space": False,
        "added_tokens_decoder": {str(i): {"content": t, **tok_entry, "special": True} for t, i in specials.items()},
        "additional_special_tokens": additional, "bos_token": None, "clean_up_tokenization_spaces": False,
        "eos_token": "<|im_end|>", "errors": "replace", "extra_special_tokens": {}, "model_max_length": 262144,
        "pad_token": "<|endoftext|>", "processor_class": "Qwen3VLProcessor", "split_special_tokens": False,
        "tokenizer_class": "Qwen2Tokenizer", "unk_token": None}, indent=2), encoding="utf-8")
    (out / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")
    (out / "preprocessor_config.json").write_text(json.dumps(PREPROCESSOR_CONFIG, indent=2), encoding="utf-8")
    (out / "video_preprocessor_config.json").write_text(json.dumps(VIDEO_PREPROCESSOR_CONFIG, indent=2),
                                                         encoding="utf-8")


# ---------------------------------------------------------------- models

def text_encoder_config() -> dict:
    """The real text_encoder/config.json, key for key, at toy sizes."""
    sp = special_ids()
    return {
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "dtype": "bfloat16",
        "image_token_id": sp["<|image_pad|>"],
        "model_type": "qwen3_vl",
        "text_config": {
            "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": sp["<|endoftext|>"],
            "dtype": "bfloat16", "eos_token_id": sp["<|im_end|>"], "head_dim": 16, "hidden_act": "silu",
            "hidden_size": 64, "initializer_range": 0.02, "intermediate_size": 128,
            "max_position_embeddings": 262144, "model_type": "qwen3_vl_text", "num_attention_heads": 4,
            "num_hidden_layers": 2, "num_key_value_heads": 2, "rms_norm_eps": 1e-06,
            # sections sum to head_dim / 2 like the real [24, 20, 20] over 128
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [4, 2, 2], "rope_type": "default"},
            "rope_theta": 5000000, "use_cache": True, "vocab_size": VOCAB_SIZE,
        },
        "tie_word_embeddings": False,
        "transformers_version": __import__("transformers").__version__,
        "video_token_id": sp["<|video_pad|>"],
        "vision_config": {
            "deepstack_visual_indexes": [0], "depth": 1, "dtype": "bfloat16", "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 32, "in_channels": 3, "initializer_range": 0.02, "intermediate_size": 64,
            "model_type": "qwen3_vl", "num_heads": 2, "num_position_embeddings": 16, "out_hidden_size": 64,
            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
        },
        "vision_end_token_id": sp["<|vision_end|>"],
        "vision_start_token_id": sp["<|vision_start|>"],
    }


TRANSFORMER_CONFIG = dict(patch_size=1, in_channels=16, out_channels=16, num_layers=2, attention_head_dim=32,
                          num_attention_heads=2, context_in_dim=64, mlp_ratio=3, axes_dims_rope=(8, 12, 12),
                          eps=1e-6, causal_condition=True)
REAL_LATENTS_MEAN = [0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651,
                     0.2732, 0.7562, -0.6163, -1.0277, 3.8363]
REAL_LATENTS_STD = [3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904,
                    2.8778, 3.7675, 3.732, 5.0756]
VAE_CONFIG = dict(base_dim=8, decoder_base_dim=12, z_dim=16, dim_mult=[1, 2, 4, 8, 8], num_res_blocks=2,
                  attn_scales=[], temperal_downsample=[False, True, True, True], dropout=0.0,
                  latents_mean=REAL_LATENTS_MEAN, latents_std=REAL_LATENTS_STD, is_residual=True, in_channels=4,
                  out_channels=4, patch_size=None, scale_factor_temporal=8, scale_factor_spatial=16)
# Picked once so most of the decoder's raw output lands inside the [-1, 1] clamp and a few percent saturates
# (measured with the defaults: std 0.53, 7.3% clamped), so both the float and the u8 comparison mean something.
VAE_CONV_OUT_GAIN = 0.75
SCHEDULER_CONFIG = dict(base_image_seq_len=256, base_shift=0.5, invert_sigmas=False, max_image_seq_len=8192,
                        max_shift=0.9, num_train_timesteps=1000, shift=1.0, shift_terminal=0.02,
                        stochastic_sampling=False, time_shift_type="exponential", use_beta_sigmas=False,
                        use_dynamic_shifting=True, use_exponential_sigmas=False, use_karras_sigmas=False)
MODEL_INDEX = {
    "_class_name": "QwenImage21Pipeline",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}


def randomize(model: torch.nn.Module, gen: torch.Generator, conv_out_gain: dict | None = None):
    """Seeded weights at a scale that keeps activations O(1): fan-in scaled matrices and kernels, norm scales
    around their identity (1 for plain RMSNorm/gamma, 0 for the zero-centred ones), small biases."""
    zero_centred = {"txt_in.text_norm.weight"}
    with torch.no_grad():
        for name, p in model.named_parameters():
            r = torch.randn(p.shape, generator=gen, dtype=torch.float32)
            leaf = name.rsplit(".", 1)[-1]
            if name in zero_centred:
                v = 0.2 * r
            elif leaf == "gamma" or (leaf == "weight" and p.dim() == 1):
                v = 1.0 + 0.2 * r
            elif leaf == "bias":
                v = 0.1 * r
            elif p.dim() >= 2:
                fan_in = p[0].numel()
                v = r / fan_in ** 0.5
                if "embed_tokens" in name or "pos_embed" in name:
                    v = r
            else:
                v = r
            if conv_out_gain and name in conv_out_gain:
                v = v * conv_out_gain[name]
            p.copy_(v.to(p.dtype))


def build_models(out: Path, gen: torch.Generator):
    from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Transformer2DModel
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    # text encoder: write the real-format config first and build the model from what transformers reads back,
    # so the saved weights match the config a loader (C or Python) will parse
    te_dir = out / "text_encoder"
    te_dir.mkdir(parents=True, exist_ok=True)
    te_cfg = text_encoder_config()
    (te_dir / "config.json").write_text(json.dumps(te_cfg, indent=2))
    config = AutoConfig.from_pretrained(str(te_dir))
    te = Qwen3VLForConditionalGeneration(config).eval()
    randomize(te, gen)
    te.to(torch.bfloat16).save_pretrained(str(te_dir), max_shard_size="150KB")
    (te_dir / "config.json").write_text(json.dumps(te_cfg, indent=2))  # save_pretrained rewrote it in v5 keys
    del te

    tr = QwenImage21Transformer2DModel(**TRANSFORMER_CONFIG).eval()
    randomize(tr, gen)
    tr.to(torch.bfloat16).save_pretrained(str(out / "transformer"), max_shard_size="200KB")
    del tr

    vae = AutoencoderKLQwenImage21(**VAE_CONFIG).eval()
    randomize(vae, gen, conv_out_gain={"decoder.conv_out.weight": VAE_CONV_OUT_GAIN})
    vae.save_pretrained(str(out / "vae"))
    del vae

    FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG).save_pretrained(str(out / "scheduler"))
    import diffusers
    (out / "model_index.json").write_text(json.dumps({**MODEL_INDEX, "_diffusers_version": diffusers.__version__},
                                                     indent=2))


# ---------------------------------------------------------------- checks

def check_tokenizer(out: Path, enc: dict):
    """The ids the processor produced must be what the raw tokenizers library gives for the template text, so
    a C tokenizer reading tokenizer.json only has to match the library."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(out / "processor" / "tokenizer.json"))
    ids = tok.encode(enc["template_text"], add_special_tokens=False).ids
    if ids != enc["input_ids"]:
        raise SystemExit(f"processor ids differ from tokenizers on the template:\n{ids}\n{enc['input_ids']}")
    sys_ids = tok.encode(f"<|im_start|>system\n{enc['system_prompt']}<|im_end|>\n", add_special_tokens=False).ids
    if len(sys_ids) != enc["drop_idx"] or enc["input_ids"][:len(sys_ids)] != sys_ids:
        raise SystemExit("drop_idx is not the length of the tokenized system block")


def check_storage_dtypes(out: Path):
    from safetensors import safe_open

    want = {"text_encoder": "bfloat16", "transformer": "bfloat16", "vae": "float32"}
    for sub, dtype in want.items():
        files = sorted((out / sub).glob("*.safetensors"))
        for f in files:
            with safe_open(str(f), framework="pt") as st:
                for k in st.keys():
                    got = str(st.get_slice(k).get_dtype()).lower()
                    if got not in (dtype, {"bfloat16": "bf16", "float32": "f32"}[dtype]):
                        raise SystemExit(f"{f.name}:{k} is stored as {got}, expected {dtype}")


def crosscheck_pipeline(out: Path, prompt: str, latents_init: torch.Tensor, width: int, height: int, steps: int):
    """QwenImage21Pipeline.from_pretrained end to end on the fixture, f32, same latents."""
    from diffusers import QwenImage21Pipeline

    # the processor goes in explicitly only because a venv without torchvision cannot load it by itself
    pipe = QwenImage21Pipeline.from_pretrained(str(out), torch_dtype=torch.float32,
                                               processor=qref.load_processor(out))
    pipe.set_progress_bar_config(disable=True)
    final = {}

    def cb(p, i, t, kw):
        final["latents"] = kw["latents"].detach().clone()
        return {}

    with torch.no_grad():
        image = pipe(prompt=prompt, latents=latents_init[None], width=width, height=height,
                     num_inference_steps=steps, true_cfg_scale=1.0, callback_on_step_end=cb).images[0]
    return np.array(image), final["latents"][0].float()


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=96)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if args.width % 32 or args.height % 32:
        raise SystemExit("width and height must be multiples of 32")
    torch.set_num_threads(args.threads)
    t_start = time.time()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    write_processor(out / "processor")
    build_models(out, gen)
    check_storage_dtypes(out)

    # reference, from the files just written, in f32
    f32 = torch.float32
    encs, te_info = qref.encode_prompts(out, [args.prompt], f32, dump=True)
    enc = encs[0]
    check_tokenizer(out, enc)
    latents_init = qref.pack_initial_latents(args.seed + 1, TRANSFORMER_CONFIG["in_channels"], args.width,
                                             args.height)
    run = {"prompt_embeds": enc["prompt_embeds"], "latents_init": latents_init, "width": args.width,
           "height": args.height, "steps": args.steps}
    dens, tr_info = qref.denoise(out, [run], f32, dump=True, redundancy=False)
    decs, vae_info = qref.decode(out, [{"final": dens[0]["final"], "width": args.width, "height": args.height}],
                                 dump=True)
    den, dec = dens[0], decs[0]

    rgba_e2e, final_e2e = crosscheck_pipeline(out, args.prompt, latents_init, args.width, args.height, args.steps)
    lat_diff = float((final_e2e - den["final"]).abs().max())
    rgba_equal = bool(np.array_equal(rgba_e2e, dec["rgba"].numpy()))
    if not rgba_equal or lat_diff > 1e-5:
        raise SystemExit(f"staged reference disagrees with the end-to-end pipeline: latents max diff {lat_diff}, "
                         f"rgba equal {rgba_equal}")

    raw = dec["vae_out_raw"]
    sched_cfg = dict(qref.load_scheduler(out).config)
    constants = qref.model_constants(te_info["config"], tr_info["config"], vae_info["config"], sched_cfg)
    meta = {"model": "qwen-image-2.1 tiny random (make_qwenimage_tiny.py)", "width": args.width,
            "height": args.height, "steps": args.steps, "seed": args.seed, "latents_init": latents_init,
            "latents_note": f"torch.randn((1, 1, C, h, w), generator=torch.Generator('cpu').manual_seed("
                            f"{args.seed + 1}), float32), then QwenImage21Pipeline._pack_latents",
            "compute_dtype": {"text_encoder": "float32", "transformer": "float32", "vae": "float32"},
            "constants": constants,
            "timings": {"total_s": time.time() - t_start, "threads": args.threads},
            "peak_rss": {"total": qref.peak_rss_gb()}}
    ref_dir = out / "ref"
    qref.write_run(ref_dir, enc, den, dec, meta, dump=True)
    ref = json.loads((ref_dir / "ref.json").read_text())
    ref["pipeline_crosscheck"] = {
        "what": "QwenImage21Pipeline.from_pretrained(OUTDIR, torch_dtype=float32) run end to end with the same "
                "prompt and latents",
        "final_latents_max_abs_diff": lat_diff, "rgba_equal": rgba_equal}
    ref["tokenizer_crosscheck"] = "processor ids == tokenizers.Tokenizer.from_file(tokenizer.json).encode(template)"
    ref["vae_out_raw_stats"] = {"std": float(raw.std()), "min": float(raw.min()), "max": float(raw.max()),
                                "frac_clamped": float((raw.abs() > 1).float().mean())}
    ref["storage_dtypes"] = {"text_encoder": "bfloat16", "transformer": "bfloat16", "vae": "float32"}
    (ref_dir / "ref.json").write_text(json.dumps(ref, indent=1))
    print(f"wrote {out}: {len(enc['input_ids'])} ids (drop {enc['drop_idx']}), "
          f"{latents_init.shape[0]} image tokens, {args.steps} steps, vae raw std {float(raw.std()):.3f}, "
          f"{100 * ref['vae_out_raw_stats']['frac_clamped']:.1f}% clamped, pipeline cross-check ok "
          f"(latents diff {lat_diff:.2e}), {time.time() - t_start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
