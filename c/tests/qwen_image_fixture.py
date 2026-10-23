"""A fake Qwen-Image pipeline directory and the stub engine, for the tests.

The directory has the real checkpoint's layout (model_index.json at the root,
no config.json, one folder per component, processor/tokenizer.json) and
safetensors files whose headers name real tensor names with tiny shapes. The
data is zeros: nothing here reads it, the planner and `coli info` only parse
headers. The engine is tools/qwenimage_stub.py.
"""
import json
import os
import struct
from pathlib import Path

C_DIR = Path(__file__).resolve().parent.parent
STUB = C_DIR / "tools" / "qwenimage_stub.py"

INDEX = {
    "_class_name": "QwenImage21Pipeline",
    "_diffusers_version": "0.37.0.dev0",
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}

# (name, dtype, shape) per component. The text encoder carries the three kinds
# of tensor the planner tells apart: language model (loaded), vision tower and
# lm_head (left on disk).
TENSORS = {
    "text_encoder": [
        ("model.language_model.embed_tokens.weight", "BF16", (1000, 64)),
        ("model.language_model.layers.0.self_attn.q_proj.weight", "BF16", (64, 64)),
        ("model.language_model.layers.0.mlp.gate_proj.weight", "BF16", (192, 64)),
        ("model.language_model.layers.0.input_layernorm.weight", "BF16", (64,)),
        ("model.language_model.norm.weight", "BF16", (64,)),
        ("model.visual.blocks.0.attn.qkv.weight", "BF16", (96, 32)),
        ("lm_head.weight", "BF16", (1000, 64)),
    ],
    "transformer": [
        ("img_in.weight", "BF16", (128, 64)),
        ("transformer_blocks.0.attn.to_q.weight", "BF16", (128, 128)),
        ("transformer_blocks.0.img_mlp.net.0.proj.weight", "BF16", (384, 128)),
        ("norm_out.linear.bias", "BF16", (256,)),
    ],
    "vae": [
        ("decoder.conv_in.weight", "F32", (64, 16, 3, 3, 3)),
        ("decoder.conv_in.bias", "F32", (64,)),
    ],
}
_SIZES = {"BF16": 2, "F32": 4}


def elements(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def expected_resident(component):
    """What the planner must report for one component of this fixture."""
    total = 0
    for name, dtype, shape in TENSORS[component]:
        if component == "text_encoder" and not name.startswith("model.language_model."):
            continue
        if component == "vae":
            total += elements(shape) * _SIZES[dtype]
        elif len(shape) == 2:
            total += elements(shape) + shape[0] * 4
        else:
            total += elements(shape) * 4
    return total


def write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, dtype, shape in tensors:
        size = elements(shape) * _SIZES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.truncate(8 + len(raw) + offset)


def make_fake_pipeline(root, tokenizer=True, license_text="Qwen RESEARCH LICENSE AGREEMENT\n"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_index.json").write_text(json.dumps(INDEX), encoding="utf-8")
    for part in ("processor", "scheduler", "text_encoder", "transformer", "vae"):
        (root / part).mkdir(exist_ok=True)
    if tokenizer:
        (root / "processor" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "scheduler" / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "FlowMatchEulerDiscreteScheduler"}), encoding="utf-8")
    (root / "transformer" / "config.json").write_text(
        json.dumps({"_class_name": "QwenImage21Transformer2DModel"}), encoding="utf-8")
    write_safetensors(root / "text_encoder" / "model-00001-of-00001.safetensors",
                      TENSORS["text_encoder"])
    write_safetensors(root / "transformer" / "diffusion_pytorch_model.safetensors",
                      TENSORS["transformer"])
    write_safetensors(root / "vae" / "diffusion_pytorch_model.safetensors", TENSORS["vae"])
    if license_text:
        (root / "LICENSE").write_text(license_text, encoding="utf-8")
    return str(root)


def stub_env(delay=0.0, **extra):
    env = dict(os.environ)
    env["QWENIMAGE_STUB_DELAY"] = str(delay)
    env.pop("COLI_IMAGE_PROTOCOL", None)
    env.update({key: str(value) for key, value in extra.items()})
    return env
