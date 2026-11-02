#!/usr/bin/env python3
"""mimo.c against Xiaomi's own code on the REAL MiMo-V2.6 checkpoint.

The tiny oracle (tools/make_mimo_tiny.py + make_mimo_ref.py) pins the math on a
checkpoint stored like the release. This pins the release itself: its MXFP4 nibble
order, its FP8 block scales, its pre-sharded qkv, its sinks, its ViT. Maintainer tool,
run by hand on a machine that has the checkpoint; nothing here runs in CI.

  python3 tools/mimo_real_check.py --model DIR --engine ./mimo [--layers N] [--image PNG]

The vendor model is built from the checkpoint's own modeling_mimo_v2.py (its SHA-256
must match the one the tiny oracle pinned) with the first N layers (default: all),
dense weights dequantized to f32 from the same bytes the engine reads. Its experts are
NOT all materialized -- 256 experts of a Flash layer are 26 GB in f32 -- but built one
at a time when routed: each is the vendor's own MiMoV2MLP, loaded, called and dropped,
so the math is still theirs and the RAM stays at one expert.

What is compared, at every prompt position: the logits (max absolute difference,
relative to the logit range), the argmax, and the KL divergence of the engine's
distribution from the vendor's. With --image, the ViT's output rows too.
"""
import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mimo_ref as R  # noqa: E402  (reuses the pinned vendor loader and dequantizers)


class Shards:
    """Tensor lookup across the checkpoint's shards, read on demand."""

    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index = json.loads((self.dir / "model.safetensors.index.json").read_text())
        self.where = index["weight_map"]
        self.headers = {}

    def header(self, shard):
        if shard not in self.headers:
            with open(self.dir / shard, "rb") as fh:
                size = struct.unpack("<Q", fh.read(8))[0]
                self.headers[shard] = (json.loads(fh.read(size)), 8 + size)
        return self.headers[shard]

    def has(self, name):
        return name in self.where

    def raw(self, name):
        shard = self.where[name]
        header, base = self.header(shard)
        meta = header[name]
        lo, hi = meta["data_offsets"]
        dtype = {"F32": np.float32, "U8": np.uint8, "BF16": np.uint16,
                 "F8_E4M3": np.uint8}[meta["dtype"]]
        with open(self.dir / shard, "rb") as fh:
            fh.seek(base + lo)
            data = fh.read(hi - lo)
        return meta["dtype"], np.frombuffer(data, dtype=dtype).reshape(meta["shape"])

    def f32(self, name):
        dtype, arr = self.raw(name)
        if dtype == "BF16":
            return R.bf16_to_f32(arr)
        if dtype == "F32":
            return arr.astype(np.float32)
        raise SystemExit(f"{name}: {dtype} is not a float tensor")


def truncated_config(config, layers, vision):
    config = dict(config)
    config["num_hidden_layers"] = layers
    config["hybrid_layer_pattern"] = config["hybrid_layer_pattern"][:layers]
    config["moe_layer_freq"] = config["moe_layer_freq"][:layers]
    config.pop("quantization_config", None)
    config.pop("audio_config", None)
    if not vision:
        config.pop("vision_config", None)
    return config


def load_dense(model, shards, config):
    """Every non-expert weight of the truncated model, dequantized to f32."""
    import torch
    chunks = config["num_key_value_heads"]
    state = {}
    for name, param in model.named_parameters():
        if ".mlp.experts." in name:
            continue
        if not shards.has(name):
            if name.startswith("visual.merger.") and name.endswith(".bias"):
                state[name] = torch.zeros(tuple(param.shape))    # the release stores none
                continue
            raise SystemExit(f"{name}: not in the checkpoint")
        dtype, arr = shards.raw(name)
        if name.endswith("qkv_proj.weight"):
            layer = int(name.split(".")[2])
            swa = config["hybrid_layer_pattern"][layer] == 1
            heads = config["swa_num_attention_heads" if swa else "num_attention_heads"]
            kv = config["swa_num_key_value_heads" if swa else "num_key_value_heads"]
            value = R.qkv_contiguous(arr, shards.f32(name + "_scale_inv"), heads, kv,
                                     config["head_dim"], config["v_head_dim"], chunks)
        elif dtype == "F8_E4M3":
            value = R.fp8_block_dequant(arr, shards.f32(name + "_scale_inv"))
        elif dtype in ("BF16", "F32"):
            value = shards.f32(name)
        else:
            raise SystemExit(f"{name}: unexpected {dtype}")
        if tuple(value.shape) != tuple(param.shape):
            if name == "visual.patch_embed.proj.weight":
                value = value.reshape(param.shape)
            else:
                raise SystemExit(f"{name}: {value.shape} vs {tuple(param.shape)}")
        state[name] = torch.from_numpy(np.ascontiguousarray(value))
    return state


def lazy_experts(model, modeling, shards, config):
    """Replace each layer's expert list with one that builds the vendor's MiMoV2MLP
    for an expert only when the router picks it."""
    import torch

    class LazyExperts(torch.nn.Module):
        def __init__(self, layer, count):
            super().__init__()
            self.layer, self.count = layer, count
            self.cfg = model.config

        def __len__(self):
            return self.count

        def __iter__(self):
            for e in range(self.count):
                yield self[e]

        def __getitem__(self, e):
            layer, cfg = self.layer, self.cfg

            class Once:
                def __call__(self, x):
                    mlp = modeling.MiMoV2MLP(cfg, intermediate_size=cfg.moe_intermediate_size)
                    prefix = f"model.layers.{layer}.mlp.experts.{e}."
                    weights = {}
                    for part in ("gate_proj", "up_proj", "down_proj"):
                        _, packed = shards.raw(prefix + part + ".weight")
                        _, scale = shards.raw(prefix + part + ".weight_scale")
                        weights[part + ".weight"] = torch.from_numpy(
                            R.mxfp4_dequant(packed, scale))
                    mlp.load_state_dict(weights)
                    with torch.no_grad():
                        return mlp.float()(x)
            return Once()

    for index, layer in enumerate(model.model.layers):
        if config["moe_layer_freq"][index]:
            layer.mlp.experts = LazyExperts(index, config["n_routed_experts"])


def engine_logits(engine, model_dir, ids, layers, env_extra, vocab):
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as tmp:
        path = tmp.name
    env = dict(os.environ, MIMO_LOGITS=path, MIMO_LAYERS=str(layers), **env_extra)
    try:
        subprocess.run([engine, str(model_dir), "--ids", " ".join(map(str, ids)), "--ngen", "0"],
                       env=env, check=True, capture_output=True, text=True, timeout=7200)
        return np.fromfile(path, dtype=np.float32).reshape(len(ids), vocab)
    finally:
        os.unlink(path)


def image_engine(args, model_dir, config, engine_env, vocab, tokenizer):
    """The picture through the gateway's preprocessing and the official processor
    (must agree on the grid), then the engine: its ViT rows and the logits of a prompt
    that carries the picture. Runs before the vendor model is built, so the two never
    hold RAM at the same time."""
    from PIL import Image
    from transformers import Qwen2VLImageProcessor
    import qwen38_image
    data = Path(args.image).read_bytes()
    ours, grid_h, grid_w = qwen38_image.preprocess(data, model_dir=str(model_dir),
                                                   max_tokens=args.image_tokens)
    # the pixel window as the gateway sets it; recent transformers read min_pixels /
    # max_pixels over `size` when the checkpoint's config names them, so give both
    raw = json.loads((model_dir / "preprocessor_config.json").read_text())
    low, high = raw.get("min_pixels", 3136), raw.get("max_pixels", 12845056)
    if args.image_tokens:
        high = min(high, args.image_tokens * 32 * 32)
        low = min(low, high)
    processor = Qwen2VLImageProcessor.from_pretrained(
        str(model_dir), min_pixels=low, max_pixels=high,
        size={"shortest_edge": low, "longest_edge": high})
    batch = processor(images=[Image.open(args.image).convert("RGB")], return_tensors="pt")
    official = batch["pixel_values"].float().numpy()
    thw = batch["image_grid_thw"]
    if thw.tolist() != [[1, grid_h, grid_w]]:
        raise SystemExit(f"grid: gateway {grid_h}x{grid_w}, official {thw.tolist()}")
    print(f"[real] image {grid_h}x{grid_w} patches: gateway vs official processor, max "
          f"|diff| {np.abs(official - ours).max():.3e} (resampling: Pillow against the "
          f"processor's own)")
    n_img = grid_h * grid_w // 4
    prefix = tokenizer.encode("<|im_start|>user\n<|vision_start|>", add_special_tokens=False).ids
    suffix = tokenizer.encode("<|vision_end|>Cosa vedi in questa immagine?<|im_end|>"
                              "<|im_start|>assistant\n<think>", add_special_tokens=False).ids
    ids = prefix + [config["image_token_id"]] * n_img + suffix
    with tempfile.TemporaryDirectory() as tmp:
        patches = Path(tmp) / "patches.f32"
        ours.tofile(patches)
        dump, logits_path = Path(tmp) / "rows.f32", Path(tmp) / "logits.f32"
        env = dict(os.environ, MIMO_LOGITS=str(logits_path), MIMO_VISION_DUMP=str(dump),
                   MIMO_LAYERS=str(config["num_hidden_layers"]), **engine_env)
        subprocess.run([args.engine, str(model_dir), "--ids", " ".join(map(str, ids)),
                        "--ngen", "0", "--image", str(patches), "--grid", str(grid_h),
                        str(grid_w)], env=env, check=True, capture_output=True, text=True,
                       timeout=7200)
        rows = np.fromfile(dump, dtype=np.float32).reshape(n_img, -1)
        logits = np.fromfile(logits_path, dtype=np.float32).reshape(len(ids), vocab)
    return {"patches": ours, "thw": thw, "ids": ids, "rows": rows, "logits": logits}


def image_vendor(model, image, config):
    import torch
    pixels = torch.from_numpy(image["patches"])
    with torch.no_grad():
        rows = model.visual(pixel_values=pixels, grid_thw=image["thw"]).float().numpy()
        theirs = model(input_ids=torch.tensor([image["ids"]]), pixel_values=pixels,
                       image_grid_thw=image["thw"], use_cache=False).logits[0].float().numpy()
    diff = np.abs(image["rows"] - rows)
    print(f"[real] ViT rows ({rows.shape[0]} x {rows.shape[1]}): max |diff| {diff.max():.3e}, "
          f"relative {diff.max() / np.abs(rows).max():.2e}")
    return compare(f"text with the picture, {config['num_hidden_layers']} layers",
                   image["logits"], theirs)


def compare(name, ours, theirs):
    diff = np.abs(ours - theirs)
    span = float(theirs.max() - theirs.min())
    same = int((ours.argmax(-1) == theirs.argmax(-1)).sum())

    def logsoftmax(x):
        x = x - x.max(-1, keepdims=True)
        return x - np.log(np.exp(x).sum(-1, keepdims=True))
    p = np.exp(logsoftmax(theirs.astype(np.float64)))
    kl = float((p * (logsoftmax(theirs.astype(np.float64)) - logsoftmax(ours.astype(np.float64)))).sum(-1).max())
    print(f"[real] {name}: argmax {same}/{len(ours)}, max |diff| {diff.max():.3e} "
          f"({diff.max() / span:.2e} of the logit range), max KL {kl:.2e}")
    return same == len(ours)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--layers", type=int, default=0, help="0: every layer")
    parser.add_argument("--prompt", default="<|im_start|>user\nCiao! Qual e' la capitale "
                        "d'Italia, e perche'?<|im_end|><|im_start|>assistant\n<think>")
    parser.add_argument("--image", default=None, help="a picture for the ViT check")
    parser.add_argument("--image-tokens", type=int, default=256,
                        help="ceiling on the picture's tokens, as the gateway's")
    parser.add_argument("--dense-bits", default="0", help="the engine's MIMO_DENSE_BITS")
    parser.add_argument("--ids", default=None, help="prompt token ids instead of --prompt")
    parser.add_argument("--vendor-dir", default=None,
                        help="where modeling_mimo_v2.py is (default: the model dir)")
    args = parser.parse_args()

    import torch
    from tokenizers import Tokenizer
    model_dir = Path(args.model)
    vendor_dir = Path(args.vendor_dir) if args.vendor_dir else model_dir
    for name, expected in R.VENDOR_FILES.items():
        digest = hashlib.sha256((vendor_dir / name).read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f"{name} in the checkpoint is not the pinned vendor file")
    full = json.loads((model_dir / "config.json").read_text())
    layers = args.layers or full["num_hidden_layers"]
    config = truncated_config(full, layers, vision=bool(args.image))
    tokenizer = None
    if args.ids:
        ids = [int(t) for t in args.ids.split()]
    else:
        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    print(f"[real] {layers} of {full['num_hidden_layers']} layers, {len(ids)} prompt tokens",
          file=sys.stderr)
    engine_env = {"MIMO_DENSE_BITS": args.dense_bits, "MIMO_IDOT": "0"}
    ours = engine_logits(args.engine, model_dir, ids, layers,
                         engine_env,
                         full["vocab_size"])
    image = None
    if args.image:
        if tokenizer is None:
            tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        image = image_engine(args, model_dir, config, engine_env, full["vocab_size"], tokenizer)
    config_module, modeling = R.load_vendor(vendor_dir)
    cfg = config_module.MiMoV2Config(**dict(config, use_cache=False))
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    with torch.device("meta"):
        model = modeling.MiMoV2ForCausalLM(cfg)
    shards = Shards(model_dir)
    lazy_experts(model, modeling, shards, config)
    state = load_dense(model, shards, config)
    model.load_state_dict(state, strict=False, assign=True)
    del state
    # the rotary tables are buffers computed at construction, i.e. on meta: rebuild
    # them on the CPU from the same config, then refuse anything still on meta
    model.model.rotary_emb = modeling.MiMoV2RotaryEmbedding(config=cfg, is_swa=False)
    model.model.swa_rotary_emb = modeling.MiMoV2RotaryEmbedding(config=cfg, is_swa=True)
    if getattr(model, "visual", None) is not None:
        head_dim = model.visual.blocks[0].attn.head_dim
        model.visual.rotary_pos_emb = modeling.MiMoVisionRotaryEmbedding(head_dim // 2)
    left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
            if t.is_meta]
    if left:
        raise SystemExit(f"still on meta after loading: {left[:8]}")
    R.check_rope(model, config)
    model = model.float().eval()

    with torch.no_grad():
        theirs = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    ok = compare(f"text, {layers} layers", ours, theirs)
    if image:
        ok = image_vendor(model, image, config) and ok
    if tokenizer:
        top = [tokenizer.decode([int(t)]) for t in theirs.argmax(-1)[-5:]]
        print(f"[real] last positions, vendor argmax: {top}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
