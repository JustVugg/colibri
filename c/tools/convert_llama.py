#!/usr/bin/env python3
"""Stage a Llama (model_type "llama") HF checkpoint as a colibri container.

The colibri container is a directory of safetensors shards + config.json
(see convert_qwen36.py). For a dense Llama that is exactly what the HF
checkpoint already is, and llama.c reads BF16/F16/F32 directly into f32, so
there is nothing to re-encode: the shards are staged byte-for-byte (hardlink
when possible, else copy) and their SHA-256 recorded in llama_meta.json.

What this tool adds is the check llama.c would otherwise make one tensor at a
time at load: every tensor the engine reads must be present, float, with the
exact shape the config implies and a byte span that matches it. The inventory
below is the one llama.c's model_init loads -- keep them in step. Anything the
engine would not read is refused too (except known-inert tensors), so a
checkpoint of a different architecture cannot pass for a Llama.

Usage:
  python tools/convert_llama.py --model ./Llama-3.2-1B --out ./llama-colibri
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path

ESZ = {"BF16": 2, "F16": 2, "F32": 4}
# Present in some exports, never read by llama.c.
INERT = re.compile(r"model\.layers\.\d+\.self_attn\.rotary_emb\.inv_freq")
COPY = ("config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json")


class ConvertError(ValueError):
    pass


def expected_tensors(config):
    """name -> shape, mirroring llama.c load_cfg/model_init."""
    if config.get("model_type") != "llama":
        raise ConvertError(f"model_type {config.get('model_type')!r} is not 'llama'")
    for key in ("attention_bias", "mlp_bias"):
        if config.get(key):
            raise ConvertError(f"{key}=true is not supported by llama.c")
    D = config["hidden_size"]
    H = config["num_attention_heads"]
    KV = config.get("num_key_value_heads", H)
    hd = config.get("head_dim") or D // H
    F = config["intermediate_size"]
    V = config["vocab_size"]
    if H % KV:
        raise ConvertError("num_attention_heads is not a multiple of num_key_value_heads")
    want = {"model.embed_tokens.weight": [V, D], "model.norm.weight": [D]}
    if not config.get("tie_word_embeddings", False):
        want["lm_head.weight"] = [V, D]
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        want.update({
            p + "input_layernorm.weight": [D],
            p + "post_attention_layernorm.weight": [D],
            p + "self_attn.q_proj.weight": [H * hd, D],
            p + "self_attn.k_proj.weight": [KV * hd, D],
            p + "self_attn.v_proj.weight": [KV * hd, D],
            p + "self_attn.o_proj.weight": [D, H * hd],
            p + "mlp.gate_proj.weight": [F, D],
            p + "mlp.up_proj.weight": [F, D],
            p + "mlp.down_proj.weight": [D, F],
        })
    return want


def read_header(path):
    size = path.stat().st_size
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ConvertError(f"{path.name}: truncated header")
        (n,) = struct.unpack("<Q", raw)
        if n > min(size - 8, 100 << 20):
            raise ConvertError(f"{path.name}: header length {n} exceeds file")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, size - 8 - n


def validate(model_dir):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    want = expected_tensors(config)
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise ConvertError(f"{model_dir}: no .safetensors shards")
    seen = {}
    for shard in shards:
        header, data_bytes = read_header(shard)
        for name, t in header.items():
            if name in seen:
                raise ConvertError(f"{name}: present in {seen[name]} and {shard.name}")
            seen[name] = shard.name
            if name not in want:
                if INERT.fullmatch(name) or (name == "lm_head.weight" and
                                             config.get("tie_word_embeddings", False)):
                    continue
                raise ConvertError(f"{name}: not a tensor llama.c reads -- refusing")
            dtype, shape, (a, b) = t["dtype"], t["shape"], t["data_offsets"]
            if dtype not in ESZ:
                raise ConvertError(f"{name}: dtype {dtype} is not BF16/F16/F32")
            if shape != want[name]:
                raise ConvertError(f"{name}: shape {shape}, config implies {want[name]} -- refusing")
            numel = 1
            for d in shape:
                numel *= d
            if not 0 <= a <= b <= data_bytes or b - a != numel * ESZ[dtype]:
                raise ConvertError(f"{name}: byte span [{a},{b}) does not hold "
                                   f"{numel} x {dtype} -- refusing")
    missing = sorted(set(want) - set(seen))
    if missing:
        raise ConvertError(f"missing {len(missing)} tensor(s), first: {missing[0]}")
    return config, shards, len(want)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stage(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="local HF checkpoint directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    src, out = Path(args.model), Path(args.out)
    try:
        config, shards, count = validate(src)
    except (ConvertError, KeyError, json.JSONDecodeError) as error:
        sys.exit(f"convert_llama: {error}")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"source": str(src.resolve()), "tensors": count, "shards": {}}
    for shard in shards:
        stage(shard, out / shard.name)
        digest = sha256(out / shard.name)
        if digest != sha256(shard):
            sys.exit(f"convert_llama: {shard.name} changed while staging")
        meta["shards"][shard.name] = digest
    for name in COPY:
        if (src / name).is_file():
            shutil.copyfile(src / name, out / name)
    (out / "llama_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"validated {count} tensors in {len(shards)} shard(s); staged to {out}")
    for name, digest in meta["shards"].items():
        print(f"  {digest}  {name}")


if __name__ == "__main__":
    main()
