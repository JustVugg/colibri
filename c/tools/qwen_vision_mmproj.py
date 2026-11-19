#!/usr/bin/env python3
"""A Qwen vision tower (Qwen3.5, 3.6 and 3.8: model.visual.* in a checkpoint) as llama.cpp's mmproj
GGUF: projector qwen3vl_merger, the names unsloth's mmproj-*.gguf files carry, float32.

qwen36 reads such a file beside a GGUF (gguf.json's "mmproj") or, with Q36_MMPROJ, in place of a
converted container's own tower: the CI holds the tiny 27B-shaped model to its transformers
tokens through it, which checks the names and the patch embedding's layout against the reference.

  python3 tools/qwen_vision_mmproj.py --model DIR --out mmproj.gguf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_flint_tiny import _gguf_bytes  # noqa: E402

U32, F32, BOOL, STR = 4, 6, 7, 8
GGML_F32 = 0

# the checkpoint's name in the tower -> the mmproj's (llama.cpp's qwen3vl_merger)
BLOCK = {"norm1": "ln1", "norm2": "ln2", "attn.qkv": "attn_qkv", "attn.proj": "attn_out",
         "mlp.linear_fc1": "ffn_up", "mlp.linear_fc2": "ffn_down"}
TOP = {"pos_embed.weight": "v.position_embd.weight", "patch_embed.proj.bias": "v.patch_embd.bias",
       "merger.norm.weight": "v.post_ln.weight", "merger.norm.bias": "v.post_ln.bias",
       "merger.linear_fc1.weight": "mm.0.weight", "merger.linear_fc1.bias": "mm.0.bias",
       "merger.linear_fc2.weight": "mm.2.weight", "merger.linear_fc2.bias": "mm.2.bias"}


def tower(model_dir):
    """the tower's tensors, float32, by their name after model.visual."""
    from safetensors import safe_open
    out = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as st:
            for k in st.keys():
                for prefix in ("model.visual.", "visual."):
                    if k.startswith(prefix):
                        out[k[len(prefix):]] = st.get_tensor(k).float().numpy()
    return out


def vision_config(model_dir):
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    vc = cfg.get("vision_config")
    if not vc:
        sys.exit(f"{model_dir}: no vision_config in config.json")
    if vc.get("deepstack_visual_indexes"):
        sys.exit("a tower with deepstack mergers (Qwen3-VL): not the Qwen3.5 family's")
    return vc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="a checkpoint directory with model.visual.* and config.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    vc, w = vision_config(a.model), tower(a.model)
    if not w:
        sys.exit(f"{a.model}: no model.visual.* tensors")
    D, depth, T, P = vc["hidden_size"], vc["depth"], vc.get("temporal_patch_size", 2), vc["patch_size"]
    tensors = []

    def put(name, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        tensors.append((name, GGML_F32, list(reversed(arr.shape)), arr.tobytes()))

    used = set()
    pe = w["patch_embed.proj.weight"]   # [D, C, T, P, P]: one [D, C, P, P] kernel a frame
    used.add("patch_embed.proj.weight")
    if pe.shape != (D, vc.get("in_channels", 3), T, P, P):
        sys.exit(f"patch_embed.proj.weight {pe.shape}: not [hidden, channels, frames, patch, patch]")
    for t in range(T):
        put("v.patch_embd.weight" + (f".{t}" if t else ""), pe[:, :, t])
    for name, gg in TOP.items():
        if name in w:
            put(gg, w[name])
            used.add(name)
    for i in range(depth):
        for hf, gg in BLOCK.items():
            for part in ("weight", "bias"):
                k = f"blocks.{i}.{hf}.{part}"
                if k in w:
                    put(f"v.blk.{i}.{gg}.{part}", w[k])
                    used.add(k)
    left = sorted(set(w) - used)
    if left:
        sys.exit(f"tensors with no mmproj name: {left[:6]}")
    kv = [("general.architecture", STR, "clip"), ("general.type", STR, "mmproj"),
          ("clip.has_vision_encoder", BOOL, True), ("clip.projector_type", STR, "qwen3vl_merger"),
          ("clip.use_gelu", BOOL, True)]
    for k, v in (("block_count", depth), ("embedding_length", D), ("feed_forward_length", vc["intermediate_size"]),
                 ("attention.head_count", vc["num_heads"]), ("patch_size", P),
                 ("spatial_merge_size", vc["spatial_merge_size"]), ("projection_dim", vc["out_hidden_size"])):
        kv.append(("clip.vision." + k, U32, int(v)))
    kv.append(("clip.vision.attention.layer_norm_epsilon", F32, 1e-6))
    Path(a.out).write_bytes(_gguf_bytes(kv, tensors))
    print(f"[mmproj] {a.out}: {len(tensors)} tensors, {depth} blocks of {D}, {T} frames a patch")


if __name__ == "__main__":
    main()
