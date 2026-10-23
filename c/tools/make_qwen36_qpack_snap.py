#!/usr/bin/env python3
"""Build a qwen36 snap view over a Swiftlet qpack container (READ-ONLY source).

The engine's contract is the one convert_qwen36.py established: a snapshot
directory with a FLAT config.json, a qwen36_meta.json whose head dims were
derived from the actual weight shapes, the *.safetensors holding the dense
weights, and tokenizer.json.  A Swiftlet qpack container already carries all
of the bytes -- model.safetensors (dense half, MLX affine triples that
c/qwen36.c expands at load), tokenizer.json, and a nested multimodal
config.json -- but not the two engine-side metadata files, and the container
itself must never be written to (it is a production artifact shared with
Swiftlet, hash-pinned by hashes.json).

So the bridge is a VIEW: a fresh directory holding the generated flat
config.json + qwen36_meta.json next to symlinks into the container.  No
tensor bytes are copied or converted on disk; st_init follows the symlink and
the engine's affine dense loader does the expansion in memory.

Usage:
  python tools/make_qwen36_qpack_snap.py \
      --container ~/models/qwen3.6-35b.qpack --out ~/build/qwen36-35b-snap

Run the engine against it:
  SNAP=<out> QWEN36_QPACK=<container> ./qwen36 16 4 prompt.txt

Head-dim derivation mirrors convert_qwen36.py ("derived from the actual
weight shapes, authoritative"), with one extra step: quantized tensors
declare PACKED shapes (uint32 words along the input axis), so logical dims
are unpacked through the config's MLX quantization spec (default bits/group
plus per-module overrides, the same matching Swiftlet's Checkpoint.quantSpec
uses).  Stdlib only -- no torch, no numpy, no safetensors dependency; the
safetensors header is 8 bytes of length plus JSON.
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path


def read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        if hlen > (512 << 20):
            sys.exit(f"{path}: implausible safetensors header length {hlen}")
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header


class QuantSpec:
    """Config `quantization` block: default bits/group + per-module overrides."""

    def __init__(self, cfg: dict):
        block = cfg.get("quantization") or cfg.get("quantization_config") or {}
        self.default = (int(block.get("bits", 4)), int(block.get("group_size", 64)))
        self.overrides = {
            key: (int(spec["bits"]), int(spec["group_size"]))
            for key, spec in block.items()
            if isinstance(spec, dict) and "bits" in spec
        }

    def for_module(self, module: str):
        for key, spec in self.overrides.items():
            if module.endswith(key) or key.endswith(module):
                return spec
        return self.default


class DenseShapes:
    """Logical (unpacked) shapes for the dense tensors, prefix-resolved."""

    def __init__(self, header: dict, quant: QuantSpec):
        self.header = header
        self.quant = quant

    def resolve(self, name: str):
        if name in self.header:
            return name
        prefixed = "language_model." + name
        return prefixed if prefixed in self.header else None

    def logical_shape(self, name: str):
        resolved = self.resolve(name)
        if resolved is None:
            return None
        info = self.header[resolved]
        shape = list(info["shape"])
        if info["dtype"] in ("U32", "I32") and self.resolve(
                name.removesuffix(".weight") + ".scales") is not None:
            bits, _ = self.quant.for_module(
                resolved.removesuffix(".weight"))
            shape[-1] *= 32 // bits
        return shape


def derive_meta(cfg_full: dict, mcfg: dict, shapes: DenseShapes) -> dict:
    n_layers = int(mcfg["num_hidden_layers"])
    layer_types = mcfg.get("layer_types")
    if not layer_types:
        interval = int(mcfg.get("full_attention_interval", 4))
        layer_types = ["full_attention" if (i + 1) % interval == 0
                       else "linear_attention" for i in range(n_layers)]
    full_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    rope = mcfg.get("rope_parameters") or {}
    meta = {
        "model_type": cfg_full.get("model_type"),
        "hidden": int(mcfg["hidden_size"]),
        "n_layers": n_layers,
        "n_active": len(full_idx),
        "layer_types": layer_types,
        "num_experts": int(mcfg["num_experts"]),
        "topk": int(mcfg["num_experts_per_tok"]),
        "moe_inter": int(mcfg["moe_intermediate_size"]),
        "shared_inter": int(mcfg.get("shared_expert_intermediate_size",
                                     mcfg.get("moe_intermediate_size", 0))),
        "rms_eps": float(mcfg.get("rms_norm_eps", 1e-6)),
        "scoring_func": mcfg.get("scoring_func", "softmax"),
        "n_group": int(mcfg.get("n_group", 1)),
        "topk_group": int(mcfg.get("topk_group", 1)),
        "norm_topk_prob": bool(mcfg.get("norm_topk_prob", False)),
        "attn_output_gate": bool(mcfg.get("attn_output_gate", False)),
        "rope_theta": float(rope.get("rope_theta",
                                     mcfg.get("rope_theta", 10000000.0))),
        "mrope_section": rope.get("mrope_section", [16, 16, 16]),
        "partial_rotary_factor": float(mcfg.get("partial_rotary_factor", 0.25)),
        "expert_gs": 0,   # snapshot-expert group scaling; unused with a qpack store
        # HF Qwen3.6 stores the rmsnorm_row weights zero-centered and the
        # engine applies (1 + w); mlx-lm materialises the +1 at conversion, so
        # an MLX-derived container carries FULL gamma.  This flag makes the
        # engine undo the shift at load (c/qwen36.c load_norm_n) -- without it
        # every normalised activation is ~doubled and the model is noise.
        "zero_centered_norms": False,
    }
    n_q = int(mcfg["num_attention_heads"])
    n_kv = int(mcfg["num_key_value_heads"])
    meta["q_heads"] = n_q
    meta["kv_heads"] = n_kv
    if not full_idx:
        sys.exit("no full_attention layer to derive head dims from")
    fi = full_idx[0]

    def logical(proj):
        return shapes.logical_shape(f"model.layers.{fi}.self_attn.{proj}.weight")

    qp, kp, vp, op = (logical(p) for p in ("q_proj", "k_proj", "v_proj", "o_proj"))
    qn = shapes.logical_shape(f"model.layers.{fi}.self_attn.q_norm.weight")
    if qp is None or kp is None or vp is None or op is None:
        sys.exit(f"layer {fi}: attention projections missing from model.safetensors")
    meta["q_head_dim"] = qp[0] // n_q
    meta["k_head_dim"] = kp[0] // n_kv
    meta["v_head_dim"] = vp[0] // n_kv
    meta["o_in"] = op[1]
    if qn is not None:
        meta["qk_rope_head_dim"] = qn[0]
    meta["head_dim"] = meta["k_head_dim"]
    meta["rope_dim"] = meta.get("qk_rope_head_dim", meta["head_dim"] // 4)
    # DeltaNet dims are explicit in config (same source convert_qwen36.py uses).
    meta["dn_vheads"] = int(mcfg.get("linear_num_value_heads",
                                     mcfg.get("num_value_heads", 0)))
    meta["dn_kheads"] = int(mcfg.get("linear_num_key_heads",
                                     mcfg.get("num_key_heads", 0)))
    meta["dn_kdim"] = int(mcfg.get("linear_key_head_dim",
                                   mcfg.get("key_head_dim", 0)))
    meta["dn_vdim"] = int(mcfg.get("linear_value_head_dim",
                                   mcfg.get("value_head_dim", 0)))
    meta["dn_convk"] = int(mcfg.get("linear_conv_kernel_dim",
                                    mcfg.get("conv_kernel_size", 0)))
    meta["dn_conv_dim"] = (meta["dn_kheads"] * meta["dn_kdim"] * 2 +
                           meta["dn_vheads"] * meta["dn_vdim"])
    return meta


def place_link(target: Path, link: Path):
    """Symlink preferred (zero copies, container stays the single source of
    truth); hard link as the fallback for hosts without symlink privileges."""
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(target)
        return "symlink"
    except OSError:
        os.link(target, link)   # let a cross-device failure surface loudly
        return "hardlink"


def main():
    # __doc__ is None under python -OO (docstrings stripped): fall back to a
    # literal rather than crashing before argparse can even print usage.
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else
        "Build a qwen36 snap view over a Swiftlet qpack container")
    ap.add_argument("--container", required=True,
                    help="qpack container directory (never written to)")
    ap.add_argument("--out", required=True,
                    help="snap view directory to create")
    args = ap.parse_args()
    container = Path(args.container).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if container == out or container in out.parents:
        sys.exit("refusing: --out must live OUTSIDE the read-only container")
    st_path = container / "model.safetensors"
    cfg_path = container / "config.json"
    for required in (st_path, cfg_path):
        if not required.is_file():
            sys.exit(f"{required}: missing -- not a qpack container?")
    manifest = container / "manifest.json"
    if manifest.is_file():
        magic = json.loads(manifest.read_text(encoding="utf-8")).get("magic")
        if magic != "QPACK":
            sys.exit(f"{manifest}: magic {magic!r} is not QPACK")
    else:
        print(f"[warn] {manifest} missing; proceeding on model.safetensors alone")

    cfg_full = json.loads(cfg_path.read_text(encoding="utf-8"))
    mcfg = cfg_full.get("text_config", cfg_full)
    quant = QuantSpec(cfg_full)
    shapes = DenseShapes(read_safetensors_header(st_path), quant)
    meta = derive_meta(cfg_full, mcfg, shapes)

    out.mkdir(parents=True, exist_ok=True)
    # The engine reads FLAT keys from config.json (load_cfg), but multimodal
    # checkpoints nest them under text_config -- same flattening
    # convert_qwen36.py performs, original kept as config.hf.json.
    (out / "config.json").write_text(json.dumps(mcfg, indent=2), encoding="utf-8")
    (out / "config.hf.json").write_text(json.dumps(cfg_full, indent=2),
                                        encoding="utf-8")
    (out / "qwen36_meta.json").write_text(json.dumps(meta, indent=2),
                                          encoding="utf-8")
    links = {"model.safetensors": st_path}
    tok = container / "tokenizer.json"
    if tok.is_file():
        links["tokenizer.json"] = tok
    else:
        print(f"[warn] {tok} missing; set TOK=<path> when running the engine")
    for name, target in links.items():
        kind = place_link(target, out / name)
        print(f"[link] {name} -> {target} ({kind})")
    print(f"[meta] {out / 'qwen36_meta.json'} "
          f"(q_head_dim={meta['q_head_dim']} k/v={meta['k_head_dim']}/"
          f"{meta['v_head_dim']} o_in={meta['o_in']} "
          f"n_active={meta['n_active']}/{meta['n_layers']})")
    print(f"\nRun:  SNAP={out} QWEN36_QPACK={container} ./qwen36 16 4 prompt.txt")


if __name__ == "__main__":
    main()
