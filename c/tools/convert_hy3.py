"""Convert tencent/Hy3-FP8 (or Hy3 BF16) -> Colibri int4 safetensors container.

Fork of convert_fp8_to_int4.py with Hy3 tensor naming:
  - num_experts MoE layers, shared_mlp -> shared_experts on write
  - MTP layer 80 -> out-mtp-*.safetensors (--mtp, int8)
  - no DSA indexer

Usage:
  python3 tools/convert_hy3.py --indir hy3_tiny --outdir hy3_tiny_i4 --ebits 4
  python3 tools/convert_hy3.py --repo tencent/Hy3-FP8 --outdir /path/hy3_i4
  python3 tools/convert_hy3.py --repo tencent/Hy3 --bf16 --outdir /path/hy3_i4
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

import numpy as np

# Reuse quant + download machinery from GLM converter
sys.path.insert(0, os.path.dirname(__file__))
from convert_fp8_to_int4 import (  # noqa: E402
    quant_int2, quant_int4, quant_int8, free_gb, layer_idx,
)

SHARED_RE = re.compile(r"\.mlp\.shared_mlp\.")


def dequant(f, name):
    """Hy3-FP8: per-tensor *.weight_scale (scalar). GLM-FP8: block *.weight_scale_inv."""
    import torch
    dt = f.get_slice(name).get_dtype()
    if dt not in ("F8_E4M3", "float8_e4m3fn"):
        return f.get_tensor(name).to(torch.float32).numpy()
    w = f.get_tensor(name).to(torch.float32)
    if (name + "_scale_inv") in f.keys():
        sc = f.get_tensor(name + "_scale_inv").to(torch.float32)
        if sc.ndim == 0:
            return (w * sc).numpy()
        o, i = w.shape
        sc = sc.repeat_interleave(128, 0).repeat_interleave(128, 1)[:o, :i]
        return (w * sc).numpy()
    if (name + "_scale") in f.keys():
        return (w * f.get_tensor(name + "_scale").to(torch.float32)).numpy()
    raise KeyError(f"FP8 tensor {name} missing weight_scale or weight_scale_inv")


def rename_out(name):
    """Colibri loader expects shared_experts.* (GLM convention)."""
    return SHARED_RE.sub(".mlp.shared_experts.", name)


def classify(name, n_layers, keep_mtp=False):
    if name.endswith("_scale_inv") or name.endswith("_scale"):
        return "consumed"
    li = layer_idx(name)
    if keep_mtp:
        if li != n_layers:
            return "skip"
    else:
        if li >= n_layers:
            return "skip"
        if "eh_proj" in name and li == n_layers:
            return "skip"
    if name.endswith("e_score_correction_bias") or name.endswith("expert_bias"):
        return "f32"
    if name.endswith("mlp.gate.weight") or name.endswith("mlp.router.gate.weight"):
        return "f32"
    if name.endswith("norm.weight") or name == "model.norm.weight":
        return "f32"
    if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
        return "f32"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "io"
    if ".mlp.experts." in name and name.endswith(".weight"):
        return "x"
    if name.endswith(".weight"):
        return "q"
    return "f32"


def convert_shard(path, out_dict, n_layers, ebits, io_bits, xbits, keep_mtp=False):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            kind = classify(name, n_layers, keep_mtp)
            if kind in ("skip", "consumed"):
                continue
            w = dequant(f, name)
            out_name = rename_out(name)
            if kind == "f32":
                out_dict[out_name] = w.astype(np.float32)
            else:
                bits = io_bits if kind == "io" else xbits if kind == "x" else ebits
                if w.ndim != 2:
                    out_dict[out_name] = w.astype(np.float32)
                    continue
                q, s = (quant_int2(w, bits) if bits <= 2 else
                        quant_int4(w, bits) if bits <= 4 else quant_int8(w, bits))
                out_dict[out_name] = q
                out_dict[out_name + ".qs"] = s


def convert_local(indir, outdir, n_layers, ebits, io_bits, xbits):
    from safetensors.numpy import save_file
    shards = sorted(glob.glob(os.path.join(indir, "*.safetensors")))
    os.makedirs(outdir, exist_ok=True)
    for i, sp in enumerate(shards):
        out = {}
        convert_shard(sp, out, n_layers, ebits, io_bits, xbits)
        save_file(out, os.path.join(outdir, f"out-{i:05d}.safetensors"))
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "generation_config.json", "chat_template.jinja"]:
        src = os.path.join(indir, fn)
        if os.path.exists(src):
            shutil.copy(src, outdir)
    print(f"converted {len(shards)} shards -> {outdir}")


def main():
    ap = argparse.ArgumentParser(description="Hy3 -> Colibri int4 container")
    ap.add_argument("--repo", default="tencent/Hy3-FP8")
    ap.add_argument("--indir", default=None)
    ap.add_argument("--outdir", required=False)
    ap.add_argument("--ebits", type=int, default=None)
    ap.add_argument("--io-bits", type=int, default=8)
    ap.add_argument("--xbits", type=int, default=None)
    ap.add_argument("--n-layers", type=int, default=80)
    ap.add_argument("--min-free-gb", type=float, default=20.0)
    ap.add_argument("--bf16", action="store_true", help="use tencent/Hy3 BF16 repo")
    ap.add_argument("--mtp", action="store_true")
    a = ap.parse_args()
    if a.bf16 and a.repo == "tencent/Hy3-FP8":
        a.repo = "tencent/Hy3"
    if a.ebits is None:
        a.ebits = 8 if a.mtp else 4
    if a.xbits is None:
        a.xbits = a.ebits

    if a.indir:
        if not a.outdir:
            sys.exit("--outdir required with --indir")
        convert_local(a.indir, a.outdir, a.n_layers, a.ebits, a.io_bits, a.xbits)
        return

    if not a.outdir:
        sys.exit("--outdir required")

    # Delegate to GLM converter main loop with Hy3 classify/dequant patched in
    import convert_fp8_to_int4 as cvt
    cvt.classify = lambda name, n_layers, keep_mtp=False, keep_idx=False: classify(
        name, n_layers, keep_mtp)
    cvt.dequant = dequant
    argv = [
        "convert_hy3.py",
        "--repo", a.repo,
        "--outdir", a.outdir,
        "--ebits", str(a.ebits),
        "--io-bits", str(a.io_bits),
        "--xbits", str(a.xbits),
        "--n-layers", str(a.n_layers),
        "--min-free-gb", str(a.min_free_gb),
    ]
    if a.mtp:
        argv.append("--mtp")
    sys.argv = argv
    cvt.main()


if __name__ == "__main__":
    main()
