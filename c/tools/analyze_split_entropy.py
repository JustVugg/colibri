#!/usr/bin/env python3
"""Measure whether packed INT4 experts can be split losslessly across VRAM/RAM.

VRAM stores a fixed-width group id ("base").  RAM stores the exact symbol's
local index inside that group ("residual").  The practical estimate uses a
global codebook and fixed-width local indices, so it is GPU-decodable without
pretending that an entropy lower bound is an implementation.
"""

import argparse
import glob
import json
import math
import os
import re

import numpy as np
from safetensors import safe_open


EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate|up|down)_proj\.weight$")
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2,
    "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def compositions(total, parts, minimum=1):
    if parts == 1:
        if total >= minimum:
            yield (total,)
        return
    for first in range(minimum, total - minimum * (parts - 1) + 1):
        for tail in compositions(total - first, parts - 1, minimum):
            yield (first,) + tail


def best_fixed_width_groups(prob, base_bits):
    """Best global unequal groups under fixed-width local indices.

    For a given group-size tuple, the most probable symbols are assigned to
    slots with the cheapest local-index width.  This is exact because the
    objective is linear in symbol probability.
    """
    groups = 1 << base_bits
    sorted_prob = np.sort(prob)[::-1]
    best = None
    for sizes in compositions(16, groups):
        ordered_sizes = tuple(sorted(sizes,
                                     key=lambda n: (math.ceil(math.log2(n)), n)))
        widths = [
            math.ceil(math.log2(size))
            for size in ordered_sizes
            for _ in range(size)
        ]
        residual = float(np.dot(sorted_prob, np.asarray(widths)))
        candidate = (residual, ordered_sizes)
        if best is None or candidate < best:
            best = candidate
    ranked_symbols = np.argsort(prob)[::-1].tolist()
    symbol_to_group = [0] * 16
    symbol_to_local = [0] * 16
    cursor = 0
    for group, size in enumerate(best[1]):
        for local, symbol in enumerate(ranked_symbols[cursor:cursor + size]):
            symbol_to_group[symbol] = group
            symbol_to_local[symbol] = local
        cursor += size
    return {
        "base_bits": base_bits,
        "residual_bits": best[0],
        "total_bits": base_bits + best[0],
        "group_sizes": best[1],
        "group_widths": [math.ceil(math.log2(n)) for n in best[1]],
        "symbol_to_group": symbol_to_group,
        "symbol_to_local": symbol_to_local,
    }


def tensor_nbytes(tensor_slice):
    shape = tensor_slice.get_shape()
    count = math.prod(shape)
    return count * DTYPE_BYTES[str(tensor_slice.get_dtype())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--layers", default="3,10,30,50,76",
                    help="comma-separated layers sampled for the histogram")
    ap.add_argument("--expert-vram-gb", type=float, default=176.57)
    ap.add_argument("--offset-weights", type=int, default=4096,
                    help="one 32-bit residual offset per this many weights")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    wanted = {int(v) for v in args.layers.split(",") if v}
    paths = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
    if not paths:
        raise SystemExit(f"no safetensors found under {args.model_dir}")

    total_weight_bytes = 0
    total_scale_bytes = 0
    selected = []
    tensor_sizes = {}
    for path in paths:
        with safe_open(path, framework="numpy") as sf:
            for name in sf.keys():
                tensor_sizes[name] = tensor_nbytes(sf.get_slice(name))

    expert_names = []
    for path in paths:
        with safe_open(path, framework="numpy") as sf:
            for name in sf.keys():
                match = EXPERT.match(name)
                if not match or str(sf.get_slice(name).get_dtype()) != "U8":
                    continue
                expert_names.append(name)
                total_weight_bytes += tensor_sizes[name]
                if int(match.group(1)) in wanted:
                    selected.append((path, name))
    total_scale_bytes = sum(
        tensor_sizes.get(name + ".qs", 0) for name in expert_names)

    hist = np.zeros(16, dtype=np.int64)
    sampled_bytes = 0
    by_layer = {}
    for index, (path, name) in enumerate(selected, 1):
        with safe_open(path, framework="numpy") as sf:
            packed = np.asarray(sf.get_tensor(name), dtype=np.uint8).reshape(-1)
        local = (np.bincount(packed & 15, minlength=16) +
                 np.bincount(packed >> 4, minlength=16))
        hist += local
        sampled_bytes += packed.nbytes
        layer = int(EXPERT.match(name).group(1))
        by_layer.setdefault(layer, np.zeros(16, dtype=np.int64))
        by_layer[layer] += local
        if index % 256 == 0:
            print(f"sampled {index}/{len(selected)} tensors "
                  f"({sampled_bytes / 1e9:.1f} GB)", flush=True)

    count = int(hist.sum())
    prob = hist / count
    nz = prob[prob > 0]
    entropy = float(-np.sum(nz * np.log2(nz)))
    modes = {bits: best_fixed_width_groups(prob, bits) for bits in (1, 2)}

    weight_count = total_weight_bytes * 2
    offset_bytes = math.ceil(weight_count / args.offset_weights) * 4
    budget = args.expert_vram_gb * 1e9
    max_base = 8 * max(
        0, budget - total_scale_bytes - offset_bytes) / weight_count
    base2_fraction = min(1.0, max(0.0, max_base - 1.0))
    practical_residual = (
        (1 - base2_fraction) * modes[1]["residual_bits"] +
        base2_fraction * modes[2]["residual_bits"])
    ideal_residual = max(0.0, entropy - max_base)

    result = {
        "model_dir": os.path.abspath(args.model_dir),
        "sample_layers": sorted(wanted),
        "sample_tensors": len(selected),
        "sample_packed_gb": sampled_bytes / 1e9,
        "expert_weight_gb": total_weight_bytes / 1e9,
        "expert_scale_gb": total_scale_bytes / 1e9,
        "symbol_histogram": hist.tolist(),
        "entropy_bits_per_weight": entropy,
        "rans_ideal_ratio_to_int4": entropy / 4,
        "fixed_width_modes": modes,
        "split_at_budget": {
            "expert_vram_budget_gb": args.expert_vram_gb,
            "offset_metadata_gb": offset_bytes / 1e9,
            "max_average_base_bits": max_base,
            "base2_weight_fraction": base2_fraction,
            "ideal_residual_bits": ideal_residual,
            "ideal_residual_gb": weight_count * ideal_residual / 8e9,
            "practical_residual_bits": practical_residual,
            "practical_residual_gb": weight_count * practical_residual / 8e9,
            "practical_total_bits": max_base + practical_residual,
        },
        "layer_entropy": {
            str(layer): float(-np.sum(
                (h[h > 0] / h.sum()) * np.log2(h[h > 0] / h.sum())))
            for layer, h in sorted(by_layer.items())
        },
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
            fh.write("\n")


if __name__ == "__main__":
    main()
