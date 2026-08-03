"""Plan integer ROCm layer offload for a GGUF model.

llama.cpp's memory fit mode may use fractional MoE-layer offload.  That is a
good safety fallback, but it can be slower on a single consumer GPU because
every token crosses the host/device boundary more often.  This module keeps
the memory calculation explicit and recommends the largest *whole-layer*
offload that leaves room for the KV cache, compute buffers, and a margin.

It intentionally reads only the GGUF header and tensor-info table; tensor
payloads are never loaded into Python.
"""

import argparse
import json
import math
import re
import struct
import subprocess
import sys
from pathlib import Path


MiB = 1024 * 1024
GiB = 1024 * MiB

# (values per block, bytes per block) for the GGML types used by current GGUF
# files.  Unknown types are rejected instead of silently underestimating VRAM.
GGML_BLOCKS = {
    0: (1, 4),       # F32
    1: (1, 2),       # F16
    2: (32, 18),     # Q4_0
    3: (32, 20),     # Q4_1
    6: (32, 22),     # Q5_0
    7: (32, 24),     # Q5_1
    8: (32, 34),     # Q8_0
    9: (32, 36),     # Q8_1
    10: (256, 84),   # Q2_K
    11: (256, 110),  # Q3_K
    12: (256, 144),  # Q4_K
    13: (256, 176),  # Q5_K
    14: (256, 210),  # Q6_K
    15: (256, 292),  # Q8_K
    16: (256, 66),   # IQ2_XXS
    17: (256, 74),   # IQ2_XS
    18: (256, 98),   # IQ3_XXS
    19: (256, 50),   # IQ1_S
    20: (32, 18),    # IQ4_NL
    21: (256, 110),  # IQ3_S
    22: (256, 82),   # IQ2_S
    23: (256, 136),  # IQ4_XS
    24: (1, 1),      # I8
    25: (1, 2),      # I16
    26: (1, 4),      # I32
    27: (1, 8),      # I64
    28: (1, 8),      # F64
    29: (256, 56),   # IQ1_M
    30: (1, 2),      # BF16
}

KV_BYTES_PER_VALUE = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q4_0": 18.0 / 32.0,
    "q4_1": 20.0 / 32.0,
    "iq4_nl": 18.0 / 32.0,
    "q5_0": 22.0 / 32.0,
    "q5_1": 24.0 / 32.0,
}
LAYER_RE = re.compile(r"(?:^|\.)blk\.(\d+)(?:\.|$)")


class _Reader:
    def __init__(self, stream):
        self.stream = stream

    def read(self, size):
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("truncated GGUF header")
        return data

    def u8(self):
        return self.read(1)[0]

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def i8(self):
        return struct.unpack("<b", self.read(1))[0]

    def i16(self):
        return struct.unpack("<h", self.read(2))[0]

    def i32(self):
        return struct.unpack("<i", self.read(4))[0]

    def i64(self):
        return struct.unpack("<q", self.read(8))[0]

    def f32(self):
        return struct.unpack("<f", self.read(4))[0]

    def f64(self):
        return struct.unpack("<d", self.read(8))[0]

    def string(self):
        return self.read(self.u64()).decode("utf-8", "replace")

    def value(self, value_type):
        if value_type == 0:
            return self.u8()
        if value_type == 1:
            return self.i8()
        if value_type == 2:
            return struct.unpack("<H", self.read(2))[0]
        if value_type == 3:
            return self.i16()
        if value_type == 4:
            return self.u32()
        if value_type == 5:
            return self.i32()
        if value_type == 6:
            return self.f32()
        if value_type == 7:
            return bool(self.u8())
        if value_type == 8:
            return self.string()
        if value_type == 9:
            item_type = self.u32()
            return [self.value(item_type) for _ in range(self.u64())]
        if value_type == 10:
            return self.u64()
        if value_type == 11:
            return self.i64()
        if value_type == 12:
            return self.f64()
        raise ValueError(f"unsupported GGUF metadata type: {value_type}")


def _tensor_bytes(shape, ggml_type):
    try:
        values_per_block, bytes_per_block = GGML_BLOCKS[ggml_type]
    except KeyError as error:
        raise ValueError(f"unsupported GGUF tensor type: {ggml_type}") from error
    values = math.prod(shape)
    if values % values_per_block:
        raise ValueError(f"tensor shape is not aligned to GGML block: {shape}")
    return values // values_per_block * bytes_per_block


def read_gguf_header(model):
    """Return model metadata and exact tensor-info byte totals by layer."""
    model = Path(model)
    with model.open("rb") as stream:
        reader = _Reader(stream)
        if reader.read(4) != b"GGUF":
            raise ValueError(f"not a GGUF file: {model}")
        version = reader.u32()
        if version not in (2, 3):
            raise ValueError(f"unsupported GGUF version: {version}")
        tensor_count = reader.u64()
        metadata_count = reader.u64()
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            metadata[key] = reader.value(reader.u32())

        layer_bytes = {}
        shared_bytes = 0
        tensor_bytes = 0
        for _ in range(tensor_count):
            name = reader.string()
            dimensions = [reader.u64() for _ in range(reader.u32())]
            ggml_type = reader.u32()
            reader.u64()  # tensor data offset; payload is deliberately not read
            size = _tensor_bytes(dimensions, ggml_type)
            tensor_bytes += size
            match = LAYER_RE.search(name)
            if match:
                layer = int(match.group(1))
                layer_bytes[layer] = layer_bytes.get(layer, 0) + size
            else:
                shared_bytes += size

    architecture = metadata.get("general.architecture", "")
    def meta(name, default=None):
        return metadata.get(f"{architecture}.{name}", default)

    block_count = int(meta("block_count", max(layer_bytes, default=-1) + 1))
    return {
        "path": str(model.resolve()),
        "version": version,
        "architecture": architecture,
        "tensor_count": tensor_count,
        "metadata": metadata,
        "block_count": block_count,
        "layer_bytes": layer_bytes,
        "shared_bytes": shared_bytes,
        "tensor_bytes": tensor_bytes,
        "head_count_kv": int(meta("attention.head_count_kv", 0)),
        "key_length": int(meta("attention.key_length", 0)),
        "value_length": int(meta("attention.value_length", 0)),
    }


def kv_cache_bytes(header, context, cache_type_k="q8_0", cache_type_v="q8_0", sequences=1):
    """Estimate K+V storage for all layers and sequences at one context."""
    try:
        k_factor = KV_BYTES_PER_VALUE[cache_type_k]
        v_factor = KV_BYTES_PER_VALUE[cache_type_v]
    except KeyError as error:
        raise ValueError(f"unsupported KV cache type: {error.args[0]}") from error
    required = (int(header["block_count"]) * int(context) * int(sequences) *
                int(header["head_count_kv"]))
    return int(required * (int(header["key_length"]) * k_factor +
                          int(header["value_length"]) * v_factor))


def recommend_layers(header, vram_bytes, context=2048, target_mib=512,
                     compute_mib=256, cache_type_k="q8_0", cache_type_v="q8_0",
                     sequences=1):
    """Return a conservative whole-layer plan for one GPU.

    ``compute_mib`` is deliberately explicit: it covers attention/temporary
    buffers that are not model tensors. The remaining target margin protects
    against allocator fragmentation and desktop use.
    """
    kv_bytes = kv_cache_bytes(header, context, cache_type_k, cache_type_v, sequences)
    fixed_reserve = int(target_mib * MiB) + int(compute_mib * MiB)
    block_count = int(header["block_count"])
    if block_count <= 0:
        raise ValueError("GGUF model has no repeating layers")
    layers = header["layer_bytes"]
    chosen_repeating = 0
    used = 0
    # llama.cpp fills repeating layers from the end of the model when -ngl is
    # used. The output layer is counted by -ngl but is part of shared tensors.
    for repeating in range(1, block_count + 1):
        layer = block_count - repeating
        size = int(layers.get(layer, 0))
        # llama.cpp's -ngl count includes the output layer, while KV entries
        # belong to repeating transformer layers. Layers left on the host
        # keep their KV rows on the host as well (--kv-offload remains on).
        gpu_kv = math.ceil(kv_bytes * repeating / block_count)
        required = (int(header["shared_bytes"]) + used + size +
                    fixed_reserve + gpu_kv)
        if required > int(vram_bytes):
            break
        used += size
        chosen_repeating = repeating
    gpu_kv = math.ceil(kv_bytes * chosen_repeating / block_count)
    return {
        "vram_bytes": int(vram_bytes),
        "target_margin_bytes": int(target_mib * MiB),
        "compute_reserve_bytes": int(compute_mib * MiB),
        "kv_cache_bytes": kv_bytes,
        "gpu_kv_cache_bytes": gpu_kv,
        "host_kv_cache_bytes": max(0, kv_bytes - gpu_kv),
        "shared_model_bytes": int(header["shared_bytes"]),
        "whole_layer_bytes": used,
        "gpu_layers": chosen_repeating + 1,  # + output layer
        "gpu_repeating_layers": chosen_repeating,
        "total_layers": block_count,
        "fits": chosen_repeating > 0,
        "cache_type_k": cache_type_k,
        "cache_type_v": cache_type_v,
        "context": int(context),
        "sequences": int(sequences),
    }


def llama_server_args(model, plan, host="127.0.0.1", port=8088):
    """Build a speed-oriented llama-server command from a plan."""
    if not plan["fits"]:
        raise ValueError("VRAM budget cannot hold one complete model layer")
    return [
        "-m", str(model),
        "-ngl", str(plan["gpu_layers"]),
        "-c", str(plan["context"]),
        "-fa", "on", "--jinja",
        "--fit", "off",  # keep whole-layer placement; avoid partial MoE fit
        "--host", str(host), "--port", str(port),
        "-ctk", plan["cache_type_k"], "-ctv", plan["cache_type_v"],
        "-np", str(plan["sequences"]),
    ]


def _discover_vram():
    try:
        from resource_plan import discover_gpus
        gpus = discover_gpus()
    except (ImportError, OSError, ValueError):
        gpus = []
    if not gpus:
        raise ValueError("VRAM was not provided and no ROCm GPU was detected")
    return int(gpus[0]["free_bytes"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="plan whole-layer ROCm offload for a GGUF")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--vram-gib", type=float, default=0,
                        help="usable VRAM in GiB (0 = detect first ROCm GPU)")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--target-mib", type=int, default=512,
                        help="free VRAM margin after fitting")
    parser.add_argument("--compute-mib", type=int, default=256,
                        help="temporary/attention buffer reserve")
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    header = read_gguf_header(args.model)
    vram = int(args.vram_gib * GiB) if args.vram_gib > 0 else _discover_vram()
    plan = recommend_layers(
        header, vram, context=args.context, target_mib=args.target_mib,
        compute_mib=args.compute_mib, cache_type_k=args.cache_type_k,
        cache_type_v=args.cache_type_v, sequences=args.sequences,
    )
    plan["architecture"] = header["architecture"]
    plan["model"] = header["path"]
    plan["command"] = llama_server_args(args.model, plan, args.host, args.port)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"architecture  {plan['architecture']}")
        print(f"VRAM          {plan['vram_bytes'] / GiB:.2f} GiB")
        print(f"KV cache      {plan['kv_cache_bytes'] / MiB:.1f} MiB total, "
              f"{plan['gpu_kv_cache_bytes'] / MiB:.1f} MiB VRAM, "
              f"{plan['host_kv_cache_bytes'] / MiB:.1f} MiB RAM "
              f"({plan['cache_type_k']} K / {plan['cache_type_v']} V, "
              f"context {plan['context']})")
        print(f"whole layers  {plan['gpu_layers']}/{plan['total_layers']}")
        print(f"headroom      {plan['target_margin_bytes'] / MiB:.0f} MiB target + "
              f"{plan['compute_reserve_bytes'] / MiB:.0f} MiB compute")
        print("llama-server  " + subprocess.list2cmdline(plan["command"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"rocm_tune: {error}", file=sys.stderr)
        raise SystemExit(2)
