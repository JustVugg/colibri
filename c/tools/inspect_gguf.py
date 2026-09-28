#!/usr/bin/env python3
"""Bounded, dependency-free GGUF metadata/tensor inventory.

This deliberately does not decode tensor payloads.  It is used while bringing
up new Colibri engines to establish the exact metadata, tensor names, shapes,
types, and hybrid layer schedule without allocating tokenizer arrays.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path


META_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
              10: 8, 11: 8, 12: 8}
META_FORMATS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
MAX_STRING = 256 * 1024 * 1024
MAX_ITEMS = 1_000_000_000
LAYER_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


class GGUFError(ValueError):
    pass


class Reader:
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.size = path.stat().st_size

    def close(self):
        self.file.close()

    def read(self, size: int) -> bytes:
        if size < 0 or self.file.tell() + size > self.size:
            raise GGUFError("unexpected end of GGUF file")
        value = self.file.read(size)
        if len(value) != size:
            raise GGUFError("unexpected end of GGUF file")
        return value

    def scalar(self, fmt: str):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = self.scalar("Q")
        if length > MAX_STRING:
            raise GGUFError(f"unreasonably large GGUF string ({length} bytes)")
        return self.read(length).decode("utf-8", errors="replace")

    def skip(self, size: int):
        if size < 0 or self.file.tell() + size > self.size:
            raise GGUFError("GGUF skip exceeds file size")
        self.file.seek(size, os.SEEK_CUR)


def read_value(reader: Reader, value_type: int, capture: bool = True, depth: int = 0):
    if depth > 16:
        raise GGUFError("GGUF metadata nesting is too deep")
    if value_type in META_FORMATS:
        value = reader.scalar(META_FORMATS[value_type])
        return value if capture else None
    if value_type == 8:
        value = reader.string()
        return value if capture else None
    if value_type != 9:
        raise GGUFError(f"unknown GGUF metadata type {value_type}")
    element_type = reader.scalar("I")
    count = reader.scalar("Q")
    if count > MAX_ITEMS:
        raise GGUFError(f"unreasonably large GGUF array ({count} items)")
    if not capture and element_type in META_SIZES:
        reader.skip(count * META_SIZES[element_type])
        return None
    # Large token arrays are irrelevant to architecture discovery. Strings
    # remain length-prefixed, so they must be walked but need not be retained.
    keep = capture and count <= 4096
    values = [] if keep else None
    for _ in range(count):
        value = read_value(reader, element_type, keep, depth + 1)
        if keep:
            values.append(value)
    return values if keep else {"count": count, "element_type": element_type}


def inspect(path: Path) -> dict:
    reader = Reader(path)
    try:
        if reader.read(4) != b"GGUF":
            raise GGUFError("not a GGUF file")
        version = reader.scalar("I")
        if version not in (2, 3):
            raise GGUFError(f"unsupported GGUF version {version}")
        tensor_count = reader.scalar("Q")
        metadata_count = reader.scalar("Q")
        if tensor_count > 10_000_000 or metadata_count > 10_000_000:
            raise GGUFError("unreasonable GGUF table size")
        metadata = {}
        for _ in range(metadata_count):
            key = reader.string()
            value_type = reader.scalar("I")
            capture = (key.startswith("general.") or key.startswith("nemotron_h_moe.")
                       or key.startswith("tokenizer.ggml."))
            metadata[key] = read_value(reader, value_type, capture)
        tensors = []
        layer_kinds: dict[int, set[str]] = {}
        type_counts: dict[int, int] = {}
        for _ in range(tensor_count):
            name = reader.string()
            n_dims = reader.scalar("I")
            if not 1 <= n_dims <= 8:
                raise GGUFError(f"invalid dimension count for {name}: {n_dims}")
            dims = [reader.scalar("Q") for _ in range(n_dims)]
            tensor_type = reader.scalar("I")
            offset = reader.scalar("Q")
            tensors.append({"name": name, "dims": dims, "type": tensor_type,
                            "offset": offset})
            type_counts[tensor_type] = type_counts.get(tensor_type, 0) + 1
            match = LAYER_RE.match(name)
            if match:
                layer = int(match.group(1))
                suffix = match.group(2)
                kind = "ssm" if suffix.startswith("ssm_") or suffix in ("ssm_a", "ssm_d") else \
                       "attention" if suffix.startswith("attn_") and suffix != "attn_norm.weight" else \
                       "moe" if suffix.startswith("ffn_") else "other"
                layer_kinds.setdefault(layer, set()).add(kind)
        alignment = int(metadata.get("general.alignment", 32))
        data_offset = (reader.file.tell() + alignment - 1) & ~(alignment - 1)
        schedule = []
        for layer in sorted(layer_kinds):
            kinds = layer_kinds[layer]
            mixer = "ssm" if "ssm" in kinds else "attention" if "attention" in kinds else "unknown"
            schedule.append({"layer": layer, "mixer": mixer, "moe": "moe" in kinds})
        return {"path": str(path.resolve()), "file_size": reader.size,
                "version": version, "tensor_count": tensor_count,
                "metadata_count": metadata_count, "metadata": metadata, "data_offset": data_offset,
                "tensor_type_counts": {str(k): v for k, v in sorted(type_counts.items())},
                "schedule": schedule, "tensors": tensors}
    finally:
        reader.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--json", action="store_true", help="emit the complete inventory as JSON")
    parser.add_argument("--tensors", action="store_true", help="include tensor rows in text output")
    args = parser.parse_args()
    try:
        result = inspect(args.gguf)
    except (OSError, GGUFError) as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    meta = result["metadata"]
    print(f"architecture: {meta.get('general.architecture', '<missing>')}")
    print(f"version: {result['version']}  tensors: {result['tensor_count']}  metadata: {result['metadata_count']}")
    print(f"file: {result['file_size']} bytes  tensor types: {result['tensor_type_counts']}")
    for key in sorted(meta):
        if key.startswith("nemotron_h_moe.") or key.startswith("tokenizer.ggml."):
            print(f"{key} = {meta[key]}")
    print("schedule: " + " ".join(
        f"{row['layer']}:{'moe' if row['mixer'] == 'unknown' and row['moe'] else row['mixer']}"
        for row in result["schedule"]))
    if args.tensors:
        for tensor in result["tensors"]:
            print(f"{tensor['name']} {tensor['dims']} type={tensor['type']} offset={tensor['offset']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
