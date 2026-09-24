#!/usr/bin/env python3
"""Deterministic GGUF v3 fixture builder for tests/test_gguf_reader.py.

Builds a small, valid GGUF v3 file exercising every metadata value type
(scalar ints/floats/bool/string, flat arrays and a nested array) plus tensors
of the raw types (F32, F16, BF16) and one block-quantized type (Q8_0).

This module is deliberately self-contained: it defines its own copy of the
GGUF spec constants instead of importing gguf_reader.py. The reader tests are
therefore a genuine cross-check of two independent implementations of the same
public spec, not a round-trip of one implementation against itself.

The file layout mirrors ggml-org/ggml/docs/gguf.md: header, metadata KV,
tensor infos, padding to general.alignment, then aligned tensor data whose
offsets are relative to the data section start. Shapes are stored reversed
relative to the PyTorch convention, like real GGUF writers do.
"""

import os
import random
import struct

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
GGUF_VERSION = 3

METADATA_UINT8 = 0
METADATA_INT8 = 1
METADATA_UINT16 = 2
METADATA_INT16 = 3
METADATA_UINT32 = 4
METADATA_INT32 = 5
METADATA_FLOAT32 = 6
METADATA_BOOL = 7
METADATA_STRING = 8
METADATA_ARRAY = 9
METADATA_UINT64 = 10
METADATA_INT64 = 11
METADATA_FLOAT64 = 12


def _align(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _encode_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _encode_scalar(value_type, value):
    if value_type == METADATA_UINT8:
        return struct.pack("<B", value)
    if value_type == METADATA_INT8:
        return struct.pack("<b", value)
    if value_type == METADATA_UINT16:
        return struct.pack("<H", value)
    if value_type == METADATA_INT16:
        return struct.pack("<h", value)
    if value_type == METADATA_UINT32:
        return struct.pack("<I", value)
    if value_type == METADATA_INT32:
        return struct.pack("<i", value)
    if value_type == METADATA_FLOAT32:
        return struct.pack("<f", value)
    if value_type == METADATA_UINT64:
        return struct.pack("<Q", value)
    if value_type == METADATA_INT64:
        return struct.pack("<q", value)
    if value_type == METADATA_FLOAT64:
        return struct.pack("<d", value)
    raise ValueError("not a scalar metadata type: %d" % value_type)


def _encode_value(value_type, value):
    if value_type == METADATA_STRING:
        return _encode_string(value)
    if value_type == METADATA_BOOL:
        return b"\x01" if value else b"\x00"
    if value_type == METADATA_ARRAY:
        element_type, items = value
        out = struct.pack("<I", element_type) + struct.pack("<Q", len(items))
        for item in items:
            out += _encode_value(element_type, item)
        return out
    return _encode_scalar(value_type, value)


def fixture_metadata(seed, alignment=DEFAULT_ALIGNMENT):
    rng = random.Random(seed)
    return [
        (METADATA_UINT32, "general.alignment", alignment),
        (METADATA_STRING, "general.architecture", "gguf-test"),
        (METADATA_STRING, "general.name", "gguf-reader-fixture"),
        (METADATA_UINT32, "general.file_type", 1),
        (METADATA_FLOAT32, "test.float32", 1.5),
        (METADATA_FLOAT64, "test.float64", -2.25),
        (METADATA_BOOL, "test.bool", True),
        (METADATA_UINT8, "test.uint8", 200),
        (METADATA_INT8, "test.int8", -100),
        (METADATA_UINT16, "test.uint16", 60000),
        (METADATA_INT16, "test.int16", -30000),
        (METADATA_UINT32, "test.uint32", 4000000000),
        (METADATA_INT32, "test.int32", -2000000000),
        (METADATA_UINT64, "test.uint64", 2**63),
        (METADATA_INT64, "test.int64", -(2**63)),
        (METADATA_ARRAY, "test.array.string",
         (METADATA_STRING, ["one", "two", "three"])),
        (METADATA_ARRAY, "test.array.int",
         (METADATA_INT32, [rng.randint(-100, 100) for _ in range(8)])),
        (METADATA_ARRAY, "test.array.float",
         (METADATA_FLOAT32,
          [struct.unpack("<f", struct.pack("<f", rng.uniform(-1.0, 1.0)))[0]
           for _ in range(4)])),
        (METADATA_ARRAY, "test.array.nested",
         (METADATA_ARRAY, [
             (METADATA_INT32, [rng.randint(0, 9) for _ in range(3)]),
             (METADATA_INT32, [rng.randint(0, 9) for _ in range(2)]),
         ])),
    ]


def _f16(value):
    return struct.pack("<e", value)


def _bf16(value):
    return struct.pack("<H", ((struct.unpack("<I", struct.pack("<f", value))[0] + 0x8000) >> 16) & 0xFFFF)


def fixture_tensors(seed):
    rng = random.Random(seed + 1)
    specs = [
        ("tensor.f32.2d", 0, [8, 16]),       # F32  (logical [16, 8])
        ("tensor.f16.2d", 1, [8, 16]),       # F16
        ("tensor.bf16.1d", 30, [16]),        # BF16
        ("tensor.q8_0.2d", 8, [32, 16]),     # Q8_0, dims[0] % 32 == 0
    ]
    tensors = []
    for name, ggml_type, dims in specs:
        if ggml_type in (0, 1, 30):
            values = [rng.uniform(-1.0, 1.0) for _ in range(_numel(dims))]
            if ggml_type == 0:
                payload = b"".join(struct.pack("<f", v) for v in values)
            elif ggml_type == 1:
                payload = b"".join(_f16(v) for v in values)
            else:
                payload = b"".join(_bf16(v) for v in values)
        elif ggml_type == 8:
            blocks = dims[0] // 32
            payload = b""
            for _ in range(blocks):
                scale = rng.uniform(0.01, 0.5)
                payload += _f16(scale)
                payload += struct.pack("<32b", *[rng.randint(-127, 127) for _ in range(32)])
            if len(dims) > 1:
                payload *= dims[1]
        else:
            raise ValueError("unsupported fixture type %d" % ggml_type)
        tensors.append({"name": name, "ggml_type": ggml_type,
                        "dims": dims, "payload": payload})
    return tensors


def _numel(dims):
    total = 1
    for dim in dims:
        total *= dim
    return total


def write_gguf_fixture(path, seed=20260825, alignment=DEFAULT_ALIGNMENT):
    """Write the deterministic GGUF fixture and return (metadata, tensors)."""
    metadata = fixture_metadata(seed, alignment)
    tensors = fixture_tensors(seed)

    out = GGUF_MAGIC
    out += struct.pack("<I", GGUF_VERSION)
    out += struct.pack("<Q", len(tensors))
    out += struct.pack("<Q", len(metadata))
    for value_type, key, value in metadata:
        out += _encode_string(key)
        out += struct.pack("<I", value_type)
        out += _encode_value(value_type, value)

    data_offset = 0
    data_base = _align(len(out) + sum(
        8 + len(t["name"].encode("utf-8")) + 4 + 8 * len(t["dims"]) + 4 + 8
        for t in tensors), alignment)
    for tensor in tensors:
        out += _encode_string(tensor["name"])
        out += struct.pack("<I", len(tensor["dims"]))
        for dim in tensor["dims"]:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", tensor["ggml_type"])
        out += struct.pack("<Q", data_offset)
        data_offset += len(tensor["payload"])
        data_offset = _align(data_offset, alignment)

    out += b"\x00" * (data_base - len(out))
    for tensor in tensors:
        out += tensor["payload"]
        out += b"\x00" * (_align(len(tensor["payload"]), alignment) - len(tensor["payload"]))

    with open(path, "wb") as f:
        f.write(out)
    return metadata, tensors


if __name__ == "__main__":
    import sys
    output = sys.argv[1] if len(sys.argv) > 1 else "gguf_fixture.gguf"
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
    metadata, tensors = write_gguf_fixture(output)
    print("wrote %s: %d metadata KV, %d tensors" % (output, len(metadata), len(tensors)))
