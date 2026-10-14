"""Stdlib edits of a safetensors container, for tests that need a hostile one.

A model directory is as attacker-chosen as its config.json: whoever ships it picks
every byte of every tensor and every length in every header. These helpers copy a
known-good tiny fixture and change exactly one thing, so a test can ask what the
engine does with that one thing.
"""
import json
import math
import shutil
import struct
from pathlib import Path

_NAN = {"F32": struct.pack("<I", 0x7FC00000), "F16": struct.pack("<H", 0x7E00),
        "BF16": struct.pack("<H", 0x7FC0)}


def _read_header(path):
    with open(path, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size)), 8 + size


def _shard_of(directory, name):
    for shard in sorted(Path(directory).glob("*.safetensors")):
        header, start = _read_header(shard)
        if name in header:
            return shard, header, start
    raise KeyError(f"{name} is in no shard of {directory}")


def shape_of(directory, name):
    return list(_shard_of(directory, name)[1][name]["shape"])


def dtype_of(directory, name):
    return _shard_of(directory, name)[1][name]["dtype"]


def copy_fixture(source, destination):
    shutil.copytree(source, destination)
    return Path(destination)


def fill_nan(directory, name):
    """Overwrite every element of tensor `name` with a quiet NaN of its dtype."""
    shard, header, start = _shard_of(directory, name)
    entry = header[name]
    pattern = _NAN[entry["dtype"]]
    first, last = entry["data_offsets"]
    with open(shard, "r+b") as handle:
        handle.seek(start + first)
        handle.write(pattern * ((last - first) // len(pattern)))


def add_f32(directory, beside, name, count, value=0x41414141):
    """Append an F32 tensor `name` of `count` elements to the shard holding `beside`.

    The default fill is 0x41414141, 'AAAA': the bytes a heap overflow would spray.
    """
    shard, header, start = _shard_of(directory, beside)
    data = shard.read_bytes()[start:]
    header[name] = {"dtype": "F32", "shape": [count],
                    "data_offsets": [len(data), len(data) + 4 * count]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data
                      + struct.pack("<I", value) * count)


def tensor_bytes(directory, name):
    shard, header, start = _shard_of(directory, name)
    first, last = header[name]["data_offsets"]
    with open(shard, "rb") as handle:
        handle.seek(start + first)
        return handle.read(last - first)


def put(directory, name, dtype, shape, payload, beside=None):
    """Store tensor `name` as `dtype`, `shape`, `payload`: in its shard, or beside `beside`.

    The header stays honest: dtype, shape and offsets agree with the bytes, and the
    other tensors move up so the data section has no hole. Only the values the test
    chose are hostile.
    """
    shard, header, start = _shard_of(directory, beside or name)
    data = shard.read_bytes()[start:]
    tensors = sorted((entry["data_offsets"][0], key) for key, entry in header.items()
                     if key not in ("__metadata__", name))
    body, at = [], 0
    for _, key in tensors:
        first, last = header[key]["data_offsets"]
        body.append(data[first:last])
        header[key]["data_offsets"] = [at, at + last - first]
        at += last - first
    header[name] = {"dtype": dtype, "shape": list(shape),
                    "data_offsets": [at, at + len(payload)]}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(body) + payload)


def shrink(directory, name, count):
    """Keep the first `count` elements of tensor `name`, declared as a flat vector."""
    dtype = dtype_of(directory, name)
    put(directory, name, dtype, [count],
        tensor_bytes(directory, name)[:len(_NAN[dtype]) * count])


def reshape(directory, name, shape):
    """Declare tensor `name` as `shape`, its data cut or zero-padded to fit."""
    dtype = dtype_of(directory, name)
    size = len(_NAN[dtype]) * math.prod(shape)
    data = tensor_bytes(directory, name)[:size]
    put(directory, name, dtype, shape, data + bytes(size - len(data)))
