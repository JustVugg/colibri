#!/usr/bin/env python3
"""GGUF reader, pure stdlib.

Parses the binary header, the metadata key/value store and the tensor index of
a GGUF file (https://github.com/ggml-org/ggml/blob/master/docs/gguf.md),
computes each tensor's absolute byte range and reads raw tensor bytes on
demand. No third-party dependency: converter profiles import this module and
own the per-ggml-type decoding.

Tensor shapes are reported exactly as stored on disk: GGUF writers store the
PyTorch/HuggingFace shape reversed (the fastest-varying dimension first), so a
checkpoint weight logged as [out, in] appears here as dims [in, out].

Only the ggml types listed in GGML_TYPE_RAW and GGML_TYPE_BLOCK are understood.
A tensor of any other type makes parsing refuse loudly rather than mis-size
it; converter profiles extend coverage by adding entries to those tables.
"""

import os
import struct

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
SUPPORTED_VERSIONS = (1, 2, 3)

MAX_METADATA_ITEMS = 1 << 22
MAX_METADATA_DEPTH = 64
MAX_TENSORS = 1 << 24

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_MXFP4 = 39

GGML_TYPE_RAW = {
    GGML_TYPE_F32: ("F32", 4),
    GGML_TYPE_F16: ("F16", 2),
    GGML_TYPE_I8: ("I8", 1),
    GGML_TYPE_I16: ("I16", 2),
    GGML_TYPE_I32: ("I32", 4),
    GGML_TYPE_I64: ("I64", 8),
    GGML_TYPE_F64: ("F64", 8),
    GGML_TYPE_BF16: ("BF16", 2),
}

# Block sizes and bytes-per-block verified against ggml-org/ggml
# src/ggml-common.h (block struct static_asserts) and src/ggml.c
# (ggml_type_traits.type_size/blck_size), QK_K = 256.
GGML_TYPE_BLOCK = {
    GGML_TYPE_Q4_0: ("Q4_0", 32, 18),
    GGML_TYPE_Q4_1: ("Q4_1", 32, 20),
    GGML_TYPE_Q5_0: ("Q5_0", 32, 22),
    GGML_TYPE_Q5_1: ("Q5_1", 32, 24),
    GGML_TYPE_Q8_0: ("Q8_0", 32, 34),
    GGML_TYPE_Q8_1: ("Q8_1", 32, 36),
    GGML_TYPE_Q2_K: ("Q2_K", 256, 84),
    GGML_TYPE_Q3_K: ("Q3_K", 256, 110),
    GGML_TYPE_Q4_K: ("Q4_K", 256, 144),
    GGML_TYPE_Q5_K: ("Q5_K", 256, 176),
    GGML_TYPE_Q6_K: ("Q6_K", 256, 210),
    GGML_TYPE_Q8_K: ("Q8_K", 256, 292),
    GGML_TYPE_IQ2_XXS: ("IQ2_XXS", 256, 66),
    GGML_TYPE_IQ2_XS: ("IQ2_XS", 256, 74),
    GGML_TYPE_IQ3_XXS: ("IQ3_XXS", 256, 98),
    GGML_TYPE_IQ1_S: ("IQ1_S", 256, 50),
    GGML_TYPE_IQ4_NL: ("IQ4_NL", 32, 18),
    GGML_TYPE_IQ3_S: ("IQ3_S", 256, 110),
    GGML_TYPE_IQ2_S: ("IQ2_S", 256, 82),
    GGML_TYPE_IQ4_XS: ("IQ4_XS", 256, 136),
    GGML_TYPE_IQ1_M: ("IQ1_M", 256, 56),
    GGML_TYPE_TQ1_0: ("TQ1_0", 256, 54),
    GGML_TYPE_TQ2_0: ("TQ2_0", 256, 66),
    GGML_TYPE_MXFP4: ("MXFP4", 32, 17),
}

_GGML_TYPE_NAMES = {}
for _id, (_name, _bytes) in GGML_TYPE_RAW.items():
    _GGML_TYPE_NAMES[_id] = _name
for _id, (_name, _block, _bytes) in GGML_TYPE_BLOCK.items():
    _GGML_TYPE_NAMES[_id] = _name

GGML_TYPE_NAMES = dict(_GGML_TYPE_NAMES)

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

_SCALAR_FORMATS = {
    METADATA_UINT8: ("B", int),
    METADATA_INT8: ("b", int),
    METADATA_UINT16: ("H", int),
    METADATA_INT16: ("h", int),
    METADATA_UINT32: ("I", int),
    METADATA_INT32: ("i", int),
    METADATA_FLOAT32: ("f", float),
    METADATA_UINT64: ("Q", int),
    METADATA_INT64: ("q", int),
    METADATA_FLOAT64: ("d", float),
}


class GGUFError(ValueError):
    pass


def ggml_type_name(ggml_type):
    return GGML_TYPE_NAMES.get(ggml_type, "T%d" % ggml_type)


def ggml_block_spec(ggml_type):
    if ggml_type in GGML_TYPE_RAW:
        return 1, GGML_TYPE_RAW[ggml_type][1]
    if ggml_type in GGML_TYPE_BLOCK:
        name, block, bytes_per_block = GGML_TYPE_BLOCK[ggml_type]
        return block, bytes_per_block
    raise GGUFError("unsupported ggml type %d (%s)" % (ggml_type, ggml_type_name(ggml_type)))


def _aligned(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _tensor_bytes(dims, ggml_type):
    numel = 1
    for dim in dims:
        numel *= dim
    block_elems, block_bytes = ggml_block_spec(ggml_type)
    row = dims[0] if dims else 1
    if block_elems > 1:
        if row % block_elems != 0:
            raise GGUFError(
                "ggml type %s requires dims[0] (%d) to be a multiple of %d"
                % (ggml_type_name(ggml_type), row, block_elems))
        blocks = row // block_elems
        if len(dims) > 1:
            row_bytes = blocks * block_bytes
            rest = 1
            for dim in dims[1:]:
                rest *= dim
            return row_bytes * rest, numel
        return blocks * block_bytes, numel
    return numel * block_bytes, numel


class GGUFReader:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        self.file_size = os.fstat(self.f.fileno()).st_size
        self.version = None
        self.alignment = DEFAULT_ALIGNMENT
        self.metadata = {}
        self.tensors = []
        self._tensor_index = {}
        self._data_base = None
        try:
            self._parse()
        except GGUFError:
            self.f.close()
            raise
        except (OSError, struct.error) as exc:
            self.f.close()
            raise GGUFError("%s: %s" % (path, exc)) from exc

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _read_exact(self, count, context):
        if count < 0 or self.f.tell() + count > self.file_size:
            raise GGUFError(
                "%s: truncated at offset %d reading %s (%d bytes, file is %d)"
                % (self.path, self.f.tell(), context, count, self.file_size))
        data = self.f.read(count)
        if len(data) != count:
            raise GGUFError(
                "%s: short read at offset %d reading %s"
                % (self.path, self.f.tell() - len(data), context))
        return data

    def _read_u64(self, context):
        return struct.unpack("<Q", self._read_exact(8, context))[0]

    def _read_u32(self, context):
        return struct.unpack("<I", self._read_exact(4, context))[0]

    def _read_string(self, context):
        length = self._read_u64("%s length" % context)
        if length > (1 << 31):
            raise GGUFError("%s: unreasonable %s length %d" % (self.path, context, length))
        raw = self._read_exact(length, context)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GGUFError("%s: %s is not valid UTF-8: %s" % (self.path, context, exc)) from exc

    def _read_scalar(self, value_type):
        fmt, cast = _SCALAR_FORMATS[value_type]
        size = struct.calcsize(fmt)
        return cast(struct.unpack("<" + fmt, self._read_exact(size, "metadata value"))[0])

    def _read_metadata_value(self, value_type, depth=0):
        if value_type == METADATA_STRING:
            return self._read_string("metadata string")
        if value_type == METADATA_BOOL:
            raw = self._read_exact(1, "metadata bool")[0]
            if raw not in (0, 1):
                raise GGUFError(
                    "%s: invalid bool metadata value %d" % (self.path, raw))
            return bool(raw)
        if value_type == METADATA_ARRAY:
            # Arrays may nest (the spec allows it); bound the recursion so a
            # hostile file cannot drive Python into RecursionError, which would
            # escape the GGUFError contract every other failure keeps.
            if depth >= MAX_METADATA_DEPTH:
                raise GGUFError(
                    "%s: metadata array nesting exceeds %d levels"
                    % (self.path, MAX_METADATA_DEPTH))
            element_type = self._read_u32("array element type")
            count = self._read_u64("array length")
            if count > MAX_METADATA_ITEMS:
                raise GGUFError(
                    "%s: metadata array of %d items exceeds cap %d"
                    % (self.path, count, MAX_METADATA_ITEMS))
            if element_type not in _SCALAR_FORMATS and element_type not in (
                    METADATA_STRING, METADATA_BOOL, METADATA_ARRAY):
                raise GGUFError(
                    "%s: unknown metadata array element type %d"
                    % (self.path, element_type))
            return [self._read_metadata_value(element_type, depth + 1)
                    for _ in range(count)]
        if value_type in _SCALAR_FORMATS:
            return self._read_scalar(value_type)
        raise GGUFError("%s: unknown metadata value type %d" % (self.path, value_type))

    def _parse(self):
        magic = self._read_exact(4, "magic")
        if magic != GGUF_MAGIC:
            raise GGUFError(
                "%s: bad magic %r (expected %r): not a GGUF file"
                % (self.path, magic, GGUF_MAGIC))
        self.version = self._read_u32("version")
        if self.version not in SUPPORTED_VERSIONS:
            raise GGUFError(
                "%s: unsupported GGUF version %d (supported: %s)"
                % (self.path, self.version, SUPPORTED_VERSIONS))
        tensor_count = self._read_u64("tensor count")
        kv_count = self._read_u64("metadata count")
        if kv_count > MAX_METADATA_ITEMS:
            raise GGUFError(
                "%s: metadata count %d exceeds cap %d"
                % (self.path, kv_count, MAX_METADATA_ITEMS))
        if tensor_count > MAX_TENSORS:
            raise GGUFError(
                "%s: tensor count %d exceeds cap %d"
                % (self.path, tensor_count, MAX_TENSORS))
        for _ in range(kv_count):
            key = self._read_string("metadata key")
            value_type = self._read_u32("metadata value type")
            if value_type not in _SCALAR_FORMATS and value_type not in (
                    METADATA_STRING, METADATA_BOOL, METADATA_ARRAY):
                raise GGUFError(
                    "%s: unknown metadata value type %d for key %r"
                    % (self.path, value_type, key))
            self.metadata[key] = self._read_metadata_value(value_type)
        alignment = self.metadata.get("general.alignment")
        if alignment is not None:
            if not isinstance(alignment, int) or alignment <= 0 or alignment % 8 != 0:
                raise GGUFError(
                    "%s: general.alignment %r must be a positive multiple of 8"
                    % (self.path, alignment))
            self.alignment = int(alignment)
        for _ in range(tensor_count):
            name = self._read_string("tensor name")
            n_dims = self._read_u32("tensor rank")
            if n_dims > 8:
                raise GGUFError(
                    "%s: tensor %r has %d dims (max 8)" % (self.path, name, n_dims))
            dims = [self._read_u64("tensor dim") for _ in range(n_dims)]
            ggml_type = self._read_u32("tensor type")
            offset = self._read_u64("tensor offset")
            if name in self._tensor_index:
                raise GGUFError(
                    "%s: duplicate tensor name %r" % (self.path, name))
            tensor = {
                "name": name,
                "rank": n_dims,
                "dims": dims,
                "ggml_type": ggml_type,
                "type_name": ggml_type_name(ggml_type),
                "offset": offset,
            }
            self.tensors.append(tensor)
            self._tensor_index[name] = tensor
        self._data_base = _aligned(self.f.tell(), self.alignment)
        for tensor in self.tensors:
            if tensor["offset"] % self.alignment != 0:
                raise GGUFError(
                    "%s: tensor %r offset %d is not aligned to %d"
                    % (self.path, tensor["name"], tensor["offset"], self.alignment))
            try:
                nbytes, numel = _tensor_bytes(tensor["dims"], tensor["ggml_type"])
            except GGUFError as exc:
                raise GGUFError(
                    "%s: tensor %r: %s" % (self.path, tensor["name"], exc)) from exc
            absolute = self._data_base + tensor["offset"]
            if absolute + nbytes > self.file_size:
                raise GGUFError(
                    "%s: tensor %r range [%d, %d) exceeds file size %d"
                    % (self.path, tensor["name"], absolute,
                       absolute + nbytes, self.file_size))
            tensor["numel"] = numel
            tensor["nbytes"] = nbytes
            tensor["abs_offset"] = absolute

    def tensor(self, name):
        tensor = self._tensor_index.get(name)
        if tensor is None:
            raise GGUFError(
                "%s: missing tensor %r" % (self.path, name))
        return tensor

    def read_tensor(self, tensor):
        self.f.seek(tensor["abs_offset"])
        data = self._read_exact(tensor["nbytes"], "tensor %r data" % tensor["name"])
        return data


def load(path):
    return GGUFReader(path)
