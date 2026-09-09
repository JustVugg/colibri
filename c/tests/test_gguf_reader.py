#!/usr/bin/env python3
"""tools/gguf_reader.py + tests/gguf_fixture.py: pure-python GGUF round trip.

Builds a deterministic GGUF fixture in a temp directory and asserts the reader
recovers every metadata value type (scalars, bool, string, flat arrays, nested
array), every tensor header (name/dims/type) and every tensor payload
byte-for-byte. Then exercises the refusal paths an untrusted-container reader
must hit loudly: bad magic, unsupported version, truncated file, duplicate
tensor names, unaligned tensor offset, tensor range beyond EOF, unsupported
ggml type, rank above cap, missing/invalid/honored alignment, malformed
metadata values and capped counts. No third-party dependency.
"""

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gguf_reader
from gguf_reader import GGUFError, GGUFReader
from gguf_fixture import (
    METADATA_ARRAY,
    METADATA_BOOL,
    METADATA_FLOAT32,
    METADATA_STRING,
    METADATA_UINT32,
    _encode_string,
    _encode_value,
    write_gguf_fixture,
)


def _raw_header(magic=b"GGUF", version=3, tensor_count=0, kv_count=0):
    return magic + struct.pack("<I", version) + struct.pack("<Q", tensor_count) + struct.pack("<Q", kv_count)


def _raw_kv(key, value_type, value):
    """One encoded metadata KV (value_type + value), like gguf_fixture does."""
    return _encode_string(key) + struct.pack("<I", value_type) + _encode_value(value_type, value)


def _raw_kv_value_type_only(key, value_type):
    """Metadata KV with only the declared type (no value bytes): the reader
    must refuse on the type alone, before needing a value."""
    return _encode_string(key) + struct.pack("<I", value_type)


def _raw_kv_string_value(key, raw_bytes):
    """Metadata STRING KV whose value bytes are provided verbatim (for
    malformed strings: oversized length, invalid UTF-8)."""
    return _encode_string(key) + struct.pack("<I", METADATA_STRING) + raw_bytes


def _raw_kv_array(key, element_type, count=0):
    """Metadata ARRAY KV with an explicit element type and no items."""
    return (_encode_string(key) + struct.pack("<I", METADATA_ARRAY)
            + struct.pack("<I", element_type) + struct.pack("<Q", count))


def _tensor_info(name=b"t", n_dims=1, dims=(16,), ggml_type=0, offset=0):
    return (
        struct.pack("<Q", len(name)) + name
        + struct.pack("<I", n_dims)
        + b"".join(struct.pack("<Q", d) for d in dims)
        + struct.pack("<I", ggml_type)
        + struct.pack("<Q", offset)
    )


def _unwrap(value):
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], int) and isinstance(value[1], list):
        return [_unwrap(item) for item in value[1]]
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


class GGUFRoundTripTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "model.gguf"

    def test_metadata_round_trip(self):
        expected_meta, expected_tensors = write_gguf_fixture(str(self.path))
        expected = {key: _unwrap(value) for value_type, key, value in expected_meta}
        with GGUFReader(str(self.path)) as reader:
            self.assertEqual(reader.version, 3)
            self.assertEqual(reader.alignment, 32)
            self.assertEqual(reader.metadata, expected)

    def test_non_default_alignment_honored(self):
        expected_meta, expected_tensors = write_gguf_fixture(str(self.path), alignment=64)
        with GGUFReader(str(self.path)) as reader:
            self.assertEqual(reader.alignment, 64)
            for tensor in reader.tensors:
                self.assertEqual(tensor["offset"] % 64, 0)
                self.assertEqual(tensor["abs_offset"] % 64, 0)

    def test_alignment_missing_defaults_to_32(self):
        header = _raw_header()
        self.path.write_bytes(header)
        with GGUFReader(str(self.path)) as reader:
            self.assertEqual(reader.alignment, 32)
            self.assertEqual(reader.metadata, {})
            self.assertEqual(reader.tensors, [])

    def test_tensor_headers_and_payload(self):
        expected_meta, expected_tensors = write_gguf_fixture(str(self.path))
        with GGUFReader(str(self.path)) as reader:
            self.assertEqual(len(reader.tensors), len(expected_tensors))
            for reader_tensor, expected in zip(reader.tensors, expected_tensors):
                self.assertEqual(reader_tensor["name"], expected["name"])
                self.assertEqual(reader_tensor["dims"], expected["dims"])
                self.assertEqual(reader_tensor["ggml_type"], expected["ggml_type"])
                self.assertEqual(reader_tensor["abs_offset"] % reader.alignment, 0)
                self.assertEqual(reader.read_tensor(reader_tensor), expected["payload"])

    def test_single_tensor_lookup(self):
        write_gguf_fixture(str(self.path))
        with GGUFReader(str(self.path)) as reader:
            self.assertEqual(reader.tensor("tensor.f32.2d")["name"], "tensor.f32.2d")
            with self.assertRaises(GGUFError):
                reader.tensor("does.not.exist")


class GGMLTypeTableTest(unittest.TestCase):
    """Pins the block geometry table to values verified against ggml sources.

    Every entry in GGML_TYPE_BLOCK is (block_elems, bytes_per_block), checked
    against ggml-org/ggml src/ggml-common.h static_asserts and the GGML_QUANT_SIZES
    table in llama.cpp's gguf-py (QK_K = 256). A wrong value here silently
    mis-sizes every tensor of that type, so the table is pinned verbatim.
    """

    def _expected_block_table(self):
        QK_K = 256
        return {
            gguf_reader.GGML_TYPE_Q4_0: (32, 18),
            gguf_reader.GGML_TYPE_Q4_1: (32, 20),
            gguf_reader.GGML_TYPE_Q5_0: (32, 22),
            gguf_reader.GGML_TYPE_Q5_1: (32, 24),
            gguf_reader.GGML_TYPE_Q8_0: (32, 34),
            gguf_reader.GGML_TYPE_Q8_1: (32, 36),
            gguf_reader.GGML_TYPE_Q2_K: (QK_K, 84),
            gguf_reader.GGML_TYPE_Q3_K: (QK_K, 110),
            gguf_reader.GGML_TYPE_Q4_K: (QK_K, 144),
            gguf_reader.GGML_TYPE_Q5_K: (QK_K, 176),
            gguf_reader.GGML_TYPE_Q6_K: (QK_K, 210),
            gguf_reader.GGML_TYPE_Q8_K: (QK_K, 292),
            gguf_reader.GGML_TYPE_IQ2_XXS: (QK_K, 66),
            gguf_reader.GGML_TYPE_IQ2_XS: (QK_K, 74),
            gguf_reader.GGML_TYPE_IQ3_XXS: (QK_K, 98),
            gguf_reader.GGML_TYPE_IQ1_S: (QK_K, 50),
            gguf_reader.GGML_TYPE_IQ4_NL: (32, 18),
            gguf_reader.GGML_TYPE_IQ3_S: (QK_K, 110),
            gguf_reader.GGML_TYPE_IQ2_S: (QK_K, 82),
            gguf_reader.GGML_TYPE_IQ4_XS: (QK_K, 136),
            gguf_reader.GGML_TYPE_IQ1_M: (QK_K, 56),
            gguf_reader.GGML_TYPE_TQ1_0: (QK_K, 54),
            gguf_reader.GGML_TYPE_TQ2_0: (QK_K, 66),
            gguf_reader.GGML_TYPE_MXFP4: (32, 17),
        }

    def test_block_table_matches_verified_values(self):
        actual = {tid: (b, n) for tid, (name, b, n) in gguf_reader.GGML_TYPE_BLOCK.items()}
        self.assertEqual(actual, self._expected_block_table())

    def test_every_block_type_resolves(self):
        for ggml_type in gguf_reader.GGML_TYPE_BLOCK:
            name, block, bpb = gguf_reader.GGML_TYPE_BLOCK[ggml_type]
            self.assertEqual(gguf_reader.ggml_block_spec(ggml_type), (block, bpb))
            self.assertEqual(gguf_reader.ggml_type_name(ggml_type), name)


class GGUFRefusalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = self.tmp.name

    def _write(self, data, name):
        path = os.path.join(self.dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _expect_error(self, data, name, message=None):
        path = self._write(data, name)
        if message is None:
            with self.assertRaises(GGUFError):
                GGUFReader(path)
        else:
            with self.assertRaisesRegex(GGUFError, message):
                GGUFReader(path)

    def test_bad_magic(self):
        self._expect_error(_raw_header(magic=b"NOPE"), "bad_magic.gguf")

    def test_unsupported_version(self):
        self._expect_error(_raw_header(version=99), "bad_version.gguf")

    def test_truncated_real_fixture(self):
        write_gguf_fixture(os.path.join(self.dir, "full.gguf"))
        data = Path(os.path.join(self.dir, "full.gguf")).read_bytes()
        self._expect_error(data[: len(data) // 2], "truncated.gguf")

    def test_duplicate_tensor_names(self):
        info = _tensor_info()
        self._expect_error(
            _raw_header(tensor_count=2) + info + info,
            "duplicate.gguf")

    def test_unaligned_tensor_offset(self):
        info = _tensor_info(offset=33)
        self._expect_error(
            _raw_header(tensor_count=1) + info,
            "unaligned.gguf")

    def test_tensor_range_beyond_eof(self):
        info = _tensor_info(dims=(16,), ggml_type=0, offset=32)
        self._expect_error(
            _raw_header(tensor_count=1) + info,
            "oob.gguf")

    def test_unsupported_ggml_type(self):
        info = _tensor_info(dims=(16,), ggml_type=100)
        self._expect_error(
            _raw_header(tensor_count=1) + info,
            "unsupported_type.gguf")

    def test_rank_above_cap(self):
        info = _tensor_info(n_dims=9, dims=(1,) * 9)
        self._expect_error(
            _raw_header(tensor_count=1) + info,
            "high_rank.gguf")

    def test_unaligned_block_tensor_dims(self):
        info = _tensor_info(dims=(15,), ggml_type=8)
        self._expect_error(
            _raw_header(tensor_count=1) + info,
            "bad_block_dims.gguf", "requires dims\\[0\\] \\(15\\) to be a multiple of 32")

    def test_metadata_unknown_value_type(self):
        kv = _raw_kv_value_type_only("test.key", 200)
        self._expect_error(_raw_header(kv_count=1) + kv, "unknown_meta_type.gguf",
                           "unknown metadata value type 200")

    def test_metadata_array_unknown_element_type(self):
        kv = _raw_kv_array("test.array", 200)
        self._expect_error(_raw_header(kv_count=1) + kv, "unknown_elem_type.gguf",
                           "unknown metadata array element type 200")

    def test_metadata_bool_invalid_value(self):
        kv = _raw_kv("test.bool", METADATA_BOOL, True)[:-1] + b"\x02"
        self._expect_error(_raw_header(kv_count=1) + kv, "bad_bool.gguf",
                           "invalid bool metadata value")

    def test_metadata_string_not_utf8(self):
        kv = _raw_kv_string_value("test.key", struct.pack("<Q", 2) + b"\xff\xfe")
        self._expect_error(_raw_header(kv_count=1) + kv, "bad_utf8.gguf",
                           "not valid UTF-8")

    def test_metadata_string_oversized_length(self):
        kv = _raw_kv_string_value("test.key", struct.pack("<Q", 1 << 32))
        self._expect_error(_raw_header(kv_count=1) + kv, "long_string.gguf",
                           "unreasonable.*length")

    def test_alignment_not_multiple_of_8(self):
        kv = _raw_kv("general.alignment", METADATA_UINT32, 12)
        self._expect_error(_raw_header(kv_count=1) + kv, "alignment_not_mult8.gguf",
                           "must be a positive multiple of 8")

    def test_alignment_zero(self):
        kv = _raw_kv("general.alignment", METADATA_UINT32, 0)
        self._expect_error(_raw_header(kv_count=1) + kv, "alignment_zero.gguf",
                           "must be a positive multiple of 8")

    def test_alignment_wrong_type(self):
        kv = _raw_kv("general.alignment", METADATA_FLOAT32, 32.0)
        self._expect_error(_raw_header(kv_count=1) + kv, "alignment_float.gguf",
                           "general.alignment .* must be a positive multiple of 8")

    def test_metadata_count_above_cap(self):
        self._expect_error(
            _raw_header(kv_count=gguf_reader.MAX_METADATA_ITEMS + 1),
            "many_meta.gguf", "metadata count .* exceeds cap")

    def test_tensor_count_above_cap(self):
        self._expect_error(
            _raw_header(tensor_count=gguf_reader.MAX_TENSORS + 1),
            "many_tensors.gguf", "tensor count .* exceeds cap")


if __name__ == "__main__":
    unittest.main()
