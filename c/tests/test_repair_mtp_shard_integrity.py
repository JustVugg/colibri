"""Local shard integrity before the MTP repair rewrites untouched tensor bytes."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

try:
    import numpy
except ImportError:
    numpy = None


TOOL = Path(__file__).resolve().parent.parent / "tools" / "repair_mtp_int8.py"


@unittest.skipIf(numpy is None, "MTP repair requires the optional NumPy runtime")
class RepairShardIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("mtp_repair_integrity", TOOL)
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "local.safetensors"

    def shard(self, header, payload):
        raw = json.dumps(header).encode()
        raw += b" " * ((8 - len(raw) % 8) % 8)
        self.path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return self.path.read_bytes()

    def tensor(self, offsets, dtype="F32", shape=None):
        return {"weight": {"dtype": dtype, "shape": [2] if shape is None else shape,
                           "data_offsets": offsets}}

    def test_truncated_tensor_is_refused_without_rewriting_input(self):
        original = self.shard(self.tensor([0, 8]), b"ab")
        with self.assertRaisesRegex(ValueError, "payload"):
            self.tool.read_shard(self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_negative_reversed_or_past_end_offsets_are_refused(self):
        for offsets in ([-1, 1], [6, 2], [9, 10], [False, True]):
            with self.subTest(offsets=offsets):
                original = self.shard(self.tensor(offsets), b"12345678")
                with self.assertRaises(ValueError):
                    self.tool.read_shard(self.path)
                self.assertEqual(self.path.read_bytes(), original)

    def test_a_valid_json_prefix_cannot_hide_a_truncated_header(self):
        # The original reader accepted {} after declaring an eight-byte header,
        # because json.loads accepts a shorter but complete JSON document.
        self.path.write_bytes(struct.pack("<Q", 8) + b"{}")
        original = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "header"):
            self.tool.read_shard(self.path)
        self.assertEqual(self.path.read_bytes(), original)

    def test_already_rejected_headers_remain_rejected(self):
        cases = (b"1234", struct.pack("<Q", 8) + b"[]      ")
        for raw in cases:
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                # These inputs already failed before the change; preserve that
                # refusal rather than count an improved exception as a new bug.
                with self.assertRaises((ValueError, struct.error, AttributeError)):
                    self.tool.read_shard(self.path)
                self.assertEqual(self.path.read_bytes(), raw)

    def test_healthy_packed_and_float_tensors_roundtrip_unchanged(self):
        floats = struct.pack("<ff", 1.25, -3.5)
        packed = b"\x88\x97"
        metadata = {"format": "pt", "fixture": "tiny-local-shard"}
        header = {"float.weight": {"dtype": "F32", "shape": [2], "data_offsets": [2, 10]},
                  "packed.weight": {"dtype": "U8", "shape": [2, 2], "data_offsets": [0, 2]},
                  "__metadata__": metadata}
        self.shard(header, packed + floats)
        tensors, meta = self.tool.read_shard(self.path)
        self.assertEqual(tensors["float.weight"], ("F32", [2], floats))
        self.assertEqual(tensors["packed.weight"], ("U8", [2, 2], packed))
        self.assertEqual(meta, metadata)
        output = self.path.with_name("roundtrip.safetensors")
        self.tool.write_shard(output, tensors, meta)
        self.assertEqual(self.tool.read_shard(output), (tensors, metadata))

    def test_empty_tensor_at_payload_end_remains_valid(self):
        header = {"full": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]},
                  "empty": {"dtype": "F32", "shape": [0], "data_offsets": [2, 2]}}
        self.shard(header, b"ab")
        tensors, metadata = self.tool.read_shard(self.path)
        self.assertEqual(tensors["full"], ("U8", [2], b"ab"))
        self.assertEqual(tensors["empty"], ("F32", [0], b""))
        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()
