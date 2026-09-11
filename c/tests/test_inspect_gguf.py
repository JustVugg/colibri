import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
SPEC = importlib.util.spec_from_file_location("inspect_gguf", TOOLS / "inspect_gguf.py")
inspect_gguf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inspect_gguf)


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def metadata_string(key, value):
    return string(key) + struct.pack("<I", 8) + string(value)


def metadata_u32(key, value):
    return string(key) + struct.pack("<II", 4, value)


def metadata_array_u32(key, values):
    return (string(key) + struct.pack("<IIQ", 9, 4, len(values)) +
            struct.pack("<" + "I" * len(values), *values))


def tensor(name, dims=(4,), tensor_type=0, offset=0):
    return (string(name) + struct.pack("<I", len(dims)) +
            struct.pack("<" + "Q" * len(dims), *dims) +
            struct.pack("<IQ", tensor_type, offset))


def fixture(path):
    metadata = [
        metadata_string("general.architecture", "nemotron_h_moe"),
        metadata_u32("nemotron_h_moe.block_count", 3),
        metadata_array_u32("nemotron_h_moe.attention.head_count_kv", [0, 2, 0]),
        metadata_array_u32("nemotron_h_moe.feed_forward_length", [0, 0, 16]),
        # A large irrelevant scalar array exercises the bounded skip path.
        metadata_array_u32("test.skipped", list(range(5000))),
    ]
    tensors = [
        tensor("blk.0.ssm_in.weight", (4, 8), 30),
        tensor("blk.1.attn_q.weight", (4, 4), 30),
        tensor("blk.2.ffn_up_exps.weight", (4, 8, 2), 40),
    ]
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata)) +
                     b"".join(metadata) + b"".join(tensors))


class InspectGGUFTests(unittest.TestCase):
    def test_inventory_and_hybrid_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            fixture(path)
            result = inspect_gguf.inspect(path)
        self.assertEqual(result["metadata"]["general.architecture"], "nemotron_h_moe")
        self.assertEqual(result["tensor_type_counts"], {"30": 2, "40": 1})
        self.assertEqual([row["mixer"] for row in result["schedule"]],
                         ["ssm", "attention", "unknown"])
        self.assertEqual([row["moe"] for row in result["schedule"]],
                         [False, False, True])

    def test_truncated_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 0))
            with self.assertRaises(inspect_gguf.GGUFError):
                inspect_gguf.inspect(path)

    def test_wrong_magic_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.gguf"
            path.write_bytes(b"nope")
            with self.assertRaises(inspect_gguf.GGUFError):
                inspect_gguf.inspect(path)


if __name__ == "__main__":
    unittest.main()
