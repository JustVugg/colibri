"""A truncated raw checkpoint tensor must fail rather than spin at EOF."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/convert_inkling_dense_int4.py"


@unittest.skipUnless(importlib.util.find_spec("numpy"), "converter requires numpy")
class InklingDenseShortReadTest(unittest.TestCase):
    def run_converter(self, root, payload):
        header = json.dumps({"model.norm.weight": {"dtype": "F32", "shape": [1],
                                                   "data_offsets": [0, 4]}}).encode()
        source = root / "model.safetensors"
        original = struct.pack("<Q", len(header)) + header + payload
        source.write_bytes(original)
        output = root / "dense-int4g64.safetensors"
        result = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(root),
                                 "--out", str(output)], capture_output=True, text=True, timeout=3)
        self.assertEqual(source.read_bytes(), original)
        return result, output

    def test_empty_or_short_raw_tensor_exits_without_publication(self):
        for payload in (b"", b"\x00\x00"):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                result, output = self.run_converter(Path(directory), payload)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("short read", result.stderr)
                self.assertIn("model.norm.weight", result.stderr)
                self.assertFalse(output.exists())

    def test_complete_raw_tensor_is_copied_byte_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            result, output = self.run_converter(Path(directory), struct.pack("<f", 2.5))
            self.assertEqual(result.returncode, 0, result.stderr)
            data = output.read_bytes()
            header_length = struct.unpack("<Q", data[:8])[0]
            header = json.loads(data[8:8 + header_length])
            self.assertEqual(header["model.norm.weight"]["data_offsets"], [0, 4])
            self.assertEqual(data[8 + header_length:], struct.pack("<f", 2.5))


if __name__ == "__main__":
    unittest.main()
