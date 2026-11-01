"""Explicit Inkling output paths must never replace or truncate source shards."""
import importlib.util
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/convert_inkling_dense_int4.py"


@unittest.skipUnless(importlib.util.find_spec("numpy"), "converter requires numpy")
class InklingDenseSourceAliasTest(unittest.TestCase):
    def test_output_and_part_aliases_are_refused_without_changing_source(self):
        for mode in ("direct", "symlink", "part-symlink", "part-hardlink"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "model.safetensors"
                header = json.dumps({"model.norm.weight": {"dtype": "F32", "shape": [1],
                                                           "data_offsets": [0, 4]}}).encode()
                original = struct.pack("<Q", len(header)) + header + struct.pack("<f", 2.5)
                source.write_bytes(original)
                output = source if mode == "direct" else root / "dense-output.safetensors"
                try:
                    if mode == "symlink": output.symlink_to(source)
                    elif mode == "part-symlink": Path(str(output) + ".part").symlink_to(source)
                    elif mode == "part-hardlink": os.link(source, str(output) + ".part")
                except (OSError, NotImplementedError) as error:
                    self.skipTest(f"filesystem link unavailable: {error}")
                result = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(root),
                                         "--out", str(output)], capture_output=True, text=True, timeout=3)
                self.assertEqual(source.read_bytes(), original,
                                 "source checkpoint bytes must remain intact")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("source shard", result.stderr)

    def test_separate_output_converts_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = json.dumps({"model.norm.weight": {"dtype": "F32", "shape": [1],
                                                       "data_offsets": [0, 4]}}).encode()
            source = root / "model.safetensors"
            original = struct.pack("<Q", len(header)) + header + struct.pack("<f", 2.5)
            source.write_bytes(original)
            output = root / "dense-output.safetensors"
            result = subprocess.run([sys.executable, str(SCRIPT), "--dir", str(root),
                                     "--out", str(output)], capture_output=True, text=True, timeout=3)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(source.read_bytes(), original)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
