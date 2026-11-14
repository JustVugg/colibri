"""SFT output files must describe this run rather than an earlier dataset."""
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/prepare_sft.py"


@unittest.skipUnless(importlib.util.find_spec("numpy"), "prepare_sft requires numpy")
class PrepareSftOutputsTest(unittest.TestCase):
    def prepare(self, root, samples):
        result = subprocess.run([sys.executable, str(SCRIPT), "--output", str(root),
                                 "--synthetic", str(samples), "32", "--seed", "7"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads((root / "metadata.json").read_text())

    def test_rerun_removes_empty_split_files_and_keeps_unrelated_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = self.prepare(root, 4)
            self.assertEqual(original["samples"], {"valid": 1, "train": 3})
            for suffix in ("bin", "msk", "idx"):
                self.assertTrue((root / f"valid.{suffix}").is_file())
            (root / "notes.txt").write_text("keep this")
            current = self.prepare(root, 1)
            self.assertEqual(current["samples"], {"valid": 0, "train": 1})
            for suffix in ("bin", "msk", "idx"):
                self.assertFalse((root / f"valid.{suffix}").exists(),
                                 "trainer reads split files, not metadata.json")
            self.assertEqual((root / "notes.txt").read_text(), "keep this")
            # Independent standard-library reader checks current training bytes.
            index = (root / "train.idx").read_bytes()
            magic, count, start, end = struct.unpack("<4q", index)
            self.assertEqual((magic, count, start), (0x434F4C4953465431, 1, 0))
            self.assertEqual(len((root / "train.bin").read_bytes()), end * 4)
            self.assertEqual(len((root / "train.msk").read_bytes()), end)
            self.assertEqual(current["tokens"]["train"], end)

    def test_first_single_sample_run_needs_no_existing_empty_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.prepare(root, 1)
            self.assertEqual({path.name for path in root.iterdir()},
                             {"train.bin", "train.msk", "train.idx", "metadata.json"})


if __name__ == "__main__":
    unittest.main()
