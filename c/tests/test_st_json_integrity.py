"""Corrupt container JSON must not become a partial tensor index."""
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import tempfile
import unittest

C_DIR = Path(__file__).resolve().parents[1]
TENSOR = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}


class SafetensorsJsonIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cc = shlex.split(os.environ.get("CC", "cc"))
        if not cc or not shutil.which(cc[0]):
            raise unittest.SkipTest("a C compiler is required")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root = Path(cls.tmp.name)
        source = cls.root / "load.c"
        source.write_text('#include "st.h"\nint main(int n,char **v){'
                          'if(n!=2)return 2; shards s; st_init(&s,v[1]);'
                          'float x=0; st_read_f32_cap(&s,"tensor",&x,1,0);'
                          'printf("%g\\n",x); return 0;}\n')
        cls.binary = cls.root / ("load.exe" if os.name == "nt" else "load")
        # -D_FILE_OFFSET_BITS=64 as the Makefile builds st.h (compat.h requires it on Windows)
        subprocess.run(cc + ["-D_FILE_OFFSET_BITS=64", "-O1", "-Wall", "-Wextra", "-Wno-unused-function",
                             "-Wno-unused-parameter", "-Wno-misleading-indentation",
                             "-I", str(C_DIR), str(source), "-o", str(cls.binary), "-lm"],
                       check=True, capture_output=True, text=True)

    def load(self, header, index=None):
        with tempfile.TemporaryDirectory(dir=self.root) as directory:
            directory = Path(directory)
            header = header.encode() if isinstance(header, str) else header
            (directory / "a.safetensors").write_bytes(struct.pack("<Q", len(header))
                                                       + header + struct.pack("<f", 2.5))
            if index is not None:
                valid = json.dumps({"tensor": TENSOR}).encode()
                (directory / "b.safetensors").write_bytes(struct.pack("<Q", len(valid))
                                                           + valid + struct.pack("<f", 7.5))
                (directory / "model.safetensors.index.json").write_text(index)
            return subprocess.run([str(self.binary), str(directory)], capture_output=True,
                                  text=True, timeout=10)

    def test_corrupt_header(self):
        valid = json.dumps({"tensor": TENSOR})
        for header in (valid[:-1], valid + " trailing", valid[:-1] + ",}",
                       valid.encode() + b"\0garbage"):
            with self.subTest(header=header):
                result = self.load(header)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("safetensors header", result.stderr)

    def test_header_requires_utf8(self):
        # JSON strings in a safetensors header are UTF-8, not arbitrary bytes.
        # Exercise both tensor names and ignored metadata values: ignoring a
        # field must not turn a corrupt header into a successful model load.
        invalid = (b"\x80", b"\xc0\xaf", b"\xc2", b"\xe0\x80\xaf",
                   b"\xed\xa0\x80", b"\xe2\x82", b"\xf0\x80\x80\xaf",
                   b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80", b"\xff")
        for raw in invalid:
            for position in ("name", "metadata"):
                with self.subTest(raw=raw, position=position):
                    if position == "name":
                        header = (b'{"' + raw + b'":' + json.dumps(
                            {"dtype": "F32", "shape": [0], "data_offsets": [0, 0]}).encode()
                                  + b',"tensor":' + json.dumps(TENSOR).encode() + b'}')
                    else:
                        header = (b'{"__metadata__":{"note":"' + raw + b'"},"tensor":'
                                  + json.dumps(TENSOR).encode() + b'}')
                    with self.assertRaises(UnicodeDecodeError):
                        header.decode("utf-8")
                    result = self.load(header)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn("safetensors header", result.stderr)

    def test_utf8_metadata_boundaries(self):
        # Independent encoder supplies shortest-form Unicode boundary scalars.
        text = "".join(chr(cp) for cp in (0x7f, 0x80, 0x7ff, 0x800, 0xd7ff,
                                        0xe000, 0xffff, 0x10000, 0x10ffff))
        header = json.dumps({"__metadata__": {"note": text}, "tensor": TENSOR},
                            ensure_ascii=False).encode("utf-8")
        self.assertEqual(json.loads(header.decode("utf-8"))["__metadata__"]["note"], text)
        result = self.load(header)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2.5")

    def test_corrupt_stamp(self):
        for stamp in ('{"tensor":"int8"', '{"tensor":"int8"} trailing'):
            with self.subTest(stamp=stamp):
                result = self.load(json.dumps({"__metadata__": {"colibri.fmt": stamp},
                                               "tensor": TENSOR}))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("malformed stamp", result.stderr)

    def test_corrupt_overlay_index(self):
        result = self.load(json.dumps({"tensor": TENSOR}),
                           '{"weight_map":{"tensor":"b.safetensors"}')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("duplicate tensor name", result.stderr)

    def test_valid_header_and_padding(self):
        result = self.load(json.dumps({"tensor": TENSOR}) + "   \n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2.5")

    def test_valid_stamp(self):
        result = self.load(json.dumps({"__metadata__": {"colibri.fmt": '{"tensor":"int8"}'},
                                       "tensor": TENSOR}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2.5")

    def test_valid_overlay_index(self):
        result = self.load(json.dumps({"tensor": TENSOR}),
                           '{"weight_map":{"tensor":"b.safetensors"}}')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "7.5")


if __name__ == "__main__":
    unittest.main()
