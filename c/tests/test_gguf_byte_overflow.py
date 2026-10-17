"""Untrusted GGUF shapes must fit the reader's signed byte-count type."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GgufByteOverflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.build.name) / 'reader'
        src = Path(cls.build.name) / 'reader.c'
        src.write_text('#include "gguf.h"\nint main(int argc,char **argv){'
                       'gguf_ctx g;if(argc!=2)return 2;gguf_header_only=1;'
                       'if(gguf_open(&g,argv[1])){fprintf(stderr,"%s\\n",g.err);return 1;}'
                       'gguf_close(&g);return 0;}\n')
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE',
                        '-Wall', '-Wextra', '-Wno-unused-function', '-I', str(ROOT),
                        str(src), '-lm', '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_byte_count(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.gguf'
            # Both rejected shapes have valid, nonoverflowing element counts.
            for dims, dtype, expected in (((1 << 40, 1 << 22), 0, 1),
                                          ((1 << 40, 1 << 21), 28, 1),
                                          ((16, 8), 0, 0),
                                          ((32, 8), 8, 0)):
                with self.subTest(dims=dims, dtype=dtype):
                    tensor = struct.pack('<Q', 1) + b'x' + struct.pack('<IQQIQ', 2, *dims, dtype, 0)
                    path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + tensor)
                    result = subprocess.run([str(self.exe), str(path)], capture_output=True)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected:
                        self.assertIn(b'byte size', result.stderr)


if __name__ == '__main__':
    unittest.main()
