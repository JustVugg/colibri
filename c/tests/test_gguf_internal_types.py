"""A GGUF may only declare ggml's own tensor types, never the engine's interleaved forms.

GQ_*_R16 (ggml_quants.h, ids GQ_Q2_0_R16 .. GQ_R16_END - 1) are layouts the engine builds at load, and only for whole
groups of 16 rows. A file that declares one is read as if it already were interleaved: a matrix
whose rows are not a multiple of 16 is then read past its bytes (gq_r16_dequant_row reads a
whole group). gguf.h must refuse those ids like any other type it cannot hold.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

FILE_TYPES = {"F32": 0, "Q4_0": 2, "Q8_0": 8, "Q4_K": 12, "IQ4_XS": 23, "Q2_0": 42}


class GgufInternalTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.build.name) / 'reader'
        src = Path(cls.build.name) / 'reader.c'
        src.write_text('#include "gguf.h"\nint main(int argc,char **argv){'
                       'if(argc==1){printf("%d %d\\n",GQ_Q2_0_R16,GQ_R16_END);return 0;}'
                       'gguf_ctx g;if(argc!=2)return 2;gguf_header_only=1;'
                       'if(gguf_open(&g,argv[1])){fprintf(stderr,"%s\\n",g.err);return 1;}'
                       'gguf_close(&g);return 0;}\n')
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE',
                        '-Wall', '-Wextra', '-Wno-unused-function', '-I', str(ROOT),
                        str(src), '-lm', '-o', str(cls.exe)], check=True)
        # the interleaved ids as ggml_quants.h numbers them today (a form added later is covered too)
        lo, end = map(int, subprocess.run([str(cls.exe)], capture_output=True, text=True, check=True).stdout.split())
        cls.r16_ids, cls.past = range(lo, end), end

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def _open(self, dtype, dims=(256, 1)):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.gguf'
            tensor = (struct.pack('<Q', 1) + b'w' + struct.pack('<I', len(dims))
                      + struct.pack('<%dQ' % len(dims), *dims) + struct.pack('<IQ', dtype, 0))
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + tensor)
            return subprocess.run([str(self.exe), str(path)], capture_output=True)

    def test_interleaved_ids_are_refused(self):
        for dtype in self.r16_ids:
            for dims in ((256, 1), (256, 16)):
                with self.subTest(dtype=dtype, dims=dims):
                    result = self._open(dtype, dims)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertIn(b'cannot hold rows', result.stderr)

    def test_ids_past_the_table_are_refused(self):
        for dtype in (43, 63, self.past, 0xFFFFFFFF):
            with self.subTest(dtype=dtype):
                result = self._open(dtype)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn(b'cannot hold rows', result.stderr)

    def test_file_types_still_open(self):
        for name, dtype in FILE_TYPES.items():
            with self.subTest(type=name):
                result = self._open(dtype)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
