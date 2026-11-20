"""A GGUF's 64-bit values past the reader's signed types: a UINT64 above INT64_MAX (#2851) and,
with the header read alone (GGUF_CHECK), a tensor that ends past the largest int64 offset (#2852)."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def s(b):
    return struct.pack('<Q', len(b)) + b


def gguf(kvs=b'', n_kv=0, tensors=b'', n_tensors=0):
    b = b'GGUF' + struct.pack('<IQQ', 3, n_tensors, n_kv) + kvs + tensors
    return b + bytes(-len(b) % 32)


class GgufWideValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        src = Path(cls.build.name) / 'reader.c'
        # reader FILE [h]: open (h: the header alone, as GGUF_CHECK), then print each key's int and float
        src.write_text(
            '#include "gguf.h"\n#include <inttypes.h>\n'
            'int main(int argc,char **argv){gguf_ctx g;if(argc<2)return 2;gguf_header_only=argc>2;'
            'if(gguf_open(&g,argv[1])){fprintf(stderr,"%s\\n",g.err);return 1;}'
            'for(int i=0;i<g.n_kv;i++)printf("%s %" PRId64 " %.1f\\n",g.kv[i].key,'
            'gguf_get_int(&g,g.kv[i].key,-7),gguf_get_float(&g,g.kv[i].key,-7.0));'
            'for(int i=0;i<g.n_tensors;i++)printf("tensor %s %" PRIu64 "\\n",g.t[i].name,g.t[i].off);'
            'gguf_close(&g);return 0;}\n')
        cls.exe = Path(cls.build.name) / 'reader'
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE', '-D_FILE_OFFSET_BITS=64',
                        '-Wall', '-Wextra', '-Wno-unused-function', '-I', str(ROOT), str(src), '-lm',
                        '-o', str(cls.exe)],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def read(self, data, header_only=False):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.gguf'
            path.write_bytes(data)
            args = [str(self.exe), str(path)] + (['h'] if header_only else [])
            return subprocess.run(args, capture_output=True, text=True)

    def test_unsigned_64_bit_values(self):
        cases = ((1 << 63, '-7', '9223372036854775808.0'),       # past INT64_MAX: the int getter's default
                 ((1 << 64) - 1, '-7', '18446744073709551616.0'),
                 ((1 << 63) - 1, '9223372036854775807', '9223372036854775808.0'),
                 (300, '300', '300.0'))
        kvs = b''.join(s(b'k%d' % i) + struct.pack('<IQ', 10, v) for i, (v, _, _) in enumerate(cases))
        kvs += s(b'neg') + struct.pack('<Iq', 11, -5)             # INT64
        r = self.read(gguf(kvs, len(cases) + 1))
        self.assertEqual(r.returncode, 0, r.stderr)
        got = dict((ln.split()[0], ln.split()[1:]) for ln in r.stdout.splitlines())
        for i, (v, as_int, as_float) in enumerate(cases):
            with self.subTest(value=v):
                self.assertEqual(got['k%d' % i], [as_int, as_float])
        self.assertEqual(got['neg'], ['-5', '-5.0'])

    def tensor(self, rel, n=32):
        # one F32 expert tensor of shape (n, 1, 2): two slices, the second at rel + 4n
        return s(b'blk.0.ffn_gate_exps.weight') + struct.pack('<IQQQIQ', 3, n, 1, 2, 0, rel)

    def test_header_alone_keeps_offsets_in_int64(self):
        # the data start at 128 here and the tensor holds 256 bytes
        for rel, ok in ((1 << 40, True),           # no bytes in the file: still read, the header alone
                        ((1 << 63) - 256, False),  # the issue's case: its second slice starts past INT64_MAX
                        ((1 << 63) - 416, True)):  # the last 32-byte place where it ends within INT64_MAX
            with self.subTest(rel=rel):
                data = gguf(tensors=self.tensor(rel), n_tensors=1)
                r = self.read(data, header_only=True)
                self.assertEqual(r.returncode == 0, ok, r.stdout + r.stderr)
                if not ok:
                    self.assertIn('ends past the largest file offset', r.stderr)
        r = self.read(gguf(tensors=self.tensor(1 << 40), n_tensors=1))
        self.assertEqual(r.returncode, 1)
        self.assertIn('lies outside the file', r.stderr)


if __name__ == '__main__':
    unittest.main()
