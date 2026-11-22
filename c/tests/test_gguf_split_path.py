"""Split discovery is based on the filename, independent of parent names."""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def write_part(path, name):
    tensor = struct.pack('<Q', len(name)) + name.encode()
    tensor += struct.pack('<IQIQ', 1, 1, 0, 0)  # dims, ne[0], F32, offset
    header = b'GGUF' + struct.pack('<IQQ', 3, 1, 0) + tensor
    path.write_bytes(header + bytes(-len(header) % 32) + struct.pack('<f', 1))


class GgufSplitPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.build.name) / 'reader'
        src = Path(cls.build.name) / 'reader.c'
        src.write_text('#include "gguf.h"\nint main(int argc,char **argv){'
                       'gguf_ctx g; if(argc!=2)return 2;'
                       'if(gguf_open(&g,argv[1]))return 3;'
                       'int ok=g.n_files==2 && gguf_find(&g,"a") && gguf_find(&g,"b");'
                       'gguf_close(&g); return ok?0:4;}\n')
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE',
                        '-Wall', '-Wextra', '-Wno-unused-function', '-I', str(ROOT),
                        str(src), '-lm', '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_parent_names(self):
        with tempfile.TemporaryDirectory() as td:
            for parent in ('plain', 'models-of-interest', 'nested-of-00002.gguf'):
                with self.subTest(parent=parent):
                    folder = Path(td) / parent
                    folder.mkdir()
                    first = folder / 'model-00001-of-00002.gguf'
                    write_part(first, 'a')
                    write_part(folder / 'model-00002-of-00002.gguf', 'b')
                    self.assertEqual(subprocess.run([str(self.exe), str(first)]).returncode, 0)


if __name__ == '__main__':
    unittest.main()
