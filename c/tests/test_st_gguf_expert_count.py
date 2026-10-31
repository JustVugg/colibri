"""Header inspection must refuse fused expert dimensions before narrowing."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def ggstr(value):
    raw = value.encode()
    return struct.pack('<Q', len(raw)) + raw


class FusedExpertCount(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory()
        cls.exe = Path(cls.build.name) / 'reader'
        src = Path(cls.build.name) / 'reader.c'
        src.write_text('#define ST_GGUF\n#include "st.h"\n'
                       'int main(int argc,char **argv){if(argc!=2)return 2;'
                       'shards S={0};return st_init_gguf(&S,argv[1])?0:3;}\n')
        subprocess.run([os.environ.get('CC', 'cc'), '-std=c11', '-D_GNU_SOURCE',
                        '-Wall', '-Wextra', '-Wno-unused-function', '-I', str(ROOT),
                        str(src), '-lm', '-o', str(cls.exe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def test_fused_expert_dimension(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            (folder / 'gguf.json').write_text(json.dumps({'gguf': 'model.gguf'}))
            for experts, expected in ((1 << 32, 1), ((1 << 31) + 1, 1), (2, 0)):
                with self.subTest(experts=experts):
                    kv = ggstr('general.architecture') + struct.pack('<I', 8) + ggstr('glm-dsa')
                    tensor = ggstr('blk.0.ffn_gate_exps.weight') + struct.pack('<IQQQIQ', 3, 32, 1, experts, 0, 0)
                    (folder / 'model.gguf').write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 1, 1) + kv + tensor)
                    env = dict(os.environ, GGUF_CHECK='1')
                    result = subprocess.run([str(self.exe), str(folder)], env=env, capture_output=True)
                    self.assertEqual(result.returncode, expected, result.stderr)
                    if expected:
                        self.assertIn(b'expert count', result.stderr)


if __name__ == '__main__':
    unittest.main()
