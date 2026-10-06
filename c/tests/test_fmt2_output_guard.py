import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('fmt2_guard_test', TOOLS / 'convert_fmt4_to_fmt2.py')
converter = importlib.util.module_from_spec(spec)
with mock.patch.dict(sys.modules, {'numpy': types.ModuleType('numpy')}):
    spec.loader.exec_module(converter)

class Fmt2OutputGuardTest(unittest.TestCase):
    def test_existing_directory_alias_is_rejected_before_reading_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, 'source')
            source.mkdir()
            alias = Path(directory, 'alias')
            try:
                alias.symlink_to(source, target_is_directory=True)
            except NotImplementedError as error:
                self.skipTest(str(error))
            except OSError as error:
                if os.name == 'nt' and getattr(error, 'winerror', None) == 1314:
                    self.skipTest(str(error))
                raise
            config = source / 'config.json'
            config.write_text('{"hidden_size": 128, "vocab_size": 2}')
            # Valid grouped-int4 tensor: 128 packed bytes + four f32 scales.
            header = json.dumps({'model.embed_tokens.weight': {'dtype': 'U8', 'shape': [128], 'data_offsets': [0, 128]},
                                 'model.embed_tokens.weight.qs': {'dtype': 'F32', 'shape': [4], 'data_offsets': [128, 144]}}).encode()
            header += b' ' * (-len(header) % 8)
            shard = source / 'model-00001.safetensors'
            original = struct.pack('<Q', len(header)) + header + bytes([0x99]) * 128 + struct.pack('<4f', 1, 1, 1, 1)
            shard.write_bytes(original)
            for output in (source, alias):
                with self.subTest(output=output), \
                        mock.patch.object(sys, 'argv', ['converter', '--indir', str(source), '--outdir', str(output), '--workers', '1']), \
                        mock.patch.object(converter, 'plan_shard', side_effect=AssertionError('in-place conversion reached shard planning')) as plan:
                    with self.assertRaisesRegex(SystemExit, '--indir and --outdir must differ'):
                        converter.main()
                    plan.assert_not_called()
            self.assertEqual(shard.read_bytes(), original)
            self.assertEqual(config.read_text(), '{"hidden_size": 128, "vocab_size": 2}')
            self.assertFalse(Path(str(shard) + '.tmp').exists())
