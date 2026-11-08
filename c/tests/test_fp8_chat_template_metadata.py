"""Exercise the local converter's metadata path without model tensors or downloads."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy  # noqa: F401
    import safetensors.numpy  # noqa: F401
except ImportError as error:
    raise unittest.SkipTest(f"converter dependencies unavailable: {error}")

CONVERTER = Path(__file__).resolve().parents[1] / "tools/convert_fp8_to_int4.py"


class Fp8ChatTemplateMetadataTest(unittest.TestCase):
    def run_converter(self, source, output, *flags):
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
        result = subprocess.run([sys.executable, str(CONVERTER), "--indir", str(source),
                                 "--outdir", str(output), *flags], env=env,
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_local_metadata_copy_preserves_template_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"glm_moe_dsa"}')
            template = '[gMASK]<sop>\n{# Unicode: λ #}\n'.encode()
            (source / "chat_template.jinja").write_bytes(template)
            self.run_converter(source, output)
            self.assertEqual((output / "chat_template.jinja").read_bytes(), template)
            self.assertEqual((source / "chat_template.jinja").read_bytes(), template)

    def test_older_checkpoint_without_separate_template_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "output"
            source.mkdir()
            (source / "config.json").write_text('{"model_type":"glm_moe_dsa"}')
            self.run_converter(source, output)
            self.assertTrue((output / "config.json").is_file())
            self.assertFalse((output / "chat_template.jinja").exists())


if __name__ == "__main__":
    unittest.main()
