"""tools/convert_inkling_dense_int4.py must finish on Windows and on a non-UTF-8 console.

docs/inkling.md sends a RAM-constrained user to this converter. Two things
stopped it from ever writing the container outside a UTF-8 POSIX shell:

  * it measured free space with os.statvfs, which does not exist on Windows,
    so every Windows run died with AttributeError right after the plan, before
    a byte was written;
  * its progress lines print "…" and the success line "✅". Where stdout cannot
    encode them (cp949/cp932 cannot encode "✅", the C locale neither), the
    print raised UnicodeEncodeError: at the first line in the C locale, and on
    a CJK Windows console after the container was written, so the run exited 1
    and never showed the quantization-error report.

The test converts a tiny shard in a child whose stdout is not UTF-8
(PYTHONUTF8=0 and the C locale, or the Windows ANSI code page).
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

TOOL = Path(__file__).resolve().parent.parent / "tools" / "convert_inkling_dense_int4.py"

CHILD_ENV = {**os.environ, "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0",
             "LC_ALL": "C", "LANG": "C"}
CHILD_ENV.pop("PYTHONIOENCODING", None)


def write_shard(path, tensors):
    header, blobs, offset = {}, [], 0
    for name, array in tensors.items():
        data = array.astype(np.float32).tobytes()
        header[name] = {"dtype": "F32", "shape": list(array.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode()
    with open(path, "wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for data in blobs:
            stream.write(data)


def read_header(path):
    with open(path, "rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed; the converter imports it")
class ConvertInklingDensePortableTest(unittest.TestCase):
    def test_converts_and_reports_in_a_non_utf8_locale(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as model:
            write_shard(Path(model, "model-00001.safetensors"), {
                "model.embed_tokens.weight": rng.standard_normal((8, 64)),
                "model.layers.0.input_layernorm.weight": np.ones(64),
            })
            result = subprocess.run([sys.executable, str(TOOL), "--dir", model],
                                    env=CHILD_ENV, capture_output=True, timeout=120)
            output = (result.stdout + result.stderr).decode("utf-8", "replace")
            self.assertEqual(result.returncode, 0, output[-3000:])
            header = read_header(Path(model, "dense-int4g64.safetensors"))
            self.assertEqual(header["__metadata__"]["model.embed_tokens.weight"], "fmt=1;orig=8x64")
            self.assertEqual(header["__metadata__"]["model.layers.0.input_layernorm.weight"], "fmt=raw")
            self.assertIn("embed/head", output)


if __name__ == "__main__":
    unittest.main()
