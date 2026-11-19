"""Regression test for replacing the source shard on Windows."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


try:
    import torch
    from safetensors.torch import load_file, save_file
except ImportError:
    torch = None
    load_file = save_file = None


TOOL = Path(__file__).resolve().parents[1] / "tools" / "make_qwen38_tiny.py"
REPO = TOOL.parents[2]


@unittest.skipIf(torch is None, "requires torch and safetensors")
class RewriteFp8ShardWindowsTest(unittest.TestCase):
    def test_rewrite_releases_source_shard_before_unlink(self):
        # On Windows this test fails with PermissionError if load_file's mapped
        # tensors are still alive when _rewrite_shard_fp8 unlinks the source.
        spec = importlib.util.spec_from_file_location("make_qwen38_tiny_test", TOOL)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory(dir=REPO) as tmp:
            out = Path(tmp)
            source = out / "model.safetensors"
            gate = "model.layers.0.mlp.experts.0.gate_proj.weight"
            down = "model.layers.0.mlp.experts.0.down_proj.weight"
            dense = "model.embed_tokens.weight"
            original = {
                gate: torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16),
                down: torch.tensor([[3.0, 4.0]], dtype=torch.bfloat16),
                dense: torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
            }
            save_file(original, str(source), metadata={"format": "pt"})
            (out / "config.json").write_text("{}\n", encoding="utf-8")
            packed = {
                gate: (torch.tensor([[1.0, 2.0]], dtype=torch.float8_e4m3fn),
                       torch.tensor([[0.25]], dtype=torch.bfloat16)),
                down: (torch.tensor([[3.0, 4.0]], dtype=torch.float8_e4m3fn),
                       torch.tensor([[0.5]], dtype=torch.bfloat16)),
            }

            module._rewrite_shard_fp8(out, packed)

            self.assertFalse(source.exists())
            index = json.loads((out / "model.safetensors.index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["weight_map"][gate], "model-00001-of-00002.safetensors")
            self.assertEqual(index["weight_map"][down], "model-00002-of-00002.safetensors")
            self.assertEqual(index["weight_map"][dense], "model-00001-of-00002.safetensors")
            rest = load_file(str(out / "model-00001-of-00002.safetensors"))
            downs = load_file(str(out / "model-00002-of-00002.safetensors"))
            self.assertEqual(rest[gate].dtype, torch.float8_e4m3fn)
            self.assertTrue(torch.equal(rest[gate], packed[gate][0]))
            self.assertEqual(rest[gate + "_scale_inv"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(rest[gate + "_scale_inv"], packed[gate][1]))
            self.assertTrue(torch.equal(rest[dense], original[dense]))
            self.assertEqual(downs[down].dtype, torch.float8_e4m3fn)
            self.assertTrue(torch.equal(downs[down], packed[down][0]))
            self.assertEqual(downs[down + "_scale_inv"].dtype, torch.bfloat16)
            self.assertTrue(torch.equal(downs[down + "_scale_inv"], packed[down][1]))
            del rest, downs


if __name__ == "__main__":
    unittest.main()
