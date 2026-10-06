#!/usr/bin/env python3
"""MTP-only conversion reads partial HF shards and never writes the base container."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
try:
    import torch
    from safetensors.torch import load_file, save_file
    from convert_qwen36 import append_mtp, convert_mtp
except ImportError:   # the plain Python suite runs without torch
    torch = None

FIXTURES = ("qwen36_tiny_mtp/model.safetensors", "qwen36_tiny_mtp_i8/model-mtp.safetensors",
            "qwen36_tiny_mtp_i4/model-mtp.safetensors")


# The fixtures come from `make qwen36-tiny-mtp-generate`; `make qwen36-tiny-mtp-check` makes
# them and runs this file, the plain suite (`make check`) skips it without them.
@unittest.skipUnless(torch is not None and all(Path(f).is_file() for f in FIXTURES),
                     "needs torch and the tiny MTP fixtures (make qwen36-tiny-mtp-generate)")
class MtpConvert(unittest.TestCase):
    def test_partial_checkpoint_and_immutable_base(self):
        source = Path("qwen36_tiny_mtp")
        weights = {k: v for k, v in load_file(str(source / "model.safetensors")).items() if k.startswith("mtp.")}
        for kind in ("i8", "i4"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                src, out = root / "hf", root / "out"
                src.mkdir()
                (src / "config.json").write_bytes((source / "config.json").read_bytes())
                save_file(weights, str(src / "head.safetensors"))
                (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                    **{k: "head.safetensors" for k in weights},
                    "model.layers.0.self_attn.q_proj.weight": "missing-trunk.safetensors"}}))
                base = Path(f"qwen36_tiny_mtp_{kind}").resolve()
                before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in base.iterdir() if p.is_file()}
                args = argparse.Namespace(model=str(src), repo=None, upload_repo=None, stream_upload=False,
                                          base_container=str(base), out=str(out))
                append_mtp(args)
                expected = load_file(str(base / "model-mtp.safetensors"))
                actual = load_file(str(out / "model-mtp.safetensors"))
                self.assertEqual(actual.keys(), expected.keys())
                for name in actual:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)
                for name, digest in before.items():
                    self.assertEqual(hashlib.sha256((base / name).read_bytes()).hexdigest(), digest)
                    if name not in ("model-mtp.safetensors", "qwen36_meta.json"):
                        self.assertTrue((out / name).is_symlink(), name)
                with self.assertRaises(ValueError):
                    append_mtp(args)  # no overwrite, including existing symlinks
                args.out = str(base)
                with self.assertRaises(ValueError):
                    append_mtp(args)

    def test_fused_and_separate_experts_all_formats(self):
        generator = torch.Generator().manual_seed(37)
        gu = torch.randn(3, 128, 64, generator=generator)
        down = torch.randn(3, 64, 64, generator=generator)
        prefix = "mtp.layers.0.mlp.experts."
        fused = {prefix + "gate_up_proj": gu, prefix + "down_proj": down,
                 "mtp.fc.weight": torch.randn(64, 128, generator=generator)}
        separate = {"mtp.fc.weight": fused["mtp.fc.weight"]}
        for e in range(3):
            for proj, matrix in (("gate", gu[e, :64]), ("up", gu[e, 64:]), ("down", down[e])):
                separate[f"{prefix}{e}.{proj}_proj.weight"] = matrix
        for bits, gs, db, dg in ((8, 0, 0, 0), (4, 0, 0, 0), (4, 64, 0, 0), (4, 64, 8, 0), (4, 64, 8, 32)):
            with self.subTest(bits=bits, gs=gs, down_bits=db, down_gs=dg):
                args = argparse.Namespace(ebits=bits, gs=gs, down_bits=db, down_gs=dg)
                a = convert_mtp(fused, fused.__getitem__, args)
                b = convert_mtp(separate, separate.__getitem__, args)
                self.assertEqual(a.keys(), b.keys())
                for name in a:
                    self.assertTrue(torch.equal(a[name], b[name]), name)


if __name__ == "__main__":
    unittest.main()
