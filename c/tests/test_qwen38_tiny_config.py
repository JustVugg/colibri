"""Tests for make_qwen38_tiny.py and make_qwen38_oracle.py transformers >= 5.16.1 compatibility."""

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

try:
    import torch  # noqa: F401  (the generators import it at load time)
except ImportError:
    torch = None

try:
    import packaging.version  # noqa: F401
except ImportError:
    packaging = None
else:
    packaging = True

_TOOLS = Path(__file__).resolve().parent.parent / "tools"
_TINY_TOOL = _TOOLS / "make_qwen38_tiny.py"
_ORACLE_TOOL = _TOOLS / "make_qwen38_oracle.py"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _has_qwen4_exp():
    if torch is None:
        return False
    try:
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig  # noqa: F401
        return True
    except Exception:
        return False


@unittest.skipUnless(torch, "torch is required to load make_qwen38_tiny.py")
class LegacyLayerTypesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool = _load_module("make_qwen38_tiny", _TINY_TOOL)
        cls.oracle = _load_module("make_qwen38_oracle", _ORACLE_TOOL)

    def test_renames_every_layer_types_list(self):
        cfg = {
            "layer_types": ["linear_attention", "indexed_attention"],
            "mtp_layer_types": ["indexed_attention"],
            "text_config": {"layer_types": ["indexed_attention", "full_attention"]},
            "mtp": {"layer_types": ["indexed_attention"], "rope_theta": 1.0},
        }
        self.assertTrue(self.tool._legacy_layer_types(cfg))
        self.assertEqual(cfg["layer_types"], ["linear_attention", "qwen_sparse_attention"])
        self.assertEqual(cfg["mtp_layer_types"], ["qwen_sparse_attention"])
        self.assertEqual(
            cfg["text_config"]["layer_types"],
            ["qwen_sparse_attention", "full_attention"],
        )
        self.assertEqual(
            cfg["mtp"],
            {"layer_types": ["qwen_sparse_attention"], "rope_theta": 1.0},
        )

    def test_5_16_1_config_is_untouched(self):
        cfg = {
            "layer_types": ["linear_attention", "qwen_sparse_attention"],
            "mtp": {"layer_types": ["full_attention"]},
        }
        before = repr(cfg)
        self.assertFalse(self.tool._legacy_layer_types(cfg))
        self.assertEqual(repr(cfg), before)

    @unittest.skipUnless(packaging, "packaging is required for version comparison")
    def test_tiny_version_gate_accepts_gte_5_16_1_and_rejects_older(self):
        fake_cls = object()
        fake_cfg = object()

        def make_fake_tf(ver):
            mod = types.ModuleType("transformers")
            mod.__version__ = ver
            mod.Qwen4ExpForCausalLM = fake_cls
            mod.Qwen4ExpTextConfig = fake_cfg
            return mod

        # Older version is rejected
        with mock.patch.dict(sys.modules, {"transformers": make_fake_tf("5.16.0")}):
            with self.assertRaises(SystemExit) as ctx:
                self.tool._classes()
            self.assertIn(">=5.16.1", str(ctx.exception))

        # Exact 5.16.1 is accepted silently
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"transformers": make_fake_tf("5.16.1")}), redirect_stderr(err):
            tf, cls, cfg = self.tool._classes()
        self.assertEqual(tf.__version__, "5.16.1")
        self.assertIs(cls, fake_cls)
        self.assertIs(cfg, fake_cfg)
        self.assertEqual(err.getvalue(), "")

        # Newer version (> 5.16.1) is accepted with an informational note on stderr
        err = io.StringIO()
        with mock.patch.dict(sys.modules, {"transformers": make_fake_tf("5.20.2")}), redirect_stderr(err):
            tf, cls, cfg = self.tool._classes()
        self.assertEqual(tf.__version__, "5.20.2")
        self.assertIn("5.20.2", err.getvalue())

    @unittest.skipUnless(packaging, "packaging is required for version comparison")
    def test_oracle_version_gate_rejects_older(self):
        mod = types.ModuleType("transformers")
        mod.__version__ = "5.15.9"
        mod.AutoConfig = object()
        mod.Qwen4ExpForCausalLM = object()
        with mock.patch.dict(sys.modules, {"transformers": mod}):
            with self.assertRaises(SystemExit) as ctx:
                self.oracle._load_model("/nonexistent")
            self.assertIn(">=5.16.1", str(ctx.exception))

    @unittest.skipUnless(_has_qwen4_exp(), "transformers with Qwen4ExpForCausalLM required")
    def test_fixture_generation_is_deterministic_and_oracle_loadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a"
            out_b = Path(tmp) / "b"
            self.tool.build(out_a, max_new=2, emit_ref=True)
            self.tool.build(out_b, max_new=2, emit_ref=True)

            # Byte-identical fixture generation across runs
            for fname in ("model.safetensors", "config.json", "ref.json"):
                self.assertEqual(
                    (out_a / fname).read_bytes(),
                    (out_b / fname).read_bytes(),
                    f"{fname} differed across deterministic runs",
                )

            # config.json uses qwen_sparse_attention, never indexed_attention
            cfg = json.loads((out_a / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                cfg["layer_types"],
                [
                    "linear_attention",
                    "qwen_sparse_attention",
                    "linear_attention",
                    "qwen_sparse_attention",
                ],
            )
            self.assertNotIn("indexed_attention", (out_a / "config.json").read_text(encoding="utf-8"))

            # make_qwen38_oracle.py loads the generated fixture and matches ref.json
            model, ver = self.oracle._load_model(str(out_a))
            ref = json.loads((out_a / "ref.json").read_text(encoding="utf-8"))
            oracle_res = self.oracle.run(model, ref["prompt_ids"], max_new=2, seed=self.tool.SEED)
            self.assertEqual(oracle_res["full_ids"], ref["full_ids"])
            self.assertEqual(oracle_res["prefill"]["logits_sha256"], ref["prefill"]["logits_sha256"])


if __name__ == "__main__":
    unittest.main()
