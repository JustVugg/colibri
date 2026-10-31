"""Actual local-converter CLI controls; no model download or inference."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy  # noqa: F401 -- safetensors.torch serialization needs it
    import torch
    from safetensors.torch import load_file, save_file
except ImportError as error:
    raise unittest.SkipTest(f"converter dependencies unavailable: {error}")

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def snapshot(directory):
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.iterdir() if path.is_file()}


class ConverterSourceDirectoryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def fixture(self, tool):
        if tool == "glm53":
            config = {"model_type": "glm5_next", "text_config": {"eos_token_id": [1]}}
            tensors = {"model.layers.0.mlp.experts.0.up_proj.weight":
                       torch.linspace(-0.5, 0.5, 128).reshape(2, 64),
                       "model.norm.weight": torch.ones(64)}
        else:
            text = {"model_type": "qwen3_5_moe_text", "hidden_size": 8,
                    "intermediate_size": 16, "num_hidden_layers": 1,
                    "num_experts": 1, "moe_intermediate_size": 4,
                    "num_experts_per_tok": 1, "num_attention_heads": 1,
                    "num_key_value_heads": 1, "layer_types": ["full_attention"]}
            config = {"model_type": "qwen3_5_moe", "text_config": text}
            prefix = "model.language_model."
            tensors = {prefix + "embed_tokens.weight": torch.arange(32).reshape(4, 8).float(),
                       prefix + "lm_head.weight": torch.arange(32).reshape(4, 8).float(),
                       prefix + "norm.weight": torch.ones(8),
                       prefix + "layers.0.input_layernorm.weight": torch.ones(8)}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tensors[prefix + "layers.0.self_attn." + name + ".weight"] = torch.eye(8)
            for name, shape in (("gate_proj", (4, 8)), ("up_proj", (4, 8)), ("down_proj", (8, 4))):
                tensors[prefix + "layers.0.mlp.experts.0." + name + ".weight"] = (
                    torch.arange(32).reshape(shape).float() / 32)
        (self.source / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (self.source / "generation_config.json").write_text('{"eos_token_id": [1]}\n')
        save_file(tensors, str(self.source / "model-00001-of-00001.safetensors"))
        return config

    def run_cli(self, tool, output):
        command = [sys.executable, str(TOOLS / f"convert_{tool}.py")]
        if tool == "glm53":
            # These owned tensor files are only a few KiB. The real converter's
            # default reserve is for multi-GB shards, not this fixture.
            command += ["--indir", str(self.source), "--outdir", str(output),
                        "--min-free-gb", "0", "--min-free-c-gb", "0", "--keep-source"]
        else:
            command += ["--model", str(self.source), "--out", str(output), "--no-readme"]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OPENBLAS_NUM_THREADS="1",
                   OMP_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)

    def overlap(self, tool, alias=False):
        self.fixture(tool)
        output = self.source
        if alias:
            output = self.root / "alias"
            try:
                output.symlink_to(self.source, target_is_directory=True)
            except NotImplementedError as error:
                self.skipTest(str(error))
            except OSError as error:
                if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                    self.skipTest(str(error))
                raise
        before = snapshot(self.source)
        result = self.run_cli(tool, output)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("must differ", result.stderr)
        self.assertEqual(snapshot(self.source), before)

    def ordinary(self, tool):
        config = self.fixture(tool)
        before = snapshot(self.source)
        output = self.root / "output"
        result = self.run_cli(tool, output)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(snapshot(self.source), before)
        if tool == "glm53":
            produced = load_file(str(output / "model-00001-of-00001.safetensors"))
            name = "model.layers.0.mlp.experts.0.up_proj.weight"
            self.assertEqual(produced[name].dtype, torch.uint8)
            self.assertEqual(produced[name + ".qs"].dtype, torch.float32)
        else:
            self.assertEqual(json.loads((output / "config.json").read_text()), config["text_config"])
            self.assertEqual(json.loads((output / "config.hf.json").read_text()), config)
            produced = load_file(str(output / "model-globals.safetensors"))
            self.assertEqual(produced["model.embed_tokens.weight"].dtype, torch.float16)

    def test_glm53_identical_directories_preserve_source(self):
        self.overlap("glm53")

    def test_glm53_directory_alias_preserves_source(self):
        self.overlap("glm53", alias=True)

    def test_glm53_separate_output_still_converts(self):
        self.ordinary("glm53")

    def test_qwen36_identical_directories_preserve_source(self):
        self.overlap("qwen36")

    def test_qwen36_directory_alias_preserves_source(self):
        self.overlap("qwen36", alias=True)

    def test_qwen36_separate_output_still_converts(self):
        self.ordinary("qwen36")


if __name__ == "__main__":
    unittest.main()
