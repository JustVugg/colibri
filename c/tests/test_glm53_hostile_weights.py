"""glm53 on a container whose weights were chosen by someone else.

An all-NaN router row: NaN > best is false for every expert, so the top-k pick
stayed at -1, and the gate weight was read as score[-1] before the id reached the
expert cache. GHSA-5xpg-vw35-2687 fixed this shape in inkling, kimi_k3 and olmoe
through rt_router_pick() in route_trace.h, which glm53 already includes.

A short f32 tensor: load_f32() sized the buffer by the numel the header declares,
while the forward pass reads norms, router, mHC and vision weights by the sizes in
config.json. One element short was a heap read past the buffer.

Runs in the GLM-5.3 oracle job of check.yml, on the streaming fixture it builds
(the same one test_glm53_context_exceeded uses):

  COLI_GLM53_FIXTURE=/tmp/glm53_stream-i4 python -m unittest -v tests.test_glm53_hostile_weights
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from safetensors_edit import copy_fixture, fill_nan, shape_of, shrink  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
BINARY = next((path for path in (HERE / "glm53.exe", HERE / "glm53") if path.exists()), None)
FIXTURE = Path(os.environ.get("COLI_GLM53_FIXTURE", ""))
ROUTER = "model.language_model.layers.3.mlp.gate.weight"   # the fixture's one MoE layer
SANITIZER = ("ERROR: AddressSanitizer", "runtime error:")

# Every tensor glm53 loads through load_f32(), once per sizing rule: layer 0 is a
# KDA layer, layer 3 the MLA + indexer + MoE one, block 0 stands for every vision
# block. The q/k/v conv pieces are copied out by a memcpy of the configured width.
F32_TENSORS = ["model.language_model." + name for name in (
    "embed_tokens.weight", "norm.weight",
    "layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight",
    "layers.0.hc_attn_fn", "layers.0.hc_attn_base", "layers.0.hc_attn_scale",
    "layers.0.hc_ffn_fn", "layers.0.hc_ffn_base", "layers.0.hc_ffn_scale",
    "layers.0.self_attn.dt_bias", "layers.0.self_attn.A_log",
    "layers.0.self_attn.o_norm.weight", "layers.0.self_attn.q_conv1d.weight",
    "layers.0.self_attn.k_conv1d.weight", "layers.0.self_attn.v_conv1d.weight",
    "layers.3.self_attn.q_a_layernorm.weight", "layers.3.self_attn.kv_a_layernorm.weight",
    "layers.3.self_attn.indexer.k_norm.weight", "layers.3.self_attn.indexer.k_norm.bias",
    "layers.3.self_attn.indexer.index_kpool_compress_ape",
    "layers.3.mlp.gate.weight", "layers.3.mlp.gate.e_score_correction_bias",
)] + ["model.visual." + name for name in (
    "patch_embed.proj.weight", "patch_embed.proj.bias", "post_layernorm.weight",
    "downsample.weight", "downsample.bias", "merger.proj.weight",
    "merger.post_projection_norm.weight", "merger.post_projection_norm.bias",
    "merger.gate_proj.weight", "merger.up_proj.weight", "merger.down_proj.weight",
    "blocks.0.norm1.weight", "blocks.0.norm2.weight",
    "blocks.0.attn.qkv.weight", "blocks.0.attn.qkv.bias",
    "blocks.0.attn.q_norm.weight", "blocks.0.attn.k_norm.weight",
    "blocks.0.attn.proj.weight", "blocks.0.attn.proj.bias",
    "blocks.0.mlp.gate_proj.weight", "blocks.0.mlp.gate_proj.bias",
    "blocks.0.mlp.up_proj.weight", "blocks.0.mlp.up_proj.bias",
    "blocks.0.mlp.down_proj.weight", "blocks.0.mlp.down_proj.bias",
)]


def image_run(snapshot):
    """The fixture's own prompt with its image: every text and vision weight is read."""
    reference = json.loads((FIXTURE / "ref.json").read_text())
    grid_h, grid_w = reference["grid"]
    command = [str(BINARY), "--model", str(snapshot),
               "--ids", ",".join(str(token) for token in reference["prompt"]),
               "--patches", str(snapshot / "patches.f32"), "--grid", f"{grid_h}x{grid_w}",
               "--greedy", "2"]
    return subprocess.run(command, capture_output=True, text=True, errors="replace",
                          timeout=300, env={**os.environ, "GLM53_BITS": "32"})


def serve_turn(snapshot, prompt=b"abc"):
    """One SERVE turn; return (the line that ends it, the engine's stderr)."""
    environment = {**os.environ, "SNAP": str(snapshot), "SERVE": "1",
                   "GLM53_BITS": "32", "GLM53_MAXT": "64"}
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen([str(BINARY)], env=environment, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=errors, bufsize=0)
        ending = None
        try:
            while b"READY" not in (line := process.stdout.readline()):
                if not line:
                    break
            else:
                process.stdin.write(f"SUBMIT 7 0 {len(prompt)} 4 0 1\n".encode() + prompt + b"\n")
                process.stdin.flush()
                for _ in range(400):
                    line = process.stdout.readline()
                    if not line:
                        break
                    text = line.decode("latin-1").rstrip("\n")
                    if text.startswith("DATA "):
                        process.stdout.read(int(text.split()[2]))
                        process.stdout.readline()
                    elif text.startswith(("ERROR ", "DONE ")):
                        ending = text
                        break
        finally:
            process.stdin.close()
            try:
                process.wait(60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()
        errors.seek(0)
        return ending, errors.read().decode("utf-8", "replace")


@unittest.skipUnless(BINARY, "glm53 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file(),
                     "COLI_GLM53_FIXTURE not set to a GLM-5.3-Flash container")
class Glm53HostileWeightsTest(unittest.TestCase):
    def test_an_all_nan_router_degrades_instead_of_indexing_minus_one(self):
        with tempfile.TemporaryDirectory() as scratch:
            snapshot = copy_fixture(FIXTURE, Path(scratch) / "model")
            fill_nan(snapshot, ROUTER)
            ending, stderr = serve_turn(snapshot)
        for marker in SANITIZER:
            self.assertNotIn(marker, stderr, stderr[-4000:])
        self.assertIsNotNone(ending, "the engine died mid-turn:\n" + stderr[-4000:])
        self.assertTrue(ending.startswith("DONE 7 "), ending)
        self.assertIn("[router] non-finite logits", stderr)


@unittest.skipUnless(BINARY, "glm53 engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "patches.f32").is_file(),
                     "COLI_GLM53_FIXTURE not set to a GLM-5.3-Flash container with an image")
class Glm53ShortTensorTest(unittest.TestCase):
    def test_the_intact_fixture_passes_the_same_checks(self):
        with tempfile.TemporaryDirectory() as scratch:
            result = image_run(copy_fixture(FIXTURE, Path(scratch) / "model"))
        for marker in SANITIZER:
            self.assertNotIn(marker, result.stderr, result.stderr[-4000:])
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        self.assertIn("greedy", result.stdout)

    def test_an_f32_tensor_one_element_short_is_refused_at_load(self):
        for name in F32_TENSORS:
            with self.subTest(name), tempfile.TemporaryDirectory() as scratch:
                snapshot = copy_fixture(FIXTURE, Path(scratch) / "model")
                shrink(snapshot, name, math.prod(shape_of(snapshot, name)) - 1)
                result = image_run(snapshot)
                for marker in SANITIZER:
                    self.assertNotIn(marker, result.stderr, result.stderr[-4000:])
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(f"{name}: ", result.stderr)
                self.assertIn("the config needs", result.stderr)


if __name__ == "__main__":
    unittest.main()
