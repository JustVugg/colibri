"""OLMoE checks every dense tensor's shape against config.json at load.

load_t() allocated each dense tensor from the element count in the
safetensors header, and the forward then used it with the config's
dimensions: q/k/v/o as [hidden, hidden], the router as [num_experts, hidden],
embed_tokens and lm_head as [vocab_size, hidden], every norm as [hidden].
Nothing compared the two. A tensor shorter than its config shape was read
past its end (a key projection written for grouped-query attention, which
this engine does not run, is one), and one of another shape with enough
elements (more rows, a transposed router) was used as if it had the right
one. The routed experts are already held to an exact size in
load_expert_merged(); this holds the dense tensors to the same rule.

Each test copies the tiny OLMoE container, rewrites ONE dense tensor with
another shape and runs the real engine on the container's own reference.
Needs a converted tiny OLMoE container and a built engine, the ones the OLMoE
tiny oracle job builds.
"""
import json
import os
import shutil
import struct
import subprocess
import tempfile
import unittest
from math import prod
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", ""))
REFERENCE = Path(os.environ.get("COLI_OLMOE_REFERENCE", ""))
ELEMENT = {"F32": 4, "F16": 2, "BF16": 2}


def reshape(container, name, shape):
    """Rewrite `name` in its shard with `shape`, its bytes cut or repeated to fit."""
    for shard in sorted(container.glob("*.safetensors")):
        raw = shard.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8:8 + size])
        if name not in header:
            continue
        data = raw[8 + size:]
        rebuilt, blob = {}, bytearray()
        for key, entry in sorted(((k, v) for k, v in header.items() if k != "__metadata__"),
                                 key=lambda item: item[1]["data_offsets"][0]):
            begin, end = entry["data_offsets"]
            payload = data[begin:end]
            if key == name:
                want = prod(shape) * ELEMENT[entry["dtype"]]
                payload = (payload * (want // len(payload) + 1))[:want]
                entry = {**entry, "shape": list(shape)}
            rebuilt[key] = {**entry, "data_offsets": [len(blob), len(blob) + len(payload)]}
            blob += payload
        if "__metadata__" in header:
            rebuilt["__metadata__"] = header["__metadata__"]
        text = json.dumps(rebuilt).encode()
        text += b" " * (-len(text) % 8)
        shard.write_bytes(struct.pack("<Q", len(text)) + text + bytes(blob))
        return
    raise AssertionError(f"{name} is not in {container}")


def run(changes):
    with tempfile.TemporaryDirectory() as tmp:
        container = Path(tmp) / "olmoe"
        shutil.copytree(FIXTURE, container)
        for name, shape in changes:
            reshape(container, name, shape)
        environment = {**os.environ, "SNAP": str(container)}
        return subprocess.run([str(ENGINE), "8", "8", str(REFERENCE)], env=environment,
                              capture_output=True, text=True, errors="replace",
                              timeout=300)


@unittest.skipUnless(ENGINE.exists(), "olmoe engine is not built")
@unittest.skipUnless(FIXTURE.name and (FIXTURE / "config.json").is_file()
                     and REFERENCE.name and REFERENCE.is_file(),
                     "COLI_OLMOE_FIXTURE / COLI_OLMOE_REFERENCE not set to a converted "
                     "OLMoE container and its reference")
class OlmoeDenseShapeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        config = json.loads((FIXTURE / "config.json").read_text())
        cls.hidden, cls.vocab = config["hidden_size"], config["vocab_size"]
        cls.experts = config["num_experts"]

    def assertRefused(self, result, name, shape, expected):
        message = f"{name}: shape {list(shape)}, expected {list(expected)} from config.json"
        self.assertEqual(result.returncode, 1, result.stderr[-2000:])
        self.assertIn(message, result.stderr, result.stderr[-2000:])
        self.assertNotIn("Matching tokens", result.stdout)

    def refused(self, name, shape, expected):
        self.assertRefused(run([(name, shape)]), name, shape, expected)

    def test_the_container_as_converted_runs_token_exact(self):
        result = run([])
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertRegex(result.stdout, r"Matching tokens: (\d+)/\1\n")

    def test_a_short_query_projection_is_refused(self):
        D = self.hidden
        self.refused("model.layers.1.self_attn.q_proj.weight", [D // 2, D], [D, D])

    def test_a_grouped_query_key_projection_is_refused(self):
        D = self.hidden
        self.refused("model.layers.0.self_attn.k_proj.weight", [D // 4, D], [D, D])

    def test_a_value_projection_with_extra_rows_is_refused(self):
        D = self.hidden
        self.refused("model.layers.2.self_attn.v_proj.weight", [2 * D, D], [D, D])

    def test_an_output_projection_with_short_columns_is_refused(self):
        D = self.hidden
        self.refused("model.layers.3.self_attn.o_proj.weight", [D, D // 2], [D, D])

    def test_a_query_projection_with_extra_columns_is_refused(self):
        D = self.hidden
        self.refused("model.layers.0.self_attn.q_proj.weight", [D, 2 * D], [D, D])

    def test_a_short_router_is_refused(self):
        D, E = self.hidden, self.experts
        self.refused("model.layers.0.mlp.gate.weight", [E // 2, D], [E, D])

    def test_a_transposed_router_is_refused(self):
        D, E = self.hidden, self.experts
        self.refused("model.layers.1.mlp.gate.weight", [D, E], [E, D])

    def test_a_short_lm_head_is_refused(self):
        D, V = self.hidden, self.vocab
        self.refused("lm_head.weight", [V - 8, D], [V, D])

    def test_a_short_embedding_is_refused(self):
        D, V = self.hidden, self.vocab
        self.refused("model.embed_tokens.weight", [V // 2, D], [V, D])

    def test_short_norms_are_refused(self):
        D = self.hidden
        for name in ("model.norm.weight",
                     "model.layers.0.input_layernorm.weight",
                     "model.layers.0.post_attention_layernorm.weight",
                     "model.layers.0.self_attn.q_norm.weight",
                     "model.layers.0.self_attn.k_norm.weight"):
            with self.subTest(name=name):
                self.refused(name, [D // 2], [D])

    def test_a_norm_with_an_extra_axis_is_refused(self):
        D = self.hidden
        self.refused("model.layers.2.input_layernorm.weight", [D, 1], [D])

    def test_a_projection_with_an_extra_axis_is_refused(self):
        D = self.hidden
        self.refused("model.layers.3.self_attn.k_proj.weight", [D, D, 1], [D, D])

    def test_a_flat_projection_is_refused(self):
        D = self.hidden
        self.refused("model.layers.2.self_attn.q_proj.weight", [D * D], [D, D])


if __name__ == "__main__":
    unittest.main()
