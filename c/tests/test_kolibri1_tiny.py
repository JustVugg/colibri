#!/usr/bin/env python3
"""tools/make_kolibri1_tiny.py: the tiny Kolibri-1 fixture really has the release's
tensor census and layout quirks, and is byte-deterministic.

The 31 name patterns below were read from the 32 safetensors headers of the real
checkpoint (116,303 tensors), pattern -> (dtype, rank).
"""
import hashlib
import json
import re
import struct
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import numpy  # noqa: F401  (the generator needs it)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"

# pattern -> (dtype, rank, tensors per layer for E experts, or fixed count)
RELEASE_PATTERNS = {
    "lm_head.weight": ("BF16", 2),
    "model.embed_tokens.weight": ("BF16", 2),
    "model.norm.weight": ("BF16", 1),
    "model.layers.N.input_layernorm.weight": ("BF16", 1),
    "model.layers.N.post_attention_layernorm.weight": ("BF16", 1),
    "model.layers.N.post_attn_norm.weight": ("BF16", 1),
    "model.layers.N.post_ffn_norm.weight": ("BF16", 1),
    "model.layers.N.self_attn.q_norm.weight": ("BF16", 1),
    "model.layers.N.self_attn.k_norm.weight": ("BF16", 1),
    "model.layers.N.mlp.gate.weight": ("BF16", 2),
    "model.layers.N.moe.router.expert_bias": ("BF16", 1),
}
for proj in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj",
             "mlp.shared_experts.down_proj", "mlp.experts.E.gate_proj",
             "mlp.experts.E.up_proj", "mlp.experts.E.down_proj"):
    RELEASE_PATTERNS[f"model.layers.N.{proj}.weight"] = ("F8_E4M3", 2)
    RELEASE_PATTERNS[f"model.layers.N.{proj}.weight_scale_inv"] = ("F32", 2)
assert len(RELEASE_PATTERNS) == 31


def pattern(name):
    name = re.sub(r"\.layers\.\d+\.", ".layers.N.", name)
    return re.sub(r"\.experts\.\d+\.", ".experts.E.", name)


def read_header(path):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    assert n % 8 == 0
    header.pop("__metadata__", None)
    return header


def read_all(directory):
    """{name: (shard file, dtype, shape, [begin, end])} over every shard."""
    tensors = {}
    for shard in sorted(Path(directory).glob("*.safetensors")):
        for name, meta in read_header(shard).items():
            assert name not in tensors, f"duplicate {name}"
            tensors[name] = (shard.name, meta["dtype"], meta["shape"], meta["data_offsets"])
    return tensors


def sha_dir(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(directory).iterdir())}


def generate(out, *extra):
    subprocess.run([sys.executable, str(TOOL), "--out", str(out), "--force", *extra],
                   check=True, capture_output=True, text=True)


@unittest.skipUnless(HAVE_NUMPY, "numpy is needed by the fixture generator")
class Kolibri1TinyFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.rel, cls.rel2, cls.simple, cls.alt = (base / n for n in ("rel", "rel2", "simple", "alt"))
        generate(cls.rel, "--layout", "release")
        generate(cls.rel2, "--layout", "release")
        generate(cls.simple)
        generate(cls.alt, "--layout", "release", "--layer-types", "alt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_census_equals_the_release_name_patterns(self):
        for directory in (self.rel, self.simple, self.alt):
            tensors = read_all(directory)
            seen = {}
            for name, (_, dtype, shape, _) in tensors.items():
                p = pattern(name)
                seen.setdefault(p, set()).add((dtype, len(shape)))
            self.assertEqual(set(seen), set(RELEASE_PATTERNS), directory.name)
            for p, kinds in seen.items():
                self.assertEqual(kinds, {RELEASE_PATTERNS[p]}, p)
            # 3 + L * (22 + 6 E) tensors; the real release is 3 + 50 * (22 + 6 * 384)
            self.assertEqual(len(tensors), 3 + 6 * (22 + 6 * 8))

    def test_scale_grids_are_ceil_blocks_and_f32(self):
        tensors = read_all(self.rel)
        for name, (_, dtype, shape, span) in tensors.items():
            if not name.endswith(".weight_scale_inv"):
                continue
            wshape = tensors[name[:-len("_scale_inv")]][2]
            self.assertEqual(shape, [-(-wshape[0] // 128), -(-wshape[1] // 128)], name)
            self.assertEqual(dtype, "F32")
            self.assertEqual(span[1] - span[0], 4 * shape[0] * shape[1])
        # q is heads*head_dim wide, not hidden: the grid is not square
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.weight"][2], [1024, 256])
        self.assertEqual(tensors["model.layers.0.self_attn.q_proj.weight_scale_inv"][2], [8, 2])
        self.assertEqual(tensors["model.layers.0.self_attn.o_proj.weight_scale_inv"][2], [2, 8])
        # the MoE width 192 ends in a partial block
        self.assertEqual(tensors["model.layers.0.mlp.experts.0.gate_proj.weight"][2], [192, 256])
        self.assertEqual(tensors["model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv"][2], [2, 2])

    def test_bytes_are_deterministic(self):
        self.assertEqual(sha_dir(self.rel), sha_dir(self.rel2))
        with tempfile.TemporaryDirectory() as again:
            generate(again, "--layout", "release")
            self.assertEqual(sha_dir(self.rel), sha_dir(again))

    def test_layouts_place_the_same_tensor_bytes(self):
        def blobs(directory):
            out = {}
            for shard in sorted(Path(directory).glob("*.safetensors")):
                raw = shard.read_bytes()
                n = struct.unpack("<Q", raw[:8])[0]
                header = json.loads(raw[8:8 + n])
                header.pop("__metadata__", None)
                for name, meta in header.items():
                    a, b = meta["data_offsets"]
                    out[name] = raw[8 + n + a:8 + n + b]
            return out
        self.assertEqual(blobs(self.rel), blobs(self.simple))
        self.assertEqual(blobs(self.rel), blobs(self.alt))

    def test_release_shards_lead_with_f32_then_bf16_then_f8_sorted_by_name(self):
        shards = sorted(self.rel.glob("*.safetensors"))
        self.assertEqual(len(shards), 4)
        for shard in shards:
            header = read_header(shard)
            by_offset = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
            groups = [{"F32": 0, "BF16": 1, "F8_E4M3": 2}[m["dtype"]] for _, m in by_offset]
            self.assertEqual(groups, sorted(groups), shard.name)
            self.assertEqual(groups[0], 0, f"{shard.name}: F32 scales must lead the shard")
            for g in (0, 1, 2):
                names = [n for (n, _), gg in zip(by_offset, groups) if gg == g]
                self.assertEqual(names, sorted(names), f"{shard.name} group {g}")

    def test_release_expert_placement_mix(self):
        tensors = read_all(self.rel)
        classes = Counter()
        for layer in range(6):
            for e in range(8):
                base = f"model.layers.{layer}.mlp.experts.{e}."
                where = {p: tensors[base + p + ".weight"] for p in ("down_proj", "gate_proj", "up_proj")}
                shard = {p: w[0] for p, w in where.items()}
                for p in where:      # a scale always sits in the shard of its weight
                    self.assertEqual(tensors[base + p + ".weight_scale_inv"][0], shard[p])
                if len(set(shard.values())) == 1:
                    d, g, u = (where[p][3] for p in ("down_proj", "gate_proj", "up_proj"))
                    self.assertTrue(d[1] == g[0] and g[1] == u[0], base)   # down|gate|up in file order
                    classes["C"] += 1
                elif shard["gate_proj"] < shard["down_proj"] == shard["up_proj"]:
                    classes["G"] += 1
                elif shard["up_proj"] > shard["down_proj"] == shard["gate_proj"]:
                    classes["U"] += 1
                else:
                    self.fail(f"unexpected placement {shard}")
        self.assertEqual(dict(classes), {"C": 28, "G": 10, "U": 10})
        # a split expert's two halves are in ADJACENT shard files (previous / next)
        manifest = json.loads((self.rel / "fixture.json").read_text())
        self.assertEqual(manifest["expert_classes"], {"C": 28, "G": 10, "U": 10})

    def test_index_matches_the_shards(self):
        for directory in (self.rel, self.simple):
            tensors = read_all(directory)
            index = json.loads((directory / "model.safetensors.index.json").read_text())
            self.assertEqual({n: t[0] for n, t in tensors.items()}, index["weight_map"])

    def test_layer_types_release_pattern_and_alt(self):
        release = json.loads((self.rel / "config.json").read_text())
        alt = json.loads((self.alt / "config.json").read_text())
        full = lambda cfg: [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
        self.assertEqual(full(release), [4])                       # l % 5 == 4
        self.assertTrue(all(i % 5 == 4 for i in full(release)))
        self.assertEqual(full(alt), [2, 5])                        # valid, not the release pattern
        self.assertFalse(all(i % 5 == 4 for i in full(alt)))
        for cfg in (release, alt):
            self.assertEqual(cfg["model_type"], "kolibri1")
            self.assertEqual(cfg["num_attention_heads"] * cfg["head_dim"], 1024)
            self.assertNotEqual(cfg["num_attention_heads"] * cfg["head_dim"], cfg["hidden_size"])


if __name__ == "__main__":
    unittest.main()
