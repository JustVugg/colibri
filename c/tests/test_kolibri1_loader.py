#!/usr/bin/env python3
"""kolibri1 --validate: the loader accepts the tiny Kolibri-1 fixtures and refuses
every malformed checkpoint by name (negative matrix).

Needs the ./kolibri1 binary (make kolibri1) and numpy (the fixture generator);
without either the tests skip. Each negative case edits a COPY of a generated
fixture, runs `kolibri1 <dir> --validate`, and checks for a non-zero exit and for
the offending tensor / config key in the diagnostics.
"""
import json
import os
import shutil
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

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "make_kolibri1_tiny.py"
BINARY = Path(os.environ.get("KOLIBRI1_BIN", ROOT / "kolibri1")).resolve()


# ---- tiny safetensors editing ------------------------------------------------

def read_shard(path):
    """[(name, meta, bytes)] in file order."""
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    out = []
    for name, meta in sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0]):
        a, b = meta["data_offsets"]
        out.append((name, {"dtype": meta["dtype"], "shape": meta["shape"]}, raw[8 + n + a:8 + n + b]))
    return out


def write_shard(path, entries):
    header, off = {"__metadata__": {"format": "pt"}}, 0
    for name, meta, blob in entries:
        header[name] = {"dtype": meta["dtype"], "shape": meta["shape"], "data_offsets": [off, off + len(blob)]}
        off += len(blob)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for _, _, b in entries:
            fh.write(b)


def find_shard(directory, tensor):
    for shard in sorted(Path(directory).glob("*.safetensors")):
        if any(n == tensor for n, _, _ in read_shard(shard)):
            return shard
    raise KeyError(tensor)


def edit(directory, tensor, fn):
    """Replace/drop one tensor: fn((name, meta, bytes)) -> entry or None."""
    shard = find_shard(directory, tensor)
    entries = []
    for e in read_shard(shard):
        if e[0] == tensor:
            e = fn(e)
            if e is None:
                continue
        entries.append(e)
    write_shard(shard, entries)
    return shard


def edit_config(directory, **kw):
    p = Path(directory) / "config.json"
    cfg = json.loads(p.read_text())
    cfg.update(kw)
    p.write_text(json.dumps(cfg, indent=2) + "\n")


def edit_index(directory, fn):
    p = Path(directory) / "model.safetensors.index.json"
    idx = json.loads(p.read_text())
    fn(idx)
    p.write_text(json.dumps(idx, indent=1) + "\n")


def run(directory, *flags, timeout=120):
    return subprocess.run([str(BINARY), str(directory), "--validate", *flags], capture_output=True,
                          text=True, timeout=timeout, env={**os.environ, "OMP_NUM_THREADS": "2"})


@unittest.skipUnless(HAVE_NUMPY and BINARY.exists(), "needs numpy and the ./kolibri1 binary (make kolibri1)")
class Kolibri1Loader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = Path(cls.tmp.name)
        cls.base = {}
        for name, args in (("rel", ["--layout", "release"]), ("simple", []),
                           ("alt", ["--layout", "release", "--layer-types", "alt"])):
            d = base / name
            subprocess.run([sys.executable, str(TOOL), "--out", str(d), "--force", *args],
                           check=True, capture_output=True)
            cls.base[name] = d
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, which="rel"):
        type(self).n += 1
        d = Path(self.tmp.name) / f"case{self.n}"
        shutil.copytree(self.base[which], d)
        return d

    def refused(self, d, needle, *flags):
        r = run(d, *flags)
        self.assertNotEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn(needle, r.stderr, r.stderr)
        return r

    # ---- positive ---------------------------------------------------------------

    def test_release_layout_validates_and_reports_the_placement_mix(self):
        for flags in ((), ("--release",)):
            r = run(self.base["rel"], *flags)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("validate OK", r.stdout)
            self.assertIn("tensors 423 in 4 shards", r.stdout)
            self.assertIn("contiguous (1 read) 28, 2 runs 20, 3 runs 0", r.stdout)
            self.assertIn("split across shards 20 (gate in previous shard 10, up in next shard 10, other 0)", r.stdout)
            self.assertIn("6 layers (5 sliding window 17, 1 full/NoPE)", r.stdout)

    def test_simple_layout_validates(self):
        for flags in ((), ("--release",)):
            r = run(self.base["simple"], *flags)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("contiguous (1 read)", r.stdout)

    def test_expect_flags(self):
        self.assertEqual(run(self.base["rel"], "--expect-tensors", "423", "--expect-split", "20").returncode, 0)
        r = run(self.base["rel"], "--expect-tensors", "424")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--expect-tensors", r.stderr)
        r = run(self.base["rel"], "--expect-split", "19")
        self.assertNotEqual(r.returncode, 0)

    def test_generic_loader_accepts_any_valid_layer_types_and_release_rejects_them(self):
        r = run(self.base["alt"])
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("2 full/NoPE", r.stdout)
        r = run(self.base["alt"], "--release")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("layer_types[2]", r.stderr)
        self.assertIn("release pattern", r.stderr)

    # ---- config refusals ----------------------------------------------------------

    def test_wrong_layer_types(self):
        d = self.copy()
        cfg = json.loads((d / "config.json").read_text())
        bad = list(cfg["layer_types"])
        bad[1] = "global_attention"
        edit_config(d, layer_types=bad)
        self.refused(d, "layer_types[1]")
        d = self.copy()
        edit_config(d, layer_types=cfg["layer_types"][:5])
        self.refused(d, "layer_types")

    def test_norm_topk_prob_true(self):
        d = self.copy(); edit_config(d, norm_topk_prob=True)
        self.refused(d, "norm_topk_prob")

    def test_tie_word_embeddings_true(self):
        d = self.copy(); edit_config(d, tie_word_embeddings=True)
        self.refused(d, "tie_word_embeddings")

    def test_other_semantic_config_refusals(self):
        for key, value, needle in (("attention_bias", True, "attention_bias"),
                                   ("hidden_act", "gelu", "hidden_act"),
                                   ("rope_scaling", {"type": "yarn", "factor": 4.0}, "rope_scaling"),
                                   ("model_type", "qwen3", "model_type"),
                                   ("num_experts_per_tok", 99, "num_experts_per_tok"),
                                   ("use_sliding_window", False, "use_sliding_window"),
                                   ("sliding_window", 0, "sliding_window")):
            with self.subTest(key=key):
                d = self.copy(); edit_config(d, **{key: value})
                self.refused(d, needle)
        d = self.copy()
        edit_config(d, quantization_config={"quant_method": "fp8", "weight_block_size": [64, 64]})
        self.refused(d, "weight_block_size")

    # ---- tensor refusals ----------------------------------------------------------

    def test_missing_tensor(self):
        d = self.copy()
        edit(d, "model.layers.2.mlp.experts.3.up_proj.weight", lambda e: None)
        # the index still lists it: both views name it; the loader reports the tensor
        edit_index(d, lambda i: i["weight_map"].pop("model.layers.2.mlp.experts.3.up_proj.weight"))
        r = run(d)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("model.layers.2.mlp.experts.3.up_proj.weight", r.stderr)

    def test_extra_tensor(self):
        d = self.copy()
        shard = sorted(d.glob("*.safetensors"))[0]
        entries = read_shard(shard)
        entries.append(("model.mtp.0.proj.weight", {"dtype": "F32", "shape": [2, 2]}, bytes(16)))
        write_shard(shard, entries)
        edit_index(d, lambda i: i["weight_map"].update({"model.mtp.0.proj.weight": shard.name}))
        self.refused(d, "model.mtp.0.proj.weight")

    def test_bf16_scale(self):
        d = self.copy()
        name = "model.layers.1.self_attn.k_proj.weight_scale_inv"
        edit(d, name, lambda e: (e[0], {"dtype": "BF16", "shape": e[1]["shape"]}, e[2][: len(e[2]) // 2]))
        self.refused(d, name)
        r = run(d)
        self.assertIn("F32", r.stderr)

    def test_transposed_scale_grid(self):
        d = self.copy()
        name = "model.layers.3.self_attn.q_proj.weight_scale_inv"      # [8,2], not square
        edit(d, name, lambda e: (e[0], {"dtype": "F32", "shape": e[1]["shape"][::-1]}, e[2]))
        r = self.refused(d, name)
        self.assertIn("TRANSPOSED", r.stderr)

    def test_nan_scale(self):
        d = self.copy()
        name = "model.layers.4.mlp.experts.5.gate_proj.weight_scale_inv"
        edit(d, name, lambda e: (e[0], e[1], struct.pack("<f", float("nan")) + e[2][4:]))
        r = self.refused(d, name)
        self.assertIn("non-finite", r.stderr)
        d = self.copy()
        name = "model.layers.0.self_attn.v_proj.weight_scale_inv"
        edit(d, name, lambda e: (e[0], e[1], e[2][:4] + struct.pack("<f", float("inf")) + e[2][8:]))
        self.refused(d, name)

    def test_wrong_shape(self):
        d = self.copy()
        name = "model.layers.2.self_attn.v_proj.weight"                # [256,256] -> [512,128], same bytes
        edit(d, name, lambda e: (e[0], {"dtype": e[1]["dtype"], "shape": [512, 128]}, e[2]))
        self.refused(d, name)

    def test_f32_projection(self):
        d = self.copy()
        name = "model.layers.5.self_attn.o_proj.weight"
        edit(d, name, lambda e: (e[0], {"dtype": "F32", "shape": e[1]["shape"]}, bytes(4 * len(e[2]))))
        r = self.refused(d, name)
        self.assertIn("F32", r.stderr)

    def test_bf16_where_f8_is_expected(self):
        d = self.copy()
        name = "model.layers.0.mlp.shared_experts.up_proj.weight"
        edit(d, name, lambda e: (e[0], {"dtype": "BF16", "shape": e[1]["shape"]}, bytes(2 * len(e[2]))))
        self.refused(d, name)

    def test_vendor_all_f32_fixture_is_refused(self):
        d = self.copy("simple")
        name = "model.embed_tokens.weight"
        edit(d, name, lambda e: (e[0], {"dtype": "F32", "shape": e[1]["shape"]}, bytes(2 * len(e[2]))))
        self.refused(d, name)

    # ---- duplicate names, overlays and the index (generic vs --release) --------------

    def duplicate(self, d, name, into):
        """Copy tensor `name` also into shard `into` (an overlay candidate)."""
        src = find_shard(d, name)
        blob = [e for e in read_shard(src) if e[0] == name][0]
        dst = d / into
        entries = read_shard(dst)
        entries.append(blob)
        write_shard(dst, entries)
        return src.name

    def test_duplicate_name_without_index_is_refused(self):
        d = self.copy()
        (d / "model.safetensors.index.json").unlink()
        name = "model.layers.0.input_layernorm.weight"
        src = self.duplicate(d, name, "model-00004-of-00004.safetensors")
        self.assertNotEqual(src, "model-00004-of-00004.safetensors")
        self.refused(d, f"duplicate tensor name '{name}'")
        self.refused(d, "--release requires model.safetensors.index.json", "--release")

    def test_overlay_chosen_by_the_index_generic_allows_release_refuses(self):
        d = self.copy()
        name = "model.layers.0.input_layernorm.weight"
        self.duplicate(d, name, "model-00004-of-00004.safetensors")
        r = run(d)                                   # index names the original shard: st.h's overlay rule
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("present in two shards", r.stderr)
        r = self.refused(d, f"duplicate tensor name '{name}'", "--release")
        self.assertIn("overlay", r.stderr)
        # the index naming the overlay copy instead is the same story
        edit_index(d, lambda i: i["weight_map"].update({name: "model-00004-of-00004.safetensors"}))
        self.assertEqual(run(d).returncode, 0)
        self.refused(d, "duplicate tensor name", "--release")

    def test_duplicate_whose_index_names_a_third_shard_is_refused_everywhere(self):
        d = self.copy()
        name = "model.layers.0.input_layernorm.weight"
        src = self.duplicate(d, name, "model-00004-of-00004.safetensors")
        third = [p.name for p in sorted(d.glob("*.safetensors")) if p.name not in (src, "model-00004-of-00004.safetensors")][0]
        edit_index(d, lambda i: i["weight_map"].update({name: third}))
        self.refused(d, "duplicate tensor name")
        self.refused(d, "duplicate tensor name", "--release")

    def test_index_entry_naming_the_wrong_shard(self):
        d = self.copy()
        name = "model.layers.2.mlp.gate.weight"
        right = find_shard(d, name).name
        wrong = [p.name for p in sorted(d.glob("*.safetensors")) if p.name != right][0]
        edit_index(d, lambda i: i["weight_map"].update({name: wrong}))
        r = self.refused(d, name)
        self.assertIn(wrong, r.stderr)
        self.refused(d, name, "--release")

    def test_index_entry_naming_a_shard_that_is_not_there(self):
        d = self.copy()
        name = "model.layers.2.mlp.gate.weight"
        edit_index(d, lambda i: i["weight_map"].update({name: "model-00009-of-00009.safetensors"}))
        self.refused(d, name)

    def test_index_missing_an_entry_and_stale_entry(self):
        d = self.copy()
        name = "model.layers.1.post_ffn_norm.weight"
        edit_index(d, lambda i: i["weight_map"].pop(name))
        self.refused(d, name)
        d = self.copy()
        edit_index(d, lambda i: i["weight_map"].update({"model.layers.9.ghost.weight": "model-00001-of-00004.safetensors"}))
        self.refused(d, "model.layers.9.ghost.weight")

    def test_release_requires_the_index(self):
        d = self.copy()
        (d / "model.safetensors.index.json").unlink()
        r = run(d)                                   # generic: no index is fine (st.h scans the headers)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.refused(d, "model.safetensors.index.json", "--release")


if __name__ == "__main__":
    unittest.main()
