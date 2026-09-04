"""make_qwen36_qpack_snap: snap view over a qpack container, plus the C round-trip.

Builds a synthetic MLX-affine container (nested multimodal config.json with
per-module quantization overrides, a hand-written model.safetensors with
packed U32 triples) and checks:

  * the view's config.json is the FLATTENED text config (load_cfg reads
    hidden_size at the top level);
  * qwen36_meta.json derives head dims from the LOGICAL weight shapes --
    o_proj's input axis is stored packed (uint32 words), so a derivation
    that forgets to unpack reports o_in 8x too small and fails here;
  * layer_types / n_active / DeltaNet dims / rope_theta (nested under
    rope_parameters) survive the trip;
  * model.safetensors and tokenizer.json are links into the container, and
    the container itself is never written;
  * refusal when --out points inside the container.

Round-trip (the independent cross-implementation evidence): the fixture
carries a seeded-random Q4 tensor whose dequantized rows are computed HERE in
pure Python (bf16 scalars decoded exactly, scale*q + bias per group) and
handed to tests/test_qwen36_dense_affine's compare mode, which loads the SAME
bytes through the engine's own st_init + load_t_n path -- through the snap
view's symlink, exactly as a real run reads them.  Skips loudly if the C gate
binary is not built (`make test-c` builds it first in a green `make test`).
"""
import json
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_DIR = HERE.parent
TOOL = C_DIR / "tools" / "make_qwen36_qpack_snap.py"
GATE = C_DIR / "tests" / ("test_qwen36_dense_affine.exe" if os.name == "nt"
                          else "test_qwen36_dense_affine")


def bf16_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0] >> 16


def bf16_value(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits << 16))[0]


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]):
    header, offset = {}, 0
    for name, dtype, shape, blob in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    encoded = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for _, _, _, blob in tensors:
            f.write(blob)


class MakeQwen36QpackSnapTest(unittest.TestCase):
    HIDDEN, N_LAYERS = 32, 4
    N_Q, N_KV, HEAD_DIM = 2, 1, 8
    Q_HEAD_DIM = 16          # attn_output_gate: q_proj carries value+gate
    GS = 16                  # container-wide affine group size

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.container = root / "tiny.qpack"
        cls.view = root / "snap-view"
        cls.container.mkdir()
        cls.rng = random.Random(42)
        cls._write_config()
        cls._write_model()
        (cls.container / "tokenizer.json").write_text(
            json.dumps({"model": {"vocab": {"a": 0}}}), encoding="utf-8")
        (cls.container / "manifest.json").write_text(
            json.dumps({"magic": "QPACK", "version": 1,
                        "modelName": "tiny", "sourceCheckpoint": "tiny"}),
            encoding="utf-8")
        cls.before = sorted(p.name for p in cls.container.iterdir())
        result = subprocess.run(
            [sys.executable, str(TOOL), "--container", str(cls.container),
             "--out", str(cls.view)],
            capture_output=True, text=True, cwd=C_DIR)
        assert result.returncode == 0, result.stderr + result.stdout
        cls.tool_output = result.stdout

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- fixture ---------------------------------------------------------

    @classmethod
    def _write_config(cls):
        text_config = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": cls.HIDDEN,
            "num_hidden_layers": cls.N_LAYERS,
            "vocab_size": 64,
            "rms_norm_eps": 1e-6,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "num_attention_heads": cls.N_Q,
            "num_key_value_heads": cls.N_KV,
            "head_dim": cls.HEAD_DIM,
            "attn_output_gate": True,
            "partial_rotary_factor": 0.25,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "linear_num_value_heads": 2,
            "linear_num_key_heads": 1,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "rope_parameters": {"rope_theta": 12345678.0,
                                "mrope_section": [2, 1, 1]},
        }
        config = {
            "model_type": "qwen3_5_moe",
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "text_config": text_config,
            "quantization": {
                "bits": 4, "group_size": cls.GS, "mode": "affine",
                "language_model.model.layers.3.mlp.gate":
                    {"bits": 8, "group_size": cls.GS},
            },
        }
        (cls.container / "config.json").write_text(json.dumps(config),
                                                   encoding="utf-8")

    @classmethod
    def _quantized(cls, name: str, rows: int, cols: int, bits: int):
        """Author a random affine triple; returns (tensors, logical rows)."""
        per_word, mask = 32 // bits, (1 << bits) - 1
        words = [cls.rng.getrandbits(32) for _ in range(rows * cols // per_word)]
        groups = cols // cls.GS
        scales = [bf16_bits(cls.rng.uniform(0.01, 0.2))
                  for _ in range(rows * groups)]
        biases = [bf16_bits(cls.rng.uniform(-1.0, 1.0))
                  for _ in range(rows * groups)]
        logical = []
        for r in range(rows):
            row = []
            for c in range(cols):
                word = words[r * (cols // per_word) + c // per_word]
                q = (word >> (bits * (c % per_word))) & mask
                g = r * groups + c // cls.GS
                row.append(bf16_value(scales[g]) * q + bf16_value(biases[g]))
            logical.append(row)
        tensors = [
            (name + ".weight", "U32", [rows, cols // per_word],
             struct.pack(f"<{len(words)}I", *words)),
            (name + ".scales", "BF16", [rows, groups],
             struct.pack(f"<{len(scales)}H", *scales)),
            (name + ".biases", "BF16", [rows, groups],
             struct.pack(f"<{len(biases)}H", *biases)),
        ]
        return tensors, logical

    @classmethod
    def _write_model(cls):
        p = "language_model.model.layers.3.self_attn."
        tensors = []
        for proj, rows in (("q_proj", cls.N_Q * cls.Q_HEAD_DIM),
                           ("k_proj", cls.N_KV * cls.HEAD_DIM),
                           ("v_proj", cls.N_KV * cls.HEAD_DIM)):
            t, _ = cls._quantized(p + proj, rows, cls.HIDDEN, bits=4)
            tensors += t
        # o_proj input axis = n_q*head_dim = 16 LOGICAL columns -> 2 packed
        # words; the o_in derivation must report 16, not 2.
        t, _ = cls._quantized(p + "o_proj", cls.HIDDEN,
                              cls.N_Q * cls.HEAD_DIM, bits=4)
        tensors += t
        tensors.append((p + "q_norm.weight", "BF16", [cls.HEAD_DIM],
                        struct.pack(f"<{cls.HEAD_DIM}H",
                                    *[bf16_bits(1.0)] * cls.HEAD_DIM)))
        # Q8 override module: 8 experts x hidden router.
        t, _ = cls._quantized("language_model.model.layers.3.mlp.gate",
                              8, cls.HIDDEN, bits=8)
        tensors += t
        # Round-trip tensor for the C compare-mode gate.
        t, cls.roundtrip = cls._quantized("language_model.model.roundtrip",
                                          4, cls.HIDDEN, bits=4)
        tensors += t
        write_safetensors(cls.container / "model.safetensors", tensors)

    # ---- gates -----------------------------------------------------------

    def test_flat_config(self):
        cfg = json.loads((self.view / "config.json").read_text())
        self.assertEqual(cfg["hidden_size"], self.HIDDEN)
        self.assertEqual(cfg["num_hidden_layers"], self.N_LAYERS)
        self.assertEqual(cfg["vocab_size"], 64)
        self.assertNotIn("text_config", cfg)
        hf = json.loads((self.view / "config.hf.json").read_text())
        self.assertIn("text_config", hf)

    def test_meta_head_dims_unpacked(self):
        meta = json.loads((self.view / "qwen36_meta.json").read_text())
        self.assertEqual(meta["q_heads"], self.N_Q)
        self.assertEqual(meta["kv_heads"], self.N_KV)
        self.assertEqual(meta["q_head_dim"], self.Q_HEAD_DIM)
        self.assertEqual(meta["k_head_dim"], self.HEAD_DIM)
        self.assertEqual(meta["v_head_dim"], self.HEAD_DIM)
        self.assertEqual(meta["o_in"], self.N_Q * self.HEAD_DIM)  # unpacked!
        self.assertEqual(meta["head_dim"], self.HEAD_DIM)
        self.assertEqual(meta["qk_rope_head_dim"], self.HEAD_DIM)

    def test_meta_layers_dn_rope(self):
        meta = json.loads((self.view / "qwen36_meta.json").read_text())
        self.assertEqual(meta["n_layers"], self.N_LAYERS)
        self.assertEqual(meta["n_active"], 1)
        self.assertEqual(meta["layer_types"],
                         ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(meta["num_experts"], 8)
        self.assertEqual(meta["topk"], 2)
        self.assertEqual(meta["moe_inter"], 16)
        self.assertEqual(meta["dn_vheads"], 2)
        self.assertEqual(meta["dn_kheads"], 1)
        self.assertEqual(meta["dn_conv_dim"], 2 * 1 * 8 + 2 * 8)
        self.assertEqual(meta["rope_theta"], 12345678.0)
        self.assertTrue(meta["attn_output_gate"])
        self.assertEqual(meta["expert_gs"], 0)
        # MLX-derived containers carry full-gamma norm weights; the engine
        # must be told to undo the shift (load_norm_n) or the model is noise.
        self.assertIs(meta["zero_centered_norms"], False)

    def test_links_and_readonly_container(self):
        for name in ("model.safetensors", "tokenizer.json"):
            link = self.view / name
            self.assertTrue(link.exists(), name)
            self.assertEqual(os.stat(link).st_ino,
                             os.stat(self.container / name).st_ino,
                             f"{name} is a copy, not a link")
        after = sorted(p.name for p in self.container.iterdir())
        self.assertEqual(self.before, after, "container was written to")

    def test_refuses_out_inside_container(self):
        result = subprocess.run(
            [sys.executable, str(TOOL), "--container", str(self.container),
             "--out", str(self.container / "view")],
            capture_output=True, text=True, cwd=C_DIR)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("OUTSIDE", result.stderr + result.stdout)

    def test_roundtrip_against_engine_loader(self):
        if not GATE.exists():
            self.skipTest(f"{GATE} not built (run `make test-c` first): the "
                          "C-vs-Python round-trip was NOT checked")
        expected = struct.pack(
            f"<{4 * self.HIDDEN}f",
            *[v for row in self.roundtrip for v in row])
        expected_path = Path(self.tmp.name) / "roundtrip.f32"
        expected_path.write_bytes(expected)
        result = subprocess.run(
            [str(GATE), str(self.view), "model.roundtrip.weight",
             str(4 * self.HIDDEN), str(expected_path)],
            capture_output=True, text=True, cwd=C_DIR)
        self.assertEqual(result.returncode, 0,
                         result.stderr + result.stdout)
        self.assertIn("round-trip", result.stdout)


if __name__ == "__main__":
    unittest.main()
