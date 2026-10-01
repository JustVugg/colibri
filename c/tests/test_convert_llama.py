"""convert_llama.py must accept exactly the tensor inventory llama.c loads and
refuse anything else: a wrong shape, a short byte span, a missing tensor, a
tensor llama.c would never read, or a non-llama config."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import convert_llama  # noqa: E402

CONFIG = {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1,
          "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
          "intermediate_size": 16, "vocab_size": 32, "tie_word_embeddings": False}


def write_shard(path, tensors):
    header, off = {}, 0
    for name, (dtype, shape, nbytes) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + nbytes]}
        off += nbytes
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * off)


def good_tensors(config):
    return {name: ("BF16", shape, 2 * _numel(shape))
            for name, shape in convert_llama.expected_tensors(config).items()}


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


class ConvertLlamaTest(unittest.TestCase):
    def check(self, config, tensors):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "config.json").write_text(json.dumps(config))
            write_shard(d / "model.safetensors", tensors)
            return convert_llama.validate(d)[2]

    def test_exact_inventory_is_accepted_and_tied_head_is_optional(self):
        self.assertEqual(self.check(CONFIG, good_tensors(CONFIG)), 12)
        tied = dict(CONFIG, tie_word_embeddings=True)
        self.assertEqual(self.check(tied, good_tensors(tied)), 11)
        # A tied checkpoint may still carry lm_head.weight; llama.c ignores it.
        self.assertEqual(self.check(tied, good_tensors(CONFIG)), 11)

    def test_every_mismatch_is_refused_by_name(self):
        k = "model.layers.0.self_attn.k_proj.weight"
        cases = {
            "shape": {k: ("BF16", [8, 8], 128)},              # q-sized k (no GQA)
            "span": {k: ("BF16", [4, 8], 62)},                # one byte short
            "dtype": {k: ("I8", [4, 8], 32)},
            "extra": {"model.layers.0.mlp.gate.weight": ("BF16", [2, 8], 32)},
        }
        for label, change in cases.items():
            tensors = good_tensors(CONFIG)
            tensors.update(change)
            with self.subTest(label), self.assertRaisesRegex(convert_llama.ConvertError,
                                                             list(change)[0].replace(".", r"\.")):
                self.check(CONFIG, tensors)
        tensors = good_tensors(CONFIG)
        del tensors["lm_head.weight"]
        with self.assertRaisesRegex(convert_llama.ConvertError, "missing"):
            self.check(CONFIG, tensors)
        with self.assertRaisesRegex(convert_llama.ConvertError, "not 'llama'"):
            self.check(dict(CONFIG, model_type="mistral"), good_tensors(CONFIG))


if __name__ == "__main__":
    unittest.main()
