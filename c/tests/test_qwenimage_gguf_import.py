"""tools/gguf_import.py on Qwen-Image-2.1's GGUFs, from their headers alone (GGUF_CHECK).

unsloth's denoisers carry no metadata: the importer must recognize one by its tensors and write
the diffusers folder around it (the transformer's config from the shapes, the base model's or
Turbo's schedule, the text encoder from a Qwen3-VL GGUF with Qwen-Image's own tokenizer split,
the VAE), and say plainly what is missing or wrong. The engine's run on such a folder is
tests/test_qwenimage_gguf_tiny.py; this one needs only the standard library.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
IMPORT = HERE / "tools" / "gguf_import.py"
sys.path.insert(0, str(HERE))
import image_engine  # noqa: E402

Q4_K, Q8_0, F32, BF16 = 12, 8, 0, 30


def nbytes(shape, ty):
    n = 1
    for d in shape:
        n *= d
    blck, size = {Q4_K: (256, 144), Q8_0: (32, 34), F32: (1, 4), BF16: (1, 2)}[ty]
    return n // blck * size


def header_gguf(path, kv, tensors):
    """The header of a GGUF v3 (no tensor data: GGUF_CHECK). kv: [(key, type, value)];
    tensors: [(name, shape rows first, ggml type)]."""
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def val(t, v):
        if t == 8:
            return s(v)
        if t == 9:
            et, items = v
            return struct.pack("<IQ", et, len(items)) + b"".join(val(et, i) for i in items)
        return struct.pack({4: "<I", 6: "<f", 7: "<?"}[t], v)
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, t, v in kv:
        out += s(k) + struct.pack("<I", t) + val(t, v)
    off = 0
    for name, shape, ty in tensors:
        dims = list(reversed(shape))
        out += s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        off = (off + nbytes(shape, ty) + 31) // 32 * 32
    Path(path).write_bytes(out)


def denoiser(path, prefix="model.diffusion_model.", blocks=2, dim=256, fused=True):
    p, M = prefix, 3 * dim
    t = [(p + "img_in.weight", (dim, 64), BF16), (p + "txt_in.in_layer.weight", (dim, 256), BF16),
         (p + "txt_in.out_layer.weight", (dim, dim), BF16), (p + "txt_in.text_norm.weight", (256,), BF16),
         (p + "modulation.1.weight", (4 * dim, dim), Q4_K), (p + "proj_out.weight", (64, dim), Q8_0),
         (p + "norm_out.linear.weight", (dim, dim), F32),
         (p + "time_text_embed.timestep_embedder.linear_1.weight", (dim, 256), Q8_0),
         (p + "time_text_embed.timestep_embedder.linear_2.weight", (dim, dim), Q8_0)]
    for l in range(blocks):
        b = f"{p}transformer_blocks.{l}."
        t += [(b + n, (dim, dim), Q4_K) for n in ("attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "attn.to_out.0.weight")]
        t += [(b + "attn.norm_q.weight", (128,), F32), (b + "attn.norm_k.weight", (128,), F32), (b + "img_mlp.out.weight", (dim, M), Q4_K)]
        t += [(b + "img_mlp.gate_up.weight", (2 * M, dim), Q4_K)] if fused else \
             [(b + "img_mlp.gate_layer.weight", (M, dim), Q4_K), (b + "img_mlp.proj.weight", (M, dim), Q4_K)]
    header_gguf(path, [], t)


def text_encoder(path, hidden=256, arch="qwen3vl"):
    toks = [chr(0x100 + i) for i in range(250)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [f"[PAD{i}]" for i in range(3)]
    types = [1] * 250 + [3, 3, 3] + [1, 1, 1]
    p = arch + "."
    kv = [("general.architecture", 8, arch), (p + "block_count", 4, 2), (p + "embedding_length", 4, hidden),
          (p + "feed_forward_length", 4, 512), (p + "attention.head_count", 4, 2), (p + "attention.head_count_kv", 4, 1),
          (p + "attention.key_length", 4, 128), (p + "rope.freq_base", 6, 5000000.0),
          (p + "attention.layer_norm_rms_epsilon", 6, 1e-6),
          ("tokenizer.ggml.model", 8, "gpt2"), ("tokenizer.ggml.pre", 8, "qwen2"), ("tokenizer.ggml.tokens", 9, (8, toks)),
          ("tokenizer.ggml.token_type", 9, (4, types)), ("tokenizer.ggml.merges", 9, (8, [])),
          ("tokenizer.ggml.eos_token_id", 4, 252)]
    t = [("token_embd.weight", (256, hidden), Q4_K), ("output_norm.weight", (hidden,), F32)]
    for l in range(2):
        t += [(f"blk.{l}.attn_q.weight", (256, hidden), Q4_K), (f"blk.{l}.attn_norm.weight", (hidden,), F32)]
    header_gguf(path, kv, t)


def vae_file(path, layout="diffusers", z=64):
    if layout == "diffusers":
        head = {"decoder.conv_in.weight": {"dtype": "BF16", "shape": [1152, z, 3, 3], "data_offsets": [0, 0]},
                "post_quant_conv.weight": {"dtype": "BF16", "shape": [z, z, 1, 1], "data_offsets": [0, 0]}}
    else:
        head = {"decoder.conv1.weight": {"dtype": "BF16", "shape": [1152, z, 1, 3, 3], "data_offsets": [0, 0]}}
    h = json.dumps(head).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h)


class QwenImageImport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def run_import(self, gguf, out, *extra):
        return subprocess.run([sys.executable, str(IMPORT), str(gguf), str(out), *map(str, extra)],
                              env=dict(os.environ, GGUF_CHECK="1"), capture_output=True, text=True)

    def test_folder_from_the_denoiser_and_its_neighbours(self):
        denoiser(self.d / "qwen-image-2.1-Q4_K_M.gguf")
        text_encoder(self.d / "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf")
        vae_file(self.d / "qwen_image_2.1_vae_bf16.safetensors")
        p = self.run_import(self.d / "qwen-image-2.1-Q4_K_M.gguf", self.d / "out")
        self.assertEqual(p.returncode, 0, p.stderr)
        out = self.d / "out"
        index = json.loads((out / "model_index.json").read_text())
        self.assertEqual(index["_class_name"], "QwenImage21Pipeline")
        self.assertNotIn("sample_sigmas", index)
        self.assertEqual(index["colibri_gguf"]["variant"], "base")
        cfg = json.loads((out / "transformer" / "config.json").read_text())
        self.assertEqual((cfg["num_layers"], cfg["num_attention_heads"], cfg["attention_head_dim"], cfg["mlp_ratio"],
                          cfg["in_channels"], cfg["context_in_dim"], cfg["axes_dims_rope"]), (2, 2, 128, 3, 64, 256, [16, 56, 56]))
        link = json.loads((out / "transformer" / "gguf.json").read_text())
        self.assertEqual(link["arch"], "qwen_image")   # the file says nothing: st_gguf.h reads it from here
        sched = json.loads((out / "scheduler" / "scheduler_config.json").read_text())
        self.assertTrue(sched["use_dynamic_shifting"])
        self.assertEqual(sched["shift_terminal"], 0.02)
        te = json.loads((out / "text_encoder" / "config.json").read_text())["text_config"]
        self.assertEqual((te["hidden_size"], te["num_hidden_layers"], te["num_key_value_heads"], te["head_dim"],
                          te["vocab_size"], te["rms_norm_eps"]), (256, 2, 1, 128, 256, 1e-6))
        self.assertEqual(json.loads((out / "text_encoder" / "gguf.json").read_text())["arch"], "qwen3vl")
        tok = json.loads((out / "processor" / "tokenizer.json").read_text())
        self.assertEqual(tok["normalizer"], {"type": "NFC"})
        # Qwen's split, digits one at a time (what tok.h keys ndig1 on), not GPT-2's bare ByteLevel
        self.assertIn("|\\p{N}|", tok["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"])
        self.assertEqual([a["content"] for a in tok["added_tokens"]], ["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
        vae = json.loads((out / "vae" / "config.json").read_text())
        self.assertEqual((vae["z_dim"], len(vae["latents_mean"]), len(vae["latents_std"])), (64, 64, 64))
        self.assertTrue((out / "vae" / "diffusion_pytorch_model.safetensors").is_file())
        self.assertTrue(image_engine.is_image_model_dir(out))

    def test_turbo_from_the_name_and_by_flag(self):
        denoiser(self.d / "qwen-image-2.1-turbo-Q2_K.gguf")
        denoiser(self.d / "plain.gguf", prefix="")   # the checkpoint's names without unsloth's prefix
        text_encoder(self.d / "te.gguf")
        vae_file(self.d / "vae.safetensors")
        for src, extra, turbo in (("qwen-image-2.1-turbo-Q2_K.gguf", (), True), ("plain.gguf", (), False),
                                  ("plain.gguf", ("--variant", "turbo"), True),
                                  ("qwen-image-2.1-turbo-Q2_K.gguf", ("--variant", "base"), False)):
            out = self.d / f"out_{src}_{len(extra)}"
            p = self.run_import(self.d / src, out, "--text-encoder", self.d / "te.gguf", "--vae", self.d / "vae.safetensors", *extra)
            self.assertEqual(p.returncode, 0, p.stderr)
            index = json.loads((out / "model_index.json").read_text())
            sched = json.loads((out / "scheduler" / "scheduler_config.json").read_text())
            self.assertEqual("sample_sigmas" in index, turbo, (src, extra))
            self.assertEqual(sched["use_dynamic_shifting"], not turbo)
            if turbo:
                self.assertEqual(index["sample_sigmas"][0], 1.0)
                self.assertEqual(len(index["sample_sigmas"]), 8)
                self.assertIsNone(sched["shift_terminal"])

    def test_split_gate_and_proj_are_read_too(self):
        denoiser(self.d / "d.gguf", fused=False)
        text_encoder(self.d / "te.gguf")
        vae_file(self.d / "vae.safetensors")
        p = self.run_import(self.d / "d.gguf", self.d / "out", "--text-encoder", self.d / "te.gguf", "--vae", self.d / "vae.safetensors")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(json.loads((self.d / "out" / "transformer" / "config.json").read_text())["mlp_ratio"], 3)

    def test_what_is_missing_or_wrong_is_said(self):
        denoiser(self.d / "d.gguf")
        p = self.run_import(self.d / "d.gguf", self.d / "o1")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("Qwen3-VL-8B-Instruct", p.stderr)
        text_encoder(self.d / "te.gguf")
        p = self.run_import(self.d / "d.gguf", self.d / "o2", "--text-encoder", self.d / "te.gguf")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("unsloth/Qwen-Image-2.1-FP8", p.stderr)
        vae_file(self.d / "comfy.safetensors", layout="comfy")
        p = self.run_import(self.d / "d.gguf", self.d / "o3", "--text-encoder", self.d / "te.gguf", "--vae", self.d / "comfy.safetensors")
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("ComfyUI", p.stderr)
        vae_file(self.d / "z16.safetensors", z=16)
        p = self.run_import(self.d / "d.gguf", self.d / "o4", "--text-encoder", self.d / "te.gguf", "--vae", self.d / "z16.safetensors")
        self.assertIn("z_dim 16", p.stderr)
        text_encoder(self.d / "wide.gguf", hidden=512)
        p = self.run_import(self.d / "d.gguf", self.d / "o5", "--text-encoder", self.d / "wide.gguf")
        self.assertIn("512 wide", p.stderr)
        text_encoder(self.d / "q3.gguf", arch="qwen3")
        p = self.run_import(self.d / "d.gguf", self.d / "o6", "--text-encoder", self.d / "q3.gguf")
        self.assertIn("not qwen3vl", p.stderr)
        p = self.run_import(self.d / "te.gguf", self.d / "o7")   # the text encoder on its own
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("--text-encoder", p.stderr)
        p = self.run_import(self.d / "d.gguf", self.d / "o8", "--mode", "copy", "--text-encoder", self.d / "te.gguf")
        self.assertIn("--mode link", p.stderr)

    def test_official_folder_as_text_encoder_and_vae(self):
        off = self.d / "Qwen-Image-2.1"
        for sub in ("text_encoder", "processor", "vae", "transformer", "scheduler"):
            (off / sub).mkdir(parents=True)
        (off / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
        (off / "vae" / "config.json").write_text("{}")
        (off / "processor" / "tokenizer.json").write_text("{}")
        denoiser(self.d / "d.gguf")
        p = self.run_import(self.d / "d.gguf", self.d / "out", "--text-encoder", off)
        self.assertEqual(p.returncode, 0, p.stderr)
        index = json.loads((self.d / "out" / "model_index.json").read_text())
        parts = index["colibri_parts"]
        self.assertEqual(Path(parts["text_encoder"]), (off / "text_encoder").resolve())
        self.assertEqual(Path(parts["processor"]), (off / "processor").resolve())
        self.assertEqual(Path(parts["vae"]), (off / "vae").resolve())   # the same folder's VAE, unasked
        self.assertFalse((self.d / "out" / "text_encoder").exists())
        self.assertEqual(image_engine._part_dir(self.d / "out", index, "vae"), (off / "vae").resolve())
        self.assertEqual(image_engine._part_dir(self.d / "out", index, "transformer"), self.d / "out" / "transformer")

    def test_plan_reads_the_gguf_parts(self):
        denoiser(self.d / "d.gguf")
        text_encoder(self.d / "te.gguf")
        vae_file(self.d / "vae.safetensors")
        p = self.run_import(self.d / "d.gguf", self.d / "out", "--text-encoder", self.d / "te.gguf", "--vae", self.d / "vae.safetensors")
        self.assertEqual(p.returncode, 0, p.stderr)
        old = os.environ.get("GGUF_CHECK")
        os.environ["GGUF_CHECK"] = "1"
        try:
            inv = image_engine.inspect_image_model(self.d / "out")
        finally:
            if old is None:
                del os.environ["GGUF_CHECK"]
            else:
                os.environ["GGUF_CHECK"] = old
        dit, te = inv["components"]["transformer"], inv["components"]["text_encoder"]
        self.assertEqual((dit["precision"], te["precision"]), ("gguf", "gguf"))
        self.assertTrue(dit["gguf"].endswith("d.gguf"))
        # Q4_K blocks plus a scale a row for every 256-wide Q4_K matrix, int8 rows for the rest
        q4 = sum(nbytes((r, c), Q4_K) + 4 * r for r, c in [(1024, 256)] + [(256, 256)] * 8 + [(256, 768)] * 2 + [(1536, 256)] * 2)
        int8 = sum(r * c + 4 * r for r, c in [(256, 64), (256, 256), (256, 256), (64, 256), (256, 256), (256, 256), (256, 256)])
        vec = 4 * (256 + 4 * 128)
        self.assertEqual(dit["resident_bytes"], q4 + int8 + vec)
        # the text encoder's embedding rows are read per token, never held
        self.assertEqual(te["resident_bytes"], nbytes((256, 256), Q4_K) * 2 + 4 * 256 * 2 + 4 * 256 * 3)
        self.assertTrue(inv["research_license"])
        self.assertTrue(inv["tokenizer"])


if __name__ == "__main__":
    unittest.main()
