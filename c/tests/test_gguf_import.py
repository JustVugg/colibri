"""GGUF -> colibri directory: both modes give the same model, the planner agrees.

A small but complete qwen35moe GGUF is written with gguf-py (every tensor the
engine reads, mixed ggml types as a UD file has them, a byte tokenizer), then
imported with tools/gguf_import.py in link and in copy mode. The engine must
produce the same tokens from the bare file, the link directory and the copy
container, on the float and on the integer path; the planner must see the
same experts in both directories; and in the container every expert slot must
start on a 4 KiB boundary of the file.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:   # the CI's Python tests run without numpy
    raise unittest.SkipTest("numpy not installed: %s" % exc)

HERE = Path(__file__).resolve().parent.parent
ENGINE = HERE / ("qwen36.exe" if sys.platform == "win32" else "qwen36")
sys.path.insert(0, str(HERE / "tests"))
sys.path.insert(0, str(HERE))
try:
    import gguf
    from gguf.constants import GGMLQuantizationType as QT
    from test_ggml_quants import make_blocks, TYPES
except ImportError:  # pragma: no cover
    gguf = None

H, LAYERS, E, TOPK, I, VOCAB = 256, 4, 8, 2, 256, 300
QH, KVH, HD, KH, VH, KD = 4, 2, 64, 4, 8, 32


def blocks(name, shape, rng, std=0.02):
    """Synthetic blocks of `name` with realistic magnitude (centred)."""
    blck, size, f16 = TYPES[name]
    rows = int(np.prod(shape[:-1])); nb = shape[-1] // blck
    raw = make_blocks(name, rows * nb, rng)
    probe = gguf.quants.dequantize(raw.reshape(-1), QT[name])
    s = float(np.std(probe)) or 1.0
    for off in f16:
        v = np.frombuffer(raw[:, off:off + 2].tobytes(), dtype=np.float16).astype(np.float32) * (std / s)
        raw[:, off:off + 2] = v.astype(np.float16).view(np.uint8).reshape(-1, 2)
    return raw.reshape(*shape[:-1], nb * size)


def write_model(path, rng, dense=False, same_gate_up=False):
    """dense: llama.cpp's qwen35 (a dense model of the family): an MLP per layer instead of the
    experts, and no output.weight (the head tied to token_embd), as the small checkpoints ship.
    same_gate_up: every layer's expert up in its gate's type (layer 0 IQ2_XXS, the others
    Q4_K), the pairing the Vulkan tier keeps as the file's blocks."""
    arch = "qwen35" if dense else "qwen35moe"
    w = gguf.GGUFWriter(str(path), arch)
    p = arch + "."
    moe = (("feed_forward_length", I),) if dense else \
        (("expert_count", E), ("expert_used_count", TOPK), ("expert_feed_forward_length", I), ("expert_shared_feed_forward_length", I))
    for k, v in (("block_count", LAYERS), ("embedding_length", H), ("attention.head_count", QH),
                 ("attention.head_count_kv", KVH), ("attention.key_length", HD), ("attention.value_length", HD),
                 *moe, ("full_attention_interval", 4),
                 ("rope.dimension_count", HD // 4), ("ssm.conv_kernel", 4), ("ssm.state_size", KD),
                 ("ssm.group_count", KH), ("ssm.time_step_rank", VH), ("ssm.inner_size", VH * KD),
                 ("context_length", 4096)):
        w.add_uint32(p + k, v)
    w.add_float32(p + "rope.freq_base", 1e7); w.add_float32(p + "attention.layer_norm_rms_epsilon", 1e-6)
    toks = [f"<0x{b:02X}>" if b < 256 else f"t{b}" for b in range(VOCAB - 3)] + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    w.add_tokenizer_model("gpt2"); w.add_token_list(toks)
    w.add_token_types([1] * (VOCAB - 3) + [3, 3, 3]); w.add_token_merges([])
    w.add_eos_token_id(VOCAB - 1); w.add_bos_token_id(VOCAB - 3)

    def f32(name, shape): w.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))
    def q(name, shape, t): w.add_tensor(name, blocks(t, shape, rng), raw_dtype=QT[t])
    q("token_embd.weight", (VOCAB, H), "Q8_0")
    if not dense:
        q("output.weight", (VOCAB, H), "Q6_K")
    w.add_tensor("output_norm.weight", np.ones(H, np.float32))
    conv = 2 * KH * KD + VH * KD
    for L in range(LAYERS):
        b = f"blk.{L}."
        w.add_tensor(b + "attn_norm.weight", np.ones(H, np.float32))
        w.add_tensor(b + "post_attention_norm.weight", np.ones(H, np.float32))
        if dense:
            q(b + "ffn_gate.weight", (I, H), "Q4_K"); q(b + "ffn_up.weight", (I, H), "Q4_K")
            q(b + "ffn_down.weight", (H, I), "Q6_K" if L % 2 else "Q5_K")
        else:
            f32(b + "ffn_gate_inp.weight", (E, H)); f32(b + "ffn_gate_inp_shexp.weight", (H,))
            q(b + "ffn_gate_shexp.weight", (I, H), "Q8_0"); q(b + "ffn_up_shexp.weight", (I, H), "Q8_0")
            q(b + "ffn_down_shexp.weight", (H, I), "Q8_0")
            q(b + "ffn_gate_exps.weight", (E, I, H), "IQ2_XXS" if L == 0 else "Q4_K")
            q(b + "ffn_up_exps.weight", (E, I, H), "IQ2_XXS" if same_gate_up and L == 0 else "Q4_K")
            q(b + "ffn_down_exps.weight", (E, H, I), "Q6_K" if L % 2 else "Q5_K")
        if (L + 1) % 4 == 0:
            q(b + "attn_q.weight", (QH * HD * 2, H), "Q8_0"); q(b + "attn_k.weight", (KVH * HD, H), "Q8_0")
            q(b + "attn_v.weight", (KVH * HD, H), "Q8_0"); q(b + "attn_output.weight", (H, QH * HD), "Q8_0")
            w.add_tensor(b + "attn_q_norm.weight", np.ones(HD, np.float32)); w.add_tensor(b + "attn_k_norm.weight", np.ones(HD, np.float32))
        else:
            q(b + "attn_qkv.weight", (conv, H), "Q8_0"); q(b + "attn_gate.weight", (VH * KD, H), "Q8_0")
            w.add_tensor(b + "ssm_a", -np.exp(rng.standard_normal(VH).astype(np.float32) * 0.1))
            f32(b + "ssm_alpha.weight", (VH, H)); f32(b + "ssm_beta.weight", (VH, H))
            f32(b + "ssm_conv1d.weight", (conv, 4)); w.add_tensor(b + "ssm_dt.bias", np.zeros(VH, np.float32))
            w.add_tensor(b + "ssm_norm.weight", np.ones(KD, np.float32))
            q(b + "ssm_out.weight", (H, VH * KD), "Q8_0")
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def engine_tokens(snap, ref, env=None):
    p = subprocess.run([str(ENGINE), "2", "8", str(ref)], env=dict(os.environ, SNAP=str(snap), OMP_NUM_THREADS="2",
                       COLI_NO_OMP_TUNE="1", **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace")
    line = [l for l in out.splitlines() if l.startswith("C engine")]
    if not line:
        raise AssertionError(f"engine produced no tokens (rc {p.returncode}):\n{out[-2000:]}")
    return line[-1].split(":", 1)[1].split("Text")[0].split()


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class GgufImport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = d / "tiny-qwen35moe.gguf"
        write_model(cls.src, np.random.default_rng(11))
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        for mode in ("link", "copy"):
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src), str(d / mode), "--mode", mode],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_directories_are_complete(self):
        for mode in ("link", "copy"):
            d = self.dir / mode
            for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "gguf.json"):
                self.assertTrue((d / f).is_file(), f"{mode}: {f}")
            cfg = json.loads((d / "config.json").read_text())
            self.assertEqual(cfg["model_type"], "qwen3_5_moe_text")
            self.assertEqual((cfg["num_hidden_layers"], cfg["num_experts"], cfg["hidden_size"]), (LAYERS, E, H))
            self.assertEqual(cfg["layer_types"][3], "full_attention")
            tok = json.loads((d / "tokenizer.json").read_text())
            self.assertNotIn("<|im_end|>", tok["model"]["vocab"])          # added tokens live in added_tokens only
            self.assertTrue(any(a["content"] == "<|im_end|>" and a["special"] for a in tok["added_tokens"]))
        self.assertTrue((self.dir / "copy" / "model.colibri.gguf").is_file())
        self.assertFalse((self.dir / "link" / "model.colibri.gguf").exists())

    def test_same_tokens_from_file_link_and_copy(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                a = engine_tokens(self.src, self.ref, env)
                self.assertEqual(engine_tokens(self.dir / "link", self.ref, env), a)
                self.assertEqual(engine_tokens(self.dir / "copy", self.ref, env), a)

    def test_header_check(self):
        """GGUF_CHECK=1 checks every tensor the loader wants without reading one: it used to
        crash on the final norm, whose 1 it took off a weight it had not read."""
        p = subprocess.run([str(ENGINE), "2", "8", str(self.ref)], env=dict(os.environ, SNAP=str(self.src), GGUF_CHECK="1",
                           OMP_NUM_THREADS="2", COLI_NO_OMP_TUNE="1"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = p.stdout.decode("utf-8", "replace")
        self.assertEqual(p.returncode, 0, out[-2000:])
        self.assertIn("GGUF CHECK OK", out)

    def test_container_slots_are_4k_aligned(self):
        r = gguf.GGUFReader(str(self.dir / "copy" / "model.colibri.gguf"))
        self.assertEqual(r.fields["colibri.layout"].contents(), "expert-packed-v1")
        packed = [t for t in r.tensors if t.name.endswith("ffn_exps_colibri")]
        self.assertEqual(len(packed), LAYERS)
        for t in packed:
            self.assertEqual(t.data_offset % 4096, 0, t.name)
            self.assertEqual(int(t.shape[0]) % 4096, 0, t.name)      # slot stride
        # each matrix's [cols, rows], which the engines reading through st_gguf.h need
        dims = [int(x) for x in r.fields["colibri.expert_dims"].contents()]
        self.assertEqual(dims, [H, I, H, I, I, H] * LAYERS)
        # the dense tensors are the source's, byte for byte
        src = gguf.GGUFReader(str(self.src))
        s = {t.name: t for t in src.tensors}
        for t in r.tensors:
            if t.name in s:
                self.assertTrue(np.array_equal(np.array(t.data).view(np.uint8), np.array(s[t.name].data).view(np.uint8)), t.name)

    def test_planner_sees_the_same_experts(self):
        import resource_plan as rp
        a, b = rp.analyze_model(str(self.dir / "link")), rp.analyze_model(str(self.dir / "copy"))
        for k in ("expert_count", "expert_layers", "expert_bytes", "per_cap_bytes", "dense_bytes"):
            self.assertEqual(a[k], b[k], k)
        self.assertEqual(a["expert_count"], LAYERS * E)


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
@unittest.skipUnless(ENGINE.is_file(), "qwen36 is not built")
class GgufImportDense(unittest.TestCase):
    """llama.cpp's qwen35 (the family's dense models): the importer writes a dense config with
    the head tied to the embedding, and the engine gives the same tokens from the bare file
    and from the link directory, on the float and on the integer path."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = Path(cls.tmp.name)
        cls.src = d / "tiny-qwen35.gguf"
        write_model(cls.src, np.random.default_rng(12), dense=True)
        cls.ref = d / "ref.json"
        cls.ref.write_text(json.dumps({"prompt_ids": [5, 6, 7, 8], "full_ids": [5, 6, 7, 8] + [0] * 12}))
        p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(cls.src), str(d / "link"), "--mode", "link"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert p.returncode == 0, p.stdout.decode()
        cls.dir = d

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dense_config(self):
        cfg = json.loads((self.dir / "link" / "config.json").read_text())
        self.assertEqual(cfg["model_type"], "qwen3_5_text")
        self.assertEqual((cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]), (LAYERS, H, I))
        self.assertNotIn("num_experts", cfg)
        self.assertTrue(cfg["tie_word_embeddings"])

    def test_same_tokens_from_file_and_link(self):
        for env in ({}, {"GQ_EXACT": "1"}):
            with self.subTest(path="exact" if env else "integer"):
                self.assertEqual(engine_tokens(self.dir / "link", self.ref, env), engine_tokens(self.src, self.ref, env))


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class ContainerLayers(unittest.TestCase):
    """The container of a model whose first layers are dense (GLM-5.3-Flash, Inkling): those
    layers have no slot and zeros in the layout keys, the others their packed experts; and an
    engine that does not read containers (qwen4exp) is refused before anything is written."""
    def test_dense_layers_get_no_slot(self):
        import tools.gguf_import as gi
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m.gguf"
            w = gguf.GGUFWriter(str(src), "glm5-next")
            w.add_tensor("blk.0.ffn_gate.weight", blocks("Q8_0", (64, 64), rng), raw_dtype=QT["Q8_0"])
            for k, shape in (("gate", (4, 32, 64)), ("up", (4, 32, 64)), ("down", (4, 64, 32))):
                w.add_tensor(f"blk.1.ffn_{k}_exps.weight", blocks("Q8_0", shape, rng), raw_dtype=QT["Q8_0"])
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            parts, readers, meta, tensors = gi.open_model(src)
            out = Path(d) / "c.gguf"
            gi.write_packed(parts, readers, meta, tensors, 2, out)
            for rd in readers:
                rd.close()
            r = gguf.GGUFReader(str(out))
            self.assertEqual([t.name for t in r.tensors if t.name.endswith("ffn_exps_colibri")], ["blk.1.ffn_exps_colibri"])
            stride = [int(x) for x in r.fields["colibri.expert_stride"].contents()]
            self.assertEqual(stride[0], 0)
            self.assertEqual(stride[1] % 4096, 0)
            self.assertEqual([int(x) for x in r.fields["colibri.expert_dims"].contents()], [0] * 6 + [64, 32, 64, 32, 32, 64])
            self.assertIn("blk.0.ffn_gate.weight", [t.name for t in r.tensors])

    def test_no_container_for_qwen4exp(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "q.gguf"
            w = gguf.GGUFWriter(str(src), "qwen4exp")
            w.add_tensor("token_embd.weight", np.zeros((4, 32), np.float32))
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            p = subprocess.run([sys.executable, str(HERE / "tools" / "gguf_import.py"), str(src), str(Path(d) / "out"), "--mode", "copy"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("--mode link", p.stdout.decode())
            self.assertFalse((Path(d) / "out").exists())


@unittest.skipIf(gguf is None, "gguf-py (pip install gguf) is not installed")
class PlannerExperts(unittest.TestCase):
    """The planner's inventory of a GGUF: one entry per routed expert under a name every family
    counts (resource_plan._GGUF_EXPERT), and the model's own layers from its config, so that
    a file whose nextn_predict_layers does not count its blocks (antirez's V4) loses none."""
    def test_layers_from_the_config(self):
        import resource_plan as rp
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "m.gguf"
            w = gguf.GGUFWriter(str(src), "deepseek4")
            w.add_uint32("deepseek4.block_count", 3); w.add_uint32("deepseek4.nextn_predict_layers", 1)
            for L in range(3):
                for k, shape in (("gate", (4, 32, 64)), ("up", (4, 32, 64)), ("down", (4, 64, 32))):
                    w.add_tensor(f"blk.{L}.ffn_{k}_exps.weight", blocks("Q8_0", shape, rng), raw_dtype=QT["Q8_0"])
            w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
            def layers_seen(n):
                return sorted({int(m.group(1)) for name, *_ in rp._gguf_tensor_sizes([src], n)
                               for m in [rp._GGUF_EXPERT.match(name)] if m})
            self.assertEqual(layers_seen(3), [0, 1, 2])
            self.assertEqual(layers_seen(None), [0, 1])   # the file's own count: the last block taken for MTP
            per = [size for name, size, *_ in rp._gguf_tensor_sizes([src], 3) if name.startswith("model.layers.0.mlp.experts.1.")]
            self.assertEqual(sum(per), 3 * 32 * 2 * 34)   # gate, up, down: 32 rows of two Q8_0 blocks each


class Qwen3MoeConfig(unittest.TestCase):
    """llama.cpp's qwen3moe (Qwen3-30B-A3B, Qwen3-Coder) as the qwen36 engine's config, from the
    metadata of unsloth's Qwen3-Coder-30B-A3B-Instruct GGUF: every layer attention, no shared
    expert, the head untied when the file has output.weight."""
    def test_coder_metadata(self):
        import tools.gguf_import as gi
        p = "qwen3moe."
        meta = {p + k: v for k, v in {
            "block_count": 48, "context_length": 262144, "embedding_length": 2048, "feed_forward_length": 5472,
            "attention.head_count": 32, "attention.head_count_kv": 4, "rope.freq_base": 10000000.0,
            "attention.layer_norm_rms_epsilon": 1e-6, "expert_used_count": 8, "attention.key_length": 128,
            "attention.value_length": 128, "expert_count": 128, "expert_feed_forward_length": 768,
            "expert_shared_feed_forward_length": 0}.items()}
        tensors = {"token_embd.weight": {"dims": [2048, 151936]}, "output.weight": {"dims": [2048, 151936]}}
        cfg, layers = gi.qwen3moe_config(meta, tensors)
        self.assertEqual(layers, 48)
        self.assertEqual(cfg["model_type"], "qwen3_moe")
        self.assertEqual((cfg["num_experts"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]), (128, 8, 768))
        self.assertEqual((cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["head_dim"]), (32, 4, 128))
        self.assertEqual(cfg["layer_types"], ["full_attention"] * 48)
        self.assertFalse(cfg["tie_word_embeddings"])
        self.assertEqual(cfg["vocab_size"], 151936)


class Qwen3DenseConfig(unittest.TestCase):
    """llama.cpp's qwen3 (the dense Qwen3) as the qwen36 engine's config: Qwen3-0.6B's numbers,
    whose head (128) is not the width over the heads (64), and whose embedding is tied."""
    def test_small_tied(self):
        import tools.gguf_import as gi
        p = "qwen3."
        meta = {p + k: v for k, v in {
            "block_count": 28, "context_length": 40960, "embedding_length": 1024, "feed_forward_length": 3072,
            "attention.head_count": 16, "attention.head_count_kv": 8, "rope.freq_base": 1000000.0,
            "attention.layer_norm_rms_epsilon": 1e-6, "attention.key_length": 128,
            "attention.value_length": 128}.items()}
        tensors = {"token_embd.weight": {"dims": [1024, 151936]}}
        cfg, layers = gi.qwen3_config(meta, tensors)
        self.assertEqual(layers, 28)
        self.assertEqual((cfg["model_type"], cfg["architectures"]), ("qwen3", ["Qwen3ForCausalLM"]))
        self.assertEqual((cfg["intermediate_size"], cfg["head_dim"]), (3072, 128))
        for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size"):
            self.assertNotIn(k, cfg)
        self.assertEqual(cfg["layer_types"], ["full_attention"] * 28)
        self.assertTrue(cfg["tie_word_embeddings"])
        # the family registry plans it as a dense qwen36 model: KV on every layer, no experts
        from family_registry import family_by_id
        fam = family_by_id("qwen36")
        self.assertIn("qwen3", fam.model_types)

    def test_named_by_the_file(self):
        """A GGUF no variant of the family names keeps its own name (general.name) in the
        banner and the API id, not the family's (it was served as qwen3.6-colibri)."""
        import tools.gguf_import as gi
        from family_registry import resolve_model, display_for, default_model_id
        meta = {"general.name": "Qwen3-0.6B", "general.basename": "Qwen3", "general.size_label": "0.6B"}
        self.assertEqual(gi.model_name(meta), {"name": "Qwen3-0.6B", "size": "0.6B"})
        self.assertEqual(gi.model_name({"general.basename": "Qwen3", "general.size_label": "8B"}), {"name": "Qwen3-8B", "size": "8B"})
        self.assertEqual(gi.model_name({}), {})
        cfg, _ = gi.qwen3_config({"qwen3." + k: v for k, v in {
            "block_count": 28, "embedding_length": 1024, "feed_forward_length": 3072,
            "attention.head_count": 16, "attention.head_count_kv": 8, "attention.key_length": 128}.items()},
            {"token_embd.weight": {"dims": [1024, 151936]}})
        with tempfile.TemporaryDirectory() as d:
            named = dict(cfg, colibri_gguf={"arch": "qwen3", "mode": "link", "source": "x.gguf", **gi.model_name(meta)})
            (Path(d) / "config.json").write_text(json.dumps(named))
            r = resolve_model(d)
            self.assertEqual(display_for(r), ("Qwen3-0.6B", "0.6B"))
            self.assertEqual(default_model_id(r), "qwen3-0.6b-colibri")
            # a GGUF of a model colibri names keeps that name and id (Qwen3-Coder-30B-A3B)
            coder, _ = gi.qwen3moe_config({"qwen3moe." + k: v for k, v in {
                "block_count": 48, "embedding_length": 2048, "attention.head_count": 32, "attention.head_count_kv": 4,
                "attention.key_length": 128, "expert_count": 128, "expert_used_count": 8,
                "expert_feed_forward_length": 768}.items()}, {"token_embd.weight": {"dims": [2048, 151936]}})
            coder["colibri_gguf"] = {"arch": "qwen3moe", "name": "Qwen3-Coder-30B-A3B-Instruct"}
            (Path(d) / "config.json").write_text(json.dumps(coder))
            r = resolve_model(d)
            self.assertEqual(display_for(r), ("Qwen3-Coder-30B-A3B", "30B"))
            self.assertEqual(default_model_id(r), "qwen3-coder-30b-a3b-colibri")


class DeepSeekV4Config(unittest.TestCase):
    """llama.cpp's deepseek4 as the deepseek_v4 engine's config.json, from unsloth's
    DeepSeek-V4-Flash-0731 header numbers; layers counted by the attention blocks (antirez's
    main files say nextn_predict_layers 1 and keep the MTP head apart)."""
    def test_flash_metadata(self):
        import tools.gguf_import as gi
        p = "deepseek4."
        meta = {p + k: v for k, v in {
            "block_count": 43, "context_length": 1048576, "embedding_length": 4096, "attention.head_count": 64,
            "attention.head_count_kv": 1, "attention.key_length": 512, "rope.dimension_count": 64,
            "attention.q_lora_rank": 1024, "attention.output_group_count": 8, "attention.output_lora_rank": 1024,
            "attention.sliding_window": 128, "attention.indexer.head_count": 64, "attention.indexer.key_length": 128,
            "attention.indexer.top_k": 512, "expert_count": 256, "expert_used_count": 6, "expert_shared_count": 1,
            "expert_feed_forward_length": 2048, "hash_layer_count": 3, "hyper_connection.count": 4,
            "hyper_connection.sinkhorn_iterations": 20, "expert_weights_scale": 1.5, "expert_gating_func": 4,
            "attention.compress_ratios": [0, 0] + [4, 128] * 21 + [0], "attention.compress_rope_freq_base": 160000.0,
            "swiglu_clamp_exp": [10.0] * 43, "nextn_predict_layers": 1, "rope.scaling.factor": 16.0,
            "rope.scaling.original_context_length": 65536}.items()}
        tensors = {"token_embd.weight": {"dims": [4096, 129280]}}
        tensors.update({f"blk.{b}.attn_q_a.weight": {} for b in range(43)})
        cfg, layers = gi.deepseek4_config(meta, tensors)
        self.assertEqual(layers, 43)
        self.assertEqual((cfg["model_type"], cfg["scoring_func"], cfg["num_hash_layers"]), ("deepseek_v4", "sqrtsoftplus", 3))
        self.assertEqual((cfg["o_groups"], cfg["q_lora_rank"], cfg["head_dim"]), (8, 1024, 512))
        self.assertEqual(cfg["compress_ratios"][:4], [0, 0, 4, 128])
        self.assertEqual(cfg["num_nextn_predict_layers"], 0)
        self.assertEqual(cfg["rope_scaling"]["original_max_position_embeddings"], 65536)

    def test_pre_tokenizers(self):
        import tools.gguf_import as gi
        ds = gi.pre_tokenizer("joyai-llm")
        self.assertEqual([x["type"] for x in ds["pretokenizers"]], ["Split", "Split", "Split", "ByteLevel"])
        self.assertIn("][A-Za-z]+|", ds["pretokenizers"][2]["pattern"]["Regex"])   # what tok.h keys DeepSeek on
        self.assertIn("\\p{Lu}", gi.pre_tokenizer("inkling")["pretokenizers"][0]["pattern"]["Regex"])   # o200k
        self.assertEqual(gi.pre_tokenizer("qwen2"), {"type": "ByteLevel", "add_prefix_space": False, "use_regex": True})


class InklingConfig(unittest.TestCase):
    """llama.cpp's inkling as the inkling engine's config.json, from unsloth's Inkling-Small
    header: global layers every sixth, two dense blocks, KV heads from the per-layer array."""
    def test_small_metadata(self):
        import tools.gguf_import as gi
        p = "inkling."
        local = [i % 6 != 5 for i in range(42)]
        meta = {p + k: v for k, v in {
            "block_count": 42, "context_length": 1048576, "embedding_length": 4096, "feed_forward_length": 16384,
            "attention.head_count": 32, "attention.head_count_kv": [4 if l else 8 for l in local],
            "attention.key_length": 128, "vocab_size": 201024, "expert_count": 256, "expert_used_count": 6,
            "expert_feed_forward_length": 2048, "expert_shared_count": 2, "expert_weights_scale": 8.0,
            "attention.sliding_window": 512, "attention.sliding_window_pattern": local, "d_rel": 16,
            "rel_extent": 1024, "shortconv_kernel": 4, "dense_block_count": 2, "logit_scale_denom": 16.0,
            "log_scaling_n_floor": 128000, "log_scaling_alpha": 0.1, "unpadded_vocab_size": 200058}.items()}
        meta["tokenizer.ggml.eos_token_id"] = 200006
        cfg, layers = gi.inkling_config(meta, {"token_embd.weight": {"dims": [4096, 201024]}})
        self.assertEqual((layers, cfg["model_type"]), (42, "inkling"))
        self.assertEqual((cfg["num_key_value_heads"], cfg["swa_num_key_value_heads"]), (8, 4))
        self.assertEqual(cfg["local_layer_ids"], [i for i in range(42) if i % 6 != 5])
        self.assertEqual(cfg["mlp_layer_types"][:3], ["dense", "dense", "sparse"])
        self.assertEqual((cfg["intermediate_size"], cfg["moe_intermediate_size"], cfg["n_shared_experts"]), (16384, 2048, 2))
        self.assertEqual((cfg["unpadded_vocab_size"], cfg["eos_token_id"]), (200058, 200006))


class GlmDsaConfig(unittest.TestCase):
    """llama.cpp's glm-dsa as the colibri engine's config.json, from unsloth's GLM-5.3 header:
    MLA's head dims from the _mla lengths, the NextN block past the model's layers, every
    layer's indexer its own when its weights are in the file."""
    def test_glm53_metadata(self):
        import tools.gguf_import as gi
        p = "glm-dsa."
        meta = {p + k: v for k, v in {
            "block_count": 79, "context_length": 1048576, "embedding_length": 6144, "feed_forward_length": 12288,
            "attention.head_count": 64, "attention.head_count_kv": 1, "rope.freq_base": 8000000.0,
            "attention.layer_norm_rms_epsilon": 1e-5, "expert_count": 256, "expert_used_count": 8,
            "expert_group_count": 1, "expert_group_used_count": 1, "expert_gating_func": 2,
            "attention.key_length": 576, "attention.value_length": 512, "leading_dense_block_count": 3,
            "vocab_size": 154880, "attention.q_lora_rank": 2048, "attention.kv_lora_rank": 512,
            "attention.key_length_mla": 256, "attention.value_length_mla": 256, "expert_feed_forward_length": 2048,
            "expert_shared_count": 1, "expert_weights_scale": 2.5, "expert_weights_norm": True,
            "rope.dimension_count": 64, "nextn_predict_layers": 1, "attention.indexer.head_count": 32,
            "attention.indexer.key_length": 128, "attention.indexer.top_k": 2048}.items()}
        meta["tokenizer.ggml.eos_token_id"] = 154820
        tensors = {"token_embd.weight": {"dims": [6144, 154880]}, "output.weight": {"dims": [6144, 154880]},
                   "blk.78.nextn.eh_proj.weight": {"dims": [12288, 6144]},
                   **{f"blk.{i}.indexer.attn_q_b.weight": {"dims": [2048, 4096]} for i in range(79) if i != 7}}
        cfg, layers = gi.glmdsa_config(meta, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["num_nextn_predict_layers"]), (78, "glm_moe_dsa", 1))
        self.assertEqual((cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"], cfg["v_head_dim"]), (192, 64, 256))
        self.assertEqual((cfg["q_lora_rank"], cfg["kv_lora_rank"], cfg["first_k_dense_replace"]), (2048, 512, 3))
        self.assertEqual((cfg["routed_scaling_factor"], cfg["rope_parameters"]["rope_theta"]), (2.5, 8000000.0))
        self.assertEqual(len(cfg["indexer_types"]), 78)
        self.assertEqual(cfg["indexer_types"][7], "shared")
        self.assertEqual(cfg["indexer_types"].count("full"), 77)
        self.assertFalse(cfg["tie_word_embeddings"])
        del tensors["blk.78.nextn.eh_proj.weight"]   # a file without its NextN block: every block a layer
        self.assertEqual(gi.glmdsa_config(meta, tensors)[1], 79)


class Mimo2Config(unittest.TestCase):
    """llama.cpp's mimo2 as the mimo engine's config.json, from MiMo-V2.6-Flash's header: the
    layer kinds and their KV heads from the pattern, the sinks from the sliding layers'
    tensors, qkv as one [Q|K|V], the rotary width the engine rounds to 64."""
    def test_flash_metadata(self):
        import tools.gguf_import as gi
        p = "mimo2."
        pattern = [0] + [1, 1, 1, 1, 0] * 9 + [1, 1] + [1, 1, 1]   # 48 layers + 3 NextN
        meta = {p + k: v for k, v in {
            "block_count": 51, "context_length": 1048576, "embedding_length": 4096, "feed_forward_length": 16384,
            "attention.head_count": 64, "attention.head_count_kv": [4 if x == 0 else 8 for x in pattern],
            "rope.freq_base": 10000000.0, "rope.freq_base_swa": 10000.0, "expert_count": 256, "expert_used_count": 8,
            "attention.key_length": 192, "attention.value_length": 128, "attention.sliding_window": 128,
            "attention.sliding_window_pattern": pattern, "expert_feed_forward_length": 2048,
            "rope.dimension_count": 64, "attention.layer_norm_rms_epsilon": 1e-6,
            "attention.value_scale": 0.707, "nextn_predict_layers": 3}.items()}
        tensors = {"token_embd.weight": {"dims": [4096, 152576]}, "output.weight": {"dims": [4096, 152576]},
                   **{f"blk.{i}.ffn_gate_exps.weight": {} for i in range(1, 48)},
                   **{f"blk.{i}.attn_sinks.weight": {} for i in range(48) if pattern[i] == 1}}
        cfg, layers = gi.mimo2_config(meta, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["qkv_shards"]), (48, "mimo_v2", 1))
        self.assertEqual((cfg["num_key_value_heads"], cfg["swa_num_key_value_heads"]), (4, 8))
        self.assertEqual(cfg["hybrid_layer_pattern"], pattern[:48])
        self.assertEqual(cfg["moe_layer_freq"], [0] + [1] * 47)
        self.assertEqual((cfg["add_full_attention_sink_bias"], cfg["add_swa_attention_sink_bias"]), (False, True))
        self.assertEqual(int(cfg["head_dim"] * cfg["partial_rotary_factor"]), 64)
        self.assertEqual((cfg["rope_theta"], cfg["swa_rope_theta"], cfg["sliding_window"]), (10000000.0, 10000.0, 128))


class DeepSeek41Config(unittest.TestCase):
    """antirez's deepseek41: the checkpoint's config.json from deepseek41.config as it is, and
    the engram sidecar (dsv41_engram.json, which tools/prepare_dsv41.py builds from the
    checkpoint) from the file's engram metadata; the planner keeps the tables out of RAM."""
    TEXT = {"num_hidden_layers": 2, "engram_layer_ids": [1], "engram_max_ngram_size": 3,
            "engram_n_heads": 2, "engram_head_dim": 32, "engram_pad_token_id": 2}

    def meta(self, **over):
        p = "deepseek41.engram."
        m = {"deepseek41.config": json.dumps({"model_type": "deepseek_v41", "text_config": self.TEXT}),
             p + "encoding": "e4m3_e8m0_32_row264", p + "layer_ids": [1], p + "rows": [97 + 101 + 103 + 107],
             p + "primes": [97, 101, 103, 107], p + "multipliers": [76632096046245, 4839876093313, 35959672319349],
             p + "token_map": [0, 1, 5, 3]}
        m.update({p + k: v for k, v in over.items()})
        return m

    def test_config_and_sidecar(self):
        import tools.gguf_import as gi
        tensors = {f"blk.{b}.attn_q_a.weight": {} for b in range(2)}
        cfg, layers = gi.deepseek41_config(self.meta(), tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["text_config"]), (2, "deepseek_v41", self.TEXT))
        e = gi.deepseek41_engram(self.meta(), cfg)
        self.assertEqual(e["primes"], [[[97, 101], [103, 107]]])
        self.assertEqual(e["offsets"], [[0, 97, 198, 301]])
        self.assertEqual(e["multipliers"], [["76632096046245", "4839876093313", "35959672319349"]])   # past a double's 53 bits: strings
        self.assertEqual((e["pad_id"], e["head_dim"], e["num_embeddings"]), (5, 32, [408]))   # the pad token's compressed id

    def test_inconsistent_metadata_is_refused(self):
        import tools.gguf_import as gi
        cfg = json.loads(self.meta()["deepseek41.config"])
        for over in ({"layer_ids": [0]}, {"encoding": "e4m3_row256"}, {"rows": [400]}, {"primes": [97, 101, 103]}):
            with self.assertRaises(SystemExit, msg=over):
                gi.deepseek41_engram(self.meta(**over), cfg)
        with self.assertRaises(SystemExit):   # a block short of the config's layers
            gi.deepseek41_config(self.meta(), {"blk.0.attn_q_a.weight": {}})

    def test_planner_keeps_the_tables_on_disk(self):
        from family_registry import _dsv41_resident_inventory as resident
        self.assertEqual(resident("blk.1.engram_embd.weight", 10 ** 11, {}), 0)          # a GGUF's table
        self.assertEqual(resident("layers.1.engram.embed.weight", 10 ** 11, {}), 0)      # the checkpoint's
        self.assertEqual(resident("blk.1.engram_kv.weight", 1000, {}), 1000)             # its projection is resident


class OlmoeConfig(unittest.TestCase):
    """llama.cpp's olmoe as the olmoe engine's config.json, from the metadata llama.cpp's
    converter writes for OLMoE-1B-7B: no norm_topk_prob in the file, so false, as the official
    checkpoints have it and as llama.cpp computes them."""
    META = {"olmoe.block_count": 16, "olmoe.context_length": 4096, "olmoe.embedding_length": 2048,
            "olmoe.feed_forward_length": 1024, "olmoe.attention.head_count": 16,
            "olmoe.attention.head_count_kv": 16, "olmoe.rope.freq_base": 10000.0,
            "olmoe.attention.layer_norm_rms_epsilon": 1e-5, "olmoe.expert_count": 64,
            "olmoe.expert_used_count": 8, "tokenizer.ggml.eos_token_id": 50279}

    def test_1b_7b_metadata(self):
        import tools.gguf_import as gi
        tensors = {"token_embd.weight": {"dims": [2048, 50304]}, "output.weight": {"dims": [2048, 50304]}}
        cfg, layers = gi.olmoe_config(self.META, tensors)
        self.assertEqual((layers, cfg["model_type"], cfg["vocab_size"], cfg["tie_word_embeddings"]), (16, "olmoe", 50304, False))
        self.assertEqual((cfg["num_experts"], cfg["num_experts_per_tok"], cfg["intermediate_size"]), (64, 8, 1024))
        self.assertIs(cfg["norm_topk_prob"], False)
        cfg, _ = gi.olmoe_config({**self.META, "olmoe.expert_weights_norm": True}, tensors)
        self.assertIs(cfg["norm_topk_prob"], True)
        with self.assertRaises(SystemExit):
            gi.olmoe_config({k: v for k, v in self.META.items() if k != "olmoe.expert_count"}, tensors)


class Qwen4ExpLayers(unittest.TestCase):
    """Which layers attend, from either way a qwen4exp GGUF says it: unsloth's Qwen3.8-Flash-Next
    files give full_attention_interval (no attention.recurrent_layers), newer llama.cpp
    writes recurrent_layers; compress_ratios, when not all 0, must agree."""
    def meta(self, **extra):
        p = "qwen4exp."
        m = {p + k: v for k, v in {
            "block_count": 8, "embedding_length": 2560, "attention.head_count": 24, "attention.head_count_kv": 2,
            "attention.key_length": 256, "rope.dimension_count": 64, "expert_count": 512, "expert_used_count": 10,
            "expert_feed_forward_length": 640, "ssm.conv_kernel": 4, "ssm.state_size": 128, "ssm.group_count": 16,
            "ssm.time_step_rank": 48, "ssm.inner_size": 6144, "hyper_connection.count": 4,
            "hyper_connection.low_rank": 320, "attention.indexer.head_count": 4, "attention.indexer.key_length": 128,
            "attention.indexer.top_k": 2048, "ple.layers": [1], "ple.ngram_size": 3, "ple.heads_per_ngram": 8,
            "ple.conv_kernel": 4, "embedding_length_per_layer_input": 160}.items()}
        m.update({p + k: v for k, v in extra.items()})
        return m

    def config(self, meta):
        import tools.gguf_import as gi
        tensors = {"token_embd.weight": {"dims": [2560, 248320]},
                   **{f"blk.{i}.indexer.k_proj.weight": {"dims": [2560, 128]} for i in range(8)}}
        return gi.qwen4exp_config(meta, tensors)[0]

    def test_interval_and_recurrent_layers_agree(self):
        ratios = [0, 0, 0, 4, 0, 0, 0, 4]
        old = self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": ratios}))
        new = self.config(self.meta(**{"attention.recurrent_layers": [1, 1, 1, 0, 1, 1, 1, 0],
                                       "attention.compress_ratios": ratios}))
        self.assertEqual(old["layer_types"], ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"])
        self.assertEqual(old, new)
        self.assertEqual(old["indexer_compress_ratio"], 4)

    def test_contradicting_metadata_is_refused(self):
        with self.assertRaises(SystemExit):
            self.config(self.meta(full_attention_interval=4, **{"attention.compress_ratios": [0, 0, 4, 0, 0, 0, 4, 0]}))


if __name__ == "__main__":
    unittest.main()
