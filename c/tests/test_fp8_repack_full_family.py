"""tools/repack_fp8_passthrough.py: FULL-FAMILY EMISSION coverage tests.

Dispatch: builder B3, task M1_MINT_COVERAGE_AUDIT_2026-08-18 /
fp8_integration_gate_v0_1.md item 6 ("the mint must emit ALL families ...
so the output is standalone-loadable"). Synthetic fixtures ONLY -- no real
Z.ai shard is read or written by this suite, per the tool's own hard
constraint (see repack_fp8_passthrough.py's module docstring) and per this
task's own scope limit: no read of the real 755 GB checkpoint, no model
decode.

Builds ONE small synthetic checkpoint directory covering every tensor family
M1's own standalone-load checklist names (M1 audit section 3, quoting
colibri.c's model_init req[] array, colibri.c:1843-1922):
  - model-level: model.embed_tokens.weight, lm_head.weight, model.norm.weight
  - a DENSE layer (layer 0): input_layernorm, post_attention_layernorm,
    self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
    kv_a_layernorm,kv_b_proj,o_proj}.weight, mlp.{gate_proj,up_proj,
    down_proj}.weight
  - a SPARSE/MoE layer (layer 1): the same norm/self_attn set, plus
    mlp.gate.weight (router), mlp.gate.e_score_correction_bias,
    mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight, and
    mlp.experts.<e>.{gate_proj,up_proj,down_proj}.weight per expert
  - a DSA indexer tensor and a beyond-n_layers ("MTP-shaped") tensor, both
    of which must be EXCLUDED (M1: separately, safely gated -- out of this
    audit's family list) -- included here as a negative control so the
    "emitted set == M1's checklist" assertion is an EXACT match, not a
    superset check that would miss over-selection.

Also builds config.json, tokenizer.json (both MANDATORY -- F3) and
generation_config.json (best-effort) in --indir, so the same fixture covers
both the tensor-family checklist and the standalone-load aux-file checklist.

Runs the real CLI end to end (subprocess, --n-layers 2, no --mtp) and checks,
per the task's evidence bar:
  1. every family in M1's checklist is emitted, name-by-name (exact set
     equality against the checklist, built independently of the tool's own
     classify() so a shared bug in both wouldn't hide);
  2. byte-identity of BOTH pass-through payloads (fp8-repack AND raw
     BF16/F32) via a SHA-256 hash of input bytes vs output bytes;
  3. a 1-D norm (several: model.norm.weight, per-layer input_layernorm,
     self_attn.q_a_layernorm) passes _emission_stamp_gate without crashing
     the run (the M1 fix-shape requirement);
  4. config.json and tokenizer.json (both MANDATORY) and generation_config.json
     (best-effort) land in --outdir, byte-identical to --indir;
  5. kv_b_proj is stamped fp8; router/norms/embed/lm_head are present but
     carry no stamp and no `.qs`.

F3 (2026-08-18, colibri_lab/reports/F3_MINT_AUX_FILES_2026-08-18.md): V2's
end-to-end smoke found the full mint emitted config.json but not
tokenizer.json, so the loaded-standalone claim was false -- the engine's
tok_load (c/tok.h:101, called from colibri.c:7294/7952/8134) hard-exits
without it, exactly like load_cfg does without config.json. tokenizer.json
joins config.json in STANDALONE_LOAD_REQUIRED_FILES; every other non-tensor
file in a real GLM-5.2 checkpoint (tokenizer_config.json, chat_template.jinja,
model.safetensors.index.json, README.md, LICENSE, .gitattributes) is verified
NOT read anywhere in colibri.c / c/*.h / openai_server.py and stays excluded
-- see the comment block above STANDALONE_LOAD_REQUIRED_FILES in
repack_fp8_passthrough.py for the file:line evidence per file.
"""
import glob, hashlib, json, os, struct, subprocess, sys, tempfile, unittest

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError as e:
    raise unittest.SkipTest(f"torch/safetensors not installed: {e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from glm_fp8_emit import fp8_block_quantize
import repack_fp8_passthrough as rp

TOOL = os.path.join(os.path.dirname(__file__), "..", "tools", "repack_fp8_passthrough.py")

# ---- fixture geometry (small, arbitrary, internally consistent) ----------
D, DQA, DQB, DKVA, DKVB, I_, V, E = 32, 16, 24, 12, 20, 48, 50, 2
N_LAYERS = 2   # layer 0 = dense, layer 1 = sparse/MoE; both < N_LAYERS ("main")


def _sha256(t):
    """Dtype-agnostic byte hash: view(uint8) is a bit-level reinterpret, the
    same trick repack_shard itself uses for fp8 weights, so it works
    identically for BF16/F32 raw tensors and FP8 weights alike."""
    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def _attn_block(prefix):
    """self_attn.{q_a_proj,q_a_layernorm,q_b_proj,kv_a_proj_with_mqa,
    kv_a_layernorm,kv_b_proj,o_proj} -- identical set for dense and sparse
    layers per M1's checklist. Returns (raw_dict, quant_dict)."""
    raw = {
        f"{prefix}.self_attn.q_a_layernorm.weight": torch.randn(DQA).to(torch.bfloat16),
        f"{prefix}.self_attn.kv_a_layernorm.weight": torch.randn(DKVA).to(torch.bfloat16),
    }
    quant = {
        f"{prefix}.self_attn.q_a_proj.weight": torch.randn(DQA, D) * 0.02,
        f"{prefix}.self_attn.q_b_proj.weight": torch.randn(DQB, DQA) * 0.02,
        f"{prefix}.self_attn.kv_a_proj_with_mqa.weight": torch.randn(DKVA, D) * 0.02,
        f"{prefix}.self_attn.kv_b_proj.weight": torch.randn(DKVB, DKVA) * 0.02,
        f"{prefix}.self_attn.o_proj.weight": torch.randn(D, DQB) * 0.02,
    }
    return raw, quant


def _make_checkpoint(indir):
    """Writes one shard covering every family in M1's checklist (dense layer
    0, sparse layer 1, model-level tensors), PLUS two names that must be
    EXCLUDED (indexer, beyond-n_layers) as a negative control, PLUS
    config.json, tokenizer.json, generation_config.json (the external files
    the mint must reproduce in --outdir for the container to be standalone-
    loadable -- config.json and tokenizer.json mandatory, generation_config
    .json best-effort)."""
    torch.manual_seed(7)
    raw = {
        "model.embed_tokens.weight": torch.randn(V, D).to(torch.bfloat16),
        "lm_head.weight": torch.randn(V, D).to(torch.bfloat16),
        "model.norm.weight": torch.randn(D).to(torch.bfloat16),          # 1-D
        "model.layers.0.input_layernorm.weight": torch.randn(D).to(torch.bfloat16),
        "model.layers.0.post_attention_layernorm.weight": torch.randn(D).to(torch.bfloat16),
        "model.layers.1.input_layernorm.weight": torch.randn(D).to(torch.bfloat16),
        "model.layers.1.post_attention_layernorm.weight": torch.randn(D).to(torch.bfloat16),
        "model.layers.1.mlp.gate.weight": torch.randn(E, D).to(torch.bfloat16),      # router
        "model.layers.1.mlp.gate.e_score_correction_bias": torch.randn(E),           # F32 bias
    }
    quant = {
        # dense layer 0 MLP
        "model.layers.0.mlp.gate_proj.weight": torch.randn(I_, D) * 0.02,
        "model.layers.0.mlp.up_proj.weight": torch.randn(I_, D) * 0.02,
        "model.layers.0.mlp.down_proj.weight": torch.randn(D, I_) * 0.02,
        # sparse layer 1: shared expert + routed experts
        "model.layers.1.mlp.shared_experts.gate_proj.weight": torch.randn(I_, D) * 0.02,
        "model.layers.1.mlp.shared_experts.up_proj.weight": torch.randn(I_, D) * 0.02,
        "model.layers.1.mlp.shared_experts.down_proj.weight": torch.randn(D, I_) * 0.02,
    }
    for e in range(E):
        quant[f"model.layers.1.mlp.experts.{e}.gate_proj.weight"] = torch.randn(I_, D) * 0.02
        quant[f"model.layers.1.mlp.experts.{e}.up_proj.weight"] = torch.randn(I_, D) * 0.02
        quant[f"model.layers.1.mlp.experts.{e}.down_proj.weight"] = torch.randn(D, I_) * 0.02
    for prefix in ("model.layers.0", "model.layers.1"):
        r, q = _attn_block(prefix)
        raw.update(r)
        quant.update(q)
    # NEGATIVE CONTROL: must be excluded from the emitted set.
    quant["model.layers.0.self_attn.indexer.wq_b.weight"] = torch.randn(I_, D) * 0.02  # DSA indexer: "skip"
    quant["model.layers.2.self_attn.o_proj.weight"] = torch.randn(D, DQB) * 0.02       # layer 2 >= N_LAYERS: "skip"

    sd = dict(raw)
    for name, t in quant.items():
        w_fp8, scale = fp8_block_quantize(t)
        sd[name] = w_fp8
        sd[name + "_scale_inv"] = scale

    os.makedirs(indir, exist_ok=True)
    shard = os.path.join(indir, "model-00001-of-00001.safetensors")
    save_file(sd, shard)
    with open(os.path.join(indir, "config.json"), "w") as fh:
        json.dump({"model_type": "glm_moe_dsa_test_fixture", "hidden_size": D}, fh)
    with open(os.path.join(indir, "tokenizer.json"), "w") as fh:
        json.dump({"model": {"vocab": {"<|endoftext|>": 0, "a": 1}, "merges": []},
                   "added_tokens": []}, fh)
    with open(os.path.join(indir, "generation_config.json"), "w") as fh:
        json.dump({"temperature": 0.0}, fh)
    return shard, sd


def _expected_checklist():
    """M1's standalone-load checklist (audit section 3), built independently
    of classify()/repack_fp8_passthrough.py -- a shared bug in both would not
    be caught by an equality check derived from the same source."""
    names = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
    attn_suffixes = ("self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
                      "self_attn.q_b_proj.weight", "self_attn.kv_a_proj_with_mqa.weight",
                      "self_attn.kv_a_layernorm.weight", "self_attn.kv_b_proj.weight",
                      "self_attn.o_proj.weight")
    for i in (0, 1):
        p = f"model.layers.{i}"
        names.add(f"{p}.input_layernorm.weight")
        names.add(f"{p}.post_attention_layernorm.weight")
        for s in attn_suffixes:
            names.add(f"{p}.{s}")
    # layer 0: dense MLP
    for s in ("gate_proj", "up_proj", "down_proj"):
        names.add(f"model.layers.0.mlp.{s}.weight")
    # layer 1: sparse MLP
    names.add("model.layers.1.mlp.gate.weight")
    names.add("model.layers.1.mlp.gate.e_score_correction_bias")
    for s in ("gate_proj", "up_proj", "down_proj"):
        names.add(f"model.layers.1.mlp.shared_experts.{s}.weight")
        for e in range(E):
            names.add(f"model.layers.1.mlp.experts.{e}.{s}.weight")
    return names


def _read_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hlen))


class FullFamilyEmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.indir = os.path.join(self.tmp.name, "fp8src")
        self.outdir = os.path.join(self.tmp.name, "out")
        self.shard, self.sd = _make_checkpoint(self.indir)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_cli(self, extra=()):
        return subprocess.run(
            [sys.executable, TOOL, "--indir", self.indir, "--outdir", self.outdir,
             "--n-layers", str(N_LAYERS), *extra],
            capture_output=True, text=True)

    def test_dry_run_covers_the_full_checklist_without_writing(self):
        """Cross-check via the tool's own shard_inventory (no subprocess) --
        1-D norms must not raise here either (_check_geometry is never
        called on them; is_passthrough_target selects them directly)."""
        inv = rp.shard_inventory(self.shard, n_layers=N_LAYERS)
        names = {it["name"] for it in inv}
        self.assertEqual(names, _expected_checklist())
        self.assertFalse(os.path.exists(self.outdir), "--dry-run must not create --outdir")

    def test_full_mint_emits_every_family_and_only_those(self):
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        hdr = _read_header(outs[0])
        emitted = {k for k in hdr if k != "__metadata__" and not k.endswith(".qs")}
        expected = _expected_checklist()
        self.assertEqual(emitted, expected,
                         f"missing: {expected - emitted}; unexpected: {emitted - expected}")
        # negative control: indexer and beyond-n_layers names must NOT appear anywhere
        self.assertNotIn("model.layers.0.self_attn.indexer.wq_b.weight", hdr)
        self.assertNotIn("model.layers.2.self_attn.o_proj.weight", hdr)

    def test_byte_identity_fp8_and_raw_passthrough_hash_match(self):
        """Byte-identity, hash in vs out, for BOTH categories: an fp8-repack
        tensor (kv_b_proj, weight+qs) and raw pass-through tensors (a 1-D
        BF16 norm, a 2-D BF16 embed table, a 1-D F32 bias)."""
        self.assertEqual(self._run_cli().returncode, 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        self.assertEqual(len(outs), 1)
        with safe_open(self.shard, framework="pt") as f_in, \
             safe_open(outs[0], framework="pt") as f_out:
            # fp8-repack: kv_b_proj weight bytes + its renamed .qs sidecar
            name = "model.layers.0.self_attn.kv_b_proj.weight"
            self.assertEqual(_sha256(f_in.get_tensor(name).view(torch.uint8)),
                             _sha256(f_out.get_tensor(name)))
            self.assertEqual(
                _sha256(f_in.get_tensor(name + "_scale_inv").reshape(-1)),
                _sha256(f_out.get_tensor(name + ".qs")))
            # raw pass-through: 1-D norm, 2-D embed table, 1-D f32 bias
            for name in ("model.norm.weight", "model.embed_tokens.weight",
                        "model.layers.1.mlp.gate.e_score_correction_bias",
                        "model.layers.0.self_attn.q_a_layernorm.weight"):
                self.assertEqual(_sha256(f_in.get_tensor(name)),
                                 _sha256(f_out.get_tensor(name)),
                                 f"{name}: hash mismatch, not byte-identical")
                self.assertNotIn(name + ".qs", f_out.keys(), f"{name}: must carry no .qs")

    def test_1d_norms_pass_the_emission_stamp_gate_without_crashing(self):
        """The M1 fix-shape requirement, end to end: several 1-D norms are in
        this fixture (model.norm.weight, both layers' input_layernorm/
        post_attention_layernorm, q_a_layernorm, kv_a_layernorm) and the run
        must complete (rc==0), not crash on _emission_stamp_gate's old
        unconditional `len(t.shape) != 2` check."""
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        hdr = _read_header(outs[0])
        expected_1d = {
            "model.norm.weight": D,
            "model.layers.0.input_layernorm.weight": D,
            "model.layers.1.post_attention_layernorm.weight": D,
            "model.layers.0.self_attn.q_a_layernorm.weight": DQA,
            "model.layers.1.self_attn.kv_a_layernorm.weight": DKVA,
        }
        for name, size in expected_1d.items():
            self.assertEqual(hdr[name]["shape"], [size], f"{name}: shape sanity")

    def test_config_json_and_generation_config_land_in_outdir(self):
        """M1 gap: main() previously never copied config.json into --outdir
        -- a minted directory was not standalone-loadable for that reason
        alone. config.json is mandatory; generation_config.json is
        best-effort (both present in this fixture's --indir)."""
        self.assertEqual(self._run_cli().returncode, 0)
        out_cfg = os.path.join(self.outdir, "config.json")
        self.assertTrue(os.path.exists(out_cfg))
        with open(os.path.join(self.indir, "config.json")) as f_in, open(out_cfg) as f_out:
            self.assertEqual(json.load(f_in), json.load(f_out))
        out_gen = os.path.join(self.outdir, "generation_config.json")
        self.assertTrue(os.path.exists(out_gen))
        with open(os.path.join(self.indir, "generation_config.json")) as f_in, open(out_gen) as f_out:
            self.assertEqual(json.load(f_in), json.load(f_out))

    def test_config_json_missing_refuses_loudly(self):
        """The mandatory half of the same fix: no config.json in --indir must
        refuse the run, not silently mint an unloadable directory."""
        os.remove(os.path.join(self.indir, "config.json"))
        proc = self._run_cli()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("config.json", proc.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.outdir, "config.json")))

    def test_generation_config_missing_is_best_effort_not_fatal(self):
        os.remove(os.path.join(self.indir, "generation_config.json"))
        proc = self._run_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.outdir, "generation_config.json")))

    def test_tokenizer_json_lands_in_outdir_byte_identical(self):
        """F3: V2's end-to-end smoke (2026-08-18) found the full mint emitted
        config.json but not tokenizer.json, so a minted directory could not
        actually be loaded standalone -- colibri.c's tok_load (c/tok.h:101)
        hard-exits without it, exactly like load_cfg does without
        config.json. Byte-identity, not just existence: this is a COPY, not a
        regeneration, so the source and minted bytes must match exactly."""
        self.assertEqual(self._run_cli().returncode, 0)
        out_tok = os.path.join(self.outdir, "tokenizer.json")
        self.assertTrue(os.path.exists(out_tok))
        src = os.path.join(self.indir, "tokenizer.json")
        with open(src, "rb") as f_in, open(out_tok, "rb") as f_out:
            in_bytes, out_bytes = f_in.read(), f_out.read()
        self.assertEqual(in_bytes, out_bytes,
                         "tokenizer.json must be a byte-identical copy, not a regeneration")

    def test_tokenizer_json_missing_refuses_loudly(self):
        """The mandatory half of the tokenizer.json fix: no tokenizer.json in
        --indir must refuse the run, not silently mint a directory the
        engine will fail to load (V2's exact finding, now a hard refusal
        instead of a silent gap)."""
        os.remove(os.path.join(self.indir, "tokenizer.json"))
        proc = self._run_cli()
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("tokenizer.json", proc.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.outdir, "tokenizer.json")))

    def test_index_json_and_docs_not_copied(self):
        """Files confirmed NOT read by colibri.c/openai_server.py must stay
        excluded, including model.safetensors.index.json -- copying it would
        be actively wrong here since its weight_map keys the SOURCE shard
        names (model-NNNNN-of-00141.safetensors), not the mint's
        out-fp8pass-*.safetensors layout. tokenizer_config.json, chat_template
        .jinja, README.md, LICENSE, and .gitattributes are likewise never
        opened anywhere in the loader/server code path (see the file:line
        citations above STANDALONE_LOAD_REQUIRED_FILES)."""
        never_copy = ("model.safetensors.index.json", "tokenizer_config.json",
                      "chat_template.jinja", "README.md", "LICENSE", ".gitattributes")
        for fname in never_copy:
            with open(os.path.join(self.indir, fname), "w") as fh:
                fh.write("placeholder -- must not be copied to --outdir\n")
        self.assertEqual(self._run_cli().returncode, 0)
        for fname in never_copy:
            self.assertFalse(os.path.exists(os.path.join(self.outdir, fname)),
                             f"{fname} must NOT be copied into --outdir")

    def test_kvb_stamped_router_norms_embed_unstamped(self):
        self.assertEqual(self._run_cli().returncode, 0)
        outs = glob.glob(os.path.join(self.outdir, "out-fp8pass-*.safetensors"))
        hdr = _read_header(outs[0])
        stamp = json.loads(hdr["__metadata__"]["colibri.fmt"])
        self.assertEqual(stamp.get("model.layers.0.self_attn.kv_b_proj.weight"), rp.FORMAT_NAME)
        self.assertEqual(stamp.get("model.layers.1.self_attn.kv_b_proj.weight"), rp.FORMAT_NAME)
        for name in ("model.embed_tokens.weight", "lm_head.weight", "model.norm.weight",
                    "model.layers.1.mlp.gate.weight",
                    "model.layers.1.mlp.gate.e_score_correction_bias"):
            self.assertNotIn(name, stamp, f"{name}: raw pass-through must never be stamped")
            self.assertNotIn(name + ".qs", hdr, f"{name}: raw pass-through must carry no .qs")


if __name__ == "__main__":
    unittest.main()
