import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from resource_plan import (
    GB,
    CgroupFormatError,
    analyze_model,
    build_plan,
    cpu_socket_count,
    discover_gpus,
    environment_for_plan,
    format_plan,
    memory_available,
    windows_available_bytes,
    WINDOWS_MEMORYSTATUSEX_FIELDS,
    parse_ssd_cache,
    physical_cpu_count,
    read_ssd_probe,
    ssd_probe_state,
)


def write_shard(path, tensors):
    offset = 0
    header = {}
    for tensor in tensors:
        name, size, *metadata = tensor
        dtype = metadata[0] if metadata else "U8"
        shape = metadata[1] if len(metadata) > 1 else [size]
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + size]}
        offset += size
    raw = json.dumps(header).encode()
    # The planner reads headers and file sizes, not tensor values. Extending
    # the file retains the zero-filled payload without allocating it in Python;
    # filesystems supporting sparse extension also avoid writing gigabytes.
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(raw)))
        stream.write(raw)
        stream.truncate(stream.tell() + offset)


class ShardFixtureTest(unittest.TestCase):
    def test_extended_payload_preserves_header_offsets_size_and_zero_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.safetensors"
            write_shard(path, [("first", 3), ("second", 5)])
            with path.open("rb") as stream:
                header_size, = struct.unpack("<Q", stream.read(8))
                header = json.loads(stream.read(header_size))
                self.assertEqual(header["first"]["data_offsets"], [0, 3])
                self.assertEqual(header["second"]["data_offsets"], [3, 8])
                self.assertEqual(stream.read(), b"\0" * 8)
            self.assertEqual(path.stat().st_size, 8 + header_size + 8)


class ResourcePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(json.dumps({
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 2,
            "n_routed_experts": 2,
            "kv_lora_rank": 4,
            "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3,
            "v_head_dim": 5,
            "num_attention_heads": 2,
        }))
        write_shard(self.model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
            ("model.layers.1.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.0.up_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.gate_proj.weight", 30),
            ("model.layers.1.mlp.experts.1.up_proj.weight", 30),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyzes_dense_and_expert_storage(self):
        info = analyze_model(self.model)
        self.assertEqual(info["dense_bytes"], 300)
        self.assertEqual(info["expert_bytes"], 120)
        self.assertEqual(info["expert_count"], 2)
        self.assertEqual(info["per_cap_bytes"], 60)

    def test_memory_available_is_positive(self):
        # Regression: on native Windows CPython, /proc/meminfo does not exist,
        # so the Linux-only path returned 0 and the expert cache was sized to
        # 0 slots/layer. The value must be a sane positive number of bytes.
        self.assertGreater(memory_available(), 0)

    def test_apple_silicon_reports_unified_host_memory_without_fake_vram(self):
        # Host memory topology is a hardware fact, independent of whether the
        # selected engine can place anything on the GPU.  In particular glm53
        # is CPU-only today, but an M-series Mac must not be reported as
        # memory.unified=false merely because planning_gpus is empty.
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch("resource_plan.platform.machine", return_value="arm64"):
            plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                              available_disk=1, gpus=[], physical_cpus=8,
                              cpu_sockets=1)
        self.assertTrue(plan["memory"]["unified"])
        self.assertEqual(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertFalse(any("jointly constrained" in warning
                             for warning in plan["warnings"]))

    def test_macos_discovers_metal_gpu_without_cuda_or_rocm_probes(self):
        output = json.dumps({"SPDisplaysDataType": [{
            "_name": "Apple M1 Pro",
            "sppci_model": "Apple M1 Pro",
            "spdisplays_mtlgpufamilysupport": "spdisplays_metal4",
        }]})
        result = subprocess.CompletedProcess(args=[], returncode=0,
                                             stdout=output, stderr="")
        with mock.patch.object(sys, "platform", "darwin"), \
             mock.patch("resource_plan.subprocess.run", return_value=result) as run:
            devices = discover_gpus()
        self.assertEqual(devices, [{"index": 0, "name": "Apple M1 Pro",
                                    "total_bytes": 0, "free_bytes": None,
                                    "unified_memory": True, "backend": "metal"}])
        self.assertEqual(run.call_args.args[0],
                         ["system_profiler", "SPDisplaysDataType", "-json"])
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                  available_disk=1, gpus=devices, physical_cpus=8,
                  cpu_sockets=1)
        self.assertIn("Metal  0:Apple M1 Pro · unified memory",
                  format_plan(plan))

    def test_glm53_auto_tune_does_not_emit_generic_inert_knobs(self):
        from resource_plan import _auto_tune

        generic = _auto_tune("disk", 0.50, [], 1, False)
        self.assertIn("DRAFT", generic)
        self.assertIn("PIPE", generic)

        glm53 = _auto_tune("disk", 0.50, [], 1, False,
                           engine_group="glm53")
        self.assertEqual(glm53, {})

    def test_only_the_colibri_engine_gets_its_own_knobs(self):
        """DRAFT, PIPE, COLI_CUDA_PIPE, COLI_NUMA and PIN_GB are read by
        colibri.c and by no other engine. glm53 was excluded; every other
        sibling was still told to set them, in `coli plan`, in `coli doctor`
        and in the --auto-tier environment."""
        from family_registry import all_families
        from resource_plan import _auto_tune

        gpu = [{"index": 0, "name": "GPU", "total_bytes": 24 * GB,
                "free_bytes": 24 * GB}]
        cases = (("disk", 0.50, [], 2), ("compute", 1.0, [], 2),
                 ("compute", 1.0, gpu, 2), ("mixed", 0.80, gpu + gpu, 1))
        core = [_auto_tune(*case, False, engine_group="colibri-core") for case in cases]
        self.assertEqual({key for tune in core for key in tune},
                         {"DRAFT", "PIPE", "COLI_CUDA_PIPE", "COLI_NUMA",
                          "_numa_hint", "PIN_GB"})
        groups = {family.engine_group for family in all_families()} - {"colibri-core"}
        self.assertIn("qwen36", groups)
        for group in sorted(groups):
            for case in cases:
                with self.subTest(engine_group=group, case=case[:2]):
                    self.assertEqual(_auto_tune(*case, False, engine_group=group), {})

    def test_v41_gateway_sizes_cap_from_ram_without_auto_tier(self):
        from openai_server import cap_for_arch

        (self.model / "config.json").write_text(json.dumps({
            "model_type": "deepseek_v41", "num_hidden_layers": 2,
            "n_routed_experts": 128, "hidden_size": 128, "head_dim": 64,
            "window_size": 8, "index_head_dim": 32, "hc_mult": 4,
            "compress_ratios": [0, 2], "kv_source_layers": [1],
        }))
        write_shard(self.model / "model.safetensors", [
            ("embed.weight", GB),
            *[(f"layers.{layer}.ffn.experts.{expert}.w1.weight", 16 * 1024**2)
              for layer in range(2) for expert in range(128)],
        ])
        with mock.patch("resource_plan.memory_available", return_value=16 * GB):
            small = cap_for_arch("deepseek_v41", None, {"RAM_GB": "8"}, self.model)
            large = cap_for_arch("deepseek_v41", None, {"RAM_GB": "12"}, self.model)
            automatic = cap_for_arch("deepseek_v41", None, {}, self.model)
        self.assertGreater(small, 8)
        self.assertGreater(large, small)
        self.assertEqual(large, 128)
        self.assertEqual(automatic, 128)
        self.assertEqual(cap_for_arch("deepseek_v41", 4, {"RAM_GB": "12"}, self.model), 4)

    def test_sibling_plan_advises_no_colibri_knob(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        olmoe = Path(other.name)
        (olmoe / "config.json").write_text(json.dumps({
            "model_type": "olmoe", "num_hidden_layers": 2, "hidden_size": 32,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "num_experts": 2, "num_experts_per_tok": 2,
            "intermediate_size": 16, "vocab_size": 100,
        }))
        write_shard(olmoe / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", 30),
            ("model.layers.0.mlp.experts.1.gate_proj.weight", 30),
        ])
        glm = build_plan(self.model, context=32, available_memory=32 * GB,
                         available_disk=1, gpus=[], cpu_sockets=1)
        self.assertEqual(set(glm["tune"]), {"DRAFT", "PIN_GB"})
        plan = build_plan(olmoe, context=32, available_memory=32 * GB,
                          available_disk=1, gpus=[], cpu_sockets=1)
        self.assertEqual(plan["bottleneck_class"], glm["bottleneck_class"])
        self.assertEqual(plan["tune"], {})
        self.assertNotIn("auto-tune:", format_plan(plan))
        env = environment_for_plan(plan, {})
        for key in ("DRAFT", "PIPE", "COLI_CUDA_PIPE", "COLI_NUMA", "PIN_GB"):
            self.assertNotIn(key, env)

    def _kimi_model(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        model = Path(other.name)
        (model / "config.json").write_text(json.dumps({
            "model_type": "kimi_k3",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "q_lora_rank": 8,
            "kv_lora_rank": 8,
            "qk_nope_head_dim": 8,
            "qk_rope_head_dim": 4,
            "v_head_dim": 8,
            "num_experts": 2,
            "linear_attn_config": {
                "num_heads": 2,
                "head_dim": 8,
                "kda_layers": [1],
            },
        }))
        write_shard(model / "model.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight", 80),
            ("model.layers.0.block_sparse_moe.experts.1.w1.weight", 80),
            ("model.layers.1.block_sparse_moe.experts.0.w1.weight", 80),
            ("model.layers.1.block_sparse_moe.experts.1.w1.weight", 80),
        ])
        return model

    def _qwen_dense_model(self, num_experts=None):
        """Qwen3.8-27B's shape at toy size (#1757): the qwen3_5 architecture with one
        dense MLP per layer and no expert count in the config."""
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        model = Path(other.name)
        text = {
            "model_type": "qwen3_5_text", "num_hidden_layers": 4, "hidden_size": 32,
            "intermediate_size": 64, "num_key_value_heads": 1, "head_dim": 8,
            "linear_num_key_heads": 2, "linear_key_head_dim": 8,
            "linear_num_value_heads": 4, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        }
        if num_experts is not None:
            text["num_experts"] = num_experts
        (model / "config.json").write_text(json.dumps(
            {"model_type": "qwen3_5", "text_config": text}))
        write_shard(model / "model.safetensors", [
            ("model.language_model.embed_tokens.weight", 100),
            ("model.language_model.layers.0.mlp.gate_proj.weight", 80),
            ("model.language_model.layers.0.mlp.up_proj.weight", 80),
            ("model.language_model.layers.0.mlp.down_proj.weight", 80),
        ])
        return model

    def test_dense_qwen_model_keeps_every_weight_resident(self):
        plan = build_plan(self._qwen_dense_model(), context=32, available_memory=32 * GB,
                          available_disk=100 * GB, gpus=[])
        self.assertEqual(plan["model"]["family_id"], "qwen36")
        self.assertEqual(plan["model"]["configured_experts"], 0)
        self.assertEqual(plan["tiers"]["ram"]["cache_slots_per_layer"], 1)
        self.assertIn("dense model", plan["expected_bottleneck"])
        self.assertFalse([w for w in plan["warnings"] if "expert slot" in w])
        self.assertEqual([d["target"] for d in plan["decisions"]], ["RAM"])

    def test_zero_experts_declared_is_still_a_broken_config(self):
        with self.assertRaisesRegex(ValueError, "num_experts|expert count is zero"):
            build_plan(self._qwen_dense_model(num_experts=0), context=32,
                       available_memory=32 * GB, available_disk=100 * GB, gpus=[])

    def _glm53_model(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        model = Path(other.name)
        (model / "config.json").write_text(json.dumps({
            "model_type": "glm5_next",
            "text_config": {
                "num_hidden_layers": 2,
                "n_routed_experts": 2,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "q_lora_rank": 8,
                "kv_lora_rank": 8,
                "qk_nope_head_dim": 8,
                "v_head_dim": 8,
                "index_head_dim": 8,
                "hc_mult": 2,
                "layer_types": ["linear", "full"],
                "linear_attn_config": {
                    "num_heads": 2,
                    "head_dim": 8,
                    "short_conv_kernel_size": 2,
                },
            },
        }))
        write_shard(model / "model.safetensors", [
            ("model.language_model.embed_tokens.weight", 100),
            ("model.language_model.layers.1.mlp.experts.0.gate_proj.weight", 80),
            ("model.language_model.layers.1.mlp.experts.1.gate_proj.weight", 80),
        ])
        return model

    def test_kimi_plan_exports_the_expert_cache_knob_the_engine_reads(self):
        """Kimi K3 sizes the expert LRU from K3_EXPERT_GB (default 8 GB).
        RAM_GB is only a ceiling, and main() never takes argv as a cap, so
        --auto-tier that set RAM_GB and COLI_PLAN_CAP left the 8 GB default
        in place on a machine whose plan had hundreds of GB of warm experts."""
        plan = build_plan(self._kimi_model(), context=32, available_memory=32 * GB,
                          available_disk=1, gpus=[], cpu_sockets=1)
        cache = plan["tiers"]["ram"]["expert_cache_bytes"]
        self.assertGreater(cache, 0)
        expected = f"{cache / GB:.3f}"
        self.assertIn("K3_EXPERT_GB", plan["tune"])
        self.assertEqual(plan["tune"]["K3_EXPERT_GB"]["value"], expected)
        self.assertNotIn("GLM53_EXPERT_GB", plan["tune"])
        self.assertIn("K3_EXPERT_GB=", format_plan(plan))
        env = environment_for_plan(plan, {})
        self.assertIn("K3_EXPERT_GB", env)
        self.assertEqual(env["K3_EXPERT_GB"], expected)
        kept = environment_for_plan(plan, {"K3_EXPERT_GB": "3.5"})
        self.assertEqual(kept["K3_EXPERT_GB"], "3.5")

    def test_glm53_dense_ram_uses_loaded_components_and_each_plans_environment(self):
        model = self._glm53_model()
        write_shard(model / "model.safetensors", [
            ("model.language_model.embed_tokens.weight", 2048, "BF16", [16, 64]),
            ("model.language_model.layers.0.self_attn.q_proj.weight", 2048, "BF16", [16, 64]),
            # The eight-column projection falls back to int8 even at GLM53_BITS=4.
            ("model.language_model.layers.0.self_attn.g_b_proj.weight", 512, "F32", [16, 8]),
            ("lm_head.weight", 2048, "F16", [16, 64]),
            ("model.language_model.layers.0.mlp.gate.weight", 256, "BF16", [2, 64]),
            ("model.language_model.layers.0.input_layernorm.weight", 128, "BF16", [64]),
            ("model.visual.blocks.0.attn.qkv.weight", 2048, "BF16", [16, 64]),
            # Absorbed kv_b, unknown components and packed source weights never
            # receive a guessed checkpoint-wide shrink factor.
            ("model.language_model.layers.1.self_attn.kv_b_proj.weight", 1024, "BF16", [64, 8]),
            ("model.language_model.layers.0.unknown_projection.weight", 128, "BF16", [8, 8]),
            ("model.language_model.layers.0.mlp.up_proj.weight", 512, "U8", [16, 32]),
            ("model.language_model.layers.0.mlp.up_proj.weight.qs", 64, "F32", [16, 1]),
        ])
        kwargs = dict(context=32, available_memory=16 * GB, available_disk=16 * GB, gpus=[])
        retained = 4096 + 512 + 256 + 4096 + 2048 + 256 + 512 + 64
        expected = {4: retained + 2 * (512 + 64) + (128 + 64),
                    8: retained + 2 * (1024 + 64) + (128 + 64),
                    32: retained + 2 * 4096 + 512}
        with mock.patch.dict(os.environ, {"GLM53_BITS": "32"}):
            scan = analyze_model(model)
            self.assertEqual(scan["dense_bytes"], expected[4])
            self.assertEqual(scan["embed_bytes"], 4096)
            # The cached scan is reused for all plans. An explicit env wins over
            # os.environ, and none of these calls bakes its format into the cache.
            with mock.patch("resource_plan._tensor_sizes", side_effect=AssertionError("cache miss")):
                for bits in (4, 32, 8, 4):
                    plan = build_plan(model, env={"GLM53_BITS": str(bits)}, **kwargs)
                    self.assertEqual(plan["tiers"]["ram"]["dense_bytes"], expected[bits])
                inherited = build_plan(model, **kwargs)
                self.assertEqual(inherited["tiers"]["ram"]["dense_bytes"], expected[32])
        with mock.patch.dict(os.environ, {"GLM53_BITS": "8"}):
            self.assertEqual(analyze_model(model)["dense_bytes"], scan["dense_bytes"])

    def test_glm53_rejects_a_dense_precision_the_engine_cannot_load(self):
        model = self._glm53_model()
        for bits in ("0", "16", "", "invalid"):
            with self.subTest(bits=bits), self.assertRaisesRegex(ValueError, "GLM53_BITS"):
                build_plan(model, context=32, env={"GLM53_BITS": bits},
                           available_memory=16 * GB, available_disk=16 * GB, gpus=[])

    def test_glm53_plan_exports_the_expert_cache_knob_the_engine_reads(self):
        """glm53.c reads GLM53_EXPERT_GB, not RAM_GB. --auto-tier exported
        RAM_GB (inert) and relied on COLI_PLAN_CAP as argv, which a direct
        engine launch never sees."""
        plan = build_plan(self._glm53_model(), context=32, available_memory=32 * GB,
                          available_disk=1, gpus=[], cpu_sockets=1)
        cache = plan["tiers"]["ram"]["expert_cache_bytes"]
        self.assertGreater(cache, 0)
        expected = f"{cache / GB:.3f}"
        self.assertIn("GLM53_EXPERT_GB", plan["tune"])
        self.assertEqual(plan["tune"]["GLM53_EXPERT_GB"]["value"], expected)
        self.assertNotIn("K3_EXPERT_GB", plan["tune"])
        env = environment_for_plan(plan, {})
        self.assertIn("GLM53_EXPERT_GB", env)
        self.assertEqual(env["GLM53_EXPERT_GB"], expected)
        self.assertNotIn("K3_EXPERT_GB", env)
    def test_auto_ram_keeps_reserve_below_small_finite_headroom(self):
        # T15: detecting a 2 GB cgroup budget and later inflating it to the
        # planner's historical 8 GB floor is still over-admission. Preserve the
        # ordinary 12% reserve and never export more RAM than the finite value.
        with mock.patch("resource_plan.memory_available", return_value=2 * GB):
            plan = build_plan(self.model, available_disk=1, gpus=[])
        ram = plan["tiers"]["ram"]
        self.assertEqual(plan["memory"]["available_bytes"], 2 * GB)
        self.assertEqual(ram["budget_bytes"], int(2 * GB * 0.88))
        self.assertLessEqual(ram["budget_bytes"], ram["available_bytes"])
        self.assertEqual(environment_for_plan(plan)["RAM_GB"], "1.760")

    def test_auto_ram_never_exceeds_finite_headroom_on_unified_memory(self):
        gpu = {"index": 0, "name": "NVIDIA GB10", "total_bytes": 130 * GB,
               "free_bytes": 128 * GB, "unified_memory": True}
        with mock.patch("resource_plan.memory_available", return_value=2 * GB):
            plan = build_plan(self.model, available_disk=1, gpus=[gpu])
        ram = plan["tiers"]["ram"]
        self.assertTrue(plan["memory"]["unified"])
        self.assertEqual(ram["budget_bytes"], int(2 * GB * 0.88))
        self.assertLessEqual(ram["budget_bytes"], ram["available_bytes"])

    def test_auto_ram_rejects_known_zero_headroom(self):
        # current >= limit is authoritative exhaustion, not the old
        # unavailable-probe sentinel. Never turn it into an 8 GB launch --
        # whether the zero was probed or handed in explicitly.
        with mock.patch("resource_plan.memory_available", return_value=0), \
             self.assertRaisesRegex(ValueError, "memory budget is exhausted"):
            build_plan(self.model, available_disk=1, gpus=[])
        with self.assertRaisesRegex(ValueError, "memory budget is exhausted"):
            build_plan(self.model, available_memory=0, available_disk=1, gpus=[])

    def test_auto_ram_retains_legacy_fallback_only_when_probe_is_unknown(self):
        # None is the tri-state's "nothing could measure it": the historical
        # 8 GB fallback and the historical 0 in the report, never a refusal.
        with mock.patch("resource_plan.memory_available", return_value=None):
            plan = build_plan(self.model, available_disk=1, gpus=[])
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 8 * GB)
        self.assertEqual(plan["memory"]["available_bytes"], 0)

    def test_malformed_cgroup_input_is_a_typed_refusal_not_a_fallback(self):
        # A present but malformed controller reaches the caller with its
        # reason; it is a ValueError, so existing handlers still catch it.
        error = CgroupFormatError("malformed cgroup memory limit: /sys/fs/cgroup/memory.max")
        with mock.patch("resource_plan.memory_available", side_effect=error), \
             self.assertRaises(CgroupFormatError) as context:
            build_plan(self.model, available_disk=1, gpus=[])
        self.assertIsInstance(context.exception, ValueError)
        self.assertIn("memory.max", str(context.exception))

    def test_explicit_small_ram_budget_is_not_silently_inflated(self):
        plan = build_plan(self.model, ram_gb=2, available_memory=16 * GB,
                          available_disk=1, gpus=[])
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 2 * GB)

    def test_cpu_socket_count_is_positive(self):
        self.assertGreaterEqual(cpu_socket_count(), 1)

    # #379: read_ssd_probe reads the cached F_NOCACHE measurement colibri.c writes to
    # <model>/.coli_ssd on its first Metal+darwin startup. This is the read-only
    # Python-side half of the probe contract (S4) -- never re-measures, never
    # guesses, so every case here is pure file parsing. Trust mirrors the
    # engine: only a v2 cache recorded on THIS volume (matching st_dev) counts.
    def _write_v2_cache(self, gbs="14.322", dev=None):
        if dev is None:
            dev = os.stat(self.model).st_dev
        # byte-exact: text mode would CRLF-translate on Windows and the strict
        # reader would (correctly) reject the fixture as garbage
        (self.model / ".coli_ssd").write_bytes(f"v2 {gbs} {dev}\n".encode("ascii"))

    def test_ssd_probe_missing_file_returns_none(self):
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_reads_cached_v2_value(self):
        self._write_v2_cache("14.322")
        self.assertEqual(read_ssd_probe(self.model), 14.322)

    def test_ssd_probe_foreign_volume_v2_returns_none(self):
        # A model dir rsync'd to another drive carries the OLD volume's
        # measurement; the engine re-probes it, so doctor/plan must not show it.
        self._write_v2_cache("14.322", dev=os.stat(self.model).st_dev + 1)
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_legacy_bare_number_returns_none(self):
        # Pre-v2 caches were written before cold-range steering existed, so the
        # value may be page-cache contamination; the engine re-probes and
        # upgrades, and until then there is no number worth surfacing.
        (self.model / ".coli_ssd").write_bytes(b"14.322\n")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_unparsable_file_returns_none(self):
        (self.model / ".coli_ssd").write_bytes(b"not-a-number\n")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_empty_file_returns_none(self):
        (self.model / ".coli_ssd").write_bytes(b"")
        self.assertIsNone(read_ssd_probe(self.model))

    def test_ssd_probe_grammar_matches_c_reader_vectors(self):
        # THE parity pin (#386 fix round): parse_ssd_cache() must accept exactly
        # the grammar colibri.c's coli_ssd_cache_parse() accepts. Both suites
        # consume the same vector file; edit it and both sides re-judge.
        vectors = Path(__file__).parent / "fixtures" / "ssd_cache_vectors.txt"
        unescape = {"n": b"\n", "r": b"\r", "t": b"\t", "0": b"\x00",
                    "s": b" ", "\\": b"\\"}
        checked = 0
        for raw in vectors.read_text(encoding="utf-8").splitlines():
            if not raw or raw.startswith("#"):
                continue
            fields = raw.split("\t")
            payload = b""
            escaped = fields[-1] if len(fields) > 1 else ""
            i = 0
            while i < len(escaped):
                if escaped[i] == "\\" and i + 1 < len(escaped):
                    self.assertIn(escaped[i + 1], unescape, f"bad escape in: {raw!r}")
                    payload += unescape[escaped[i + 1]]
                    i += 2
                else:
                    payload += escaped[i].encode("utf-8")
                    i += 1
            kind, gbs, dev = parse_ssd_cache(payload)
            if fields[0] == "garbage":
                self.assertEqual((kind, gbs, dev), (None, None, None),
                                 f"garbage accepted: {payload!r} -> {(kind, gbs, dev)}")
            elif fields[0] == "legacy":
                self.assertEqual((kind, gbs, dev), ("legacy", float(fields[1]), None),
                                 f"legacy misread: {payload!r}")
            elif fields[0] == "v2":
                self.assertEqual((kind, gbs, dev), ("v2", float(fields[1]), int(fields[2])),
                                 f"v2 misread: {payload!r}")
            else:
                self.fail(f"unknown vector kind: {fields[0]}")
            checked += 1
        self.assertGreaterEqual(checked, 40, "vector file suspiciously short")

    def test_ssd_probe_state_classifies_every_case(self):
        # #386 r2, F10: doctor/plan wording keys off these states -- a cache
        # that exists but is not trusted must never read "no cached probe yet".
        self.assertEqual(ssd_probe_state(self.model), ("absent", None))
        self._write_v2_cache("14.322")
        self.assertEqual(ssd_probe_state(self.model), ("ok", 14.322))
        self._write_v2_cache("14.322", dev=os.stat(self.model).st_dev + 1)
        self.assertEqual(ssd_probe_state(self.model), ("foreign", None))
        (self.model / ".coli_ssd").write_bytes(b"14.322\n")
        self.assertEqual(ssd_probe_state(self.model), ("legacy", None))
        (self.model / ".coli_ssd").write_bytes(b"inf\n")
        self.assertEqual(ssd_probe_state(self.model), ("garbage", None))

    def test_ssd_probe_surfaces_in_plan_and_format(self):
        self._write_v2_cache("14.3")
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertEqual(plan["ssd_probe_gbs"], 14.3)
        self.assertEqual(plan["ssd_probe_state"], "ok")
        self.assertIn("14.3 GB/s", format_plan(plan))

    def test_ssd_probe_pending_states_surface_in_format(self):
        (self.model / ".coli_ssd").write_bytes(b"14.3\n")   # legacy
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertIsNone(plan["ssd_probe_gbs"])
        self.assertEqual(plan["ssd_probe_state"], "legacy")
        self.assertIn("legacy cache pending engine upgrade", format_plan(plan))

    def test_ssd_probe_absent_from_plan_and_format_when_not_cached(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1)
        self.assertIsNone(plan["ssd_probe_gbs"])
        self.assertNotIn("F_NOCACHE", format_plan(plan))

    def test_builds_bounded_three_tier_plan(self):
        gpus = [{"index": 0, "name": "test-gpu", "total_bytes": 12 * GB,
                 "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, context=32, vram_gb=20,
                          available_memory=32 * GB, available_disk=100 * GB, gpus=gpus,
                          physical_cpus=24, cpu_sockets=2)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["policy"]["name"], "quality")
        self.assertEqual(plan["cpu"]["physical_cores"], 24)
        self.assertEqual(plan["cpu"]["sockets"], 2)
        self.assertTrue(plan["policy"]["preserve_quantization"])
        self.assertFalse(plan["tiers"]["vram"]["requires_host_backing"])
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 16 * GB)
        self.assertLessEqual(plan["tiers"]["vram"]["budget_bytes"], 8 * GB)
        self.assertIn("clamped", plan["warnings"][0])
        self.assertIn("0:test-gpu", format_plan(plan))

    def test_glm_kv_slots_scale_the_planned_state_pool(self):
        one = build_plan(self.model, context=32, kv_slots=1,
                         available_memory=32 * GB, available_disk=1, gpus=[])
        four = build_plan(self.model, context=32, kv_slots=4,
                          available_memory=32 * GB, available_disk=1, gpus=[])
        self.assertEqual(four["tiers"]["ram"]["sequence_state_bytes"],
                         one["tiers"]["ram"]["sequence_state_bytes"])
        delta = (four["tiers"]["ram"]["runtime_bytes"] -
                 one["tiers"]["ram"]["runtime_bytes"])
        per_slot = (one["tiers"]["ram"]["sequence_state_bytes"] +
                    one["tiers"]["ram"]["fixed_state_bytes"])
        self.assertEqual(delta, 3 * per_slot)

    def test_olmoe_plans_instead_of_refusing(self):
        # #1066: OLMoE used to refuse in `coli plan`/`doctor` because its
        # planner_geometry was None (which under-reserved as zero-byte KV). With
        # the adapter it plans, charging an fp32 K and V cache per layer sized at
        # num_attention_heads * head_dim (mirrors olmoe.c:1019-1020), no fixed
        # recurrent state.
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            (model / "config.json").write_text(json.dumps({
                "model_type": "olmoe",
                "num_hidden_layers": 2,
                "hidden_size": 32,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "num_experts": 2,
                "num_experts_per_tok": 2,
                "intermediate_size": 16,
                "vocab_size": 100,
            }))
            write_shard(model / "model.safetensors", [
                ("model.embed_tokens.weight", 100),
                ("model.layers.0.mlp.experts.0.gate_proj.weight", 30),
                ("model.layers.0.mlp.experts.1.gate_proj.weight", 30),
            ])
            plan = build_plan(model, context=32, kv_slots=1,
                              available_memory=32 * GB, available_disk=1, gpus=[])
            ram = plan["tiers"]["ram"]
            # layers=2, ctx=32, heads=4, head_dim=32//4=8, K and V, fp32:
            self.assertEqual(ram["sequence_state_bytes"], 2 * 32 * 4 * 8 * 2 * 4)
            self.assertEqual(ram["fixed_state_bytes"], 0)

    def test_glm_dsa_state_is_charged_only_when_every_indexer_weight_exists(self):
        config = json.loads((self.model / "config.json").read_text())
        config.update({"index_head_dim": 16,
                       "indexer_types": ["full", "shared"]})
        (self.model / "config.json").write_text(json.dumps(config))
        absent = build_plan(self.model, context=32, available_memory=32 * GB,
                            available_disk=1, gpus=[])
        write_shard(self.model / "indexer.safetensors", [
            ("model.layers.0.self_attn.indexer.wq_b.weight", 4),
        ])
        present = build_plan(self.model, context=32, available_memory=32 * GB,
                             available_disk=1, gpus=[])
        self.assertEqual(
            present["tiers"]["ram"]["sequence_state_bytes"] -
            absent["tiers"]["ram"]["sequence_state_bytes"],
            1 * 32 * 16 * 4)

    def test_unified_memory_uses_one_shared_pool(self):
        gpus = [{"index": 0, "name": "NVIDIA GB10", "total_bytes": 130 * GB,
                 "free_bytes": 128 * GB, "unified_memory": True}]
        plan = build_plan(self.model, ram_gb=100, vram_gb=65,
                          available_memory=121 * GB, available_disk=1,
                          gpus=gpus, physical_cpus=8, cpu_sockets=1)
        ram = plan["tiers"]["ram"]["budget_bytes"]
        vram = plan["tiers"]["vram"]["budget_bytes"]
        self.assertTrue(plan["memory"]["unified"])
        self.assertLessEqual(ram + vram + plan["model"]["dense_bytes"], 121 * GB)
        self.assertTrue(any("share one physical memory" in warning
                            for warning in plan["warnings"]))

    def test_nvidia_unified_device_is_marked_from_name(self):
        output = "0, NVIDIA GB10, 130000, 120000\n"
        with mock.patch("resource_plan.subprocess.run",
                        return_value=subprocess.CompletedProcess(
                            args=[], returncode=0, stdout=output, stderr="")):
            from resource_plan import _discover_nvidia_gpus
            devices = _discover_nvidia_gpus()
        self.assertTrue(devices[0]["unified_memory"])

    # --- identity-only devices -------------------------------------------
    # A device can be discovered without its free memory being qualified as a
    # Colibri placement budget. That is the state of a Windows AMD device found
    # through hipInfo: the runtime may well report a free figure, but on an
    # integrated part it describes the same physical pages the host RAM tier is
    # already counting. Until a later slice qualifies that relationship, such a
    # device carries free_bytes=None -- "unknown for planning", which is NOT the
    # same claim as free_bytes=0 ("measured, and none is free").

    def _identity_only_gpu(self):
        return {"index": 0, "name": "AMD Radeon(TM) 8060S Graphics",
                "total_bytes": 78 * GB, "free_bytes": None,
                "unified_memory": True}

    def test_identity_only_gpu_is_planned_without_a_free_memory_value(self):
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=[self._identity_only_gpu()],
                          physical_cpus=8, cpu_sockets=1)
        # It is still reported: the hardware exists and the user should see it.
        names = [gpu["name"] for gpu in plan["tiers"]["vram"]["devices"]]
        self.assertIn("AMD Radeon(TM) 8060S Graphics", names)
        text = format_plan(plan)
        self.assertIn("8060S", text)
        # ...and the reader is told why it earns no tier, rather than being
        # left to read "0.0 GB hot tier" as "the card is full".
        self.assertIn("identity only", text)
        self.assertTrue(any("not qualified as a placement budget" in warning
                            for warning in plan["warnings"]))

    def test_plan_wording_is_backend_neutral_without_a_gpu(self):
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=[], physical_cpus=8,
                          cpu_sockets=1)
        text = format_plan(plan)
        self.assertIn("no supported GPU detected", text)
        self.assertNotIn("NVIDIA", text)
        # But it buys no tier.
        self.assertEqual(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertEqual(plan["tiers"]["vram"]["expert_capacity"], 0)

    def test_identity_only_gpu_decides_nothing_a_cpu_only_host_would_not(self):
        gpu_plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                              available_disk=1, gpus=[self._identity_only_gpu()],
                              physical_cpus=8, cpu_sockets=1)
        cpu_plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                              available_disk=1, gpus=[], physical_cpus=8,
                              cpu_sockets=1)
        # Presence alone must not reclassify the bottleneck or move DRAFT.
        self.assertEqual(gpu_plan["bottleneck_class"], cpu_plan["bottleneck_class"])
        self.assertNotEqual(gpu_plan["bottleneck_class"], "mixed")
        self.assertEqual(gpu_plan["tune"].get("DRAFT"), cpu_plan["tune"].get("DRAFT"))
        # Nor may it switch on the resident pipeline.
        self.assertNotIn("COLI_CUDA_PIPE", gpu_plan["tune"])
        # The strongest statement of the contract: for the same inputs, the
        # recommended environment is byte-identical to the CPU-only host's.
        # PIN_GB=all may legitimately appear in BOTH -- that is the no-GPU
        # residency advice (_auto_tune, `not has_gpu`), not a VRAM-derived
        # budget -- so equality is the assertion, not absence.
        env = environment_for_plan(gpu_plan, {"PIN": "stats.txt"})
        cpu_env = environment_for_plan(cpu_plan, {"PIN": "stats.txt"})
        self.assertEqual(env, cpu_env)
        self.assertNotIn("COLI_CUDA_PIPE", env)
        self.assertNotIn("COLI_CUDA", env)
        self.assertNotIn("COLI_GPU", env)
        self.assertNotIn("CUDA_EXPERT_GB", env)
        self.assertNotEqual(env.get("PIN_GB"), f"{gpu_plan['tiers']['vram']['budget_bytes'] / GB:.3f}")

    def test_identity_only_gpu_does_not_claim_vram_is_in_use(self):
        # The "already in use" warning divides free by total. With no qualified
        # free value there is nothing to divide, and telling the user to stop a
        # running engine would be a fabricated diagnosis.
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=[self._identity_only_gpu()],
                          physical_cpus=8, cpu_sockets=1)
        self.assertFalse(any("already in use" in warning
                             for warning in plan["warnings"]))

    def test_measured_zero_free_memory_still_plans_as_before(self):
        # free_bytes=0 is a MEASUREMENT, not the unqualified state, and keeps
        # every behaviour it had: the tier is empty because the card is full,
        # the pipeline knob is still offered, and the in-use warning still fires.
        gpus = [{"index": 0, "name": "full-gpu", "total_bytes": 12 * GB,
                 "free_bytes": 0}]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, physical_cpus=8,
                          cpu_sockets=1)
        self.assertEqual(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertEqual(plan["tune"]["COLI_CUDA_PIPE"]["value"], "1")
        self.assertTrue(any("already in use" in warning
                            for warning in plan["warnings"]))

    def test_mixed_fleet_plans_only_the_qualified_device(self):
        gpus = [self._identity_only_gpu(),
                {"index": 1, "name": "discrete", "total_bytes": 12 * GB,
                 "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, physical_cpus=8,
                          cpu_sockets=1)
        env = environment_for_plan(plan)
        # The qualified card earns a tier; the unqualified one must not be
        # named to the engine as a placement target.
        self.assertGreater(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertEqual(env.get("COLI_GPU"), "1")
        self.assertNotIn("COLI_GPUS", env)

    # --- Windows AMD discovery through hipInfo ----------------------------
    # Captured from hipInfo.exe shipped with the Windows HIP SDK (TheRock and
    # ROCm 7.1 both emit this layout). Trimmed to the fields the parser reads
    # plus a few it must ignore; the "89.39 GB" spellings are verbatim, and
    # hipInfo divides by 1024 (it prints a 65536-byte shared block as 64.00 KB).
    HIPINFO_INTEGRATED = """\
--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon(TM) 8060S Graphics
pciBusID:                         196
totalGlobalMem:                   89.39 GB
sharedMemPerBlock:                64.00 KB
isIntegrated:                     1
gcnArchName:                      gfx1151
peers:
non-peers:                        device#0

memInfo.total:                    89.39 GB
memInfo.free:                     89.24 GB (100%)
"""

    HIPINFO_DISCRETE = """\
--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon RX 7900 XTX
totalGlobalMem:                   24.00 GB
isIntegrated:                     0
gcnArchName:                      gfx1100

memInfo.total:                    24.00 GB
memInfo.free:                     23.50 GB (97%)
"""

    def _run_hipinfo(self, stdout, returncode=0):
        """Discover with hipInfo located and returning `stdout`."""
        from resource_plan import _discover_amd_gpus
        completed = subprocess.CompletedProcess(args=[], returncode=returncode,
                                                stdout=stdout, stderr="")
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("resource_plan._hipinfo_executable",
                        return_value=Path("C:/sdk/bin/hipInfo.exe")), \
             mock.patch("resource_plan.subprocess.run", return_value=completed):
            return _discover_amd_gpus()

    def test_windows_integrated_amd_device_is_identity_only(self):
        devices = self._run_hipinfo(self.HIPINFO_INTEGRATED)
        self.assertEqual(len(devices), 1)
        gpu = devices[0]
        self.assertEqual(gpu["index"], 0)
        self.assertEqual(gpu["name"], "AMD Radeon(TM) 8060S Graphics")
        self.assertEqual(gpu["arch"], "gfx1151")
        self.assertTrue(gpu["unified_memory"])
        self.assertEqual(gpu["total_bytes"], int(89.39 * 1024 ** 3))
        # The whole point of the slice: hipInfo DID report free memory, and it
        # deliberately did not become a planning budget.
        self.assertIsNone(gpu["free_bytes"])

    def test_windows_positive_hipinfo_free_memory_never_becomes_a_budget(self):
        # Belt and braces on the regression that matters most: the fixture says
        # 89.24 GB free (100%), so any leak of that number into planning would
        # show up as a non-zero VRAM tier here.
        self.assertIn("memInfo.free", self.HIPINFO_INTEGRATED)
        devices = self._run_hipinfo(self.HIPINFO_INTEGRATED)
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=devices, physical_cpus=8,
                          cpu_sockets=1)
        self.assertEqual(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertNotIn("COLI_CUDA_PIPE", plan["tune"])
        self.assertNotIn("CUDA_EXPERT_GB", environment_for_plan(plan))

    def test_windows_discrete_amd_device_is_not_marked_unified(self):
        devices = self._run_hipinfo(self.HIPINFO_DISCRETE)
        self.assertEqual(len(devices), 1)
        self.assertFalse(devices[0]["unified_memory"])
        self.assertEqual(devices[0]["arch"], "gfx1100")
        # Windows AMD free memory is unqualified for every part in this slice,
        # discrete included: we have no discrete Windows AMD host to qualify it
        # against, and guessing is what this slice exists to avoid.
        self.assertIsNone(devices[0]["free_bytes"])

    HIPINFO_SECOND_DEVICE = """\
--------------------------------------------------------------------------------
device#                           1
Name:                             AMD Radeon RX 7900 XTX
totalGlobalMem:                   24.00 GB
isIntegrated:                     0
gcnArchName:                      gfx1100

memInfo.total:                    24.00 GB
memInfo.free:                     23.50 GB (97%)
"""

    def test_windows_parses_every_complete_device_block(self):
        devices = self._run_hipinfo(self.HIPINFO_INTEGRATED
                                    + self.HIPINFO_SECOND_DEVICE)
        self.assertEqual([gpu["index"] for gpu in devices], [0, 1])
        self.assertEqual([gpu["unified_memory"] for gpu in devices], [True, False])
        self.assertEqual([gpu["free_bytes"] for gpu in devices], [None, None])

    def test_windows_without_hipinfo_reports_no_device(self):
        from resource_plan import _discover_amd_gpus
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("resource_plan._hipinfo_executable", return_value=None):
            self.assertEqual(_discover_amd_gpus(), [])

    def test_windows_hipinfo_failure_invents_nothing(self):
        from resource_plan import _discover_amd_gpus
        with mock.patch.object(sys, "platform", "win32"), \
             mock.patch("resource_plan._hipinfo_executable",
                        return_value=Path("C:/sdk/bin/hipInfo.exe")), \
             mock.patch("resource_plan.subprocess.run",
                        side_effect=subprocess.CalledProcessError(1, "hipInfo")):
            self.assertEqual(_discover_amd_gpus(), [])

    def test_windows_incomplete_device_block_is_not_half_trusted(self):
        # A block that names a device but reports no memory must produce no
        # record at all rather than one with fabricated zeros.
        partial = ("--------------------------------------------------------\n"
                   "device#                           0\n"
                   "Name:                             AMD Radeon(TM) 8060S Graphics\n"
                   "isIntegrated:                     1\n")
        self.assertEqual(self._run_hipinfo(partial), [])
        headless = ("--------------------------------------------------------\n"
                    "device#                           0\n"
                    "totalGlobalMem:                   89.39 GB\n"
                    "memInfo.total:                    89.39 GB\n")
        self.assertEqual(self._run_hipinfo(headless), [])

    def test_hipinfo_lookup_prefers_the_colibri_runtime_over_a_stale_install(self):
        # This machine has had two Windows HIP installs at once. The engine
        # binds the runtime COLI_HIP_RUNTIME_DIR names, and hipInfo sits beside
        # amdhip64_7.dll in that same directory, so its answer describes the
        # runtime that will actually be used. An ambient HIP_PATH pointing at a
        # different SDK must not win.
        from resource_plan import _hipinfo_executable
        with tempfile.TemporaryDirectory() as root:
            chosen = Path(root) / "therock" / "bin"
            stale = Path(root) / "sdk" / "bin"
            for directory in (chosen, stale):
                directory.mkdir(parents=True)
                (directory / "hipInfo.exe").write_bytes(b"")
            env = {"COLI_HIP_RUNTIME_DIR": str(chosen),
                   "HIP_PATH": str(stale.parent)}
            with mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.dict(os.environ, env, clear=False):
                self.assertEqual(_hipinfo_executable(), chosen / "hipInfo.exe")
            # With no Colibri-specific runtime selected, the SDK root is used.
            with mock.patch.object(sys, "platform", "win32"), \
                 mock.patch.dict(os.environ, {"HIP_PATH": str(stale.parent)}, clear=True):
                self.assertEqual(_hipinfo_executable(), stale / "hipInfo.exe")

    def test_linux_amd_discovery_still_uses_rocm_smi(self):
        from resource_plan import _discover_amd_gpus
        output = ("device,Card Series,VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
                  "card0,Instinct MI300X,68719476736,8589934592\n")
        completed = subprocess.CompletedProcess(args=[], returncode=0,
                                                stdout=output, stderr="")
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch("resource_plan.subprocess.run",
                        return_value=completed) as run:
            devices = _discover_amd_gpus()
        self.assertEqual(run.call_args[0][0][0], "rocm-smi")
        self.assertEqual(devices[0]["total_bytes"], 68719476736)
        self.assertEqual(devices[0]["free_bytes"], 68719476736 - 8589934592)
        self.assertFalse(devices[0]["unified_memory"])

    def test_auto_tier_thread_count_uses_physical_cores(self):
        # End-to-end for #325: build_plan + environment_for_plan must export the
        # physical (not logical SMT) core count as OMP_NUM_THREADS. The original
        # suite passed physical_cpus=24 explicitly, so it never exercised the
        # real physical_cpu_count() probe whose single-core failure pinned decode.
        def lscpu(stdout):
            return subprocess.CompletedProcess(args=[], returncode=0,
                                               stdout=stdout, stderr="")
        # 1 socket, 12 cores, 2 SMT siblings -> 24 threads, 12 physical cores.

        # The parser must return 12 physical cores under BOTH lscpu layouts:
        #  - 2-col: `lscpu -p=core,socket` emits exactly [core,socket] (this is
        #           what the probe actually requests; the previous fields[1]/[2]
        #           indexing skipped every line here and fell through to the
        #           logical count -> the regression JustVugg caught).
        #  - 3-col: bare `lscpu -p` prepends a CPU column -> [cpu,core,socket].
        # Taking the last two fields is correct in both cases.
        layouts = {
            "2-col (-p=core,socket)": (
                "# core,socket\n" +
                "\n".join(f"{core},0" for core in range(12) for _ in range(2))),
            "3-col (bare -p, CPU prefix)": (
                "# CPU,Core,Socket\n" +
                "\n".join(f"{cpu},{core},0" for core in range(12) for cpu in range(2))),
        }
        for label, blob in layouts.items():
            with mock.patch("resource_plan.subprocess.run",
                            return_value=lscpu(blob)), \
                 mock.patch.object(sys, "platform", "linux"):
                plan = build_plan(self.model, available_memory=16 * GB,
                                  available_disk=1, gpus=[])
                env = environment_for_plan(plan)
            self.assertEqual(plan["cpu"]["physical_cores"], 12, label)
            self.assertEqual(env["OMP_NUM_THREADS"], "12", label)

    def test_plan_does_not_set_omp_affinity_vars(self):
        # The real #325 regression: --auto-tier set OMP_PROC_BIND=spread +
        # OMP_PLACES=cores, which ran before the engine's overwrite=0 setenv and
        # so won, collapsing the OpenMP team to one CPU on the reporter's 64-core
        # Linux box even though OMP_NUM_THREADS was correct. The plan must leave
        # affinity to the engine's own hot-thread tuning (which prefers 'close').
        plan = build_plan(self.model, available_memory=16 * GB,
                          available_disk=1, gpus=[], physical_cpus=64)
        env = environment_for_plan(plan)
        self.assertEqual(env["OMP_NUM_THREADS"], "64")
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertNotIn("OMP_PLACES", env)

    def test_plan_conserves_budget_and_experts_above_256gb(self):
        # Regression for #325's reporter: a 512 GB machine loading the whole
        # model into RAM. Verify the budget math stays exact at large RAM sizes
        # (no integer truncation, no over-allocation, no experts lost between
        # tiers). Checked at 256/512/800 GB to bracket the reporter's box.
        for ram_gb in (256, 512, 800):
            plan = build_plan(self.model, ram_gb=ram_gb, available_disk=1,
                              gpus=[], physical_cpus=64)
            ram = plan["tiers"]["ram"]
            # RAM budget never over-allocated: dense + runtime + cache <= budget.
            allocated = (ram["dense_bytes"] + ram["runtime_bytes"]
                         + ram["expert_cache_bytes"])
            self.assertLessEqual(allocated, ram["budget_bytes"],
                                 f"over-allocated RAM at {ram_gb} GB")
            # Every expert byte is accounted for exactly once across the tiers.
            tiers = plan["tiers"]
            tiered = (tiers["vram"]["hot_expert_bytes"]
                      + ram["warm_expert_bytes"]
                      + tiers["disk"]["cold_expert_bytes"])
            self.assertEqual(tiered, plan["model"]["expert_bytes"],
                             f"expert bytes lost/duplicated at {ram_gb} GB")
            # A positive RAM budget yields a non-negative cache and a sensible cap.
            self.assertGreaterEqual(ram["expert_cache_bytes"], 0)
            self.assertGreaterEqual(ram["cache_slots_per_layer"], 0)

    def test_filters_requested_devices(self):
        gpus = [{"index": 0, "name": "a", "total_bytes": 8 * GB, "free_bytes": 8 * GB}]
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=gpus, gpu_indices=[1])
        self.assertEqual(plan["tiers"]["vram"]["devices"], [])
        self.assertIn("not detected", plan["warnings"][0])

    def write_qwen38(self):
        config = {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text", "num_hidden_layers": 2,
                "hidden_size": 8, "vocab_size": 16,
                "max_position_embeddings": 128, "eos_token_id": 2,
                "layer_types": ["linear_attention", "qwen_sparse_attention"],
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "head_dim": 4, "rope_parameters": {"partial_rotary_factor": 0.5},
                "indexer_kv_heads": 1, "indexer_head_dim": 2,
                "indexer_n_heads": 1, "indexer_budget": 2,
                "indexer_compress_ratio": 1,
                "linear_num_key_heads": 1, "linear_num_value_heads": 1,
                "linear_key_head_dim": 2, "linear_value_head_dim": 2,
                "linear_conv_kernel_dim": 2, "hc_count": 4, "hc_lowrank": 4,
                "ple_layer_ids": [1], "ple_embed_dim": 8,
                "ple_conv_kernel_size": 2, "ngram_size": 3,
                "heads_per_ngram": 1, "split_ngram_parts": 1,
                "num_experts": 2, "num_experts_per_tok": 1,
                "moe_intermediate_size": 4,
                "shared_expert_intermediate_size": 4,
            },
        }
        (self.model / "config.json").write_text(json.dumps(config))
        MiB = 1 << 20
        tensors = [("model.embed_tokens.weight", 256, "BF16", [16, 8]),
                   # dense matmul matrices the engine offers to the tier: one
                   # big enough to go (4 MiB BF16 -> 2 MiB int8), one under the
                   # 1 MiB line that stays on the CPU, one PLE projection that
                   # is never offered
                   ("model.layers.0.linear_attn.in_proj_qkv.weight", 4 * MiB, "BF16", [1024, 2048]),
                   ("model.layers.0.mlp.gate.weight", 1024, "BF16", [2, 256]),
                   ("model.layers.1.ple.key_proj.weight", 4 * MiB, "BF16", [1024, 2048])]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"model.layers.0.mlp.experts.0.{projection}"
            tensors.append((prefix + ".weight", 32, "F8_E4M3"))
            tensors.append((prefix + ".weight_scale_inv", 4, "F32"))
            tensors.append((
                f"model.layers.0.mlp.experts.1.{projection}.weight", 128, "F32"
            ))
        write_shard(self.model / "model.safetensors", tensors)

    def test_qwen38_plan_prices_heterogeneous_cache_exports_cap_and_plans_vram(self):
        self.write_qwen38()
        MiB = 1 << 20
        analysis = analyze_model(self.model)
        self.assertEqual(analysis["dense_bytes"], 256 + 4 * MiB + 1024 + 4 * MiB)
        # The stage-1 trunk offload: int8 bytes of the offered matrices only.
        self.assertEqual(analysis["trunk_int8_bytes"], 2 * MiB)
        # The three native FP8 sidecars are retained once in the normalized
        # scale bank, not once per cache slot.
        self.assertEqual(analysis["expert_fixed_bytes"], 12)
        self.assertEqual(analysis["expert_bytes"], 480)
        # A slot can receive either expert. It must use the larger retained
        # representation, not the median of unlike source dtypes.
        self.assertEqual(analysis["expert_bytes_by_layer"], {0: 384})
        self.assertEqual(analysis["per_cap_bytes"], 384)
        gpu = {"index": 0, "name": "unrelated", "total_bytes": 16 * GB,
               "free_bytes": 14 * GB, "unified_memory": True}
        plan = build_plan(self.model, context=64, available_memory=16 * GB,
                          available_disk=16 * GB, gpus=[gpu])
        # Qwen3.8 has the CUDA VRAM expert tier (fp8 streaming mode): a
        # qualified device is planned, the environment names it, and the RAM
        # cache cap is still exported -- VRAM is a stage above the LRU, not
        # a replacement for it.
        self.assertEqual([device["index"] for device in plan["tiers"]["vram"]["devices"]], [0])
        self.assertGreater(plan["tiers"]["vram"]["budget_bytes"], 0)
        self.assertTrue(any(item["target"] == "VRAM" for item in plan["decisions"]))
        # The trunk goes first, out of the same VRAM, and the plan says so.
        self.assertEqual(plan["tiers"]["vram"]["trunk_bytes"], 2 * MiB)
        self.assertTrue(any(item["reason"] == "dense trunk as int8 residents"
                            for item in plan["decisions"]))
        self.assertIn("int8 trunk", format_plan(plan))
        cap = plan["tiers"]["ram"]["cache_slots_per_layer"]
        self.assertGreaterEqual(cap, 1)
        environment = environment_for_plan(plan)
        self.assertEqual(environment["COLI_PLAN_CAP"], str(cap))
        self.assertEqual(environment["COLI_CUDA"], "1")
        self.assertEqual(environment["COLI_GPU"], "0")
        for variable in ("Q38_NATIVE_FP8", "Q38_NATIVE_BF16"):
            with self.subTest(variable=variable), self.assertRaisesRegex(
                    ValueError, "requires native expert storage"):
                environment_for_plan(plan, {variable: "0"})
        plan["tiers"]["ram"]["cache_slots_per_layer"] = 0
        with self.assertRaisesRegex(ValueError, "one expert slot"):
            environment_for_plan(plan)
        selected = build_plan(self.model, context=64, gpu_indices=[0], available_memory=16 * GB,
                              available_disk=16 * GB, gpus=[gpu])
        self.assertEqual([device["index"] for device in selected["tiers"]["vram"]["devices"]], [0])
        capped = build_plan(self.model, context=64, vram_gb=4, available_memory=16 * GB,
                            available_disk=16 * GB, gpus=[gpu])
        self.assertLessEqual(capped["tiers"]["vram"]["budget_bytes"], 4 * GB - 2 * MiB)
        self.assertEqual(capped["tiers"]["vram"]["trunk_bytes"], 2 * MiB)
        # A budget too small for the trunk leaves it on the CPU: experts only.
        tiny = build_plan(self.model, context=64, vram_gb=0.001, available_memory=16 * GB,
                          available_disk=16 * GB, gpus=[gpu])
        self.assertEqual(tiny["tiers"]["vram"]["trunk_bytes"], 0)

    def test_qwen38_plan_holds_the_trunk_as_int8_rows(self):
        # q38_trunk_cpu_int8 (on by default) keeps the trunk's large matrices as
        # int8 rows: the plan's RAM is the checkpoint's BF16 less those bytes.
        self.write_qwen38()
        MiB = 1 << 20
        analysis = analyze_model(self.model)
        kwargs = dict(context=64, available_memory=16 * GB, available_disk=16 * GB, gpus=[])
        plan = build_plan(self.model, env={}, **kwargs)
        self.assertEqual(plan["tiers"]["ram"]["dense_bytes"], analysis["dense_bytes"] - 2 * MiB + 4 * 1024)
        bf16 = build_plan(self.model, env={"Q38_TRUNK_CPU_INT8": "0"}, **kwargs)
        self.assertEqual(bf16["tiers"]["ram"]["dense_bytes"], analysis["dense_bytes"])

    def test_qwen38_cpu_int8_selection_is_recomputed_for_each_plan(self):
        self.write_qwen38()
        kwargs = dict(context=64, available_memory=16 * GB, available_disk=16 * GB, gpus=[])
        baseline = analyze_model(self.model)["dense_bytes"]
        # Every call after the scan hits the same cache. These knobs must still
        # affect RAM: neither the GPU inventory nor a cached default is the CPU table.
        for env in ({"Q38_TRUNK_SKIP": "dnqkv"},
                    {"Q38_TRUNK_SKIP": "router,dnqkv,attnq"},
                    {"Q38_TRUNK_MIN_KB": "999999999"},
                    {"Q38_TRUNK_MIN_KB": "-1"}):
            with self.subTest(env=env):
                plan = build_plan(self.model, env=env, **kwargs)
                self.assertEqual(plan["tiers"]["ram"]["dense_bytes"], baseline)
        regular = build_plan(self.model, env={}, **kwargs)
        no_gpu = build_plan(self.model, env={"Q38_TRUNK_GPU": "0"}, **kwargs)
        self.assertEqual(regular["tiers"]["ram"]["dense_bytes"],
                         no_gpu["tiers"]["ram"]["dense_bytes"])
        self.assertLess(regular["tiers"]["ram"]["dense_bytes"], baseline)

    def test_qwen38_cpu_int8_threshold_includes_scales_and_loaded_dtype(self):
        self.write_qwen38()
        kwargs = dict(context=64, available_memory=16 * GB, available_disk=16 * GB, gpus=[])
        rows, cols = 256, 4092
        # At exactly 1 MiB including row scales, but below 1 MiB of values.
        # F16 is expanded to F32 by q38_load_weight, BF16 remains native.
        for dtype, width, loaded_width in (("BF16", 2, 2), ("F16", 2, 4), ("F32", 4, 4)):
            with self.subTest(dtype=dtype):
                write_shard(self.model / "model.safetensors", [
                    ("model.layers.0.linear_attn.in_proj_qkv.weight", rows * cols * width,
                     dtype, [rows, cols]),
                    # Embedding and PLE are never CPU-int8 trunk candidates,
                    # even when they meet its size threshold.
                    ("model.embed_tokens.weight", rows * cols * 2, "BF16", [rows, cols]),
                    ("model.layers.1.ple.key_proj.weight", rows * cols * 2, "BF16", [rows, cols]),
                ])
                before = build_plan(self.model, env={"Q38_TRUNK_CPU_INT8": "0"}, **kwargs)
                after = build_plan(self.model, env={}, **kwargs)
                self.assertEqual(before["tiers"]["ram"]["dense_bytes"] - after["tiers"]["ram"]["dense_bytes"],
                                 rows * cols * loaded_width - (rows * cols + 4 * rows))
                above = build_plan(self.model, env={"Q38_TRUNK_MIN_KB": "1025"}, **kwargs)
                self.assertEqual(above["tiers"]["ram"]["dense_bytes"], before["tiers"]["ram"]["dense_bytes"])

    def test_vulkan_device_only_reserves_cpu_components_and_unknown_matrices(self):
        self.write_qwen38()
        write_shard(self.model / "retained.safetensors", [
            ("model.layers.0.input_layernorm.weight", 32, "BF16", [16]),
            ("model.layers.0.linear_attn.conv1d.weight", 256, "BF16", [4, 4, 8]),
            ("model.layers.0.unknown_projection.weight", 2048, "F32", [16, 32]),
        ])
        kwargs = dict(context=64, ram_gb=4, available_memory=16 * GB,
                      available_disk=16 * GB, gpus=[],
                      vulkan={"type": "discrete", "budget_bytes": 16 * GB})
        on = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1"}
        kept = build_plan(self.model, env=dict(on, COLI_VK_DENSE_HOST="1"), **kwargs)["tiers"]["ram"]
        dropped = build_plan(self.model, env=dict(on, COLI_VK_DENSE_HOST="0"), **kwargs)["tiers"]["ram"]
        # Small BF16 vectors/conv weights expand to F32 in this loader.
        retained = 256 + 32 * 2 + 256 * 2 + 2048
        self.assertEqual(dropped["dense_bytes"], retained)
        self.assertEqual(dropped["dense_on_device_bytes"], kept["dense_bytes"] - retained)
        self.assertEqual(dropped["expert_cache_bytes"] - kept["expert_cache_bytes"],
                         dropped["dense_on_device_bytes"])

    def test_qwen36_vulkan_credit_keeps_vision_vectors_and_unsupported_storage(self):
        model = self._qwen_dense_model()
        write_shard(model / "model.safetensors", [
            ("model.language_model.embed_tokens.weight", 8192, "BF16", [64, 64]),
            ("model.language_model.layers.0.self_attn.q_proj.weight", 8192, "BF16", [64, 64]),
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", 8192, "BF16", [64, 64]),
            ("model.language_model.layers.0.mlp.shared_expert_gate.weight", 128, "BF16", [64]),
            ("model.visual.blocks.0.attn.qkv.weight", 8192, "BF16", [64, 64]),
            ("model.language_model.layers.0.self_attn.k_proj.weight", 2048, "U8", [64, 32]),
        ])
        kwargs = dict(context=32, available_memory=16 * GB, available_disk=16 * GB, gpus=[],
                      vulkan={"type": "discrete", "budget_bytes": 16 * GB})
        on = {"COLI_VULKAN": "1", "COLI_VK_DENSE_HOST": "0"}
        host = build_plan(model, env=dict(on, COLI_VK_DENSE_HOST="1"), **kwargs)["tiers"]["ram"]
        for bits, credit in (("8", 4096), ("4", 2048), ("16", 0)):
            with self.subTest(bits=bits):
                ram = build_plan(model, env=dict(on, COLI_DENSE_BITS=bits), **kwargs)["tiers"]["ram"]
                self.assertEqual(ram["dense_on_device_bytes"], credit)
                self.assertEqual(ram["dense_bytes"], host["dense_bytes"] - credit)

    def test_vulkan_device_only_dense_leaves_the_embedding_in_ram(self):
        # COLI_VULKAN=1 with the dense weights on the device only: the RAM budget
        # holds the embedding table alone, and the experts' cache takes the rest.
        self.write_qwen38()
        MiB = 1 << 20
        kwargs = dict(context=64, ram_gb=4, available_memory=16 * GB,
                      available_disk=16 * GB, gpus=[])
        # Model the device explicitly: a CPU-only runner must exercise the same
        # placement policy without relying on an installed Vulkan driver.
        dgpu = {"type": "discrete", "budget_bytes": 16 * GB}
        host = build_plan(self.model, env={"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1",
                                           "COLI_VK_DENSE_HOST": "1"}, vulkan=dgpu, **kwargs)
        only = build_plan(self.model, env={"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1",
                                           "COLI_VK_DENSE_HOST": "0"}, vulkan=dgpu, **kwargs)
        self.assertEqual(only["tiers"]["ram"]["dense_bytes"], 256)
        self.assertEqual(only["tiers"]["ram"]["dense_on_device_bytes"],
                         host["tiers"]["ram"]["dense_bytes"] - 256)
        self.assertEqual(host["tiers"]["ram"]["dense_on_device_bytes"], 0)
        self.assertGreater(only["tiers"]["ram"]["expert_cache_bytes"],
                           host["tiers"]["ram"]["expert_cache_bytes"])
        self.assertIn("on the Vulkan device only", format_plan(only))
        self.assertNotIn("on the Vulkan device only", format_plan(host))
        # unset: an integrated GPU holds them when the chain is on (no host copy:
        # the same RAM twice); a discrete one when they fit its free memory
        igpu = build_plan(self.model, env={"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1"},
                          vulkan={"type": "integrated"}, **kwargs)
        self.assertEqual(igpu["tiers"]["ram"]["dense_bytes"], 256)
        small = build_plan(self.model, env={"COLI_VULKAN": "1"},
                           vulkan={"type": "discrete", "budget_bytes": GB + MiB}, **kwargs)
        self.assertEqual(small["tiers"]["ram"]["dense_bytes"], host["tiers"]["ram"]["dense_bytes"])
        # without COLI_VULKAN nothing changes
        cpu = build_plan(self.model, env={"COLI_VK_DENSE_HOST": "0"}, **kwargs)
        self.assertEqual(cpu["tiers"]["ram"]["dense_bytes"], host["tiers"]["ram"]["dense_bytes"])

    def test_vulkan_without_a_device_keeps_dense_weights_in_ram(self):
        self.write_qwen38()
        kwargs = dict(context=64, ram_gb=4, available_memory=16 * GB,
                      available_disk=16 * GB, gpus=[])
        cpu = build_plan(self.model, env={}, **kwargs)["tiers"]["ram"]
        with mock.patch("setup_hw.detect_vulkan", return_value={"devices": []}):
            for host in (None, "0", "1"):
                with self.subTest(dense_host=host):
                    env = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1"}
                    if host is not None:
                        env["COLI_VK_DENSE_HOST"] = host
                    ram = build_plan(self.model, env=env, **kwargs)["tiers"]["ram"]
                    self.assertEqual(ram["dense_on_device_bytes"], 0)
                    self.assertEqual(ram["dense_bytes"], cpu["dense_bytes"])
                    self.assertEqual(ram["expert_cache_bytes"], cpu["expert_cache_bytes"])

    def test_vulkan_device_only_decision_follows_the_engine_rule(self):
        from resource_plan import vk_dense_device_only
        on = {"COLI_VULKAN": "1"}
        igpu, dgpu = {"type": "integrated"}, {"type": "discrete", "budget_bytes": 16 * GB}
        cases = [
            ({}, dgpu, 1, False),                                     # no Vulkan
            (dict(on), dgpu, 5 * GB, True),                           # discrete, fits
            (dict(on), dgpu, 15.5 * GB, False),                       # discrete, 1 GiB short
            (dict(on, COLI_VK_DENSE_HOST="1"), dgpu, 1, False),       # copies kept
            (dict(on, COLI_VK_CHAIN="0", COLI_VK_DENSE="0"), dgpu, 1, False),  # dense on the CPU
            (dict(on), igpu, 1, False),                               # qwen38's chain is off there
            (dict(on, COLI_VK_CHAIN="1"), igpu, 1, True),
            (dict(on, COLI_VK_DENSE="1"), igpu, 1, True),
            (dict(on, COLI_VK_DENSE_HOST="0", COLI_VK_CHAIN="1"), {"type": "cpu"}, 1, True),
            (dict(on, COLI_VK_CHAIN="1"), {"type": "cpu"}, 1, False),  # Lavapipe keeps them
        ]
        for env, device, dense, want in cases:
            with self.subTest(env=env, device=device["type"], dense=dense):
                self.assertEqual(vk_dense_device_only(dense, "qwen38", env, device)[0], want)
        # qwen36 and olmoe run the chain by default on an integrated GPU
        self.assertTrue(vk_dense_device_only(1, "qwen36", dict(on), igpu)[0])

    def test_vulkan_host_copies_use_free_budget_and_supported_engine(self):
        from resource_plan import vk_dense_device_only
        GiB = 1 << 30
        on = {"COLI_VULKAN": "1"}
        device = {"type": "discrete", "budget_bytes": 8 * GiB,
                  "heaps": [{"device_local": True, "usage": 4 * GiB}]}
        self.assertFalse(vk_dense_device_only(4 * GiB, "qwen38", on, device)[0])
        self.assertTrue(vk_dense_device_only(3 * GiB, "qwen38", on, device)[0])
        self.assertFalse(vk_dense_device_only(3 * GiB + 1, "qwen38", on, device)[0])
        forced = dict(on, COLI_VK_DENSE_HOST="0")
        for family in ("laya", "gliner_decide", "qwen_image", None):
            with self.subTest(family=family):
                self.assertFalse(vk_dense_device_only(1, family, forced, device)[0])
        self.assertFalse(vk_dense_device_only(
            1, "qwen38", dict(forced, COLI_CUDA="1"), device)[0])
        igpu = {"type": "integrated"}
        # Dense and chain are independent decisions in the runtime. Disabling
        # the per-matrix path alone still leaves qwen36's default chain on.
        self.assertTrue(vk_dense_device_only(
            1, "qwen36", dict(on, COLI_VK_DENSE="0"), igpu)[0])
        self.assertTrue(vk_dense_device_only(
            1, "qwen38", dict(on, COLI_VK_CHAIN="0", COLI_VK_TIER="0"), igpu)[0])
        for gpu in (device, igpu):
            with self.subTest(glm_device=gpu["type"]):
                off = dict(on, COLI_VK_CHAIN="0", COLI_VK_TIER="0")
                self.assertFalse(vk_dense_device_only(1, "glm", off, gpu)[0])
                self.assertTrue(vk_dense_device_only(1, "glm", dict(off, COLI_VK_DENSE="1"), gpu)[0])

    def test_vulkan_integrated_dense_copy_is_still_physical_ram(self):
        self.write_qwen38()
        kwargs = dict(context=64, ram_gb=4, available_memory=16 * GB,
                      available_disk=16 * GB, gpus=[], vulkan={"type": "integrated"})
        cpu = build_plan(self.model, env={}, **kwargs)
        on = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1"}
        kept = build_plan(self.model, env=dict(on, COLI_VK_DENSE_HOST="1"), **kwargs)
        dropped = build_plan(self.model, env=dict(on, COLI_VK_DENSE_HOST="0"), **kwargs)
        cpu_ram, kept_ram, dropped_ram = (p["tiers"]["ram"] for p in (cpu, kept, dropped))
        device_copy = dropped_ram["dense_on_device_bytes"]
        self.assertGreater(device_copy, 0)
        self.assertEqual(dropped_ram["shared_device_dense_bytes"], device_copy)
        self.assertEqual(kept_ram["shared_device_dense_bytes"], device_copy)
        self.assertEqual(dropped_ram["runtime_bytes"], cpu_ram["runtime_bytes"] + device_copy)
        self.assertEqual(kept_ram["runtime_bytes"], dropped_ram["runtime_bytes"])
        # One GPU copy replaces one CPU copy, so dropping the duplicate recovers
        # exactly its bytes without pretending that the weights vanished from RAM.
        self.assertEqual(dropped_ram["expert_cache_bytes"], cpu_ram["expert_cache_bytes"])
        self.assertEqual(dropped_ram["expert_cache_bytes"] - kept_ram["expert_cache_bytes"],
                         device_copy)
        self.assertTrue(dropped["memory"]["unified"])

    def test_qwen38_int4_sidecar_prices_the_cache_with_its_records(self):
        # tools/convert_qwen38_experts_int4.py's index for this geometry
        # (2 layers, 2 experts, hidden 8, moe_intermediate 4): codes 16+16+16
        # bytes, scales (4+4+8) floats -> 112 bytes per expert.
        self.write_qwen38()
        sidecar = self.model / "experts-int4g64"
        sidecar.mkdir()
        index = {"format": "colibri.qwen38.experts-int4g64", "version": 1, "complete": True,
                 "layers": 2, "experts": 2, "hidden_size": 8, "moe_intermediate_size": 4,
                 "record_bytes": 112}
        (sidecar / "index.json").write_text(json.dumps(index))
        gpu = {"index": 0, "name": "unrelated", "total_bytes": 16 * GB,
               "free_bytes": 14 * GB, "unified_memory": True}
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("Q38_EXPERT_INT4", None)
            plan = build_plan(self.model, context=64, available_memory=16 * GB,
                              available_disk=16 * GB, gpus=[gpu])
            model = plan["model"]
            self.assertTrue(model["qwen38_int4_experts"])
            self.assertEqual((model["per_cap_bytes"], model["expert_bytes"],
                              model["expert_fixed_bytes"]), (112, 224, 0))
            # the engine's tier declines int4 experts, so nothing is planned on the GPU
            self.assertEqual(plan["tiers"]["vram"]["devices"], [])
            self.assertNotIn("COLI_CUDA", environment_for_plan(plan))
            with self.assertRaisesRegex(ValueError, "run on the CPU"):
                build_plan(self.model, context=64, gpu_indices=[0], available_memory=16 * GB,
                           available_disk=16 * GB, gpus=[gpu])
            # what the engine would not use is not priced: FP8 forced, or a
            # conversion still running
            os.environ["Q38_EXPERT_INT4"] = "0"
            forced = build_plan(self.model, context=64, available_memory=16 * GB,
                                available_disk=16 * GB, gpus=[])
            self.assertEqual(forced["model"]["per_cap_bytes"], 384)
            os.environ.pop("Q38_EXPERT_INT4")
            (sidecar / "index.json").write_text(json.dumps(dict(index, complete=False)))
            partial = build_plan(self.model, context=64, available_memory=16 * GB,
                                 available_disk=16 * GB, gpus=[])
            self.assertNotIn("qwen38_int4_experts", partial["model"])
            self.assertEqual(partial["model"]["per_cap_bytes"], 384)

    def test_cli_emits_versioned_json(self):
        cli = Path(__file__).parents[1] / "coli"
        run = subprocess.run([
            sys.executable, str(cli), "plan", "--model", str(self.model),
            "--gpu", "none", "--json",
        ], text=True, capture_output=True, check=True)
        plan = json.loads(run.stdout)
        self.assertEqual(plan["version"], 2)
        self.assertEqual(plan["model"]["expert_count"], 2)

    def test_applies_plan_without_overriding_explicit_settings(self):
        gpus = [
            {"index": 0, "name": "a", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
            {"index": 1, "name": "b", "total_bytes": 12 * GB, "free_bytes": 10 * GB},
        ]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=2)
        env = environment_for_plan(plan, {"RAM_GB": "12", "PIN": "stats.txt",
                                               "COLI_GPUS": "1"})
        self.assertEqual(env["RAM_GB"], "12")
        self.assertEqual(env["COLI_CUDA"], "1")
        self.assertEqual(env["COLI_GPUS"], "1")
        self.assertEqual(env["OMP_NUM_THREADS"], str(plan["cpu"]["physical_cores"]))
        # The plan must NOT set OMP_PROC_BIND / OMP_PLACES on any platform:
        # the engine's own hot-thread tuning owns affinity (it prefers
        # OMP_PROC_BIND=close for the back-to-back per-expert matmuls). Setting
        # spread + cores here ran before the engine's overwrite=0 setenv and so
        # won, collapsing the team to one CPU on some libgomp topologies (#325).
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertNotIn("OMP_PLACES", env)
        self.assertEqual(env["PIN_GB"], env["CUDA_EXPERT_GB"])

        explicit_threads = environment_for_plan(plan, {"OMP_NUM_THREADS": "7",
                                                        "OMP_PROC_BIND": "close"})
        self.assertEqual(explicit_threads["OMP_NUM_THREADS"], "7")
        self.assertEqual(explicit_threads["OMP_PROC_BIND"], "close")

        if sys.platform.startswith("linux"):
            self.assertEqual(env["COLI_NUMA"], "1")
            explicit_numa = environment_for_plan(plan, {"COLI_NUMA": "0"})
            self.assertEqual(explicit_numa["COLI_NUMA"], "0")

    def test_single_socket_plan_does_not_enable_numa(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], physical_cpus=8, cpu_sockets=1)
        self.assertNotIn("COLI_NUMA", environment_for_plan(plan))

    def test_auto_tune_mtp_off_when_compute_bound(self):
        # Tiny model with 64 GB RAM and no GPU: all experts fit in RAM with no
        # warm tier, so the plan classifies as compute-bound.
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=24,
                          cpu_sockets=2)
        # With such a small model fully in RAM and no GPU, bottleneck is compute
        self.assertEqual(plan["bottleneck_class"], "compute")
        self.assertIn("DRAFT", plan["tune"])
        self.assertEqual(plan["tune"]["DRAFT"]["value"], "0")
        env = environment_for_plan(plan)
        self.assertEqual(env["DRAFT"], "0")
        explicit = environment_for_plan(plan, {"DRAFT": "3"})
        self.assertEqual(explicit["DRAFT"], "3")

    def test_auto_tune_mtp_off_when_disk_low_hit(self):
        # Use a model large enough that 8 GB RAM can't hold all experts.
        big = tempfile.TemporaryDirectory()
        bigmodel = Path(big.name)
        (bigmodel / "config.json").write_text(json.dumps({
            "model_type": "glm_moe_dsa",
            "num_hidden_layers": 2, "n_routed_experts": 4,
            "kv_lora_rank": 4, "qk_rope_head_dim": 2,
            "qk_nope_head_dim": 3, "v_head_dim": 5, "num_attention_heads": 2,
        }))
        expert_size = 3 * GB  # each expert 3 GB → 12 GB total, won't fit in 8 GB budget
        write_shard(bigmodel / "out-00000.safetensors", [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_a_proj.weight", 200),
        ])
        for i in range(4):
            write_shard(bigmodel / f"out-{i+1:05d}.safetensors", [
                (f"model.layers.1.mlp.experts.{i}.gate_proj.weight", expert_size),
            ])
        plan = build_plan(bigmodel, ram_gb=0, available_memory=4 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=8,
                          cpu_sockets=1)
        big.cleanup()
        self.assertEqual(plan["bottleneck_class"], "disk")
        self.assertLess(plan["projected_hit_rate"], 0.90)
        self.assertEqual(plan["tune"]["DRAFT"]["value"], "0")

    def test_auto_tune_pipe_multi_gpu(self):
        gpus = [
            {"index": 0, "name": "a", "total_bytes": 32 * GB, "free_bytes": 30 * GB},
            {"index": 1, "name": "b", "total_bytes": 32 * GB, "free_bytes": 30 * GB},
        ]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=2)
        self.assertEqual(plan["tune"]["COLI_CUDA_PIPE"]["value"], "2")
        env = environment_for_plan(plan)
        self.assertEqual(env["COLI_CUDA_PIPE"], "2")

    def test_auto_tune_pipe_single_gpu(self):
        gpus = [{"index": 0, "name": "a", "total_bytes": 12 * GB, "free_bytes": 10 * GB}]
        plan = build_plan(self.model, ram_gb=16, available_memory=32 * GB,
                          available_disk=1, gpus=gpus, cpu_sockets=1)
        self.assertEqual(plan["tune"]["COLI_CUDA_PIPE"]["value"], "1")

    def test_auto_tune_numa_hint_for_cpu_only(self):
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=1, gpus=[], physical_cpus=64, cpu_sockets=2)
        self.assertIn("_numa_hint", plan["tune"])
        self.assertIn("numactl", plan["tune"]["_numa_hint"])
        self.assertIn("auto-tune", format_plan(plan))

    def test_format_plan_shows_tune_and_hit_rate(self):
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=100 * GB, gpus=[], physical_cpus=24,
                          cpu_sockets=1)
        text = format_plan(plan)
        self.assertIn("hit", text)
        self.assertIn("auto-tune", text)
        self.assertIn("DRAFT", text)

    def test_cpu_binary_does_not_apply_gpu_tier(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[{"index": 0, "name": "a", "total_bytes": 8 * GB,
                                 "free_bytes": 8 * GB}])
        env = environment_for_plan(plan, cuda_enabled=False)
        self.assertIn("RAM_GB", env)
        self.assertNotIn("COLI_CUDA", env)
        disabled = environment_for_plan(plan, {"COLI_CUDA": "0"}, cuda_enabled=True)
        self.assertNotIn("COLI_GPU", disabled)
        self.assertNotIn("CUDA_EXPERT_GB", disabled)

    def test_rejects_unknown_policy_and_marks_experimental_policy(self):
        with self.assertRaisesRegex(ValueError, "unknown policy"):
            build_plan(self.model, available_memory=16 * GB, available_disk=1,
                       gpus=[], policy="fast-ish")
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="experimental-fast")
        self.assertFalse(plan["policy"]["quality_preserving"])
        self.assertFalse(plan["policy"]["preserve_router"])

    def test_balanced_policy_enables_lossless_live_repin(self):
        plan = build_plan(self.model, available_memory=16 * GB, available_disk=1,
                          gpus=[], policy="balanced")
        env = environment_for_plan(plan)
        self.assertEqual(env["COLI_POLICY"], "balanced")
        self.assertEqual(env["REPIN"], "64")
        explicit = environment_for_plan(plan, {"REPIN": "0"})
        self.assertEqual(explicit["REPIN"], "0")

    def test_plan_explains_hot_warm_and_cold_placement(self):
        plan = build_plan(self.model, ram_gb=4, vram_gb=0,
                          available_memory=4 * GB, available_disk=1, gpus=[])
        self.assertEqual([item["target"] for item in plan["decisions"]],
                         ["VRAM", "RAM", "Disk"])
        self.assertIn("quality-preserving yes", format_plan(plan))
        self.assertIn("expected_bottleneck", plan)

    def test_disk_plan_requires_probe_and_recommends_measured_tuning(self):
        info = analyze_model(self.model)
        info.update(expert_bytes=40 * GB, typical_expert_bytes=1 * GB,
                    max_expert_bytes=1 * GB, per_cap_bytes=2 * GB)
        with mock.patch("resource_plan.ssd_probe_state",
                        return_value=("missing", None)), \
             mock.patch("resource_plan.analyze_model", return_value=info):
            plan = build_plan(self.model, ram_gb=4, available_memory=4 * GB,
                              available_disk=1, gpus=[])
        self.assertEqual([action["id"] for action in plan["next_actions"]],
                         ["measure-storage", "measure-residency"])
        self.assertEqual(plan["next_actions"][0]["priority"], "required")
        text = format_plan(plan)
        self.assertIn("next actions:", text)
        self.assertIn("coli tune --model <model>", text)

    def test_trusted_storage_probe_removes_probe_action(self):
        info = analyze_model(self.model)
        info.update(expert_bytes=40 * GB, typical_expert_bytes=1 * GB,
                    max_expert_bytes=1 * GB, per_cap_bytes=2 * GB)
        with mock.patch("resource_plan.ssd_probe_state",
                        return_value=("trusted", 7.5)), \
             mock.patch("resource_plan.analyze_model", return_value=info):
            plan = build_plan(self.model, ram_gb=4, available_memory=4 * GB,
                              available_disk=1, gpus=[])
        self.assertEqual([action["id"] for action in plan["next_actions"]],
                         ["measure-residency"])

    def test_resident_plan_recommends_kernel_measurement(self):
        plan = build_plan(self.model, ram_gb=64, available_memory=64 * GB,
                          available_disk=1, gpus=[])
        self.assertEqual(plan["bottleneck_class"], "compute")
        self.assertEqual(plan["next_actions"][0]["id"], "measure-kernels")


class VulkanPartialChainTest(unittest.TestCase):
    """The partial chain (docs/vulkan.md, "A partial chain"): resource_plan.vk_chain_fit
    predicts the engine's N with vkc_fit's rule from the device's budget, and the plan
    credits only the N layers' host copies."""

    CONFIG = {
        "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
        "hidden_size": 128, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32,
        "q_lora_rank": 128, "qk_rope_head_dim": 16, "o_groups": 1, "o_lora_rank": 128,
        "sliding_window": 8, "index_n_heads": 2, "index_head_dim": 32, "index_topk": 2,
        "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
        "moe_intermediate_size": 128, "num_hash_layers": 1, "num_nextn_predict_layers": 1,
        "hc_mult": 2, "hc_sinkhorn_iters": 3, "vocab_size": 128, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-06, "hc_eps": 1e-06, "routed_scaling_factor": 1.5, "swiglu_limit": 10.0,
        "rope_theta": 10000.0, "compress_rope_theta": 40000.0,
        "rope_scaling": {"type": "yarn", "factor": 1.0, "original_max_position_embeddings": 128,
                         "beta_fast": 32, "beta_slow": 1},
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write_v4(self, ratios):
        """The tiny DeepSeek V4 fixture's geometry (tools/make_deepseek_v4_tiny.py) with
        len(ratios) layers: every tensor coli_v4_layer_plan names, and its experts."""
        c = dict(self.CONFIG, num_hidden_layers=len(ratios), compress_ratios=list(ratios) + [0])
        (self.model / "config.json").write_text(json.dumps(c))
        width = {"F8_E4M3": 1, "F8_E8M0": 1, "BF16": 2, "F32": 4, "I64": 8, "I8": 1}
        tensors = []

        def add(name, dtype, *shape):
            count = 1
            for n in shape:
                count *= n
            tensors.append((name, count * width[dtype], dtype, list(shape)))

        def fp8(name, rows, cols):
            add(name + ".weight", "F8_E4M3", rows, cols)
            add(name + ".scale", "F8_E8M0", -(-rows // 128), -(-cols // 128))
        add("embed.weight", "BF16", 128, 128)
        add("head.weight", "BF16", 128, 128)
        add("norm.weight", "BF16", 128)
        add("hc_head_fn", "F32", 2, 256)
        for i, r in enumerate(ratios):
            p = f"layers.{i}."
            add(p + "attn.attn_sink", "F32", 4)
            add(p + "attn.kv_norm.weight", "BF16", 32)
            add(p + "attn.q_norm.weight", "BF16", 128)
            for name, rows, cols in (("wkv", 32, 128), ("wo_a", 128, 128), ("wo_b", 128, 128),
                                     ("wq_a", 128, 128), ("wq_b", 128, 128)):
                fp8(p + "attn." + name, rows, cols)
            add(p + "attn_norm.weight", "BF16", 128)
            if r:
                proj = (2 if r == 4 else 1) * 32
                add(p + "attn.compressor.ape", "F32", r, proj)
                add(p + "attn.compressor.norm.weight", "BF16", 32)
                add(p + "attn.compressor.wgate.weight", "BF16", proj, 128)
                add(p + "attn.compressor.wkv.weight", "BF16", proj, 128)
            if r == 4:
                add(p + "attn.indexer.compressor.ape", "F32", 4, 64)
                add(p + "attn.indexer.compressor.norm.weight", "BF16", 32)
                add(p + "attn.indexer.compressor.wgate.weight", "BF16", 64, 128)
                add(p + "attn.indexer.compressor.wkv.weight", "BF16", 64, 128)
                add(p + "attn.indexer.weights_proj.weight", "BF16", 2, 128)
                fp8(p + "attn.indexer.wq_b", 64, 128)
            add(p + "ffn.gate.weight", "BF16", 4, 128)
            if i == 0:
                add(p + "ffn.gate.tid2eid", "I64", 128, 2)
            else:
                add(p + "ffn.gate.bias", "F32", 4)
            for name in ("w1", "w2", "w3"):
                fp8(p + "ffn.shared_experts." + name, 128, 128)
            add(p + "ffn_norm.weight", "BF16", 128)
            for site in ("attn", "ffn"):
                add(p + f"hc_{site}_base", "F32", 8)
                add(p + f"hc_{site}_fn", "F32", 8, 256)
                add(p + f"hc_{site}_scale", "F32", 3)
            for e in range(4):
                for name in ("w1", "w2", "w3"):
                    add(p + f"ffn.experts.{e}.{name}.scale", "F8_E8M0", 128, 4)
                    add(p + f"ffn.experts.{e}.{name}.weight", "I8", 128, 64)
        write_shard(self.model / "model.safetensors", tensors)
        return analyze_model(self.model)

    def test_deepseek_v4_layout_is_the_engines(self):
        # The numbers the engine printed for tools/make_deepseek_v4_tiny.py's fixture on
        # Lavapipe ("[VK] deepseek_v4 chain fit: ... the engine's 1891840 B ..., layers
        # 159720 278760 192104 B"): the same per-layer bytes and fixed bytes here.
        from resource_plan import vk_chain_fit, vk_fit_pools
        info = self.write_v4([0, 4, 8])
        env = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1"}
        fit = vk_chain_fit(info, "deepseek_v4", env, {"type": "cpu", "budget_bytes": 64 * GB})
        self.assertEqual(fit["layers"], [159720, 278760, 192104])
        self.assertEqual(fit["fixed"] - vk_fit_pools(0), 1891840)
        self.assertEqual(fit["n"], 3)
        # COLI_VK_CHAIN_ROWS lowers the chunk the fit counts (the window rings, the scratch)
        small = vk_chain_fit(info, "deepseek_v4", dict(env, COLI_VK_CHAIN_ROWS="3"), {"type": "cpu", "budget_bytes": 64 * GB})
        self.assertLess(small["fixed"], fit["fixed"])
        self.assertTrue(all(a < b for a, b in zip(small["layers"], fit["layers"])))

    def test_n_follows_the_engines_rule(self):
        from resource_plan import VkChainLayout, vk_chain_fit, vk_fit_pools
        MiB, GiB = 1 << 20, 1 << 30
        layout = VkChainLayout([100 * MiB, 200 * MiB, 300 * MiB, 400 * MiB], 50 * MiB, 70 * MiB)
        on = {"COLI_VULKAN": "1"}
        pools = vk_fit_pools(0)
        fixed = 50 * MiB + pools
        with mock.patch.dict("resource_plan._VK_CHAIN_LAYOUT", {"deepseek_v4": lambda info, env, vk: layout}):
            def fit(free, env=on, kind="discrete", heaps=()):
                return vk_chain_fit({}, "deepseek_v4", env, {"type": kind, "budget_bytes": free,
                                                             "heaps": list(heaps)})
            room = GiB + fixed   # free = reserve + fixed + what the layers may take
            self.assertEqual((fit(room + 1000 * MiB + 70 * MiB)["n"], fit(room + 1000 * MiB + 70 * MiB)["tail"]), (4, True))
            self.assertEqual((fit(room + 1000 * MiB)["n"], fit(room + 1000 * MiB)["tail"]), (4, False))
            self.assertEqual(fit(room + 1000 * MiB - 1)["n"], 3)
            self.assertEqual(fit(room + 600 * MiB)["n"], 3)
            self.assertEqual(fit(room + 300 * MiB - 1)["n"], 1)
            self.assertEqual(fit(room + 99 * MiB)["n"], 0)
            self.assertEqual(fit(GiB // 2)["n"], 0)
            # the heaps' usage is taken off the budget, as coli_vk_free_bytes does
            self.assertEqual(fit(room + 600 * MiB, heaps=[{"device_local": True, "usage": 300 * MiB}])["n"], 2)
            # the reserve is COLI_VK_TIER_RESERVE_GB
            self.assertEqual(fit(fixed + 600 * MiB, dict(on, COLI_VK_TIER_RESERVE_GB="0"))["n"], 3)
            # COLI_VK_CHAIN_LAYERS forces N, capped at L; the tail goes up with every layer
            for want, n, tail in (("2", 2, False), ("0", 0, False), ("9", 4, True), ("auto", 3, False)):
                got = fit(room + 600 * MiB, dict(on, COLI_VK_CHAIN_LAYERS=want))
                self.assertEqual((got["n"], got["tail"], got["forced"]), (n, tail, want != "auto"))
            # COLI_VK_DEVICE_CAP_MB: the device is the cap, with small pool blocks
            cap = 2 * GiB
            capped = fit(64 * GiB, dict(on, COLI_VK_DEVICE_CAP_MB=str(cap // MiB)))
            self.assertEqual(capped["free"], cap)
            self.assertLess(capped["fixed"], fixed)
            self.assertEqual(capped["n"], 3)
            # no chain, no fit: the plan as before
            self.assertIsNone(fit(64 * GiB, dict(on, COLI_VK_CHAIN="0")))
            self.assertIsNone(fit(64 * GiB, {}))

    def test_device_only_credits_the_layers_on_the_device(self):
        from resource_plan import vk_chain_fit, vk_fit_pools
        info = self.write_v4([0, 4, 8, 0, 0, 4])
        kwargs = dict(context=64, ram_gb=4, available_memory=16 * GB, available_disk=16 * GB, gpus=[])
        on = {"COLI_VULKAN": "1", "COLI_VK_CHAIN": "1", "COLI_VK_DENSE_HOST": "0",
              "COLI_VK_TIER_RESERVE_GB": "0"}
        layers = vk_chain_fit(info, "deepseek_v4", on, {"type": "discrete", "budget_bytes": 64 * GB})["layers"]
        # what the device can drop for the first k layers: fp8 matrices and the compressors'
        # bf16 projections (coli_v4_dense_device_only_tensor)
        def droppable(k):
            total = 0
            for t in info["dense_tensors"]:
                parts = t["name"].split(".")
                if parts[0] != "layers" or int(parts[1]) >= k or len(t["shape"]) != 2:
                    continue
                if t["dtype"] == "F8_E4M3" or (t["dtype"] == "BF16" and t["name"].endswith(
                        ("compressor.wkv.weight", "compressor.wgate.weight"))):
                    total += t["size"]
            return total
        full = build_plan(self.model, env=on, vulkan={"type": "discrete", "budget_bytes": 64 * GB}, **kwargs)
        self.assertEqual(full["tiers"]["ram"]["dense_on_device_bytes"], droppable(6))
        self.assertEqual(full["tiers"]["ram"]["vk_chain_layers"]["on_device"], 6)
        for k in (1, 3, 5):
            free = vk_fit_pools(0) + vk_chain_fit(info, "deepseek_v4", on, {"type": "discrete", "budget_bytes": 64 * GB})["fixed"] \
                - vk_fit_pools(0) + sum(layers[:k]) + layers[k] // 2
            device = {"type": "discrete", "budget_bytes": free}
            with self.subTest(k=k):
                plan = build_plan(self.model, env=on, vulkan=device, **kwargs)
                ram = plan["tiers"]["ram"]
                self.assertEqual(ram["vk_chain_layers"]["on_device"], k)
                self.assertEqual(ram["dense_on_device_bytes"], droppable(k))
                self.assertEqual(full["tiers"]["ram"]["dense_bytes"] + droppable(6) - droppable(k), ram["dense_bytes"])
                self.assertIn(f"the first {k} of 6 layers", format_plan(plan))
                # forced, the same credit
                forced = build_plan(self.model, env=dict(on, COLI_VK_CHAIN_LAYERS=str(k)),
                                    vulkan={"type": "discrete", "budget_bytes": 64 * GB}, **kwargs)
                self.assertEqual(forced["tiers"]["ram"]["dense_on_device_bytes"], droppable(k))
        # no layer fits: nothing on the device, the host copies stay
        none = build_plan(self.model, env=on, vulkan={"type": "discrete", "budget_bytes": 1 << 20}, **kwargs)
        self.assertEqual(none["tiers"]["ram"]["dense_on_device_bytes"], 0)
        self.assertEqual(none["tiers"]["ram"]["vk_chain_layers"]["on_device"], 0)
        # host copies kept: no credit whatever N
        kept = build_plan(self.model, env=dict(on, COLI_VK_DENSE_HOST="1", COLI_VK_CHAIN_LAYERS="3"),
                          vulkan={"type": "discrete", "budget_bytes": 64 * GB}, **kwargs)
        self.assertEqual(kept["tiers"]["ram"]["dense_on_device_bytes"], 0)
        # an integrated GPU's device copy (physical RAM, priced once) is the N layers' alone
        igpu = {"type": "integrated", "budget_bytes": 64 * GB}
        two = build_plan(self.model, env=dict(on, COLI_VK_CHAIN_LAYERS="2"), vulkan=igpu, **kwargs)["tiers"]["ram"]
        every = build_plan(self.model, env=on, vulkan=igpu, **kwargs)["tiers"]["ram"]
        self.assertEqual(two["dense_on_device_bytes"], droppable(2))
        self.assertEqual(every["dense_on_device_bytes"], droppable(6))
        self.assertGreaterEqual(two["shared_device_dense_bytes"], droppable(2))
        self.assertLess(two["shared_device_dense_bytes"], every["shared_device_dense_bytes"])


class PhysicalCpuCountTest(unittest.TestCase):
    """Regression for #325: --auto-tier pinned decode to one core because
    physical_cpu_count() silently returned 1.

    Two root causes this locks down:
      1. lscpu -p prepends a CPU column, so `-p=core,socket` emits
         CPU,Core,Socket; counting rows counted logical SMT siblings.
      2. any probe failure fell through to ``os.cpu_count() or 1`` and the
         ``or 1`` could pin a constrained/cgroup'd box to a single core.
    """

    def _lscpu(self, stdout):
        return subprocess.CompletedProcess(args=[], returncode=0,
                                           stdout=stdout, stderr="")

    def _lscpu_topology(self, sockets, cores_per_socket, threads_per_core):
        # Real lscpu shape: socket-local core IDs repeat across sockets; the
        # CPU column (always prepended) is a unique logical-CPU index.
        rows, cpu = [], 0
        for sock in range(sockets):
            for core in range(cores_per_socket):
                for _ in range(threads_per_core):
                    rows.append(f"{cpu},{core},{sock}")
                    cpu += 1
        return "# CPU,Core,Socket\n" + "\n".join(rows)

    def test_counts_physical_cores_not_smt_threads(self):
        blob = self._lscpu_topology(sockets=2, cores_per_socket=16, threads_per_core=2)
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 32)

    def test_single_socket_no_smt(self):
        blob = self._lscpu_topology(sockets=1, cores_per_socket=8, threads_per_core=1)
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 8)

    def test_skips_offline_core_socket_fields(self):
        # VMs / large NUMA boxes emit "-" for offline core or socket IDs; that
        # used to raise ValueError, discard the whole parse, and fall through
        # to the single-core fallback.
        blob = "# CPU,Core,Socket\n0,0,0\n1,-,0\n2,1,0\n3,1,0\n"
        with mock.patch("resource_plan.subprocess.run", return_value=self._lscpu(blob)), \
             mock.patch.object(sys, "platform", "linux"):
            self.assertEqual(physical_cpu_count(), 2)

    def test_lscpu_missing_falls_back_to_logical_not_silent_one(self):
        # The bug: lscpu absent -> os.cpu_count() or 1. On a constrained box
        # os.cpu_count() can be 1. We still must never silently pick 1 without
        # a warning, and when logical cores exist they must be used.
        import os
        with mock.patch("resource_plan.subprocess.run", side_effect=FileNotFoundError), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch("resource_plan.os.cpu_count", return_value=16), \
             mock.patch("sys.stderr"):
            self.assertEqual(physical_cpu_count(), 16)

    def test_zero_logical_cores_warns_and_returns_one(self):
        # The genuine degenerate case: no probe works and os.cpu_count() is
        # None/1. Must return 1 (engine needs a positive team size) but warn.
        with mock.patch("resource_plan.subprocess.run", side_effect=FileNotFoundError), \
             mock.patch.object(sys, "platform", "linux"), \
             mock.patch("resource_plan.os.cpu_count", return_value=None), \
             mock.patch("sys.stderr"):
            self.assertEqual(physical_cpu_count(), 1)

    def test_apple_silicon_prefers_performance_cores(self):
        def sysctl(command, **kwargs):
            value = "8\n" if command[-1] == "hw.perflevel0.logicalcpu" else "10\n"
            return subprocess.CompletedProcess(args=command, returncode=0,
                                               stdout=value, stderr="")
        with mock.patch("resource_plan.subprocess.run", side_effect=sysctl), \
             mock.patch.object(sys, "platform", "darwin"):
            self.assertEqual(physical_cpu_count(), 8)


if __name__ == "__main__":
    unittest.main()


class WindowsCommitLimitTest(unittest.TestCase):
    """#1375: a 14 MB malloc failing on a 128 GB machine. Free physical memory
    is not what decides whether malloc succeeds on Windows; grantable commit
    is, and it can be far lower with a small page file."""

    def test_commit_caps_the_budget_when_lower_than_physical(self):
        self.assertEqual(windows_available_bytes(110 << 30, 40 << 30), 40 << 30)

    def test_physical_is_used_when_commit_is_larger_or_unknown(self):
        self.assertEqual(windows_available_bytes(110 << 30, 200 << 30), 110 << 30)
        self.assertEqual(windows_available_bytes(110 << 30, 0), 110 << 30)

    def test_memorystatusex_layout_is_the_documented_one(self):
        # Order matters for ctypes: a skipped field shifts every later one.
        self.assertEqual([n for n, _ in WINDOWS_MEMORYSTATUSEX_FIELDS], [
            "dwLength", "dwMemoryLoad", "ullTotalPhys", "ullAvailPhys",
            "ullTotalPageFile", "ullAvailPageFile", "ullTotalVirtual",
            "ullAvailVirtual", "ullAvailExtendedVirtual"])
