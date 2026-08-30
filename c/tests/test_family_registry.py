import json
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from family_registry import (
    FAMILIES,
    FamilyCapabilities,
    FamilyConfigError,
    FamilyDescriptor,
    FamilyLimits,
    PlannerGeometry,
    RegistryError,
    UnknownFamilyError,
    PlannerUnsupportedError,
    _build_registry,
    expert_contributions,
    fixed_resident_contribution,
    family_for_config,
    planner_geometry,
    public_metadata,
    resident_contribution,
    resolve_model,
    tuning_replay_prompt,
)


def qwen_geometry(config, context, _model_dir):
    layers = config["num_hidden_layers"]
    full = sum(kind == "full_attention" for kind in config["layer_types"])
    kv = full * context * config["num_key_value_heads"] * config["head_dim"] * 2 * 4
    conv_dim = (config["linear_num_key_heads"] * config["linear_key_head_dim"] * 2 +
                config["linear_num_value_heads"] * config["linear_value_head_dim"])
    fixed = (layers - full) * (
        config["linear_num_value_heads"] * config["linear_key_head_dim"] *
        config["linear_value_head_dim"] +
        conv_dim * (config["linear_conv_kernel_dim"] - 1)) * 4
    return PlannerGeometry(kv, fixed, 0, config["num_experts"])


def minimax_geometry(config, context, _model_dir):
    state = ((config["num_hidden_layers"] + 1) * context *
             config["num_key_value_heads"] * config["head_dim"] * 2 * 4)
    sparse = config["sparse_attention_config"]
    state += sum(bool(value) for value in sparse["sparse_attention_freq"]) * \
        context * sparse["sparse_index_dim"] * 4
    return PlannerGeometry(state, 0, 0, config["num_local_experts"])


TEST_INVENTORY = lambda _name, _size, _config, _dtype=None: ()
QWEN36_FIXTURE = FamilyDescriptor(
    id="qwen36",
    model_types=("qwen3_5_moe_text",),
    display_name="Qwen3.6",
    display_scale="",
    engine_artifact="qwen36",
    engine_aliases=(),
    engine_group="qwen36",
    internal_arch="qwen36",
    build_target="qwen36",
    process_names=("qwen36",),
    default_model_id="qwen3.6-colibri",
    cli_adapter="qwen36",
    gateway_adapter="qwen36",
    planner_id="qwen36_hybrid",
    planner_geometry=qwen_geometry,
    planner_unsupported_reason="",
    expert_inventory=TEST_INVENTORY,
    config_section="root",
    limits=FamilyLimits(8192, 262144, 1024, 8192, 1, 8, "Q36_MAXT"),
    capabilities=FamilyCapabilities(False, False, False, True),
)
MINIMAX_M3_FIXTURE = FamilyDescriptor(
    id="minimax_m3",
    model_types=("minimax_m3",),
    display_name="MiniMax M3",
    display_scale="",
    engine_artifact="colibri",
    engine_aliases=(),
    engine_group="colibri-core",
    internal_arch="minimax_m3",
    build_target="colibri",
    process_names=("colibri",),
    default_model_id="minimax-m3-colibri",
    cli_adapter="minimax_m3",
    gateway_adapter="minimax_m3",
    planner_id="minimax_m3_gqa",
    planner_geometry=minimax_geometry,
    planner_unsupported_reason="",
    expert_inventory=TEST_INVENTORY,
    config_section="root",
    limits=FamilyLimits(8192, 262144, 1024, 8192, 1, 8, "CTX"),
    capabilities=FamilyCapabilities(True, False, False, True),
)


class FamilyRegistryTest(unittest.TestCase):
    def test_production_descriptors_are_complete_unique_and_serializable(self):
        self.assertGreaterEqual(len(FAMILIES), 5)
        by_id, by_type = _build_registry(FAMILIES)
        self.assertEqual(len(by_id), len(FAMILIES))
        self.assertGreaterEqual(len(by_type), len(FAMILIES))
        for family in FAMILIES:
            with self.subTest(family=family.id):
                json.dumps(public_metadata(family))
                self.assertIn(family.id, by_id)

    def test_unknown_or_invalid_config_never_falls_back_to_glm(self):
        with self.assertRaises(UnknownFamilyError):
            family_for_config({"model_type": "qwen3_moe"})
        for config in ({}, {"model_type": ""}, {"model_type": []}, None):
            with self.subTest(config=config), self.assertRaises(FamilyConfigError):
                family_for_config(config)

    def test_model_resolution_reads_text_config_without_changing_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"model_type": "kimi_k3", "text_config": {"num_hidden_layers": 2}}
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            resolved = resolve_model(root)
            self.assertEqual(resolved.descriptor.id, "kimi")
            self.assertEqual(resolved.family_config, config["text_config"])

    def test_qwen_fixture_models_gqa_and_fixed_deltanet_state(self):
        config = {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"] +
                           ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_value_heads": 8, "linear_num_key_heads": 4,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
        }
        # qwen36 is a registered family now, so the fixture would collide on its
        # model_type alias. Assert against the production descriptor instead --
        # which is the stronger test: it pins the shipped geometry, not a copy.
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        self.assertEqual(family, by_id["qwen36"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        self.assertEqual(geometry.configured_experts, 8)
        self.assertEqual(geometry.context_state_bytes, 16_384)
        self.assertEqual(geometry.fixed_state_bytes, 6 * (8 * 8 * 8 + 128 * 3) * 4)
        for model_type in ("qwen2", "qwen3_moe", "my_qwen_model"):
            self.assertNotIn(model_type, by_type)

    def test_qwen38_fixture_resolves_nested_text_config_and_sizes_all_state(self):
        config = {
            "model_type": "qwen4_exp",
            "text_config": {
                "model_type": "qwen4_exp_text",
                "num_hidden_layers": 48,
                "hidden_size": 2560,
                "vocab_size": 248320,
                "max_position_embeddings": 262144,
                "eos_token_id": 248044,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "rope_parameters": {"partial_rotary_factor": 0.25},
                "layer_types": ["linear_attention"] * 36 + ["full_attention"] * 12,
                "indexer_kv_heads": 1, "indexer_head_dim": 128,
                "indexer_n_heads": 4, "indexer_budget": 2048,
                "indexer_compress_ratio": 4,
                "linear_num_value_heads": 48,
                "linear_num_key_heads": 16,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "hc_count": 4, "hc_lowrank": 320,
                "ple_layer_ids": [2], "ple_embed_dim": 2560,
                "ple_conv_kernel_size": 4, "ngram_size": 3,
                "heads_per_ngram": 8, "split_ngram_parts": 128,
                "num_experts": 512, "moe_intermediate_size": 640,
                "num_experts_per_tok": 10,
                "shared_expert_intermediate_size": 640,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
            resolved = resolve_model(root)
        self.assertEqual(resolved.descriptor.id, "qwen38")
        self.assertEqual(resolved.family_config["model_type"], "qwen4_exp_text")
        geometry = planner_geometry(resolved, 32)
        self.assertEqual(geometry.configured_experts, 512)
        live_context = 12 * 32 * (2 * 2 * 256 + 128) * 4
        self.assertEqual(geometry.context_state_bytes,
                         live_context + 32 * 4)
        live_fixed = (36 * (48 * 128 * 128 + 10240 * 3) * 4 +
                      10240 * 9 * 4 + 2 * 8)
        expected_fixed = live_fixed * 2 + 248320 * 4
        self.assertEqual(geometry.fixed_state_bytes, expected_fixed)
        base_row = 4 * 2560 + 2 * 2560 + 4
        gated_residual_row = base_row + 2 * 4 * 2560 + 320
        qsa_row = (base_row + 2 * 24 * 256 + 2 * 2 * 256 +
                   (4 + 1) * 128 + 24 * 256)
        ple_row = base_row + 4 * 2560
        moe_fixed = 512 + 3 * 640 + 3 * 640 + 2 * 2560
        gdn_width = 2 * 16 * 128 + 48 * 128
        gdn_fixed = (2 * gdn_width + 3 * 48 * 128 +
                     2 * 48 + 2 * 48 * 128)
        ple_fixed = 2560 + 5 * 4 * 2560 + 2560
        qsa_fixed = 4 * 128 + 128 + 2048 + 4 - 1 + max(
            2 * (32 // 4), 256 + 2048 + 4 - 1)
        legacy_workspace = (
            32 * max(gated_residual_row, qsa_row, ple_row) +
            max(moe_fixed, gdn_fixed, ple_fixed, qsa_fixed)) * 4
        moe_assignment_bytes = 2 * (2560 + 640) * 4 + 8 + 2 * 4
        moe_row_bytes = (512 + 3 * 640 + 2560 + 1) * 4 + \
            10 * moe_assignment_bytes
        moe_fixed_bytes = 512 * (4 * 4 + 8) + 4
        moe_chunk_bytes = moe_fixed_bytes + 32 * moe_row_bytes
        delta_row_bytes = (gdn_width + 2 * 48 * 128 + 2 * 48) * 4
        delta_fixed_bytes = (gdn_width + 2 * 48 * 128 + 48 * 128) * 4
        delta_chunk_bytes = delta_fixed_bytes + 32 * delta_row_bytes
        batched_workspace = 32 * base_row * 4 + max(
            moe_chunk_bytes, delta_chunk_bytes)
        expected_workspace = max(legacy_workspace, batched_workspace)
        self.assertEqual(geometry.workspace_bytes, expected_workspace)

        family = resolved.descriptor
        invalid = {
            "ngram_size": 2,
            "indexer_kv_heads": 2,
            "indexer_budget": 2047,
            "hc_count": 17,
            "ple_embed_dim": 2561,
            "linear_num_value_heads": 47,
            "num_attention_heads": 25,
        }
        for key, value in invalid.items():
            broken = json.loads(json.dumps(config["text_config"]))
            broken[key] = value
            bad = type("R", (), {"descriptor": family, "family_config": broken,
                                   "model_dir": "."})()
            with self.subTest(key=key), self.assertRaises(ValueError):
                planner_geometry(bad, 32)
        short = json.loads(json.dumps(config["text_config"]))
        short["max_position_embeddings"] = 16
        bad = type("R", (), {"descriptor": family, "family_config": short,
                               "model_dir": "."})()
        with self.assertRaisesRegex(ValueError, "max_position_embeddings"):
            planner_geometry(bad, 32)

    def test_qwen38_inventory_aggregates_per_expert_and_fused_tensor_sizes(self):
        family = next(family for family in FAMILIES if family.id == "qwen38")
        resolved = type("R", (), {"descriptor": family,
                                   "family_config": {"num_experts": 512,
                                                     "hidden_size": 32,
                                                     "moe_intermediate_size": 8},
                                   "model_dir": "."})()
        prefix = "model.language_model.layers.7.mlp.experts.23"
        self.assertEqual(expert_contributions(
            resolved, prefix + ".gate_proj.weight", 256, "F8_E4M3"),
            ((7, 23, 256),))
        self.assertEqual(expert_contributions(
            resolved, prefix + ".up_proj.weight", 512, "BF16"),
            ((7, 23, 512),))
        self.assertEqual(expert_contributions(
            resolved, prefix + ".down_proj.weight", 1024, "F32"),
            ((7, 23, 1024),))
        self.assertEqual(expert_contributions(
            resolved, prefix + ".down_proj.weight", 512, "F16"),
            ((7, 23, 1024),))
        # FP8 sidecars are normalized into a fixed per-model scale bank, not
        # copied into every cache slot.
        self.assertEqual(expert_contributions(
            resolved, prefix + ".down_proj.weight_scale_inv", 4, "F32"), ())
        self.assertEqual(fixed_resident_contribution(
            resolved, prefix + ".down_proj.weight_scale_inv", 4, "F32"), 4)
        with self.assertRaisesRegex(ValueError, "unsupported dtype/size"):
            expert_contributions(
                resolved, prefix + ".down_proj.weight_scale_inv", 16, "F32")
        fused = "model.layers.7.mlp.experts.gate_up_proj"
        fused_size = 512 * 2 * 32 * 8 * 2
        contributions = expert_contributions(resolved, fused, fused_size, "BF16")
        self.assertEqual(len(contributions), 512)
        self.assertEqual(contributions[23], (7, 23, 1024))
        contributions = expert_contributions(resolved, fused, fused_size, "F16")
        self.assertEqual(contributions[23], (7, 23, 2048))

        self.assertEqual(resident_contribution(
            resolved, "model.language_model.layers.0.self_attn.q_proj.weight",
            100, "BF16"), 100)
        self.assertEqual(resident_contribution(
            resolved, "model.language_model.layers.0.self_attn.q_proj.weight",
            100, "F16"), 200)
        self.assertEqual(resident_contribution(
            resolved, "model.language_model.layers.0.self_attn.q_norm.weight",
            100, "BF16"), 200)
        self.assertEqual(resident_contribution(
            resolved, "model.visual.blocks.0.attn.qkv.weight", 100), 0)
        self.assertEqual(resident_contribution(
            resolved, "mtp.layers.0.mlp.gate.weight", 100), 0)
        self.assertEqual(resident_contribution(
            resolved,
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_7.weight",
            100), 0)
        self.assertEqual(resident_contribution(
            resolved,
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
            24), 0)

    def test_minimax_fixture_can_share_colibri_without_becoming_glm(self):
        config = {
            "model_type": "minimax_m3",
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 8,
                "sparse_attention_freq": [0, 1],
            },
        }
        by_id, by_type = _build_registry(FAMILIES + (MINIMAX_M3_FIXTURE,))
        family = by_type[config["model_type"]]
        self.assertEqual(family.engine_artifact, by_id["glm"].engine_artifact)
        self.assertEqual(family.engine_group, by_id["glm"].engine_group)
        self.assertNotEqual(family.internal_arch, by_id["glm"].internal_arch)
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        self.assertEqual(geometry.configured_experts, 4)
        self.assertEqual(geometry.context_state_bytes, 13_312)

    def test_olmoe_fixture_models_conventional_fp32_kv_cache(self):
        # OLMoE keeps a full K and V cache per layer, sized at num_attention_heads
        # * head_dim in fp32 (olmoe.c:1019-1020), no recurrent/fixed state, and no
        # model-specific workspace beyond the base runtime reserve.
        config = {
            "model_type": "olmoe",
            "num_hidden_layers": 4,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 16,
            "vocab_size": 100,
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type["olmoe"]
        self.assertEqual(family, by_id["olmoe"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        self.assertEqual(geometry.configured_experts, 8)
        # layers=4, context=32, heads=4, head_dim=32//4=8, K and V, fp32:
        # 4 * 32 * 4 * 8 * 2 * 4 = 32768
        self.assertEqual(geometry.context_state_bytes, 4 * 32 * 4 * 8 * 2 * 4)
        self.assertEqual(geometry.fixed_state_bytes, 0)
        self.assertEqual(geometry.workspace_bytes, 0)

    def test_production_planners_report_real_geometry(self):
        # #1066 is closed: every production planner now returns a real
        # PlannerGeometry instead of refusing with PlannerUnsupportedError.
        # The plan must never be a silently-invented zero-byte budget --
        # every family here has a resident KV/state cache, so at least one
        # of context_state_bytes or fixed_state_bytes must be non-zero.
        cases = {
            "olmoe": {
                "model_type": "olmoe",
                "hidden_size": 2048,
                "num_hidden_layers": 2,
                "num_attention_heads": 16,
                "num_key_value_heads": 16,
                "num_experts": 8,
            },
            "kimi_k3": {
                "model_type": "kimi_k3",
                "hidden_size": 2048,
                "num_hidden_layers": 8,
                "num_attention_heads": 16,
                "q_lora_rank": 64,
                "kv_lora_rank": 128,
                "qk_nope_head_dim": 64,
                "qk_rope_head_dim": 32,
                "v_head_dim": 128,
                "num_experts": 32,
                "linear_attn_config": {
                    "num_heads": 8,
                    "head_dim": 64,
                    "kda_layers": [1, 2, 3, 4, 5],
                },
            },
            "inkling": {
                "model_type": "inkling",
                "hidden_size": 2048,
                "num_hidden_layers": 6,
                "num_attention_heads": 16,
                "num_key_value_heads": 4,
                "head_dim": 32,
                "n_routed_experts": 8,
            },
            "deepseek_v4": {
                "model_type": "deepseek_v4",
                "hidden_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 16,
                "head_dim": 32,
                "q_lora_rank": 16,
                "o_groups": 4,
                "o_lora_rank": 64,
                "sliding_window": 8,
                "index_head_dim": 24,
                "n_routed_experts": 32,
                "compress_ratios": [0, 2, 4, 4],
            },
        }
        for model_type, config in cases.items():
            family = family_for_config({"model_type": model_type})
            resolved = type("R", (), {"descriptor": family,
                                      "family_config": config,
                                      "model_dir": "."})()
            with self.subTest(model_type=model_type):
                geometry = planner_geometry(resolved, 32)
                self.assertIsInstance(geometry, PlannerGeometry)
                self.assertGreater(
                    geometry.context_state_bytes + geometry.fixed_state_bytes,
                    0)


    def test_olmoe_geometry_matches_engine_kv_allocation(self):
        # OLMoE config shaped like the real 1B-7B model (AI2), but small.
        config = {
            "model_type": "olmoe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 512,
            "vocab_size": 50304,
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        self.assertEqual(family, by_id["olmoe"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        # Engine allocates K+V with n_heads (not n_kv_heads), fp32:
        #   head_dim = hidden // n_heads = 2048 // 16 = 128
        #   state   = layers * context * heads * head_dim * 2 * 4
        self.assertEqual(geometry.configured_experts, 8)
        self.assertEqual(geometry.context_state_bytes,
                         2 * 32 * 16 * 128 * 2 * 4)
        self.assertEqual(geometry.fixed_state_bytes, 0)
        # Workspace: the bounded per-forward scratch (attention scores,
        # logits, expert temporaries) is covered by the base runtime reserve,
        # so the planner reports no context-scaling workspace (dev #1095).
        self.assertEqual(geometry.workspace_bytes, 0)

    def test_olmoe_geometry_scales_with_context(self):
        config = {
            "model_type": "olmoe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 512,
            "vocab_size": 50304,
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        small = planner_geometry(resolved, 16)
        large = planner_geometry(resolved, 64)
        ratio = large.context_state_bytes / small.context_state_bytes
        self.assertEqual(ratio, 4)  # linear in context
        # Workspace stays at zero at every context (per dev #1095): only the
        # KV state scales with context, so the ratio above is the full story.
        self.assertEqual(small.workspace_bytes, 0)
        self.assertEqual(large.workspace_bytes, 0)

    def test_olmoe_geometry_rejects_missing_keys(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["olmoe"]
        resolved = type("R", (), {"descriptor": family, "family_config": {},
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_olmoe_geometry_uses_n_heads_not_n_kv_heads(self):
        # GQA where kv < q heads: the engine still allocates K/V per q head,
        # so the plan must follow num_attention_heads (olmoe.c:1019-1020).
        config = {
            "model_type": "olmoe",
            "hidden_size": 2048,
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,   # GQA ratio 4
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "intermediate_size": 512,
            "vocab_size": 50304,
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["olmoe"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        self.assertEqual(geometry.context_state_bytes,
                         2 * 32 * 16 * 128 * 2 * 4)  # 16 heads, not 4

    def test_kimi_geometry_matches_engine_hybrid_allocation(self):
        # Kimi K3-shaped config: 8 layers, 5 KDA + 3 MLA (small synthetic).
        config = {
            "model_type": "kimi_k3",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "num_experts": 32,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 64,
                "short_conv_kernel_size": 3,
                "kda_layers": [1, 2, 3, 4, 5],
            },
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        self.assertEqual(family, by_id["kimi"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        # MLA cache: 3 MLA layers x context x (kv_lora + qk_rope) x 4
        self.assertEqual(geometry.context_state_bytes,
                         3 * 32 * (128 + 32) * 4)
        # KDA fixed recurrent state: 5 KDA layers x heads x hd x hd x 4
        self.assertEqual(geometry.fixed_state_bytes,
                         5 * 8 * 64 * 64 * 4)
        self.assertEqual(geometry.configured_experts, 32)

    def test_kimi_geometry_kda_state_does_not_scale_with_context(self):
        config = {
            "model_type": "kimi_k3",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "num_experts": 32,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 64,
                "short_conv_kernel_size": 3,
                "kda_layers": [1, 2, 3, 4, 5],
            },
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["kimi"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        small = planner_geometry(resolved, 16)
        large = planner_geometry(resolved, 64)
        # KDA recurrent state is context-independent
        self.assertEqual(small.fixed_state_bytes, large.fixed_state_bytes)
        # MLA cache scales linearly with context
        self.assertEqual(large.context_state_bytes / small.context_state_bytes, 4)

    def test_kimi_geometry_rejects_missing_linear_attn_config(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["kimi"]
        config = {
            "model_type": "kimi_k3",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "num_experts": 32,
        }
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_kimi_geometry_rejects_missing_kda_layers(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["kimi"]
        config = {
            "model_type": "kimi_k3",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "num_experts": 32,
            "linear_attn_config": {"num_heads": 8, "head_dim": 64},
        }
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_kimi_geometry_rejects_missing_keys(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["kimi"]
        resolved = type("R", (), {"descriptor": family, "family_config": {},
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_kimi_geometry_workspace_is_max_of_kda_and_mla(self):
        config = {
            "model_type": "kimi_k3",
            "hidden_size": 2048,
            "num_hidden_layers": 8,
            "num_attention_heads": 16,
            "q_lora_rank": 64,
            "kv_lora_rank": 128,
            "qk_nope_head_dim": 64,
            "qk_rope_head_dim": 32,
            "v_head_dim": 128,
            "num_experts": 32,
            "linear_attn_config": {
                "num_heads": 8,
                "head_dim": 64,
                "short_conv_kernel_size": 3,
                "kda_layers": [1, 2, 3, 4, 5],
            },
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["kimi"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        ctx = 32
        # KDA workspace: 6*ctx*P + ctx*hd + ctx*hidden floats
        ws_kda = (6 * ctx * (8 * 64) + ctx * 64 + ctx * 2048) * 4
        # MLA workspace: qa + qv + ckv + gv + ctx buffers
        qh = 64 + 32
        ws_mla = (ctx * 64 + ctx * 16 * qh + ctx * (128 + 32) +
                  2 * ctx * 16 * 128) * 4
        self.assertEqual(geometry.workspace_bytes, max(ws_kda, ws_mla))

    def test_inkling_geometry_matches_engine_hybrid_allocation(self):
        # 6 layers: default rule (i+1)%6 -> 5 sliding + 1 global (last).
        config = {
            "model_type": "inkling",
            "hidden_size": 2048,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "swa_num_attention_heads": 16,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "sliding_window_size": 8,
            "d_rel": 4,
            "sconv_kernel_size": 3,
            "n_routed_experts": 8,
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        self.assertEqual(family, by_id["inkling"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        # 5 sliding layers: kv=2, hd=32, rows=window=8 -> 2*2*8*32*4 each
        sliding = 5 * (2 * 2 * 8 * 32 * 4)
        # 1 global layer: kv=4, hd=32, rows=context=32 -> 2*4*32*32*4
        global_ = 1 * (2 * 4 * 32 * 32 * 4)
        self.assertEqual(geometry.context_state_bytes, sliding + global_)
        # Conv states per layer: (2*kvdim + 2*hidden) * (conv_k-1) * 4
        kvdim_s = 2 * 32   # sliding kv*hd
        kvdim_g = 4 * 32   # global kv*hd
        fixed = 5 * (2 * kvdim_s + 2 * 2048) * 2 * 4
        fixed += 1 * (2 * kvdim_g + 2 * 2048) * 2 * 4
        self.assertEqual(geometry.fixed_state_bytes, fixed)
        self.assertEqual(geometry.configured_experts, 8)

    def test_inkling_geometry_sliding_ring_does_not_scale_past_window(self):
        config = {
            "model_type": "inkling",
            "hidden_size": 2048,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "swa_num_attention_heads": 16,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "sliding_window_size": 8,
            "d_rel": 4,
            "sconv_kernel_size": 3,
            "n_routed_experts": 8,
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["inkling"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        small = planner_geometry(resolved, 8)    # context == window
        large = planner_geometry(resolved, 64)   # context > window
        # Sliding ring rows are capped at window, so beyond window the
        # sliding contribution is flat; only the 1 global layer grows.
        delta = large.context_state_bytes - small.context_state_bytes
        self.assertEqual(delta, 1 * (2 * 4 * (64 - 8) * 32 * 4))
        # Fixed (conv) state is context-independent
        self.assertEqual(small.fixed_state_bytes, large.fixed_state_bytes)

    def test_inkling_geometry_local_layer_ids_override_default(self):
        config = {
            "model_type": "inkling",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "swa_num_attention_heads": 16,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "sliding_window_size": 8,
            "d_rel": 4,
            "sconv_kernel_size": 3,
            "n_routed_experts": 8,
            "local_layer_ids": [0, 2],   # only layers 0 and 2 are sliding
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["inkling"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        # 2 sliding (kv=2, rows=8) + 2 global (kv=4, rows=32)
        expected = 2 * (2 * 2 * 8 * 32 * 4) + 2 * (2 * 4 * 32 * 32 * 4)
        self.assertEqual(geometry.context_state_bytes, expected)

    def test_inkling_geometry_audio_tower_adds_fixed_reserve(self):
        base = {
            "model_type": "inkling",
            "hidden_size": 2048,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "swa_num_attention_heads": 16,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 32,
            "sliding_window_size": 8,
            "d_rel": 4,
            "sconv_kernel_size": 3,
            "n_routed_experts": 8,
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["inkling"]
        resolved = type("R", (), {"descriptor": family, "family_config": base,
                                   "model_dir": "."})()
        no_audio = planner_geometry(resolved, 32)

        with_audio_cfg = dict(base)
        with_audio_cfg["audio_config"] = {"n_mel_bins": 80, "mel_vocab_size": 16}
        resolved2 = type("R", (), {"descriptor": family,
                                   "family_config": with_audio_cfg,
                                   "model_dir": "."})()
        audio = planner_geometry(resolved2, 32)
        # audio_enc table [80*16, 2048] + norm [2048], fp32
        expected_audio = (80 * 16 * 2048 + 2048) * 4
        self.assertEqual(audio.fixed_state_bytes - no_audio.fixed_state_bytes,
                         expected_audio)
        self.assertEqual(audio.context_state_bytes, no_audio.context_state_bytes)

    def test_inkling_geometry_rejects_missing_keys(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["inkling"]
        resolved = type("R", (), {"descriptor": family, "family_config": {},
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_inkling_geometry_rejects_bad_layer_types_length(self):
        config = {
            "model_type": "inkling",
            "hidden_size": 2048,
            "num_hidden_layers": 6,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 32,
            "n_routed_experts": 8,
            "layer_types": ["hybrid_sliding"],  # wrong length
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["inkling"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_dsv4_geometry_matches_engine_context_bytes(self):
        # 4 layers: ratios [0, 2, 4, 4] -> no-compress, compressor, indexer, indexer
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 32,
            "q_lora_rank": 16,
            "o_groups": 4,
            "o_lora_rank": 64,
            "sliding_window": 8,
            "index_head_dim": 24,
            "n_routed_experts": 32,
            "compress_ratios": [0, 2, 4, 4],
        }
        by_id, by_type = _build_registry(FAMILIES)
        family = by_type[config["model_type"]]
        self.assertEqual(family, by_id["deepseek_v4"])
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        # Fixed window ring: 4 layers * window 8 * head_dim 32 * 4
        self.assertEqual(geometry.fixed_state_bytes, 4 * 8 * 32 * 4)
        # Compressor states: ceil(32/2)=16 * hd * 4  +  ceil(32/4)=8 * hd * 4 * 2
        state = 16 * 32 * 4 + 8 * 32 * 4 + 8 * 32 * 4
        # Indexer states (ratio==4): ceil(32/4)=8 * index_hd 24 * 4, for 2 layers
        state += 8 * 24 * 4 + 8 * 24 * 4
        self.assertEqual(geometry.context_state_bytes, state)
        self.assertEqual(geometry.configured_experts, 32)

    def test_dsv4_geometry_fixed_ring_does_not_scale_with_context(self):
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 32,
            "q_lora_rank": 16,
            "o_groups": 4,
            "o_lora_rank": 64,
            "sliding_window": 8,
            "index_head_dim": 24,
            "n_routed_experts": 32,
            "compress_ratios": [0, 2, 4, 4],
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["deepseek_v4"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        small = planner_geometry(resolved, 16)
        large = planner_geometry(resolved, 64)
        # Window ring is context-independent
        self.assertEqual(small.fixed_state_bytes, large.fixed_state_bytes)
        # Compressed states scale ~linearly with context (ceil divisions)
        ratio = large.context_state_bytes / small.context_state_bytes
        self.assertAlmostEqual(ratio, 4, delta=0.5)

    def test_dsv4_geometry_workspace_matches_attention_scratch(self):
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 32,
            "q_lora_rank": 16,
            "o_groups": 4,
            "o_lora_rank": 64,
            "sliding_window": 8,
            "index_head_dim": 24,
            "n_routed_experts": 32,
            "compress_ratios": [0, 2, 4, 4],
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["deepseek_v4"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 32)
        ctx = 32
        q_width = 16 * 32       # heads * head_dim
        oa_width = 4 * 64       # o_groups * o_lora_rank
        expected = (ctx * (16 + 2 * q_width + 32 + oa_width) +
                    max(16, 32)) * 4
        self.assertEqual(geometry.workspace_bytes, expected)

    def test_dsv4_geometry_rejects_bad_compress_ratios(self):
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 32,
            "q_lora_rank": 16,
            "o_groups": 4,
            "o_lora_rank": 64,
            "sliding_window": 8,
            "index_head_dim": 24,
            "n_routed_experts": 32,
            "compress_ratios": [0, 2],  # wrong length
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["deepseek_v4"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_dsv4_geometry_rejects_missing_keys(self):
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["deepseek_v4"]
        resolved = type("R", (), {"descriptor": family, "family_config": {},
                                   "model_dir": "."})()
        with self.assertRaises(ValueError):
            planner_geometry(resolved, 32)

    def test_dsv4_geometry_all_uncompressed_layers_have_only_ring(self):
        config = {
            "model_type": "deepseek_v4",
            "hidden_size": 2048,
            "num_hidden_layers": 4,
            "num_attention_heads": 16,
            "head_dim": 32,
            "q_lora_rank": 16,
            "o_groups": 4,
            "o_lora_rank": 64,
            "sliding_window": 8,
            "index_head_dim": 24,
            "n_routed_experts": 32,
            "compress_ratios": [0, 0, 0, 0],
        }
        by_id, _ = _build_registry(FAMILIES)
        family = by_id["deepseek_v4"]
        resolved = type("R", (), {"descriptor": family, "family_config": config,
                                   "model_dir": "."})()
        geometry = planner_geometry(resolved, 64)
        # No compressors: context state is zero, only the ring is resident
        self.assertEqual(geometry.context_state_bytes, 0)
        self.assertEqual(geometry.fixed_state_bytes, 4 * 8 * 32 * 4)

    def test_cli_and_gateway_dispatch_follow_the_registry(self):
        import openai_server
        from importlib.machinery import SourceFileLoader
        import importlib.util

        cli_path = Path(__file__).resolve().parent.parent / "coli"
        loader = SourceFileLoader("family_registry_cli_test", str(cli_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cli = importlib.util.module_from_spec(spec)
        loader.exec_module(cli)

        source = cli_path.read_text(encoding="utf-8")
        self.assertNotRegex(source, r"family\.(?:cli|gateway)_adapter\s+not in")
        self.assertNotIn("K3CHAT1", source)
        self.assertEqual(source.count("if not family.has_gateway_adapter:"), 3)
        self.assertEqual(source.count("if not family.has_cli_adapter:"), 1)
        self.assertEqual(set(openai_server.family_ids()),
                         {family.id for family in FAMILIES})
        self.assertEqual({family.id for family in cli.all_families()},
                         {family.id for family in FAMILIES})

        resolved = type("R", (), {"descriptor": replace(
            FAMILIES[0], has_cli_adapter=False, has_gateway_adapter=False)})()
        args = type("A", (), {"model": ".", "prompt": ["hello"],
                                "no_attach": True})()
        with mock.patch.object(cli, "need_model"), \
             mock.patch.object(cli, "resolve_model", return_value=resolved), \
             mock.patch.object(cli, "engine_for", return_value="engine"), \
             mock.patch.object(cli, "banner"):
            with self.assertRaisesRegex(SystemExit, "coli run is not wired"):
                cli.cmd_run(args)
            with self.assertRaisesRegex(SystemExit, "gateway adapter is not wired"):
                cli.cmd_chat(args)

    def test_tuning_replay_prompts_are_registry_owned(self):
        prompt = "hello {world}"
        expected = {
            # GLM-5.3 apre il ragionamento e non lo chiude: il suo
            # chat_template.jinja mette <think> dopo <|assistant|> e basta,
            # dove GLM-5.2 metteva <think></think>.
            "glm53": "[gMASK]<sop><|user|>hello {world}<|assistant|><think>",
            "glm": "[gMASK]<sop><|user|>hello {world}<|assistant|><think></think>",
            "inkling": "<|user|>hello {world}<|assistant|>",
            "kimi": "K3CHAT1\nM user 13\nhello {world}G 0\n\n",
            "olmoe": "<|user|>\nhello {world}\n<|assistant|>\n",
            # Qwen3.6's generation prompt MUST open <think>: the model was
            # never trained on the bare "assistant\\n" state and greedy argmax
            # there lands on an EOS special (measured gen=0).
            "qwen36": "<|im_start|>user\nhello {world}<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n",
            "qwen38": "<|im_start|>system\nReasoning effort is set to xhigh. Please think carefully "
                      "through the task, validate key assumptions, consider plausible alternatives, "
                      "and prioritize correctness, consistency, and clarity in the final answer."
                      "<|im_end|>\n<|im_start|>user\nhello {world}<|im_end|>\n"
                      "<|im_start|>assistant\n<think>\n",
            "deepseek_v4": "hello {world}",
        }
        self.assertEqual(
            {family.id: tuning_replay_prompt(family, prompt) for family in FAMILIES},
            expected,
        )

    def test_optional_adapters_and_prompt_template_are_registry_invariants(self):
        self.assertFalse(QWEN36_FIXTURE.has_cli_adapter)
        self.assertFalse(QWEN36_FIXTURE.has_gateway_adapter)
        self.assertEqual(tuning_replay_prompt(QWEN36_FIXTURE, "hello"), "hello")

        with self.assertRaises(RegistryError):
            _build_registry((replace(QWEN36_FIXTURE,
                                     resident_inventory="not callable"),))

        for template in ("static", "{unknown}", "{prompt", "{prompt[foo]}"):
            with self.subTest(template=template), self.assertRaises(RegistryError):
                _build_registry((replace(QWEN36_FIXTURE,
                                         tune_prompt_template=template),))

    def test_doctor_reports_unknown_family_instead_of_falling_back(self):
        from doctor import run_doctor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen3_moe"}), encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            report = run_doctor(root, engine_path=root / "colibri",
                                available_memory=16_000_000_000,
                                available_disk=1, gpus=[],
                                linkage={"linked": False, "missing": False})
        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(checks["model.family"]["status"], "fail")
        self.assertIn("unsupported model_type", checks["model.family"]["summary"])
        self.assertIsNone(report["plan"])

    def test_doctor_reports_engine_capability_failure(self):
        from doctor import run_doctor

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps({"model_type": "kimi_k3"}), encoding="utf-8")
            (root / "tokenizer.json").write_text("{}", encoding="utf-8")
            report = run_doctor(
                root, engine_path=root / "colibri",
                engine_error=UnknownFamilyError("this image contains only GLM"),
                available_memory=16_000_000_000, available_disk=1, gpus=[],
                linkage={"linked": False, "missing": False})
        checks = {item["id"]: item for item in report["checks"]}
        self.assertEqual(checks["engine.binary"]["status"], "fail")
        self.assertEqual(report["status"], "error")

    def test_tools_capability_matches_what_the_renderer_actually_does(self):
        """The `tools` flag must agree with the gateway's behaviour.

        It is descriptive -- it only feeds the capability dict -- which is
        exactly why it drifted: glm53 and kimi both render and parse tool calls
        while sharing a COMMON_CAP that said they do not, and nothing failed to
        tell anyone. A client asking what a family supports was told the
        opposite of the truth. Ask the renderer instead of trusting the flag.
        """
        import openai_server

        probe = [{"role": "user", "content": "hi"}]
        tool = [{"type": "function",
                 "function": {"name": "f", "description": "d",
                              "parameters": {"type": "object", "properties": {}}}}]
        for family in FAMILIES:
            renderer = getattr(openai_server, f"render_chat_{family.id}", None)
            if renderer is None:          # kimi/deepseek build their payload in C
                continue
            refused = False
            try:
                renderer(probe, tools=tool)
            except openai_server.APIError as exc:
                refused = getattr(exc, "code", None) == "unsupported_parameter"
            except Exception:             # nothing else here is a capability answer
                continue
            self.assertEqual(
                family.capabilities.tools, not refused,
                f"{family.id}: registry says tools={family.capabilities.tools} but the "
                f"renderer {'refuses' if refused else 'accepts'} them")

    def test_a_reasoning_family_gets_more_room_than_a_single_answer(self):
        """Chi ragiona deve avere un budget interattivo piu' largo del default.

        Il tetto sui token e' una rete di sicurezza: la fine vera la decidono
        gli stop token. Ma su un modello che riflette prima di rispondere, un
        tetto stretto non ti prende DOPO la risposta, ti prende DENTRO al
        pensiero, e il turno finisce senza che l'utente veda niente (#1278).

        glm53, inkling e kimi erano rimaste all'interattivo uguale al default,
        mentre ogni altra famiglia che ragiona lo aveva gia' alzato. Nessuno se
        n'e' accorto perche' il registro accetta qualsiasi valore >= 1: questo
        controllo esiste perche' la prossima non passi allo stesso modo.
        """
        for family in FAMILIES:
            if not family.capabilities.thinking:
                continue          # senza ragionamento il default e' la risposta intera
            self.assertGreater(
                family.limits.interactive_max_output, family.limits.default_max_output,
                f"{family.id}: reasons, but its interactive budget "
                f"({family.limits.interactive_max_output}) is no larger than the "
                f"non-interactive default ({family.limits.default_max_output}) -- "
                f"reasoning can consume it before the answer starts")

    def test_every_readme_names_every_family(self):
        """Ogni README, tradotto compreso, deve nominare ogni famiglia.

        E' la terza volta oggi che un conteggio ripetuto in due posti diverge:
        release.yml costruiva sei motori e ne copiava sette, i contatori degli
        adapter dicevano 7 con 8 famiglie registrate, e i README dichiaravano
        sei, sette e otto famiglie contemporaneamente -- l'inglese si
        contraddiceva da solo fra riga 21 e riga 586.

        Il nome della famiglia e' il controllo giusto, non il numerale: sono
        quattro lingue e il numerale si scrive in quattro modi, mentre
        `Qwen3.8-Flash-Next` si scrive uguale ovunque. Chi aggiunge una famiglia
        e dimentica le traduzioni lo scopre qui invece che da un utente che
        legge la sua lingua e non trova il modello.
        """
        repo = Path(__file__).resolve().parents[2]
        readmes = ["README.md", "README.it.md", "README.zh-CN.md", "README.zh-TW.md"]
        for name in readmes:
            path = repo / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8")
            for family in FAMILIES:
                # Il nome senza il suffisso di taglia: i README scrivono
                # "**Qwen3.6** (35B-A3B)", non "Qwen3.6-35B-A3B", perche' la
                # dimensione sta fra parentesi. Cercare il display_name intero
                # fallirebbe su una differenza di formattazione invece che su
                # una famiglia mancante, che e' quello che qui interessa.
                parts = []
                for piece in family.display_name.split("-"):
                    if re.fullmatch(r"A?\d+(\.\d+)?B", piece):
                        break
                    parts.append(piece)
                token = "-".join(parts) or family.display_name
                # assertTrue e non assertIn: assertIn stampa il README intero
                # nel messaggio di errore, e mille righe di markdown nascondono
                # la riga che dice cosa manca.
                self.assertTrue(token in text,
                                f"{name}: does not mention {family.display_name} "
                                f"({family.id}); a reader in that language cannot "
                                f"tell the family is supported")

    def test_build_install_ci_and_release_cover_registered_engines(self):
        repo = Path(__file__).resolve().parents[2]
        makefile = (repo / "c" / "Makefile").read_text(encoding="utf-8")
        ci = (repo / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        release = (repo / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8")
        docker = (repo / "docker" / "Dockerfile.slim").read_text(encoding="utf-8")
        clean = (repo / "c" / "tools" / "clean.py").read_text(encoding="utf-8")
        make_rules = re.sub(r"\\\n[ \t]*", " ", makefile)
        install_rule = re.search(r"(?m)^install:\s*(.*)$", make_rules)
        self.assertIsNotNone(install_rule)
        install_prerequisites = install_rule.group(1).split("#", 1)[0]
        for family in FAMILIES:
            with self.subTest(family=family.id):
                self.assertRegex(
                    makefile,
                    rf"(?m)^{re.escape(family.build_target)}(?:\$\(EXE\))?:")
                install_target = family.build_target
                if family.id != "deepseek_v4":
                    install_target += "$(EXE)"
                self.assertRegex(
                    install_prerequisites,
                    rf"(?<![\w.-]){re.escape(install_target)}(?![\w.-])")
                if family.id != "deepseek_v4":
                    self.assertIn(family.build_target,
                                  re.search(r'ENGINES="([^"]+)"', ci).group(1).split())
                    self.assertIn(f"cp c/{family.engine_artifact}", release)
                    # Copiarlo non basta: va anche COSTRUITO. Il contratto
                    # verificava solo meta', e con quella meta' la v1.9.0 e'
                    # uscita col nome di GLM-5.3-Flash e senza il suo binario,
                    # esattamente come la v1.5.0 con DeepSeek V4 (#858). Un
                    # `cp` di un file che nessuno ha compilato fallisce a
                    # release gia' pubblicata, cioe' nel momento peggiore.
                    build_step = re.search(r"for t in ([a-z0-9_ ]+); do",
                                           release)
                    self.assertIsNotNone(build_step,
                                         "release.yml: build loop not found")
                    self.assertIn(family.build_target, build_step.group(1).split(),
                                  f"{family.id}: release.yml copies "
                                  f"c/{family.engine_artifact} but never builds it")
                    self.assertIn(f"$(LIBEXECDIR)/{family.engine_artifact}", makefile)
                    # `make clean` must actually remove the engine. When it does
                    # not, a rebuild with different EXTRA_CFLAGS reports "up to
                    # date" and the caller silently keeps the OLD binary. The
                    # ASan step of the Qwen3.6 oracle job re-ran an
                    # un-instrumented qwen36 that way for as long as it existed
                    # (#1262): green, with the sanitizer never having run.
                    self.assertIn(f'"{family.engine_artifact}"', clean,
                                  f"{family.id}: tools/clean.py does not remove "
                                  f"c/{family.engine_artifact}, so a rebuild with "
                                  f"different flags is a silent no-op")
                else:
                    self.assertIn("deepseek-v4", ci)
                    self.assertIn("cp c/deepseek_v4", release)
        for text in (makefile, release, docker):
            self.assertIn("family_registry.py", text)


if __name__ == "__main__":
    unittest.main()
