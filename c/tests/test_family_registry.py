import json
import re
import tempfile
import unittest
from pathlib import Path

from family_registry import (
    FAMILIES,
    FamilyCapabilities,
    FamilyConfigError,
    FamilyDescriptor,
    FamilyLimits,
    PlannerGeometry,
    UnknownFamilyError,
    PlannerUnsupportedError,
    _build_registry,
    family_for_config,
    planner_geometry,
    public_metadata,
    resolve_model,
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


TEST_INVENTORY = lambda _name, _size, _config: ()
QWEN36_FIXTURE = FamilyDescriptor(
    "qwen36", ("qwen3_5_moe_text",), "Qwen3.6", "", "qwen36", (), "qwen36",
    "qwen36", "qwen36", ("qwen36",), "qwen3.6-colibri", "qwen36", "qwen36",
    "qwen36_hybrid", qwen_geometry, "", TEST_INVENTORY, "root",
    FamilyLimits(8192, 262144, 1024, 8192, 1, 8, "Q36_MAXT"),
    FamilyCapabilities(False, False, False, True))
MINIMAX_M3_FIXTURE = FamilyDescriptor(
    "minimax_m3", ("minimax_m3",), "MiniMax M3", "", "colibri", (),
    "colibri-core", "minimax_m3", "colibri", ("colibri",), "minimax-m3-colibri",
    "minimax_m3", "minimax_m3", "minimax_m3_gqa", minimax_geometry, "",
    TEST_INVENTORY, "root", FamilyLimits(8192, 262144, 1024, 8192, 1, 8, "CTX"),
    FamilyCapabilities(True, False, False, True))


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
        by_id, by_type = _build_registry(FAMILIES + (QWEN36_FIXTURE,))
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

    def test_unproven_production_planners_refuse_instead_of_inventing_zero(self):
        for model_type in ("inkling", "kimi_k3", "olmoe", "deepseek_v4"):
            family = family_for_config({"model_type": model_type})
            resolved = type("R", (), {"descriptor": family, "family_config": {},
                                       "model_dir": "."})()
            with self.subTest(model_type=model_type), \
                 self.assertRaises(PlannerUnsupportedError):
                planner_geometry(resolved, 32)

    def test_cli_and_gateway_adapter_sets_equal_the_registry(self):
        import openai_server
        from importlib.machinery import SourceFileLoader
        import importlib.util

        cli_path = Path(__file__).resolve().parent.parent / "coli"
        loader = SourceFileLoader("family_registry_cli_test", str(cli_path))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cli = importlib.util.module_from_spec(spec)
        loader.exec_module(cli)

        cli_adapters = {"glm", "inkling", "kimi", "olmoe", "deepseek_v4"}
        gateway_adapters = {"glm", "inkling", "kimi", "olmoe", "deepseek_v4"}
        self.assertEqual({family.cli_adapter for family in FAMILIES}, cli_adapters)
        self.assertEqual({family.gateway_adapter for family in FAMILIES},
                         gateway_adapters)
        self.assertEqual(set(openai_server.family_ids()),
                         {family.id for family in FAMILIES})
        self.assertEqual({family.id for family in cli.all_families()},
                         {family.id for family in FAMILIES})

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

    def test_build_install_ci_and_release_cover_registered_engines(self):
        repo = Path(__file__).resolve().parents[2]
        makefile = (repo / "c" / "Makefile").read_text(encoding="utf-8")
        ci = (repo / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        release = (repo / ".github" / "workflows" / "release.yml").read_text(
            encoding="utf-8")
        docker = (repo / "docker" / "Dockerfile.slim").read_text(encoding="utf-8")
        for family in FAMILIES:
            with self.subTest(family=family.id):
                self.assertRegex(
                    makefile,
                    rf"(?m)^{re.escape(family.build_target)}(?:\$\(EXE\))?:")
                if family.id != "deepseek_v4":
                    self.assertIn(family.build_target,
                                  re.search(r'ENGINES="([^"]+)"', ci).group(1).split())
                    self.assertIn(f"cp c/{family.engine_artifact}", release)
                    self.assertIn(f"$(LIBEXECDIR)/{family.engine_artifact}", makefile)
                else:
                    self.assertIn("deepseek-v4", ci)
                    self.assertIn("cp c/deepseek_v4", release)
        for text in (makefile, release, docker):
            self.assertIn("family_registry.py", text)


if __name__ == "__main__":
    unittest.main()
