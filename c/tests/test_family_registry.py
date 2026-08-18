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
    family_for_config,
    planner_geometry,
    public_metadata,
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


TEST_INVENTORY = lambda _name, _size, _config: ()
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
            "glm": "[gMASK]<sop><|user|>hello {world}<|assistant|><think></think>",
            "inkling": "<|user|>hello {world}<|assistant|>",
            "kimi": "K3CHAT1\nM user 13\nhello {world}G 0\n\n",
            "olmoe": "<|user|>\nhello {world}\n<|assistant|>\n",
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
