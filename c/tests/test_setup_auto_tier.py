import os
import importlib.machinery
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup_flow
from family_registry import family_by_id


class SavedAutoTierTest(unittest.TestCase):
    def test_every_backend_stores_auto_tier_for_sibling_engines(self):
        for backend in ("cpu", "vulkan", "cuda"):
            for family_id in ("qwen36", "kimi", "inkling", "deepseek_v4"):
                with self.subTest(backend=backend, family=family_id), tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
                    family = family_by_id(family_id)
                    engine_info = {"backend": backend, "launcher_dir": home, "engine": str(Path(home) / family.engine_artifact), "source": "fixture"}
                    cfg = setup_flow.make_config(model_dir=str(Path(home) / "model"), family=family, entry=None, engine_info=engine_info, hw={"cpu": {"physical_cores": 4}}, host="127.0.0.1", port=8000)
                    setup_flow.save_config(cfg)
                    loaded = setup_flow.load_config()
                    self.assertEqual(loaded["args"].count("--auto-tier"), 1, loaded["args"])
                    self.assertEqual("--gpu" in loaded["args"], backend == "cuda")
                    self.assertIn("--auto-tier", setup_flow.equivalent_command(loaded))

    def test_saved_cpu_and_vulkan_args_apply_ram_plan_to_engine_cap(self):
        loader = importlib.machinery.SourceFileLoader("coli_setup_tier", str(Path(__file__).resolve().parent.parent / "coli"))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        coli = importlib.util.module_from_spec(spec)
        loader.exec_module(coli)
        import openai_server
        import resource_plan
        # A placement plan with 24 expert slots: no accelerator or model loading.
        plan = {"policy": {"name": "quality"}, "cpu": {"physical_cores": 4},
                "model": {"family_id": "qwen36"},
                "tiers": {"ram": {"budget_bytes": 16 * 1000 ** 3,
                                    "cache_slots_per_layer": 24},
                          "vram": {"devices": [], "budget_bytes": 0}}}
        for backend in ("cpu", "vulkan"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as home, mock.patch.dict(os.environ, {"COLI_SETUP_HOME": home}, clear=True):
                family = family_by_id("qwen36")
                cfg = setup_flow.make_config(model_dir=str(Path(home) / "model"), family=family, entry=None, engine_info={"backend": backend, "launcher_dir": home, "engine": str(Path(home) / family.engine_artifact)}, hw={"cpu": {"physical_cores": 4}}, host="127.0.0.1", port=8000)
                setup_flow.save_config(cfg)
                loaded = setup_flow.load_config()
                captured = {}

                def launch(parsed):
                    parsed.no_tune_profile = True
                    env = coli.env_for_engine(parsed, "qwen36", plan=plan)
                    captured.update(env=env, cap=openai_server.cap_for_arch("qwen36", parsed.cap, env))
                    return 0

                with mock.patch.object(sys, "argv", ["coli"] + loaded["args"]), mock.patch.object(coli, "cmd_web", side_effect=launch), mock.patch.object(coli, "plan_cuda_enabled", return_value=False), mock.patch.object(resource_plan, "physical_cpu_count", return_value=4):
                    with self.assertRaises(SystemExit) as exited:
                        coli.main()
                    self.assertEqual(exited.exception.code, 0)
                self.assertEqual(captured["env"].get("RAM_GB"), "16.000")
                self.assertEqual(captured["env"].get("COLI_PLAN_CAP"), "24")
                self.assertEqual(captured["cap"], 24)
                self.assertNotEqual(captured["env"].get("COLI_CUDA"), "1")


if __name__ == "__main__":
    unittest.main()
