"""Launching Pi must not overwrite an unrelated provider with a familiar name."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("ui_agents", Path(__file__).resolve().parents[2] / "ui/agents.py")
agents = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agents)


class PiProviderCollisionTest(unittest.TestCase):
    def configure(self, providers):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        file = Path(tmp.name) / "models.json"
        original = {"providers": providers, "extra": "preserved"}
        file.write_text(json.dumps(original))
        name = agents._pi_provider(str(file), "http://127.0.0.1:8080/v1", "model", 8192, "test-key")
        return name, json.loads(file.read_text()), original, file

    def test_all_occupied_names_are_preserved(self):
        providers = {name: {"baseUrl": "https://example.invalid/" + name, "models": []}
                     for name in ("colibri", "colibri-ui", "colibri-ui-2")}
        name, cfg, original, file = self.configure(providers)
        for key, value in original["providers"].items():
            self.assertEqual(cfg["providers"][key], value)
        self.assertNotIn(name, original["providers"])
        self.assertEqual(cfg["extra"], "preserved")
        self.assertEqual(json.loads(Path(str(file) + ".bak-colibri-ui").read_text()), original)

    def test_existing_matching_endpoint_is_reused_even_with_a_custom_name(self):
        name, cfg, _, _ = self.configure({"local-engine": {"baseUrl": "http://127.0.0.1:8080/v1/", "models": []}})
        self.assertEqual(name, "local-engine")
        self.assertEqual(list(cfg["providers"]), ["local-engine"])
        self.assertEqual(cfg["providers"][name]["models"][0]["id"], "model")

    def test_first_provider_keeps_the_original_default_name(self):
        name, cfg, _, _ = self.configure({})
        self.assertEqual(name, "colibri")
        self.assertEqual(cfg["providers"][name]["apiKey"], "test-key")


if __name__ == "__main__":
    unittest.main()
