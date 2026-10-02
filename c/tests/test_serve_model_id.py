"""`coli serve` announces the model that was loaded, not its family's default.

MiMo-V2.6 Flash and Pro share a model_type, and so do Qwen3.6 and the dense
Qwen3.8-27B. The registry names each by its geometry (default_model_id), and
`coli chat` already asked it; `coli serve` took the family's default, so a Pro
server listed itself in /v1/models as mimo-v2.6-flash."""
import importlib.machinery
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

_loader = importlib.machinery.SourceFileLoader("coli_cli_serve_id", str(HERE / "coli"))
_spec = importlib.util.spec_from_loader("coli_cli_serve_id", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


class ServeModelIdTest(unittest.TestCase):
    def model(self, config):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        (Path(directory.name) / "config.json").write_text(json.dumps(config), encoding="utf-8")
        return directory.name

    def serve_id(self, config, model_id=None):
        return coli.serve_model_id(types.SimpleNamespace(model=self.model(config), model_id=model_id))

    def test_each_variant_names_itself(self):
        cases = (
            ({"model_type": "mimo_v2", "hidden_size": 4096, "num_hidden_layers": 48,
              "n_routed_experts": 256}, "mimo-v2.6-flash"),
            ({"model_type": "mimo_v2", "hidden_size": 6144, "num_hidden_layers": 70,
              "n_routed_experts": 384}, "mimo-v2.6-pro"),
            ({"model_type": "qwen3_5_text", "num_hidden_layers": 64, "hidden_size": 5120,
              "intermediate_size": 17408}, "qwen3.8-27b-colibri"),
        )
        for config, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self.serve_id(config), expected)

    def test_explicit_model_id_wins(self):
        pro = {"model_type": "mimo_v2", "hidden_size": 6144, "num_hidden_layers": 70, "n_routed_experts": 384}
        self.assertEqual(self.serve_id(pro, model_id="my-pro"), "my-pro")

    def test_family_without_variants_keeps_its_default(self):
        self.assertEqual(self.serve_id({"model_type": "glm_moe_dsa"}), "glm-5.2-colibri")


if __name__ == "__main__":
    unittest.main()
