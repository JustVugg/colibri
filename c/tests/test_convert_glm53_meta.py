"""#1478: the converter must leave a generation_config.json in the container
whenever the source declares eos ids, also when the shards come from --indir
(which used to copy no metadata at all)."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / "tools" / "convert_glm53.py"

try:
    import numpy  # noqa: F401  (the converter imports it at module level)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def load_converter():
    spec = importlib.util.spec_from_file_location("convert_glm53_t", SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_NUMPY, "numpy is not installed; the converter imports it")
class ConverterMetaTest(unittest.TestCase):
    def setUp(self):
        self.m = load_converter()
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "out"); os.makedirs(self.out)
        self.src = os.path.join(self.tmp.name, "src"); os.makedirs(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_eos_from_text_config_and_top_level(self):
        self.assertEqual(self.m.eos_ids_from_config({"text_config": {"eos_token_id": [1, 2]}}), [1, 2])
        self.assertEqual(self.m.eos_ids_from_config({"eos_token_id": 7}), [7])
        self.assertEqual(self.m.eos_ids_from_config({"eos_token_id": 7, "text_config": {"eos_token_id": [1]}}), [7])
        self.assertEqual(self.m.eos_ids_from_config({"text_config": {}}), [])
        self.assertEqual(self.m.eos_ids_from_config({"eos_token_id": "x"}), [])

    def test_generation_config_written_from_config(self):
        with open(os.path.join(self.out, "config.json"), "w") as f:
            json.dump({"model_type": "glm5_next", "text_config": {"eos_token_id": [154820, 154827, 154829]}}, f)
        self.assertEqual(self.m.ensure_generation_config(self.out), [154820, 154827, 154829])
        with open(os.path.join(self.out, "generation_config.json")) as f:
            self.assertEqual(json.load(f)["eos_token_id"], [154820, 154827, 154829])
        # already there: left alone
        self.assertIsNone(self.m.ensure_generation_config(self.out))

    def test_nothing_known_writes_nothing(self):
        with open(os.path.join(self.out, "config.json"), "w") as f:
            json.dump({"model_type": "glm5_next"}, f)
        self.assertIsNone(self.m.ensure_generation_config(self.out))
        self.assertFalse(os.path.exists(os.path.join(self.out, "generation_config.json")))

    def test_indir_metadata_is_carried(self):
        for name, body in (("config.json", {"eos_token_id": 3}), ("generation_config.json", {"eos_token_id": [3, 4]}),
                           ("tokenizer.json", {})):
            with open(os.path.join(self.src, name), "w") as f:
                json.dump(body, f)
        copied, missing = self.m.copy_meta_from_dir(self.src, self.out)
        self.assertEqual(copied, ["config.json", "generation_config.json", "tokenizer.json"])
        self.assertIn("chat_template.jinja", missing)
        with open(os.path.join(self.out, "generation_config.json")) as f:
            self.assertEqual(json.load(f)["eos_token_id"], [3, 4])


if __name__ == "__main__":
    unittest.main()
