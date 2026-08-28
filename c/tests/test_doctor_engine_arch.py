"""`coli doctor` must inspect the engine the model would actually run on (#783).

engine_path was hardcoded to the GLM binary, so doctor reported against colibri
whatever the model was. That misled in both directions: a user who had built
kimi_k3 was told "engine is not built", and anyone with a stale colibri lying
around would have been told the engine was ready for a model that never runs on
it. A diagnostic that reports on the wrong file is worse than no diagnostic.
"""
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"

ARCHES = [
    ("glm_moe_dsa", "colibri"),
    ("inkling", "inkling"),
    ("kimi_k3", "kimi_k3"),
    ("olmoe", "olmoe"),
    ("deepseek_v4", "deepseek_v4"),
]


def write_model(root, model_type):
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({"model_type": model_type}))
    (root / "tokenizer.json").write_text("{}")
    header = json.dumps({"w": {"dtype": "U8", "shape": [4],
                               "data_offsets": [0, 4]}}).encode()
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 4)


class DoctorEngineArchTest(unittest.TestCase):
    def doctor(self, model):
        result = subprocess.run(
            [sys.executable, str(CLI), "doctor", "--model", str(model), "--json"],
            cwd=HERE, text=True, capture_output=True, check=False, timeout=60)
        self.assertTrue(result.stdout.strip(),
                        f"no report on stdout; stderr={result.stderr}")
        return json.loads(result.stdout)

    def engine_check(self, report):
        for check in report["checks"]:
            if check["id"] == "engine.binary":
                return check
        self.fail("report has no engine.binary check")

    def test_engine_check_names_the_engine_for_this_model_arch(self):
        with tempfile.TemporaryDirectory() as tmp:
            for model_type, expected in ARCHES:
                model = Path(tmp) / model_type
                write_model(model, model_type)
                check = self.engine_check(self.doctor(model))
                path = check.get("details", {}).get("path", "")
                self.assertTrue(
                    Path(path).name.startswith(expected),
                    f"{model_type}: doctor inspected {path!r}, expected the "
                    f"{expected} engine")

    def test_two_arches_do_not_resolve_to_the_same_engine(self):
        """The regression this guards: every arch mapping to the GLM binary."""
        with tempfile.TemporaryDirectory() as tmp:
            seen = []
            for model_type, _ in ARCHES:
                model = Path(tmp) / model_type
                write_model(model, model_type)
                check = self.engine_check(self.doctor(model))
                seen.append(check.get("details", {}).get("path", ""))
            self.assertEqual(len(set(seen)), len(ARCHES),
                             f"engines collapsed to the same path: {seen}")


if __name__ == "__main__":
    unittest.main()
