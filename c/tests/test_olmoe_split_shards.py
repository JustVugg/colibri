"""Native OLMoE generation must agree for single- and multiple-directory shards.

Set COLI_OLMOE_FIXTURE to a tiny converted model with at least two safetensors
shards and COLI_OLMOE_ENGINE to the built executable. Only temporary copies are
used; the original model is never moved or changed.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent.parent
ENGINE = Path(os.environ.get("COLI_OLMOE_ENGINE", HERE / ("olmoe.exe" if os.name == "nt" else "olmoe")))
FIXTURE = Path(os.environ.get("COLI_OLMOE_FIXTURE", HERE / "olmoe_tiny_c"))


@unittest.skipUnless(ENGINE.is_file(), "OLMoE native executable is not built")
@unittest.skipUnless((FIXTURE / "config.json").is_file(), "tiny converted OLMoE fixture is absent")
class OlmoeSplitShardsTest(unittest.TestCase):
    def test_generation_with_all_shards_on_separate_directories(self):
        shards = sorted(FIXTURE.glob("*.safetensors"))
        self.assertGreaterEqual(len(shards), 2, "fixture needs at least two shards")
        with tempfile.TemporaryDirectory(prefix="colibri split ") as tmp:
            root = Path(tmp)
            metadata = root / "model metadata"
            first, second = root / "disk one", root / "disk two"
            for directory in (metadata, first, second):
                directory.mkdir()
            for path in FIXTURE.iterdir():
                if path.is_file() and path.suffix != ".safetensors":
                    shutil.copy2(path, metadata / path.name)
            for index, path in enumerate(shards):
                shutil.copy2(path, (first if index % 2 == 0 else second) / path.name)
            reference = root / "ref.json"
            # The reference supplies a prompt and generation length. We compare
            # the two native outputs rather than an arbitrary reference token.
            reference.write_text(json.dumps({"prompt_ids": [1, 10, 20],
                                             "full_ids": [1, 10, 20, 0, 0, 0, 0]}))

            def generate(snap, extra=None):
                environment = {key: value for key, value in os.environ.items()
                               if key not in ("CHAT", "SERVE", "PPL", "COLI_MODEL_DIRS")}
                environment.update(SNAP=str(snap), OMP_NUM_THREADS="2", PILOT="0", HOT="0")
                if extra is not None:
                    environment["COLI_MODEL_DIRS"] = ";".join(str(path) for path in extra)
                result = subprocess.run([str(ENGINE.resolve()), "2", "8", str(reference)],
                                        env=environment, capture_output=True,
                                        text=True, errors="replace", timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                match = re.search(r"C engine\s*:\s*([\d ]+)", result.stdout)
                self.assertIsNotNone(match, result.stdout + result.stderr)
                tokens = [int(token) for token in match[1].split()]
                self.assertEqual(len(tokens), 4)
                return tokens, result.stderr

            baseline, _ = generate(FIXTURE)
            split, diagnostics = generate(metadata, [first, second])
            self.assertEqual(split, baseline)
            self.assertIn("model across 3 dir(s)", diagnostics)
            self.assertIn("-> 0 shard(s)", diagnostics)


if __name__ == "__main__":
    unittest.main()
