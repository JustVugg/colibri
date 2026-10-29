"""MMLU questions and greedy mode must reach the actual engine child."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "tools/eval_mmlu_subset.py"
MODES = {"COLI_PROMPT": "ambient unrelated question", "SERVE": "1", "SERVE_BATCH": "1",
         "SCORE": "ambient.score", "ABLATE_SCORE": "ambient.ablate", "REPLAY": "1",
         "CONSIST": "1", "TF": "1", "ORACLE_STRICT": "1", "EXPERT_WORKER": "1",
         "I4_ACC512_TEST": "1", "I3_AVX512_TEST": "1", "COLI_ANS_PACK": "1"}


class MmluSubsetEnvironmentTest(unittest.TestCase):
    def test_question_mode_temperature_and_adapter_override_ambient_settings(self):
        for adapter in (None, "requested-adapter"):
            for contaminated in (False, True):
                with self.subTest(adapter=adapter, contaminated=contaminated), \
                        tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    capture = root / "environment.json"
                    cache = root / "questions.json"
                    cache.write_text(json.dumps([{"subject": "logic", "question": "Requested question?",
                                                  "choices": ["A", "B", "C", "D"], "answer": 1}]))
                    engine = root / "environment-probe"
                    engine.write_text("import json, os\n"
                                      "open(os.environ['MMLU_TEST_CAPTURE'], 'w').write(json.dumps(dict(os.environ)))\n"
                                      "print('Answer: B\\n---')\n")
                    argv = [str(SCRIPT), "--model", "requested-model", "--n", "1",
                            "--cache", str(cache), "--binary", str(engine)]
                    if adapter: argv += ["--adapter", adapter]
                    env = {"MMLU_TEST_CAPTURE": str(capture), "RAM_GB": "7", "COLI_METAL": "0"}
                    if contaminated:
                        env.update(MODES, COLI_TEMP="1.7", TEMP="1.2", ADAPTER="ambient-adapter")
                    real_popen = subprocess.Popen
                    def portable_probe(command, *args, **kwargs):
                        if command and str(command[0]) == str(engine):
                            command = [sys.executable, str(engine), *command[1:]]
                        return real_popen(command, *args, **kwargs)
                    with mock.patch.dict(os.environ, env, clear=True), \
                            mock.patch.object(sys, "argv", argv), \
                            mock.patch.object(subprocess, "Popen", side_effect=portable_probe), \
                            contextlib.redirect_stdout(io.StringIO()):
                        runpy.run_path(str(SCRIPT), run_name="__main__")
                    actual = json.loads(capture.read_text())
                    self.assertIn("Requested question?", actual["PROMPT"])
                    for key in MODES:
                        self.assertNotIn(key, actual)
                    self.assertEqual(actual.get("COLI_TEMP", actual.get("TEMP")), "0")
                    self.assertEqual(actual["TEMP"], "0")
                    self.assertEqual(actual["NGEN"], "2")
                    self.assertEqual(actual["DRAFT"], "0")
                    self.assertEqual(actual["SNAP"], "requested-model")
                    self.assertEqual(actual.get("ADAPTER"), adapter)
                    self.assertEqual((actual["RAM_GB"], actual["COLI_METAL"]), ("7", "0"))


if __name__ == "__main__":
    unittest.main()
