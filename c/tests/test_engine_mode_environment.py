"""Check tool mode selection at the actual child-process boundary.

The child only records its environment and exits; it is not a model and does
not supply scoring or performance results. Eval tokenization is a small test
fixture so this suite remains dependency-free.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

from tools import benchmark_cuda_fixture, diag_harness, eval_glm


CONFLICTING_MODES = {
    "PROMPT": "ambient prompt", "COLI_PROMPT": "ambient override",
    "SCORE": "ambient.score", "ABLATE_SCORE": "ambient.ablation",
    "SERVE": "1", "SERVE_BATCH": "1", "REPLAY": "1", "CONSIST": "1",
    "TF": "1", "ORACLE_STRICT": "1", "EXPERT_WORKER": "1",
    "I4_ACC512_TEST": "1", "I3_AVX512_TEST": "1", "COLI_ANS_PACK": "1",
}
HARDWARE = {"RAM_GB": "7", "TOPP": "0.75", "COLI_METAL": "0"}


class EngineModeEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.capture = self.root / "environment.json"
        self.engine = self.root / "environment-probe"
        self.engine.write_text(
            "#!" + sys.executable + "\n"
            "import json, os\n"
            "with open(os.environ['COLIBRI_ENV_CAPTURE'], 'w') as f:\n"
            "    json.dump(dict(os.environ), f)\n"
            "raise SystemExit(7)\n")
        actual_popen = subprocess.Popen

        def portable_probe(command, *args, **kwargs):
            # Windows cannot execute a Python shebang directly. Translate only
            # our owned probe; all environment, process, stream, and exit-status
            # behavior still comes from a real Python child on every platform.
            if command and str(command[0]) == str(self.engine):
                command = [sys.executable, str(self.engine), *command[1:]]
            return actual_popen(command, *args, **kwargs)

        patcher = mock.patch.object(subprocess, "Popen", side_effect=portable_probe)
        patcher.start()
        self.addCleanup(patcher.stop)

    def environment(self, contaminated):
        env = dict(HARDWARE, COLIBRI_ENV_CAPTURE=str(self.capture))
        if contaminated:
            env.update(CONFLICTING_MODES)
        return mock.patch.dict(os.environ, env, clear=True)

    def captured(self):
        env = json.loads(self.capture.read_text())
        for key, value in HARDWARE.items():
            self.assertEqual(env[key], value)
        return env

    def assert_mode(self, env, selected):
        self.assertIn(selected, env)
        if selected in ("PROMPT", "REPLAY"):
            self.assertEqual(env[selected], "requested text" if selected == "PROMPT" else "1")
        for key in CONFLICTING_MODES:
            if key != selected:
                self.assertNotIn(key, env, msg=f"ambient {key} conflicts with requested {selected}")

    def prompt(self, contaminated):
        runner = diag_harness.EngineRunner(self.engine, "requested.snap", self.root,
                                           default_env={"PIPE": "1"}, timeout=5)
        with self.environment(contaminated):
            _, _, rc, _ = runner.run_prompt("requested text", ngen=9,
                                            env_extra={"MTP": "0"})
        self.assertEqual(rc, 7)
        env = self.captured()
        self.assertEqual(env["SNAP"], "requested.snap")
        self.assertEqual(env["NGEN"], "9")
        self.assertEqual(env["PIPE"], "1")
        self.assertEqual(env["MTP"], "0")
        self.assertEqual(env["TOKENS"], "1")
        self.assert_mode(env, "PROMPT")

    def score(self, contaminated):
        runner = diag_harness.EngineRunner(self.engine, "requested.snap", self.root,
                                           timeout=5)
        with self.environment(contaminated):
            _, _, rc, _ = runner.run_score("requested.score")
        self.assertEqual(rc, 7)
        env = self.captured()
        self.assertEqual(env["SCORE"], "requested.score")
        self.assert_mode(env, "SCORE")

    def benchmark(self, contaminated):
        argv = ["benchmark_cuda_fixture.py", "--model", str(self.root),
                "--engine", str(self.engine), "--threads", "2"]
        with self.environment(contaminated), mock.patch.object(sys, "argv", argv):
            # Stop at the first calibration launch: no GPU arms, synthetic
            # throughput, or fabricated successful benchmark are involved.
            with self.assertRaises(subprocess.CalledProcessError) as failure:
                benchmark_cuda_fixture.main()
        self.assertEqual(failure.exception.returncode, 7)
        env = self.captured()
        self.assertEqual(env["OMP_NUM_THREADS"], "2")
        self.assertEqual(env["REF"], str(self.root / "ref_glm.json"))
        self.assert_mode(env, "REPLAY")

    def evaluation(self, contaminated):
        class FixtureTokenizer:
            @staticmethod
            def from_file(path):
                return FixtureTokenizer()

            @staticmethod
            def encode(text):
                return types.SimpleNamespace(ids=[1, 2])

        (self.root / "config.json").write_text('{"vocab_size":3}\n')
        argv = ["eval_glm.py", "--snap", str(self.root), "--glm", str(self.engine),
                "--tasks", "smoke", "--limit", "1"]
        with self.environment(contaminated), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(sys.modules, {"tokenizers": types.SimpleNamespace(
                    Tokenizer=FixtureTokenizer)}), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            # An empty exit-7 stream remains incomplete. This only probes the
            # real child environment, not evaluation accuracy.
            self.assertEqual(eval_glm.main(), 1)
        self.assert_mode(self.captured(), "SCORE")

    def test_prompt_ignores_ambient_modes(self):
        self.prompt(True)

    def test_prompt_preserves_hardware_and_explicit_tuning(self):
        self.prompt(False)

    def test_score_ignores_ambient_modes(self):
        self.score(True)

    def test_score_preserves_hardware(self):
        self.score(False)

    def test_replay_benchmark_ignores_ambient_modes(self):
        self.benchmark(True)

    def test_replay_benchmark_preserves_hardware(self):
        self.benchmark(False)

    def test_evaluation_ignores_ambient_modes(self):
        self.evaluation(True)

    def test_evaluation_preserves_hardware(self):
        self.evaluation(False)


if __name__ == "__main__":
    unittest.main()
