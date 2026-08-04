import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


class CliOutputLanguageTest(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=HERE,
            text=True,
            encoding="utf-8",
            capture_output=True,
            check=False,
            timeout=10,
        )

    def test_help_is_english(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("run GLM-5.2 locally", result.stdout)
        self.assertIn("automatically apply the RAM/VRAM plan", result.stdout)
        self.assertNotIn("modello", result.stdout.lower())
        self.assertNotIn("motore", result.stdout.lower())

    def test_serve_help_includes_allowed_host(self):
        result = self.run_cli("serve", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allowed-host", result.stdout)

    def test_tune_help_describes_measured_safe_profile(self):
        result = self.run_cli("tune", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("fastest quality-preserving execution profile", result.stdout)
        self.assertIn("--min-gain", result.stdout)

    def test_info_status_is_english(self):
        with tempfile.TemporaryDirectory() as model:
            result = self.run_cli("info", "--model", model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("config.json is missing", result.stdout)
        self.assertIn("disk", result.stdout)
        self.assertIn("engine", result.stdout)

    def test_missing_model_error_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing-model")
            result = self.run_cli("run", "--model", missing_model, "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)
        self.assertIn("set COLI_MODEL or use --model", result.stderr)


class ChatCapForwardingTest(unittest.TestCase):
    """#379/#386 r2 (F9): `coli chat` on a non-glm model spawns openai_server
    as its local server. An explicit --cap must ride along on that command
    line (it was silently eaten for years -- keeping it is a DISCLOSED
    behavior change: a long-ignored `coli chat --cap 32` now takes effect),
    and an absent --cap must stay absent so openai_server's arch-keyed
    default (cap_for_arch) applies. This drives the real cmd_chat with the
    process boundary faked, and pins the argv it builds."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_cli_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def _chat_server_cmd(self, cap):
        coli = self.coli
        captured = {}

        class FakeProc:
            def __init__(self, cmd, **_kw):
                captured["cmd"] = cmd
            def poll(self):
                return None
            def terminate(self):
                pass
            def wait(self, timeout=None):
                return 0
            def kill(self):
                pass

        class FakeSpinner:
            def __init__(self, *_a, **_k):
                pass
            def start(self):
                pass
            def stop(self):
                pass

        model = tempfile.mkdtemp()
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", model], check=False))
        (Path(model) / "config.json").write_text(json.dumps({"model_type": "inkling"}))
        args = types.SimpleNamespace(model=model, cap=cap, ngen=256, api_key=None,
                                     no_attach=True, attach=None)
        with mock.patch.object(coli, "need_model"), \
             mock.patch.object(coli, "banner"), \
             mock.patch.object(coli, "engine_for", return_value="/stub/inkling"), \
             mock.patch.object(coli, "env_for_engine", return_value={}), \
             mock.patch.object(coli, "server_probe", return_value="inkling-colibri"), \
             mock.patch.object(coli, "chat_attached"), \
             mock.patch.object(coli, "Spinner", FakeSpinner), \
             mock.patch("subprocess.Popen", FakeProc):
            coli.cmd_chat(args)
        return captured["cmd"]

    def test_explicit_cap_rides_along(self):
        cmd = self._chat_server_cmd(cap=32)
        self.assertIn("--cap", cmd)
        self.assertEqual(cmd[cmd.index("--cap") + 1], "32")
        self.assertEqual(cmd[cmd.index("--arch") + 1], "inkling")

    def test_explicit_cap_zero_rides_along(self):
        # explicit 0 = upstream RAM-auto for inkling, by request (#386 r2, F8)
        cmd = self._chat_server_cmd(cap=0)
        self.assertEqual(cmd[cmd.index("--cap") + 1], "0")

    def test_absent_cap_stays_absent(self):
        cmd = self._chat_server_cmd(cap=None)
        self.assertNotIn("--cap", cmd)


if __name__ == "__main__":
    unittest.main()
