import importlib.util
import json
import os
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

    def test_info_reads_registered_nested_text_config(self):
        with tempfile.TemporaryDirectory() as model:
            (Path(model) / "config.json").write_text(json.dumps({
                "model_type": "qwen4_exp",
                "text_config": {
                    "model_type": "qwen4_exp_text",
                    "hidden_size": 2560,
                    "num_hidden_layers": 48,
                    "num_experts": 512,
                    "num_experts_per_tok": 10,
                },
            }), encoding="utf-8")
            result = self.run_cli("info", "--model", model)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("hidden 2560 · 48 layer · 512 expert/layer · top-10",
                      result.stdout)
        self.assertNotIn("hidden None", result.stdout)

    def test_missing_model_error_is_english(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = str(Path(directory) / "missing-model")
            result = self.run_cli("run", "--model", missing_model, "hello")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model not found", result.stderr)
        self.assertIn("set COLI_MODEL or use --model", result.stderr)


class InteractivePromptTest(unittest.TestCase):
    """Pasted prompts stay intact and render predictably in the TUI box."""

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_prompt_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def test_prompt_box_preserves_newlines_and_indentation(self):
        lines = self.coli.prompt_box_lines("int main() {\n  return 0;\n}", 20)
        self.assertEqual(lines, ["int main() {", "  return 0;", "}"])

    def test_prompt_box_wraps_long_lines_without_collapsing_whitespace(self):
        lines = self.coli.prompt_box_lines("    return a_long_name;", 10)
        self.assertEqual(lines, ["    return", " a_long_na", "me;"])

    def test_prompt_input_rows_accounts_for_explicit_lines(self):
        self.assertEqual(self.coli.prompt_input_rows("one\ntwo", 80), 2)
        self.assertEqual(self.coli.prompt_input_rows("x" * 80, 80), 2)

    def test_read_prompt_collects_lines_already_queued_after_first(self):
        import io
        import select

        stream = io.StringIO("second\nthird\n")
        with mock.patch.object(self.coli, "TTY", True), \
             mock.patch.object(self.coli.sys, "platform", "freebsd"), \
             mock.patch("builtins.input", return_value="first"), \
             mock.patch.object(self.coli.sys, "stdin", stream), \
             mock.patch.object(select, "select", side_effect=(
                 ([stream], [], []), ([stream], [], []), ([], [], []),
             )):
            self.assertEqual(self.coli.read_prompt(), "first\nsecond\nthird")


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


class BannerModelLineTest(unittest.TestCase):
    """The banner's third line must describe the model that is loaded.

    It said "GLM-5.2 · 744B MoE · int4 · streaming CPU" for every checkpoint,
    because model_arch() answers "glm" for anything it does not recognise --
    the right default for choosing an engine, and a wrong statement of fact.
    """

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_banner_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def make_model(self, config, shard_bytes=0):
        directory = Path(tempfile.mkdtemp(prefix="coli-banner-"))
        self.addCleanup(lambda: __import__("shutil").rmtree(directory, ignore_errors=True))
        (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
        if shard_bytes:
            # An empty file: the size is reported by the getsize patch in
            # line(). truncate() to the real size would be sparse on ext4 and
            # APFS but NOT on NTFS, where it allocates -- the first revision of
            # this test asked a Windows runner for 372 GB and got
            # "OSError: [Errno 28] No space left on device".
            (directory / "model-00001.safetensors").write_bytes(b"")
        return directory

    def line(self, config, shard_bytes=0):
        directory = self.make_model(config, shard_bytes)
        if not shard_bytes:
            return self.coli.model_banner_line(str(directory))
        real_getsize = os.path.getsize

        def fake_getsize(path):
            return shard_bytes if str(path).endswith(".safetensors") else real_getsize(path)

        with mock.patch.object(self.coli.os.path, "getsize", fake_getsize):
            return self.coli.model_banner_line(str(directory))

    def test_each_engine_names_itself(self):
        for model_type, expected in (
            ("glm_moe_dsa", "GLM-5.2"),
            ("inkling", "Inkling"),
            ("kimi_k3", "Kimi K3"),
            ("deepseek_v4", "DeepSeek V4 Flash"),
            ("olmoe", "OLMoE"),
        ):
            with self.subTest(model_type=model_type):
                line = self.line({"model_type": model_type, "n_routed_experts": 8})
                self.assertTrue(line.startswith(expected), line)

    def test_deepseek_v4_is_not_read_as_glm(self):
        """The regression this exists for: a non-GLM checkpoint said GLM-5.2."""
        line = self.line({"model_type": "deepseek_v4", "n_routed_experts": 256})
        self.assertNotIn("GLM", line)
        self.assertNotIn("744B", line)

    def test_unknown_model_states_its_own_type(self):
        """No forcing into the roster: an unknown checkpoint speaks for itself."""
        line = self.line({"model_type": "qwen3_moe", "num_hidden_layers": 48,
                          "n_routed_experts": 128})
        self.assertIn("qwen3_moe", line)
        self.assertIn("48L x 128E", line)
        self.assertNotIn("GLM", line)

    def test_missing_model_type_does_not_invent_one(self):
        line = self.line({"num_hidden_layers": 32})
        self.assertIn("unknown model", line)
        self.assertNotIn("GLM-5.2", line)

    def test_no_model_keeps_the_generic_tagline(self):
        self.assertIn("GLM-5.2", self.coli.model_banner_line(None))

    def test_unreadable_model_falls_back_instead_of_raising(self):
        """`coli info` banners before validating the path; it must not crash."""
        self.assertIn("GLM-5.2", self.coli.model_banner_line("/nonexistent/xyz"))

    def test_size_is_reported_without_rounding_to_zero(self):
        small = self.line({"model_type": "olmoe"}, shard_bytes=4_200_000_000)
        self.assertIn("4.2 GB on disk", small)
        large = self.line({"model_type": "glm_moe_dsa"}, shard_bytes=372_000_000_000)
        self.assertIn("372 GB on disk", large)
        tiny = self.line({"model_type": "olmoe"}, shard_bytes=3_000_000)
        self.assertIn("MB on disk", tiny)

    def test_model_is_keyword_only(self):
        """banner(sub, x) must not read x as a path: other PRs add arguments."""
        with self.assertRaises(TypeError):
            self.coli.banner("run", True)


class OmpThreadsForEveryEngineTest(unittest.TestCase):
    """Launchers size shared engines; V4 delegates to its loader-aware runtime.

    #805's physical-core default still covers the memory-bound sister engines.
    DeepSeek V4 instead reserves logical CPUs for its expert-loader workers in
    v4_omp_reserve_loader_cpus(), unless the operator overrides or disables it.
    """

    @classmethod
    def setUpClass(cls):
        loader = SourceFileLoader("coli_omp_under_test", str(CLI))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        cls.coli = importlib.util.module_from_spec(spec)
        loader.exec_module(cls.coli)

    def args(self):
        return types.SimpleNamespace(model="/x", ram=None, ctx=None, ngen=None,
                                     temp=None, cap=None)

    def test_other_engines_get_physical_cores(self):
        with mock.patch("resource_plan.physical_cpu_count", return_value=6):
            for arch in ("inkling", "kimi", "olmoe"):
                with self.subTest(arch=arch):
                    env = self.coli.env_for_engine(self.args(), arch)
                    self.assertEqual(env.get("OMP_NUM_THREADS"), "6")

    def test_v4_delegates_thread_team_to_runtime(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch("resource_plan.physical_cpu_count",
                        side_effect=AssertionError("V4 launcher sized the team")):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertNotIn("OMP_NUM_THREADS", env)

    def test_v4_gets_memory_bound_affinity_defaults(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
             mock.patch.object(self.coli.sys, "platform", "linux"):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertEqual(env.get("OMP_PROC_BIND"), "close")
        self.assertEqual(env.get("OMP_PLACES"), "cores")
        self.assertEqual(env.get("OMP_WAIT_POLICY"), "active")
        self.assertEqual(env.get("GOMP_SPINCOUNT"), "200000")
        self.assertEqual(env.get("OMP_DYNAMIC"), "FALSE")

    def test_explicit_setting_still_wins(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}, clear=True), \
             mock.patch("resource_plan.physical_cpu_count",
                        side_effect=AssertionError("V4 launcher sized the team")):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        self.assertEqual(env["OMP_NUM_THREADS"], "3")

    def test_kill_switch_is_honoured(self):
        with mock.patch.dict(os.environ, {"COLI_NO_OMP_TUNE": "1"}, clear=True):
            env = self.coli.env_for_engine(self.args(), "deepseek_v4")
        for key in ("OMP_NUM_THREADS", "OMP_WAIT_POLICY", "GOMP_SPINCOUNT",
                    "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES"):
            self.assertNotIn(key, env)

    def test_qwen38_rejects_gpu_and_vram_flags_before_binary_detection(self):
        gpu_args = self.args();gpu_args.gpu="0";gpu_args.vram=0
        with self.assertRaisesRegex(SystemExit, "CPU only.*--gpu"):
            self.coli.env_for_engine(gpu_args, "qwen38")
        vram_args = self.args();vram_args.gpu=None;vram_args.vram=4
        with self.assertRaisesRegex(SystemExit, "CPU only.*--vram"):
            self.coli.env_for_engine(vram_args, "qwen38")


if __name__ == "__main__":
    unittest.main()
