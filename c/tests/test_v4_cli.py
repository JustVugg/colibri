import argparse
import importlib.machinery
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parent.parent
CLI = HERE / "coli"


def load_cli():
    loader = importlib.machinery.SourceFileLoader("coli_v4_cli_test", str(CLI))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class V4CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cli = load_cli()

    def make_model(self, model_type="deepseek_v4"):
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name)
        (root / "config.json").write_text(
            json.dumps({"model_type": model_type}), encoding="utf-8"
        )
        (root / "tokenizer.json").write_text("{}", encoding="utf-8")
        return directory, root

    def test_model_arch_detects_deepseek_v4(self):
        directory, root = self.make_model()
        try:
            self.assertEqual(self.cli.model_arch(str(root)), "deepseek_v4")
        finally:
            directory.cleanup()

    def test_engine_for_selects_deepseek_v4_binary(self):
        directory, root = self.make_model()
        try:
            expected = "deepseek_v4.exe" if os.name == "nt" else "deepseek_v4"
            self.assertEqual(Path(self.cli.engine_for(str(root))).name, expected)
        finally:
            directory.cleanup()

    def test_model_arch_selects_olmoe(self):
        directory, root = self.make_model("olmoe")
        try:
            self.assertEqual(self.cli.model_arch(str(root)), "olmoe")
        finally:
            directory.cleanup()

    def test_engine_for_selects_olmoe_binary(self):
        directory, root = self.make_model("olmoe")
        try:
            expected = "olmoe.exe" if os.name == "nt" else "olmoe"
            self.assertEqual(Path(self.cli.engine_for(str(root))).name, expected)
        finally:
            directory.cleanup()

    def test_olmoe_environment_enters_chat_mode(self):
        args = argparse.Namespace(ngen=32, temp=0.25, ram=0, ctx=0)
        env = self.cli.env_for_engine(args, "olmoe")
        self.assertEqual(env["CHAT"], "1")
        self.assertEqual(env["MAX_NEW"], "32")
        # #509 for olmoe: the engine reads COLI_TEMP like every other arch.
        # The legacy TEMP channel double-poisoned on Windows: it overrode the
        # child's %TEMP% directory with "0.25", while a real %TEMP% path read
        # back as atof("C:\...") == 0.0 and silently forced greedy decoding.
        # env starts from os.environ, so TEMP may be inherited — the launcher
        # must simply never write the temperature there.
        self.assertEqual(env["COLI_TEMP"], "0.25")
        self.assertNotEqual(env.get("TEMP"), "0.25")

    def test_olmoe_run_uses_its_engine_and_writes_one_prompt_line(self):
        directory, root = self.make_model("olmoe")
        args = argparse.Namespace(
            model=str(root), prompt=["hello", "world"], ngen=32, ram=0,
            temp=0.25, ctx=0, cap=None,
        )
        captured = {}

        def fake_run(command, **kwargs):
            captured["command"] = command
            captured.update(kwargs)
            return argparse.Namespace(returncode=0)

        try:
            with mock.patch.object(self.cli, "engine_for", return_value="/engines/olmoe"), \
                 mock.patch.object(self.cli, "need_model"), \
                 mock.patch.object(self.cli, "banner"), \
                 mock.patch("resource_plan.physical_cpu_count", return_value=6), \
                 mock.patch.object(self.cli.subprocess, "run", side_effect=fake_run):
                with self.assertRaises(SystemExit) as stopped:
                    self.cli.cmd_run(args)
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(captured["command"], ["/engines/olmoe", "16", "8"])
            self.assertEqual(captured["input"], "hello world\n")
            self.assertTrue(captured["text"])
            self.assertEqual(captured["env"]["CHAT"], "1")
            self.assertEqual(captured["env"]["MAX_NEW"], "32")
        finally:
            directory.cleanup()

    def test_v4_engine_environment_forwards_ram_and_context(self):
        args = argparse.Namespace(ngen=8, temp=0.0, ram=64, ctx=4096)
        env = self.cli.env_for_engine(args, "deepseek_v4")
        self.assertEqual(env["NGEN"], "8")
        self.assertEqual(env["RAM_GB"], "64")
        self.assertEqual(env["CTX"], "4096")

    def test_kimi_engine_environment_forwards_ram(self):
        """#855: `--ram` reached the environment for deepseek_v4 only, so on Kimi
        K3 it was set and never read -- the flag a user reaches for to bound
        memory did nothing, and the reported session ran itself out of RAM with
        it apparently in effect. kimi_k3 reads RAM_GB now, so coli must send it.
        """
        args = argparse.Namespace(ngen=8, temp=0.0, ram=242, ctx=0)
        env = self.cli.env_for_engine(args, "kimi")
        self.assertEqual(env["RAM_GB"], "242")

    def test_kimi_without_ram_stays_unset(self):
        """Absent --ram, the engine budgets from MemAvailable itself. Sending an
        empty or zero RAM_GB would read as an explicit ceiling of zero."""
        args = argparse.Namespace(ngen=8, temp=0.0, ram=0, ctx=0)
        env = self.cli.env_for_engine(args, "kimi")
        self.assertNotIn("RAM_GB", env)

    def test_ngen_default_differs_for_interactive_commands(self):
        """#889: `coli web` passed --max-tokens 1024, and openai_server clamps a
        request to that ceiling (#260), so the browser's own "max output tokens"
        control silently did nothing above 1024. One-shot runs keep 1024; a
        session where the user has a control gets a ceiling that is not in the
        way. An explicit --ngen always wins."""
        run = argparse.Namespace(ngen=None)
        self.assertEqual(self.cli.ngen_for(run), 1024)
        self.assertEqual(self.cli.ngen_for(run, interactive=True), 16384)
        explicit = argparse.Namespace(ngen=300)
        self.assertEqual(self.cli.ngen_for(explicit), 300)
        self.assertEqual(self.cli.ngen_for(explicit, interactive=True), 300)

    def test_one_shot_env_still_carries_the_historic_ngen(self):
        """The default moved to None so the two cases can be told apart. These
        sites must not become "unset": before, NGEN was always exported."""
        args = argparse.Namespace(ngen=None, temp=None, ram=0, ctx=0)
        env = self.cli.env_for_engine(args, "deepseek_v4")
        self.assertEqual(env["NGEN"], "1024")

    def test_kimi_context_uses_its_registered_environment_channel(self):
        args = argparse.Namespace(ngen=8, temp=0.0, ram=242, ctx=4096)
        env = self.cli.env_for_engine(args, "kimi")
        self.assertNotIn("CTX", env)
        self.assertEqual(env["K3_MAXT"], "4096")
        self.assertNotIn("V4_MTP", env)

    def test_windows_v4_run_passes_chinese_prompt_as_utf8_file(self):
        directory, root = self.make_model()
        prompt = "请用中文解释：存储、内存和显存如何协同推理？"
        args = argparse.Namespace(
            model=str(root), prompt=[prompt], ngen=10, ram=0,
            temp=None, ctx=0,
        )
        captured = {}

        def fake_call(command, env):
            prompt_index = command.index("--prompt-file") + 1
            prompt_path = Path(command[prompt_index])
            captured["command"] = list(command)
            captured["path"] = prompt_path
            captured["bytes"] = prompt_path.read_bytes()
            return 0

        try:
            with mock.patch.object(self.cli.sys, "platform", "win32"), \
                 mock.patch.object(self.cli, "engine_for",
                                   return_value="deepseek_v4.exe"), \
                 mock.patch.object(self.cli, "need_model"), \
                 mock.patch.object(self.cli, "banner"), \
                 mock.patch("resource_plan.physical_cpu_count", return_value=8), \
                 mock.patch.object(self.cli.subprocess, "call",
                                   side_effect=fake_call):
                with self.assertRaises(SystemExit) as stopped:
                    self.cli.cmd_run(args)
            self.assertEqual(stopped.exception.code, 0)
            self.assertEqual(captured["bytes"], prompt.encode("utf-8"))
            self.assertNotIn(prompt, captured["command"])
            self.assertFalse(captured["path"].exists())
        finally:
            directory.cleanup()

    def test_openai_renderer_uses_native_v4_multiturn_template(self):
        import openai_server

        prompt = openai_server.render_chat_v4(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
                {"role": "user", "content": "Again"},
            ],
            enable_thinking=True,
        )
        self.assertEqual(
            prompt,
            "<\uff5cbegin\u2581of\u2581sentence\uff5c>Be concise."
            "<\uff5cUser\uff5c>Hello<\uff5cAssistant\uff5c></think>Hi!"
            "<\uff5cend\u2581of\u2581sentence\uff5c>"
            "<\uff5cUser\uff5c>Again<\uff5cAssistant\uff5c><think>",
        )

    def test_openai_renderer_scaffolds_v4_tools(self):
        import openai_server

        prompt = openai_server.render_chat_v4(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {
                "name": "get_weather",
                "description": "current weather",
                "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}}, "required": ["city"]}}}],
        )
        self.assertIn("## Tools", prompt)
        self.assertIn('"get_weather"', prompt)
        # Sanity: tool_choice="none" suppresses the declaration.
        no_tools = openai_server.render_chat_v4(
            [{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="none",
        )
        self.assertNotIn("## Tools", no_tools)


if __name__ == "__main__":
    unittest.main()
