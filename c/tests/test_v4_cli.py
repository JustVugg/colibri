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

    def test_v4_engine_environment_forwards_ram_and_context(self):
        args = argparse.Namespace(ngen=8, temp=0.0, ram=64, ctx=4096)
        env = self.cli.env_for_engine(args, "deepseek_v4")
        self.assertEqual(env["NGEN"], "8")
        self.assertEqual(env["RAM_GB"], "64")
        self.assertEqual(env["CTX"], "4096")

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

    def test_openai_renderer_rejects_unwired_tools(self):
        import openai_server

        with self.assertRaises(openai_server.APIError):
            openai_server.render_chat_v4(
                [{"role": "user", "content": "hello"}],
                tools=[{"type": "function"}],
            )


if __name__ == "__main__":
    unittest.main()
