import contextlib
import importlib.machinery
import importlib.util
import io
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


C_DIR = Path(__file__).resolve().parents[1]
CLI = C_DIR / "coli"
sys.path.insert(0, str(C_DIR))

_loader = importlib.machinery.SourceFileLoader("coli_ramdisk_cli", str(CLI))
_spec = importlib.util.spec_from_loader("coli_ramdisk_cli", _loader)
coli = importlib.util.module_from_spec(_spec)
_loader.exec_module(coli)


class RamdiskCliTest(unittest.TestCase):
    def run_cli(self, *args, env=None):
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        return subprocess.run(
            [sys.executable, str(CLI), *args],
            cwd=C_DIR,
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )

    def parsed_dispatch(self, *argv):
        import ramdisk

        captured = {}

        def fake_dispatch(args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return 0

        with mock.patch.object(ramdisk, "dispatch", side_effect=fake_dispatch):
            with self.assertRaises(SystemExit) as stopped:
                coli.main(list(argv))
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("args", captured)
        return captured

    def test_nested_actions_are_scriptable(self):
        action_options = {
            "plan": ("--json",),
            "prepare": (),
            "status": ("--json",),
            "benchmark": ("--json",),
            "start": ("--base-port", "8123"),
            "stop": (),
            "destroy": ("--yes",),
        }
        for action, options in action_options.items():
            with self.subTest(action=action):
                captured = self.parsed_dispatch("ramdisk", action, *options)
                self.assertEqual(captured["args"].ramdisk_action, action)
                self.assertEqual(Path(captured["kwargs"]["cli_path"]), CLI)
                self.assertTrue(captured["kwargs"]["engine_path"])

    def test_shared_model_option_works_before_and_after_action(self):
        before = self.parsed_dispatch(
            "ramdisk", "--model", "/tmp/model-before", "plan", "--json"
        )
        after = self.parsed_dispatch(
            "ramdisk", "plan", "--model", "/tmp/model-after", "--json"
        )
        self.assertEqual(before["args"].model, "/tmp/model-before")
        self.assertEqual(after["args"].model, "/tmp/model-after")

    def test_planning_knobs_reach_dispatch(self):
        captured = self.parsed_dispatch(
            "ramdisk",
            "plan",
            "--model",
            "/tmp/model",
            "--mode",
            "partial",
            "--topology",
            "per-node",
            "--capacity-gb",
            "12.5",
            "--profile",
            "/tmp/profile",
            "--mount-root",
            "/tmp/ram-root",
            "--allow-swappable",
            "--prefault",
            "0",
            "--parallel",
            "3",
            "--json",
        )
        args = captured["args"]
        self.assertEqual(args.mode, "partial")
        self.assertEqual(args.topology, "per-node")
        self.assertEqual(args.capacity_gb, 12.5)
        self.assertEqual(args.profile, "/tmp/profile")
        self.assertEqual(args.mount_root, "/tmp/ram-root")
        self.assertTrue(args.allow_swappable)
        self.assertEqual(args.prefault, 0)
        self.assertEqual(args.parallel, 3)
        self.assertTrue(args.json)

    def test_bare_command_rejects_non_tty_without_curses(self):
        result = self.run_cli("ramdisk")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("interactive TUI requires a terminal", result.stderr)
        self.assertIn("ramdisk plan --json", result.stderr)

    def test_help_lists_scriptable_actions(self):
        result = self.run_cli("ramdisk", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for action in (
            "plan",
            "prepare",
            "status",
            "benchmark",
            "start",
            "stop",
            "destroy",
        ):
            self.assertIn(action, result.stdout)

    def test_kv_resume_notice_uses_durable_state_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            state = root / "state"
            model.mkdir()
            state.mkdir()
            self.write_kv_header(model / ".coli_kv", 3)
            self.write_kv_header(state / ".coli_kv", 7)

            output = io.StringIO()
            with mock.patch.dict(os.environ, {"COLI_STATE_DIR": str(state)}), \
                 contextlib.redirect_stdout(output):
                coli.kv_resume_notice(str(model))

        self.assertIn("7 tokens", output.getvalue())
        self.assertNotIn("3 tokens", output.getvalue())

    @staticmethod
    def write_kv_header(path, tokens):
        fields = [0] * 8
        fields[6] = tokens
        path.write_bytes(b"COLIKV1\0" + struct.pack("<8i", *fields))


if __name__ == "__main__":
    unittest.main()
