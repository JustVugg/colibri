"""File readiness for catalog post-install artifacts; no model or engine runs."""
import argparse
import base64
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest import mock

C = Path(__file__).resolve().parents[1]
if str(C) not in sys.path:
    sys.path.insert(0, str(C))
import setup_catalog
import setup_download
import setup_flow


class GeneratedOutputReadinessTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        home = self.root / "setup-home"
        env = mock.patch.dict(os.environ, {"COLI_SETUP_HOME": str(home)})
        env.start()
        self.addCleanup(env.stop)
        # Use the shipped catalog; preserve the caller's optional overlay.
        overlay = os.environ.pop("COLI_SETUP_CATALOG", None)
        if overlay is not None:
            self.addCleanup(os.environ.__setitem__, "COLI_SETUP_CATALOG", overlay)
        self.model = self.root / "model"
        self.model.mkdir()
        raw = b"".join(base64.b64encode(bytes([i])) + b" " + str(i).encode() + b"\n"
                       for i in range(256))
        (self.model / "tiktoken.model").write_bytes(raw)
        files = [{"path": name, "size": (self.model / name).stat().st_size}
                 for name in ("tiktoken.model",)]
        (self.model / setup_download.MANIFEST).write_text(json.dumps({
            "complete": True, "repo": "moonshotai/Kimi-K3", "files": files}))
        # Actual stdlib serializer with a tiny byte vocabulary, not a checkpoint.
        script = C / "tools/k3_tokenizer.py"
        spec = importlib.util.spec_from_file_location("test_actual_k3_serializer", script)
        serializer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(serializer)
        with mock.patch.object(sys, "argv", [str(script), str(self.model)]), \
             contextlib.redirect_stdout(io.StringIO()):
            serializer.main()
        self.engine = self.root / "engine-stat-fixture"
        self.launcher = self.root / "launcher-stat-fixture"
        self.engine.write_text("not an executable engine\n")
        self.launcher.write_text("not an executable launcher\n")
        entry = setup_catalog.by_id("kimi-k3")
        self.cfg = setup_flow.save_config({"status": "ready", "family": entry.family,
            "model": {"id": entry.id, "name": entry.name, "repo": entry.repo},
            "model_dir": str(self.model), "engine": str(self.engine),
            "launcher": str(self.launcher), "backend": "cpu", "gpu_pending": [],
            "host": "127.0.0.1", "port": 8000, "urls": setup_flow.urls("127.0.0.1", 8000),
            "args": ["web", "--model", str(self.model)]})

    def public_cli(self):
        # The complete CLI/parser/dispatch. Restore its process-local file-limit
        # adjustment on platforms with resource, including when dispatch raises.
        try:
            import resource
        except ImportError:
            resource = None
        limits = resource.getrlimit(resource.RLIMIT_NOFILE) if resource else None
        try:
            cli = runpy.run_path(str(C / "coli"), run_name="test_setup_generated_outputs_cli")
            cli["main"]()
        finally:
            if resource:
                resource.setrlimit(resource.RLIMIT_NOFILE, limits)

    def model_bytes(self):
        return {str(path.relative_to(self.model)): path.read_bytes()
                for path in self.model.rglob("*") if path.is_file()}

    def test_public_setup_parser_keeps_healthy_fast_path(self):
        before = self.model_bytes()
        output = io.StringIO()
        with mock.patch.object(setup_flow.setup_hw, "detect", side_effect=AssertionError("hardware probe")), \
             mock.patch.object(setup_flow.subprocess, "Popen", side_effect=AssertionError("child launch")), \
             mock.patch.object(setup_flow, "download_model", side_effect=AssertionError("download")), \
             mock.patch.object(setup_flow, "start_server", side_effect=AssertionError("server launch")), \
             mock.patch.object(sys, "argv", [str(C / "coli"), "setup", "--yes", "--no-start"]), \
             contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_error:
                self.public_cli()
        self.assertEqual(exit_error.exception.code, 0)
        self.assertIn("Already set up:", output.getvalue())
        self.assertIn("Ready. Start it", output.getvalue())
        self.assertEqual(self.model_bytes(), before)

    def test_generated_tokenizer_and_download_are_ready(self):
        self.assertTrue(setup_download.is_complete(str(self.model)))
        self.assertTrue(setup_flow.config_ready(self.cfg))

    def test_missing_generated_tokenizer_is_not_ready(self):
        (self.model / "tokenizer.json").unlink()
        self.assertTrue(setup_download.is_complete(str(self.model)))
        self.assertFalse(setup_flow.config_ready(self.cfg))

    def test_directory_at_generated_file_path_is_not_ready(self):
        (self.model / "tokenizer.json").unlink()
        (self.model / "tokenizer.json").mkdir()
        self.assertTrue(setup_download.is_complete(str(self.model)))
        self.assertFalse(setup_flow.config_ready(self.cfg))

    def test_catalog_model_without_recipe_keeps_existing_readiness(self):
        (self.model / "tokenizer.json").unlink()
        model = dict(self.cfg["model"], id="qwen36-35b")
        self.assertTrue(setup_flow.config_ready(dict(self.cfg, model=model)))

    def test_local_folder_and_unlisted_model_keep_existing_readiness(self):
        (self.model / "tokenizer.json").unlink()
        for model in ({"id": None, "repo": None},
                      {"id": "no-longer-in-catalog", "repo": self.cfg["model"]["repo"]}):
            with self.subTest(model=model):
                self.assertTrue(setup_flow.config_ready(dict(self.cfg, model=model)))

    def test_start_reports_setup_recovery_before_server_or_build(self):
        (self.model / "tokenizer.json").unlink()
        with mock.patch.object(setup_flow, "refresh_configured", side_effect=AssertionError("build probe")), \
             mock.patch.object(setup_flow, "start_server", side_effect=AssertionError("server launch")):
            with self.assertRaisesRegex(setup_flow.SetupError, "run `coli setup`"):
                setup_flow.cmd_start(argparse.Namespace(background=True, no_browser=True, json=False))

    def test_public_setup_parser_enters_resume_without_false_ready(self):
        (self.model / "tokenizer.json").unlink()
        class ResumeReached(RuntimeError):
            pass
        # Stop the real slow branch at its first hardware call: no simulated
        # successful install, download, working model, engine or server.
        before = self.model_bytes()
        output = io.StringIO()
        with mock.patch.object(setup_flow.setup_hw, "detect", side_effect=ResumeReached), \
             mock.patch.object(setup_flow.subprocess, "Popen", side_effect=AssertionError("child launch")), \
             mock.patch.object(setup_flow, "download_model", side_effect=AssertionError("download")), \
             mock.patch.object(setup_flow, "start_server", side_effect=AssertionError("server launch")), \
             mock.patch.object(sys, "argv", [str(C / "coli"), "setup", "--yes", "--no-start"]), \
             contextlib.redirect_stdout(output):
            with self.assertRaises(ResumeReached):
                self.public_cli()
        self.assertNotIn("Already set up:", output.getvalue())
        self.assertNotIn("Ready. Start it", output.getvalue())
        self.assertFalse((self.model / "tokenizer.json").exists())
        self.assertEqual(self.model_bytes(), before)


if __name__ == "__main__":
    unittest.main()
