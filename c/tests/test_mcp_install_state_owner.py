"""Only the install child may publish its durable progress/completion state."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

C_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C_DIR))
import mcp_server
import setup_flow


class InstallStateOwnerTests(unittest.TestCase):
    def setUp(self):
        owner = patch.object(mcp_server, "_INSTALL_PROCESS", None, create=True)
        owner.start()
        self.addCleanup(owner.stop)

    def test_child_completion_before_spawn_return_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
            children = []

            def spawn(command, **kwargs):
                child_code = (
                    "import sys; sys.path.insert(0,sys.argv[1]); import setup_flow; "
                    "setup_flow.write_state('done', result='native-child-result')"
                )
                child = subprocess.Popen([sys.executable, "-c", child_code, str(C_DIR)], **kwargs)
                children.append(child)
                # The OS can run a fast child before scheduling the parent
                # again after Popen; make that valid ordering deterministic.
                child.wait(timeout=5)
                return child

            with patch.object(setup_flow, "spawn_detached", side_effect=spawn):
                result = mcp_server.tool_install({"start": False}, lambda *_: None)
            state = setup_flow.read_state()
            self.assertEqual(state["phase"], "done")
            self.assertEqual(state["result"], "native-child-result")
            self.assertEqual(state["pid"], children[0].pid)
            self.assertEqual(result["install"], state)

    def test_listing_models_preserves_existing_install_state(self):
        with tempfile.TemporaryDirectory() as home:
            env = dict(os.environ, COLI_SETUP_HOME=home, HOME=home, USERPROFILE=home)
            state = Path(home) / "install-state.json"
            original = '{"phase":"done","pid":123,"result":"existing"}'
            state.write_text(original)
            result = subprocess.run(
                [sys.executable, str(C_DIR / "coli"), "setup", "--list", "--json",
                 "--dir", str(Path(home) / "models")],
                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertEqual(state.read_text(), original)

    def test_launch_gap_status_and_admission_use_the_real_child(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
            setup_flow._write_json(setup_flow.state_path(),
                                   {"phase": "done", "pid": 99999999, "result": "previous"})
            original = Path(setup_flow.state_path()).read_bytes()
            release = Path(home) / "release"
            children = []

            def spawn(command, **kwargs):
                code = (
                    "import sys,time,pathlib; sys.path.insert(0,sys.argv[1]); import setup_flow; "
                    "release=pathlib.Path(sys.argv[2]); deadline=time.monotonic()+10\n"
                    "while not release.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                    "setup_flow.write_state('done',result='current')"
                )
                child = subprocess.Popen([sys.executable, "-c", code, str(C_DIR), str(release)], **kwargs)
                children.append(child)
                return child

            try:
                with patch.object(setup_flow, "spawn_detached", side_effect=spawn):
                    first = mcp_server.tool_install({"start": False}, lambda *_: None)
                    status = mcp_server.tool_status({}, lambda *_: None)
                    self.assertEqual(first["install"]["pid"], children[0].pid)
                    self.assertEqual(status["install"]["phase"], "starting")
                    self.assertTrue(status["install"]["running"])
                    self.assertEqual(Path(setup_flow.state_path()).read_bytes(), original)
                    with self.assertRaisesRegex(mcp_server.ToolError, "already running"):
                        mcp_server.tool_install({"start": False}, lambda *_: None)
                    self.assertEqual(len(children), 1)
                    release.touch()
                    children[0].wait(timeout=5)
                    final = mcp_server.tool_status({}, lambda *_: None)
                    self.assertEqual(final["install"]["result"], "current")
                    self.assertFalse(final["install"]["running"])
            finally:
                release.touch()
                for child in children:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=5)

    def test_exit_before_first_state_never_reports_previous_completion(self):
        for code in (0, 7):
            with self.subTest(exit_code=code), tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
                setup_flow._write_json(setup_flow.state_path(),
                                       {"phase": "done", "pid": 99999999, "result": "previous"})
                children = []

                def spawn(command, **kwargs):
                    child = subprocess.Popen([sys.executable, "-c", "import sys;sys.exit(int(sys.argv[1]))", str(code)], **kwargs)
                    children.append(child)
                    child.wait(timeout=5)
                    return child

                with patch.object(setup_flow, "spawn_detached", side_effect=spawn):
                    with self.assertRaises(mcp_server.ToolError) as caught:
                        mcp_server.tool_install({"start": False}, lambda *_: None)
                result = json.loads(str(caught.exception))
                self.assertEqual(result["install"]["pid"], children[0].pid)
                self.assertEqual(result["install"]["phase"], "error")
                self.assertEqual(result["install"]["exit_code"], code)
                self.assertNotIn("result", result["install"])
                status = mcp_server.tool_status({}, lambda *_: None)
                self.assertEqual(status["install"]["pid"], children[0].pid)
                self.assertFalse(status["install"]["running"])

    def test_finished_call_does_not_borrow_a_later_admitted_job(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
            children = []

            def spawn(command, **kwargs):
                code = ("import sys;sys.path.insert(0,sys.argv[1]);import setup_flow;"
                        "setup_flow.write_state('done',result=sys.argv[2])")
                child = subprocess.Popen([sys.executable, "-c", code, str(C_DIR),
                                          str(len(children))], **kwargs)
                children.append(child)
                child.wait(timeout=5)
                return child

            read_state = mcp_server._current_install_state
            admitted_second = []

            def read_with_new_admission(process=mcp_server._CURRENT_INSTALL, **kwargs):
                if children and process is children[0] and not admitted_second:
                    admitted_second.append(True)
                    mcp_server.tool_install({"start": False}, lambda *_: None)
                return read_state(process, **kwargs)

            with patch.object(setup_flow, "spawn_detached", side_effect=spawn), \
                    patch.object(mcp_server, "_current_install_state", side_effect=read_with_new_admission):
                with self.assertRaises(mcp_server.ToolError) as caught:
                    mcp_server.tool_install({"start": False}, lambda *_: None)
            result = json.loads(str(caught.exception))
            self.assertEqual(len(children), 2)
            self.assertEqual(result["install"]["pid"], children[0].pid)
            self.assertNotIn("result", result["install"])
            status = mcp_server.tool_status({}, lambda *_: None)
            self.assertEqual(status["install"]["pid"], children[1].pid)
            self.assertEqual(status["install"]["result"], "1")

    def test_later_external_cli_state_remains_visible(self):
        with tempfile.TemporaryDirectory() as home, patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
            children = []

            def spawn(command, **kwargs):
                child = subprocess.Popen([sys.executable, "-c", "pass"], **kwargs)
                children.append(child)
                child.wait(timeout=5)
                return child

            with patch.object(setup_flow, "spawn_detached", side_effect=spawn):
                with self.assertRaises(mcp_server.ToolError):
                    mcp_server.tool_install({"start": False}, lambda *_: None)
            setup_flow._write_json(setup_flow.state_path(),
                                   {"phase": "download", "pid": os.getpid(), "model": "external"})
            active = mcp_server.tool_status({}, lambda *_: None)
            self.assertEqual(active["install"]["model"], "external")
            self.assertTrue(active["install"]["running"])
            with self.assertRaisesRegex(mcp_server.ToolError, "already running"):
                mcp_server.tool_install({"start": False}, lambda *_: None)
            setup_flow._write_json(setup_flow.state_path(),
                                   {"phase": "done", "pid": os.getpid(), "result": "external"})
            status = mcp_server.tool_status({}, lambda *_: None)
            self.assertEqual(status["install"]["result"], "external")
            self.assertFalse(status["install"]["running"])

    def test_reused_setup_publishes_ready_before_no_start_exit(self):
        for json_mode in (False, True):
            with self.subTest(json=json_mode), tempfile.TemporaryDirectory() as home, \
                    patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
                folder = Path(home)
                model = folder / "model"
                model.mkdir()
                engine = folder / "prebuilt-engine"
                # A configured external/prebuilt engine needs no rebuild and
                # --no-start must not execute it.
                engine.write_bytes(b"not executed: no-start")
                cfg = {
                    "status": "ready", "family": "glm", "model_dir": str(model),
                    "model": {"id": None, "name": "local fixture", "repo": None},
                    "engine": str(engine), "launcher": str(C_DIR / "coli"),
                    "env": {}, "args": [], "host": "127.0.0.1", "port": None,
                    "backend": "cpu", "requested_backend": "cpu",
                    "urls": setup_flow.urls("127.0.0.1", 8000),
                }
                setup_flow.save_config(cfg)
                command = [sys.executable, str(C_DIR / "coli"), "setup", "--yes",
                           "--no-browser", "--no-start"]
                if json_mode:
                    command.append("--json")
                result = subprocess.run(command, env=dict(os.environ, HOME=home, USERPROFILE=home),
                                        capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                if json_mode:
                    self.assertEqual(json.loads(result.stdout)["config"]["model_dir"], str(model))
                else:
                    self.assertIn("Already set up", result.stdout)
                state = setup_flow.read_state()
                self.assertEqual(state["phase"], "ready")
                self.assertEqual(state["model_dir"], str(model))
                self.assertNotEqual(state["pid"], os.getpid())
