"""Native port exhaustion and CLI error handling at the TCP upper boundary."""
import http.server
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_flow


class NotColibri(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_error(404)

    def log_message(self, *args):
        pass


class PortRangeTests(unittest.TestCase):
    def listener(self, port):
        sock = socket.socket()
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            sock.listen()
        except OSError as error:
            sock.close()
            self.skipTest(f"native test port {port} unavailable: {error}")
        self.addCleanup(sock.close)
        return sock

    def test_port_below_one_is_setup_error(self):
        for port in (0, -1):
            with self.subTest(port=port):
                with self.assertRaises(setup_flow.SetupError):
                    setup_flow.pick_port("127.0.0.1", port)

    def test_free_last_port_remains_selectable(self):
        sock = self.listener(65535)
        sock.close()
        self.assertEqual(setup_flow.pick_port("127.0.0.1", 65535), 65535)

    def test_occupied_last_port_is_setup_error(self):
        self.listener(65535)
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.pick_port("127.0.0.1", 65535)

    def test_exhausted_valid_tail_is_setup_error(self):
        self.listener(65534)
        self.listener(65535)
        with self.assertRaises(setup_flow.SetupError):
            setup_flow.pick_port("127.0.0.1", 65534)

    def test_ordinary_next_port_is_still_selected(self):
        # Reserve a native adjacent pair; no fake port-free implementation.
        for _ in range(100):
            first = socket.socket()
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            second = socket.socket()
            try:
                if port >= 65535:
                    continue
                second.bind(("127.0.0.1", port + 1))
                first.listen()
                second.close()
                self.assertEqual(setup_flow.pick_port("127.0.0.1", port), port + 1)
                return
            except OSError:
                pass
            finally:
                first.close()
                second.close()
        self.fail("could not reserve adjacent local ports")

    def test_actual_start_cli_reports_exhaustion_without_traceback(self):
        try:
            server = http.server.HTTPServer(("127.0.0.1", 65535), NotColibri)
        except OSError as error:
            self.skipTest(f"native test port 65535 unavailable: {error}")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        launcher = Path(setup_flow.COLI_SCRIPT)
        for json_mode in (False, True):
            with self.subTest(json_mode=json_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "models").mkdir()
                engine = root / "unused-engine"
                engine.write_text("placeholder; exhaustion must precede engine execution")
                cfg = {"version": setup_flow.CONFIG_VERSION, "status": "ready",
                       "host": "127.0.0.1", "port": 65535, "model": {},
                       "model_dir": str(root / "models"), "backend": "cpu",
                       "requested_backend": "cpu", "engine": str(engine),
                       "launcher": str(launcher), "args": ["web", "--port", "65535"]}
                (root / "setup.json").write_text(json.dumps(cfg))
                env = {key: os.environ[key] for key in
                       ("PATH", "SYSTEMROOT", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH")
                       if key in os.environ}
                env.update(HOME=tmp, COLI_SETUP_HOME=tmp, COLI_COLOR="0")
                command = [sys.executable, str(launcher), "start", "--no-browser", "--background"]
                if json_mode:
                    command.append("--json")
                result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 1)
                self.assertNotIn("Traceback", result.stderr)
                self.assertIn("setup stopped:", result.stderr)
                state = json.loads((root / "install-state.json").read_text())
                self.assertEqual(state["phase"], "error")
                self.assertIn("65535", state["message"])
                self.assertEqual(json.loads((root / "setup.json").read_text()), cfg)
                if json_mode:
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
