"""A private gateway closing mid-generation is a failed one-shot run."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
loader = SourceFileLoader("coli_gateway_stream_test", str(HERE / "coli"))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class GatewayStreamCompletionTests(unittest.TestCase):
    def _run(self, complete):
        class Peer(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                payload = {"status": "ok"} if self.path == "/health" else {"data": [{"id": "fixture"}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"partial answer"}}]}\n\n')
                if complete:
                    self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()

        real_popen = subprocess.Popen
        servers, children = [], []

        def start_peer(command, **kwargs):
            port = int(command[command.index("--port") + 1])
            server = ThreadingHTTPServer(("127.0.0.1", port), Peer)
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
            children.append(child)
            return child

        output = io.StringIO()
        try:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory, "config.json").write_text('{"model_type":"qwen4_exp"}')
                args = argparse.Namespace(model=directory, cap=None, ngen=8, effort=None, think=False)
                family = cli.resolve_model(directory).descriptor
                with patch.object(cli.subprocess, "Popen", side_effect=start_peer), \
                     patch.object(cli, "env_for_engine", return_value=dict(os.environ)), \
                     contextlib.redirect_stdout(output):
                    code = None
                    try:
                        cli.run_via_gateway(args, family, "unused-fixture-engine", "hello")
                    except SystemExit as error:
                        code = error.code
                self.assertTrue(all(child.poll() is not None for child in children))
                return code, output.getvalue()
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for child in children:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=5)

    def test_truncated_generation_is_reported_as_failure(self):
        code, output = self._run(False)
        self.assertNotIn(code, (None, 0), "EOF without the completion marker was accepted")
        self.assertIn("partial answer", output)

    def test_completed_generation_retains_success_and_output(self):
        code, output = self._run(True)
        self.assertIn(code, (None, 0))
        self.assertIn("partial answer", output)
