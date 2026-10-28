"""Overlapping MCP starts must share a single local server process."""
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcp_server
import setup_flow

LAUNCHER = '''import json, os, pathlib, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
folder = pathlib.Path(os.environ['TEST_DIR'])
with open(folder / 'launches', 'a') as handle:
    handle.write(str(os.getpid()) + '\\n')
# Model startup is not instantaneous: the PID/health endpoint arrives later.
time.sleep(1)
class Handler(BaseHTTPRequestHandler):
    server_version = 'colibri'
    sys_version = ''
    def log_message(self, *args):
        pass
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())
server = HTTPServer(('127.0.0.1', int(sys.argv[sys.argv.index('--port') + 1])), Handler)
pathlib.Path(os.environ['TEST_PID']).write_text(str(os.getpid()))
server.serve_forever()
'''


class StartAdmissionTest(unittest.TestCase):
    def test_overlapping_start_calls_launch_one_server(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            launcher = folder / 'launcher.py'
            launcher.write_text(LAUNCHER)
            pidfile = folder / 'server.pid'
            cfg = {'status': 'ready', 'model_dir': directory,
                   'engine': str(launcher), 'launcher': str(launcher),
                   'port': port, 'host': '127.0.0.1',
                   'args': ['--port', str(port)], 'model': {},
                   'env': {'TEST_DIR': directory, 'TEST_PID': str(pidfile)},
                   'urls': setup_flow.urls('127.0.0.1', port)}
            processes = []
            spawn = setup_flow.spawn_detached

            def tracked_spawn(*args, **kwargs):
                child = spawn(*args, **kwargs)
                processes.append(child)
                return child

            server = mcp_server.Server(io.StringIO())
            try:
                with mock.patch.dict(os.environ, {'COLI_SETUP_HOME': directory}), \
                        mock.patch.object(setup_flow, 'load_config', return_value=cfg), \
                        mock.patch.object(setup_flow, 'serve_pidfile', return_value=str(pidfile)), \
                        mock.patch.object(setup_flow, 'spawn_detached', side_effect=tracked_spawn):
                    for number in (1, 2):
                        server.handle({'jsonrpc': '2.0', 'id': number,
                                       'method': 'tools/call', 'params': {'name': 'start'}})
                    for worker in server.threads:
                        worker.join(timeout=12)
                        self.assertFalse(worker.is_alive())
                replies = [json.loads(line) for line in server.out.getvalue().splitlines()]
                self.assertEqual(len(processes), 1, 'overlapping starts spawned duplicate servers')
                self.assertEqual(len((folder / 'launches').read_text().splitlines()), 1)
                self.assertEqual({reply['id'] for reply in replies}, {1, 2})
                for reply in replies:
                    self.assertFalse(reply['result']['isError'], reply)
                    self.assertEqual(reply['result']['structuredContent']['state'], 'ready')
            finally:
                for child in processes:
                    if child.poll() is None:
                        child.terminate()
                    child.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
