"""Small real process fixture: no model files or optional dependencies."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--delay', type=float, default=0)
parser.add_argument('--response', default='good')
parser.add_argument('--no-evidence', action='store_true')
parser.add_argument('--fail', action='store_true')
parser.add_argument('--flood', action='store_true')
parser.add_argument('--secrets', action='store_true')
parser.add_argument('--tree', type=int, default=0)
parser.add_argument('--pid-dir')
parser.add_argument('--sleeper', action='store_true')
parser.add_argument('--exit-after', type=float, default=0)
parser.add_argument('--request-marker')
args = parser.parse_args()
if args.pid_dir:
    Path(args.pid_dir, str(os.getpid())).write_text('alive')
if args.tree:
    subprocess.Popen([sys.executable, __file__, '--port', str(args.port),
                      '--pid-dir', args.pid_dir, '--sleeper', '--tree', str(args.tree - 1)])
if args.sleeper:
    if os.name != 'nt':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(90)
    sys.exit(0)
if args.flood:
    os.write(1, b'token=very-secret-value\nAuthorization: Bearer another-secret\n')
    os.write(1, b'password=' + b'x' * 100000 + b'\ninvalid:\xff\n')
    for i in range(3000):
        print('progress ' + str(i), flush=True)
if args.secrets:
    print('Authorization: Basic dXNlcjpwYXNzd29yZA==', flush=True)
    print('{"access_token": "hidden-token", "api_key": "hidden-key"}', flush=True)
    os.write(1, b'invalid:\xff\n')
time.sleep(args.delay)
if args.fail:
    print('simulated engine failure', file=sys.stderr, flush=True)
    sys.exit(7)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if args.request_marker:
            with open(args.request_marker, 'a') as marker:
                marker.write(self.path + '\n')
        if args.response in ('slow-body', 'slow-header'):
            if args.response == 'slow-body':
                self.send_response(200)
                self.end_headers()
            else:
                self.wfile.write(b'HTTP/1.1 200 OK\r\nX-Slow: ')
            try:
                for _ in range(60):
                    self.wfile.write(b' ')
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass
            return
        if args.response == 'redirect':
            self.send_response(302)
            self.send_header('Location', '/good')
            self.end_headers()
            return
        if self.path == '/health':
            result = {'status': 'bad' if args.response == 'bad-health' else 'ok'}
        elif args.response == 'empty':
            result = {'data': []}
        elif args.response == 'malformed':
            result = {'data': [None, 1, 'test-model']}
        else:
            result = {'data': [{'id': 'other-model' if args.response == 'wrong' else 'test-model'}]}
        self.send_response(500 if args.response == 'http-error' else 200)
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass

server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
if not args.no_evidence:
    print(f'OpenAI-compatible API listening on http://127.0.0.1:{args.port}/v1', file=sys.stderr, flush=True)
if args.exit_after:
    import threading
    threading.Timer(args.exit_after, lambda: os._exit(9)).start()
server.serve_forever()
