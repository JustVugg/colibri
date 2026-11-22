"""convert_fp8_to_int4's downloads carry HF_TOKEN to the hub and not past its redirect: the hub
answers a file with a redirect to its CDN, and a plain header follows urllib's redirects
(setup_download._request's rule)."""
import ast
import os
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock


CONVERTER = Path(__file__).resolve().parent.parent / "tools" / "convert_fp8_to_int4.py"


def load_hf_request():
    tree = ast.parse(CONVERTER.read_text(encoding="utf-8"), filename=str(CONVERTER))
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_hf_request")
    namespace = {"os": os}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(CONVERTER), "exec"), namespace)
    return namespace["_hf_request"]


def serve(handler):
    server = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class TokenStaysWithTheHub(unittest.TestCase):
    def setUp(self):
        self.seen = {}
        seen = self.seen

        class Elsewhere(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["elsewhere"] = self.headers.get("Authorization")
                seen["range"] = self.headers.get("Range")
                body = b"weights"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.elsewhere = serve(Elsewhere)
        target = f"http://127.0.0.1:{self.elsewhere.server_address[1]}/blob"

        class Hub(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["hub"] = self.headers.get("Authorization")
                self.send_response(302)
                self.send_header("Location", target)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                pass

        self.hub = serve(Hub)
        self.url = f"http://127.0.0.1:{self.hub.server_address[1]}/repo/resolve/main/model.safetensors"

    def tearDown(self):
        for server in (self.hub, self.elsewhere):
            server.shutdown()
            server.server_close()

    def test_redirect_target_never_gets_the_token(self):
        hf_request = load_hf_request()
        with mock.patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}):
            request = hf_request(self.url, {"User-Agent": "colibri-convert", "Range": "bytes=0-6"})
            with urllib.request.urlopen(request, timeout=10) as response:
                self.assertEqual(response.read(), b"weights")
        self.assertEqual(self.seen["hub"], "Bearer hf_test_token")
        self.assertIsNone(self.seen["elsewhere"], "the token followed the redirect")
        self.assertEqual(self.seen["range"], "bytes=0-6", "the other headers still follow it")

    def test_no_token_no_header(self):
        hf_request = load_hf_request()
        with mock.patch.dict(os.environ, {"HF_TOKEN": ""}):
            with urllib.request.urlopen(hf_request(self.url, {}), timeout=10) as response:
                response.read()
        self.assertIsNone(self.seen["hub"])
        self.assertIsNone(self.seen["elsewhere"])


if __name__ == "__main__":
    unittest.main()
