"""`coli ui --backend https://...` speaks TLS to the gateway, never plain HTTP.

The UI's proxy (ui/server.py) and its Responses shim (ui/responses_shim.py) forward the
page's requests, the Authorization header with them. An https:// backend is a TLS front
for the gateway; reaching it in plain HTTP, on port 80 when no port is given, sends the
API key in clear. Here the "https" backend is a plain-HTTP server: a client that does
TLS fails its handshake (502 from the UI) and the server never reads a request.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import socket
import threading
import unittest
from urllib.parse import urlsplit

C = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ui_https_server", C.parent / "ui/server.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)
shim = ui.responses_shim


class PlainGateway(BaseHTTPRequestHandler):
    """A plain-HTTP 'gateway' that records every request it could read."""
    seen = []

    def _answer(self):
        type(self).seen.append((self.command, self.path, self.headers.get("Authorization")))
        body = b'{"object":"list","data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._answer()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._answer()

    def log_message(self, *args):
        pass


class UIHttpsBackend(unittest.TestCase):
    def serve(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, timeout=2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def ui_for(self, backend):
        class Quiet(ui.Handler):
            def log_message(self, *args):
                pass
        Quiet.backend = urlsplit(backend)
        return self.serve(Quiet)

    @staticmethod
    def send(server, method, path, body=b""):
        with socket.create_connection(server.server_address, timeout=10) as s:
            s.sendall(("%s %s HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer secret\r\n"
                       "Content-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
                       % (method, path, len(body))).encode() + body)
            reply = b""
            while chunk := s.recv(65536):
                reply += chunk
        return int(reply.split(b" ", 2)[1])

    def test_an_https_backend_is_never_sent_plain_http(self):
        PlainGateway.seen = []
        gateway = self.serve(PlainGateway)
        page = self.ui_for("https://127.0.0.1:%d" % gateway.server_port)
        responses = json.dumps({"model": "m", "input": "hi"}).encode()
        for method, path, body in (("GET", "/v1/models", b""),          # the proxy
                                   ("POST", "/v1/responses", responses)):  # the Responses shim
            with self.subTest(path=path):
                self.assertEqual(self.send(page, method, path, body), 502)
        self.assertEqual(PlainGateway.seen, [], "the key went to the https backend in clear")

    def test_an_http_backend_is_still_proxied(self):
        PlainGateway.seen = []
        gateway = self.serve(PlainGateway)
        page = self.ui_for("http://127.0.0.1:%d" % gateway.server_port)
        self.assertEqual(self.send(page, "GET", "/v1/models"), 200)
        self.assertEqual(PlainGateway.seen, [("GET", "/v1/models", "Bearer secret")])

    def test_default_ports_follow_the_scheme(self):
        for url, kind, port in (("https://gw.example", "HTTPSConnection", 443),
                                ("https://gw.example:8443", "HTTPSConnection", 8443),
                                ("http://gw.example", "HTTPConnection", 80),
                                ("http://127.0.0.1:8000", "HTTPConnection", 8000)):
            with self.subTest(url=url):
                conn = shim.gateway_connection(urlsplit(url))
                self.assertEqual((type(conn).__name__, conn.port), (kind, port))
                conn.close()


if __name__ == "__main__":
    unittest.main()
