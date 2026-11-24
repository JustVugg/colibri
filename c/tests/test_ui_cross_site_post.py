"""ui/server.py: a POST from a page on another site does not reach the gateway through the UI port.

The proxy drops the Origin before the gateway could look at it, and a page can POST without a
preflight (a text/plain body is a "simple" request). So the proxy checks it for the API paths:
this machine's names, the request's own Host, or a name added with --allowed-host.
"""
import http.client
import json
import sys
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))
import server as ui   # noqa: E402

POSTS = ("/v1/chat/completions", "/v1/completions", "/v1/messages", "/v1/responses")


class Upstream(BaseHTTPRequestHandler):
    hits = []

    def _answer(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        Upstream.hits.append((self.command, self.path))
        b = json.dumps({"ok": True}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_GET = do_POST = _answer

    def log_message(self, *a):
        pass


class UiCrossSitePostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        threading.Thread(target=cls.up.serve_forever, daemon=True).start()
        cls.saved = (ui.Handler.backend, ui.Handler.models_backend, ui.Handler.port,
                     ui.Handler.bind_host, ui.Handler.allowed_hosts)
        ui.Handler.backend = urllib.parse.urlparse("http://127.0.0.1:%d" % cls.up.server_port)
        ui.Handler.models_backend = None
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), ui.Handler)
        ui.Handler.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_port

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.up):
            s.shutdown()
            s.server_close()
        (ui.Handler.backend, ui.Handler.models_backend, ui.Handler.port,
         ui.Handler.bind_host, ui.Handler.allowed_hosts) = cls.saved

    def setUp(self):
        ui.Handler.bind_host, ui.Handler.allowed_hosts = "127.0.0.1", ()
        Upstream.hits.clear()

    def call(self, method, path, origin=None, host=None, ctype="text/plain"):
        body = b'{"model":"m","messages":[]}'
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest(method, path, skip_host=True)
            conn.putheader("Host", host or "127.0.0.1:%d" % self.port)
            if origin is not None:
                conn.putheader("Origin", origin)
            if method == "POST":
                conn.putheader("Content-Type", ctype)
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body if method == "POST" else None)
            response = conn.getresponse()
            response.read()
            return response.status
        finally:
            conn.close()

    def test_a_foreign_origin_never_reaches_the_gateway(self):
        for path in POSTS:
            for origin in ("https://evil.example", "http://evil.example:8790", "null", "not a url"):
                with self.subTest(path=path, origin=origin):
                    self.assertEqual(self.call("POST", path, origin), 403)
        self.assertEqual(Upstream.hits, [])

    def test_no_origin_is_a_script_and_runs(self):
        for path in POSTS[:3]:
            with self.subTest(path=path):
                self.assertEqual(self.call("POST", path), 200)

    def test_this_machine_and_this_host_run(self):
        for origin in ("http://127.0.0.1:%d" % self.port, "http://localhost:%d" % self.port,
                       "http://localhost:5173", "http://[::1]:%d" % self.port):
            with self.subTest(origin=origin):
                self.assertEqual(self.call("POST", "/v1/chat/completions", origin), 200)

    def test_the_origin_matching_the_host_runs_on_a_lan_bind(self):
        ui.Handler.bind_host = "0.0.0.0"
        host = "192.168.1.5:%d" % self.port
        self.assertEqual(self.call("POST", "/v1/chat/completions", "http://" + host, host=host), 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", "https://evil.example", host=host), 403)

    def test_an_allowed_host_name_runs(self):
        ui.Handler.allowed_hosts = ("chat.example",)
        self.assertEqual(self.call("POST", "/v1/chat/completions", "https://chat.example"), 200)
        self.assertEqual(self.call("POST", "/v1/chat/completions", "https://other.example"), 403)

    def test_a_wildcard_allows_any_origin(self):
        ui.Handler.allowed_hosts = ("*",)
        self.assertEqual(self.call("POST", "/v1/chat/completions", "https://anywhere.example"), 200)

    def test_a_get_with_a_foreign_origin_is_unchanged(self):
        self.assertEqual(self.call("GET", "/v1/models", "https://evil.example"), 200)
        self.assertEqual(self.call("GET", "/health", "https://evil.example"), 200)


if __name__ == "__main__":
    unittest.main()
