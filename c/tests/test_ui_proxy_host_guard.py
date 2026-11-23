"""ui/server.py: the API paths it passes on to the gateway keep the DNS-rebinding guard.

The gateway refuses a Host that is not loopback, but behind this proxy it sees only the Host
the proxy writes (the client's is dropped), so a page on another site whose name is re-pointed
at 127.0.0.1 would reach the gateway through the UI port. The UI server checks the Host itself
for /v1/, /health, /profile, /experts, /power and /v1/responses when it is bound to this machine.
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

API_PATHS = ("/v1/models", "/health", "/profile", "/experts", "/power")


class Upstream(BaseHTTPRequestHandler):
    hits = []

    def _answer(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            self.rfile.read(n)
        Upstream.hits.append(self.path)
        b = json.dumps({"status": "ok", "path": self.path}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    do_GET = do_POST = _answer

    def log_message(self, *a):
        pass


class UiProxyHostGuardTest(unittest.TestCase):
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

    def call(self, method, path, host, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest(method, path, skip_host=True)
            conn.putheader("Host", host)
            if body is not None:
                conn.putheader("Content-Type", "application/json")
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body)
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    # controls: what must keep working
    def test_this_machine_by_any_of_its_names(self):
        for host in ("127.0.0.1:%d" % self.port, "localhost:%d" % self.port, "[::1]:%d" % self.port,
                     "localhost:9999",   # a tunnel moves the port
                     "127.0.0.1"):
            for path in API_PATHS:
                with self.subTest(host=host, path=path):
                    status, body = self.call("GET", path, host)
                    self.assertEqual((status, json.loads(body)["path"]), (200, path))

    # the finding
    def test_another_host_name_never_reaches_the_gateway(self):
        for host in ("attacker.example:%d" % self.port, "attacker.example", "127.0.0.1.attacker.example",
                     "localhost.attacker.example:%d" % self.port, ""):
            for path in API_PATHS:
                with self.subTest(host=host, path=path):
                    status, body = self.call("GET", path, host)
                    self.assertEqual(status, 403)
                    self.assertIn("not allowed", json.loads(body)["error"])
        self.assertEqual(Upstream.hits, [])   # the gateway was never asked

    def test_a_post_with_a_body_is_refused_before_the_gateway(self):
        for path in ("/v1/chat/completions", "/v1/responses"):
            with self.subTest(path=path):
                status, _ = self.call("POST", path, "attacker.example:%d" % self.port, b'{"model":"x"}')
                self.assertEqual(status, 403)
        self.assertEqual(Upstream.hits, [])

    # the operator's way out
    def test_an_allowed_host_is_accepted_and_others_still_are_not(self):
        ui.Handler.allowed_hosts = ("ui.example.com",)
        self.assertEqual(self.call("GET", "/health", "UI.example.com:443")[0], 200)
        self.assertEqual(self.call("GET", "/health", "attacker.example")[0], 403)

    def test_a_wildcard_accepts_any_host(self):
        ui.Handler.allowed_hosts = ("*",)
        self.assertEqual(self.call("GET", "/health", "anything.example")[0], 200)

    def test_a_server_bound_beyond_this_machine_is_unchanged(self):
        for bind in ("0.0.0.0", "192.168.1.20"):
            ui.Handler.bind_host = bind
            with self.subTest(bind=bind):
                self.assertEqual(self.call("GET", "/health", "mybox.lan:8790")[0], 200)

    def test_host_names_are_read_like_the_gateway_reads_them(self):
        for value, name in (("localhost:8790", "localhost"), ("127.0.0.1", "127.0.0.1"), ("[::1]:8790", "::1"),
                            ("::1", "::1"), ("EXAMPLE.com:80", "example.com"), ("", ""), (None, "")):
            self.assertEqual(ui.host_name(value), name, value)


if __name__ == "__main__":
    unittest.main()
