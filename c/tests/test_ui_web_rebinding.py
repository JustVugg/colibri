#!/usr/bin/env python3
"""ui/web.py opens a page only at a public address, and the address it checks has to be the one it
connects to. A name can answer a public address to the check and this machine to the connection
that follows (DNS rebinding: a TTL-0 record), and the page the model asked for is then one of this
machine's own services.

No network: a stand-in resolver answers the names, and one server on 127.0.0.1 plays this
machine's service and, in the tests where 127.0.0.1 counts as public, the site."""
import ipaddress
import socket
import ssl
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))
import web   # noqa: E402

PUBLIC = "93.184.216.34"
_getaddrinfo = socket.getaddrinfo
_is_global = ipaddress.IPv4Address.is_global


class Local(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        Local.seen.append((self.path, self.headers.get("Host")))
        if self.path.startswith("/to/"):   # a redirect to the name after /to/
            self.send_response(302)
            self.send_header("Location", "http://%s:%d/" % (self.path[4:], self.server.server_port))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b"<html><body><p>SECRET from a service on 127.0.0.1 only</p></body></html>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class FakeTLS:
    """a TLS context that records the name the handshake would check, and leaves the socket plain"""
    check_hostname, verify_mode, post_handshake_auth = True, ssl.CERT_REQUIRED, None

    def __init__(self):
        self.names = []

    def set_alpn_protocols(self, protocols):
        pass

    def wrap_socket(self, sock, server_hostname=None, **kw):
        self.names.append(server_hostname)
        return sock


class PageAddressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Local)
        cls.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Local.seen = []
        # each name's answers, one per lookup, the last one kept
        self.answers = {"rebind.test": [PUBLIC, "127.0.0.1"], "site.test": ["127.0.0.1"], "example.test": [PUBLIC]}
        self.public = set()   # addresses that count as public in this test
        self.tls = FakeTLS()
        test = self
        for p in (mock.patch("socket.getaddrinfo", self.resolve),
                  mock.patch.object(ipaddress.IPv4Address, "is_global",
                                    property(lambda ip: str(ip) in test.public or _is_global.fget(ip))),
                  mock.patch("urllib.request.getproxies", lambda: {}),
                  mock.patch("ssl._create_default_https_context", lambda *a, **k: self.tls)):
            p.start()
            self.addCleanup(p.stop)

    def resolve(self, host, port, *args, **kw):
        if host not in self.answers:
            return _getaddrinfo(host, port, *args, **kw)
        if host == "rebind.test":   # from the rebinding name on, this machine is this machine again
            self.public.discard("127.0.0.1")
        seq = self.answers[host]
        ip = seq.pop(0) if len(seq) > 1 else seq[0]
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    def url(self, scheme, host, path="/"):
        return "%s://%s:%d%s" % (scheme, host, self.port, path)

    def test_a_rebinding_name_is_not_fetched(self):
        for scheme in ("http", "https"):
            with self.subTest(scheme=scheme):
                Local.seen, self.answers["rebind.test"] = [], [PUBLIC, "127.0.0.1"]
                with self.assertRaisesRegex(ValueError, "not a public address: rebind.test"):
                    web.page(self.url(scheme, "rebind.test"))
                self.assertEqual(Local.seen, [], "this machine's service was read")

    def test_a_redirect_to_a_rebinding_name_is_not_followed(self):
        self.public.add("127.0.0.1")   # the first site, until the redirect names rebind.test
        with self.assertRaisesRegex(ValueError, "not a public address: rebind.test"):
            web.page(self.url("http", "site.test", "/to/rebind.test"))
        self.assertEqual(Local.seen, [("/to/rebind.test", "site.test:%d" % self.port)])

    def test_a_public_name_is_still_fetched(self):
        # the site's name stays in the Host header and in the TLS handshake: only the address is pinned
        self.public.add("127.0.0.1")
        for scheme in ("http", "https"):
            with self.subTest(scheme=scheme):
                Local.seen = []
                got = web.page(self.url(scheme, "site.test"))
                self.assertIn("SECRET", got["text"])
                self.assertEqual(Local.seen, [("/", "site.test:%d" % self.port)])
        self.assertEqual(self.tls.names, ["site.test"])

    def test_through_a_proxy_the_proxy_is_reached(self):
        # a proxy on this machine is not the page: it is sent the whole URL and resolves the name
        with mock.patch("urllib.request.getproxies", lambda: {"http": "http://127.0.0.1:%d" % self.port}), \
                mock.patch("urllib.request.proxy_bypass", lambda host: False):
            got = web.page("http://example.test/a")
        self.assertIn("SECRET", got["text"])
        self.assertEqual(Local.seen, [("http://example.test/a", "example.test")])

    def test_this_machine_by_name_or_number(self):
        for url in (self.url("http", "127.0.0.1"), "http://localhost:%d/" % self.port):
            with self.subTest(url=url), self.assertRaisesRegex(ValueError, "not a public address"):
                web.page(url)
        self.assertEqual(Local.seen, [])


if __name__ == "__main__":
    unittest.main()
