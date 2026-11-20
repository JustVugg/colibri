#!/usr/bin/env python3
"""The chat's file tools through ui/server.py: /fs/ reaches the gateway only from the chat's own
page (its Host, its Origin and the X-Colibri-UI header), and the proxy drops the Origin, which the
gateway would otherwise refuse as not its own page."""
import json
import sys
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))


class Gateway(BaseHTTPRequestHandler):
    """stands in for the gateway: records each request and answers like POST /fs/list"""
    seen = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        Gateway.seen.append((self.path, self.headers.get("Origin"), json.loads(body)))
        out = json.dumps({"path": "/home/x", "entries": ["a.txt"], "more": 0}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class FsRouteTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import server
        cls.server = server
        cls.patches = [unittest.mock.patch.object(server.Handler, "log_message", lambda *a: None)]
        cls.gw = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
        cls.ui = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        cls.port = cls.ui.server_port
        cls.patches += [unittest.mock.patch.object(server.Handler, "port", cls.port),
                        unittest.mock.patch.object(server.Handler, "backend",
                                                   urllib.parse.urlparse("http://127.0.0.1:%d" % cls.gw.server_port))]
        for p in cls.patches:
            p.start()
        for s in (cls.gw, cls.ui):
            threading.Thread(target=s.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.gw, cls.ui):
            s.shutdown()
            s.server_close()
        for p in cls.patches:
            p.stop()

    def setUp(self):
        Gateway.seen.clear()

    def post(self, host=None, origin=None, ui=True, ctype="application/json"):
        own = "127.0.0.1:%d" % self.port
        h = {"Host": host or own, "Origin": origin or "http://" + own, "Content-Type": ctype}
        if ui:
            h["X-Colibri-UI"] = "1"
        req = urllib.request.Request("http://127.0.0.1:%d/fs/list" % self.port,
                                     data=json.dumps({"path": ".", "folder": ""}).encode(), headers=h)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_the_page_reaches_the_gateway_without_its_origin(self):
        code, d = self.post()
        self.assertEqual((code, d["entries"]), (200, ["a.txt"]))
        self.assertEqual(Gateway.seen, [("/fs/list", None, {"path": ".", "folder": ""})])

    def test_other_pages_hosts_and_bodies_are_refused_before_the_gateway(self):
        for kw in ({"ui": False}, {"origin": "http://evil.example"},
                   {"host": "evil.example:%d" % self.port}, {"ctype": "text/plain"}):
            with self.subTest(**kw):
                self.assertEqual(self.post(**kw)[0], 403)
        self.assertEqual(Gateway.seen, [])


if __name__ == "__main__":
    unittest.main()
