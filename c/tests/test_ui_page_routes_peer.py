"""ui/server.py: the page-only routes answer a caller on this machine, not whoever types a loopback Host.

/agents, /models/, /web/ and /doc/extract download and switch models, open agents and fetch pages.
Their guard is the Host, Origin and X-Colibri-UI headers, which stop another site in a browser but
are all the caller's to write: once the server listens on 0.0.0.0 a remote client sends them too.
So the caller's address must be loopback as well, unless --allow-remote-ui says otherwise.
"""
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

C = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C.parent / "ui"))
import server as ui   # noqa: E402


class PeerServer(ThreadingHTTPServer):
    """reports `peer` as the caller's address, as a request from another machine would show"""
    peer = "127.0.0.1"

    def get_request(self):
        sock, addr = super().get_request()
        return sock, (self.peer, addr[1])


class PageRoutesPeerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = (ui.Handler.backend, ui.Handler.models_backend, ui.Handler.models, ui.Handler.port,
                     ui.Handler.bind_host, ui.Handler.allowed_hosts, ui.Handler.allow_remote)
        ui.Handler.models_backend = None
        ui.Handler.models = None          # /models/... answers 404 "no models service" once past the guard
        cls.srv = PeerServer(("127.0.0.1", 0), ui.Handler)
        ui.Handler.port = cls.srv.server_port
        ui.Handler.bind_host = "0.0.0.0"  # as if the server listens beyond this machine
        cls.port = cls.srv.server_port
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        (ui.Handler.backend, ui.Handler.models_backend, ui.Handler.models, ui.Handler.port,
         ui.Handler.bind_host, ui.Handler.allowed_hosts, ui.Handler.allow_remote) = cls.saved

    def setUp(self):
        ui.Handler.allow_remote = False
        PeerServer.peer = "127.0.0.1"

    def call(self, method, path, host=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            conn.putrequest(method, path, skip_host=True)
            conn.putheader("Host", host or "127.0.0.1:%d" % self.port)
            if method == "POST":
                body = body if body is not None else b"{}"
                conn.putheader("Content-Type", "application/json")
                conn.putheader("X-Colibri-UI", "1")
                conn.putheader("Content-Length", str(len(body)))
            conn.endheaders(body if method == "POST" else None)
            response = conn.getresponse()
            data = response.read()
            return response.status, data
        finally:
            conn.close()

    ROUTES = (("GET", "/agents"), ("POST", "/agents/refresh"), ("POST", "/agents/launch"),
              ("GET", "/web/status"), ("GET", "/models/jobs"), ("POST", "/models/use"),
              ("POST", "/doc/extract"))

    def test_a_remote_peer_with_a_forged_loopback_host_is_refused(self):
        PeerServer.peer = "10.0.0.5"
        for method, path in self.ROUTES:
            with self.subTest(route=path):
                status, _ = self.call(method, path)
                self.assertEqual(status, 403)

    def test_a_peer_on_this_machine_gets_past_the_guard(self):
        for peer in ("127.0.0.1", "::1", "::ffff:127.0.0.1", "127.0.0.2"):
            PeerServer.peer = peer
            with self.subTest(peer=peer):
                self.assertEqual(self.call("GET", "/web/status")[0], 200)
                self.assertEqual(self.call("GET", "/models/jobs")[0], 404)   # past the guard: no models service here

    def test_addresses_that_are_not_loopback_are_refused(self):
        for peer in ("192.168.1.20", "172.17.0.1", "::ffff:10.0.0.5", "2001:db8::1", "not-an-address", ""):
            PeerServer.peer = peer
            with self.subTest(peer=peer):
                self.assertEqual(self.call("GET", "/web/status")[0], 403)

    def test_a_remote_peer_is_refused_whatever_the_host_says(self):
        PeerServer.peer = "10.0.0.5"
        for host in ("127.0.0.1:%d" % self.port, "localhost:%d" % self.port, "10.0.0.5:%d" % self.port):
            with self.subTest(host=host):
                self.assertEqual(self.call("GET", "/agents", host=host)[0], 403)

    def test_allow_remote_lifts_the_peer_check_and_nothing_else(self):
        ui.Handler.allow_remote = True
        PeerServer.peer = "10.0.0.5"
        self.assertEqual(self.call("GET", "/web/status")[0], 200)
        self.assertEqual(self.call("GET", "/models/jobs")[0], 404)
        # the Host check still holds: another site's name re-pointed at this server
        self.assertEqual(self.call("GET", "/web/status", host="attacker.example:%d" % self.port)[0], 403)

    def test_the_page_itself_is_not_a_page_only_route(self):
        PeerServer.peer = "10.0.0.5"
        status, body = self.call("GET", "/")
        self.assertNotEqual(status, 403)

    def test_the_refusal_names_the_way_out_in_the_log(self):
        import io
        import contextlib
        PeerServer.peer = "10.0.0.5"
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(self.call("GET", "/web/status")[0], 403)
        self.assertIn("--allow-remote-ui", err.getvalue())


class StartupTest(unittest.TestCase):
    """the real server process: the flag and its environment variable parse, and the warning shows"""

    def start(self, *args, env=None):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        environ = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        environ.pop("COLI_UI_ALLOW_REMOTE", None)
        environ.update(env or {})
        # a backend that is not loopback: no Models page to start in a test
        proc = subprocess.Popen([sys.executable, str(C.parent / "ui" / "server.py"), "--port", str(port),
                                 "--backend", "http://gateway.invalid:8000", *args],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environ)
        self.addCleanup(lambda: (proc.kill(), proc.wait(), proc.stdout.close(), proc.stderr.close()))
        line = proc.stdout.readline()      # "colibri UI: http://..." once it listens
        self.assertIn("colibri UI:", line, proc.poll())
        time.sleep(0.2)
        self.assertIsNone(proc.poll(), "the server exited")
        return proc, port

    def get(self, port, path):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        try:
            conn.request("GET", path)
            response = conn.getresponse()
            response.read()
            return response.status
        finally:
            conn.close()

    def test_the_flag_starts_the_server_and_a_local_caller_is_answered(self):
        _, port = self.start("--allow-remote-ui")
        self.assertEqual(self.get(port, "/web/status"), 200)

    def test_the_environment_variable_starts_the_server(self):
        _, port = self.start(env={"COLI_UI_ALLOW_REMOTE": "1"})
        self.assertEqual(self.get(port, "/web/status"), 200)

    def test_without_the_flag_a_local_caller_is_answered_too(self):
        _, port = self.start()
        self.assertEqual(self.get(port, "/web/status"), 200)

    def test_the_flag_on_a_wildcard_bind_warns(self):
        proc, _ = self.start("--host", "0.0.0.0", "--allow-remote-ui")
        proc.kill()
        self.assertIn("--allow-remote-ui", proc.stderr.read())


if __name__ == "__main__":
    unittest.main()
