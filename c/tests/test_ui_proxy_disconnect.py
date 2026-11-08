"""The UI proxy shares reset cancellation without aborting half-closed readers."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import select
import socket
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
import openai_server as gw
from test_openai_server import ScriptedEngine, _spawn_test_server
from test_responses_disconnect import PrefillEngine, reset_connection

spec = importlib.util.spec_from_file_location("ui_disconnect_server", C.parent / "ui/server.py")
ui = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui)


class UIProxyDisconnectTest(unittest.TestCase):
    def setUp(self):
        for name, value in (("ARCH", "glm"), ("CHAT_FLAVOR", None)):
            patcher = patch.object(gw, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def server(self, handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, timeout=2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def proxy(self, backend):
        errors, finished = [], threading.Event()

        class Proxy(ui.Handler):
            def _proxy(self, *args):
                try:
                    return super()._proxy(*args)
                except Exception as error:
                    errors.append(error)
                    self.close_connection = True
                finally:
                    finished.set()

            def log_message(self, *args):
                pass

        Proxy.backend = backend
        return self.server(Proxy), errors, finished

    def connect(self, server, streaming=False):
        data = json.dumps({"model": "test-model", "stream": streaming,
                          "messages": [{"role": "user", "content": "hello"}]}).encode()
        connection = socket.create_connection(server.server_address, timeout=3)
        connection.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                            "Authorization: Bearer secret\r\nContent-Type: application/json\r\n"
                            "Content-Length: %d\r\n\r\n" % len(data)).encode() + data)
        return connection

    def next_request(self, server):
        request = Request("http://127.0.0.1:%d/v1/chat/completions" % server.server_port,
                          data=json.dumps({"model": "test-model", "messages": [
                              {"role": "user", "content": "next"}]}).encode(),
                          headers={"Authorization": "Bearer secret", "Content-Type": "application/json"})
        with urlopen(request, timeout=3) as response:
            self.assertEqual(response.status, 200)

    def assert_monitors_finished(self):
        for thread in threading.enumerate():
            if thread.name == "responses-disconnect":
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive())

    def test_confirmed_resets_cancel_prefill_and_release_admission(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                engine = PrefillEngine()
                self.addCleanup(engine.release.set)
                backend = urlsplit(_spawn_test_server(self, engine))
                server, errors, finished = self.proxy(backend)
                connection = self.connect(server, streaming)
                try:
                    self.assertTrue(engine.entered.wait(2))
                finally:
                    reset_connection(connection)
                self.assertTrue(engine.cancelled.wait(1))
                self.assertTrue(engine.finished.wait(1))
                self.assertTrue(finished.wait(1))
                self.assertEqual(errors, [])
                self.next_request(server)
                self.assert_monitors_finished()

    def test_half_closed_request_still_reads_response_and_next_request(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                engine = PrefillEngine()
                self.addCleanup(engine.release.set)
                backend = urlsplit(_spawn_test_server(self, engine))
                server, errors, finished = self.proxy(backend)
                connection = self.connect(server, streaming)
                try:
                    self.assertTrue(engine.entered.wait(2))
                    connection.shutdown(socket.SHUT_WR)
                    self.assertFalse(engine.cancelled.wait(0.2))
                    engine.release.set()
                    response = http.client.HTTPResponse(connection)
                    try:
                        response.begin()
                        self.assertEqual(response.status, 200)
                        body = response.read()
                        if streaming:
                            self.assertIn(b"[DONE]", body)
                        else:
                            self.assertEqual(json.loads(body)["choices"][0]["message"]["content"], "answer")
                    finally:
                        response.close()
                finally:
                    engine.release.set()
                    connection.close()
                self.assertTrue(finished.wait(1))
                self.assertEqual(errors, [])
                self.next_request(server)
                self.assert_monitors_finished()

    def test_normal_completion_preserves_forwarded_request_and_closes_responses(self):
        observed = []

        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                observed.append((self.command, self.path, self.headers.get("Authorization"),
                                 self.headers.get("Origin"), self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.send_header("X-Fixture", "preserved")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *args):
                pass

        backend = self.server(Backend)
        server, errors, finished = self.proxy(urlsplit("http://127.0.0.1:%d" % backend.server_port))
        responses = []
        getresponse = http.client.HTTPConnection.getresponse

        def capture(connection):
            response = getresponse(connection)
            responses.append(response)
            return response

        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        try:
            with patch.object(http.client.HTTPConnection, "getresponse", capture):
                frontend_socket = None
                for _ in range(2):
                    finished.clear()
                    connection.request("POST", "/v1/fixture?query=kept", body=b'{"body":"kept"}',
                                       headers={"Authorization": "Bearer fixture", "Origin": "http://localhost",
                                                "Content-Type": "application/json"})
                    frontend_socket = frontend_socket or connection.sock
                    response = connection.getresponse()
                    try:
                        self.assertEqual(response.status, 201)
                        self.assertEqual(response.headers["X-Fixture"], "preserved")
                        self.assertEqual(json.load(response), {"ok": True})
                    finally:
                        response.close()
                    self.assertIs(connection.sock, frontend_socket, "normal UI keepalive connection was replaced")
                    self.assertTrue(finished.wait(1))
                self.assertEqual(len(responses), 4)
                self.assertTrue(all(response.isclosed() for response in responses))
        finally:
            connection.close()
        self.assertEqual(observed, [("POST", "/v1/fixture?query=kept", "Bearer fixture", None, b'{"body":"kept"}')] * 2)
        self.assertEqual(errors, [])
        self.assert_monitors_finished()

    def test_alive_client_keeps_upstream_header_failure_as_502(self):
        class Backend(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.close_connection = True

            def log_message(self, *args):
                pass

        backend = self.server(Backend)
        server, errors, finished = self.proxy(urlsplit("http://127.0.0.1:%d" % backend.server_port))
        connection = self.connect(server)
        response = http.client.HTTPResponse(connection)
        try:
            response.begin()
            self.assertEqual(response.status, 502)
            self.assertIn("colibri non risponde", json.loads(response.read())["error"]["message"])
        finally:
            response.close()
            connection.close()
        self.assertTrue(finished.wait(1))
        self.assertEqual(errors, [])
        self.assert_monitors_finished()

    def held_response(self, status=200, chunked=False, abort=False):
        started, gone, release = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class Backend(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if chunked else "HTTP/1.0"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(status)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                data = b"held-body"
                self.wfile.write((b"9\r\n" + data + b"\r\n") if chunked else data)
                self.wfile.flush()
                started.set()
                if abort:
                    self.close_connection = True
                    return
                while not release.wait(0.01):
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        gone.set()
                        return

            def log_message(self, *args):
                pass

        backend = self.server(Backend)
        server, errors, finished = self.proxy(urlsplit("http://127.0.0.1:%d" % backend.server_port))
        connection = self.connect(server)
        try:
            self.assertTrue(started.wait(2))
            if abort:
                self.assertTrue(finished.wait(1))
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], http.client.IncompleteRead)
            else:
                received = b""
                while b"held-body" not in received:
                    chunk = connection.recv(8192)
                    self.assertTrue(chunk)
                    received += chunk
                reset_connection(connection)
                self.assertTrue(gone.wait(1), "UI response-owned upstream socket stayed open")
                self.assertTrue(finished.wait(1))
                self.assertEqual(errors, [])
            self.assert_monitors_finished()
        finally:
            connection.close()
            release.set()

    def test_reset_closes_response_owned_http10_body(self):
        self.held_response()

    def test_reset_closes_held_error_body(self):
        self.held_response(status=503)

    def test_reset_closes_chunked_body(self):
        self.held_response(chunked=True)

    def test_live_reader_observes_upstream_chunk_failure(self):
        self.held_response(chunked=True, abort=True)


if __name__ == "__main__":
    unittest.main()
