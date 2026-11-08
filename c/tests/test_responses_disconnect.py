"""Confirmed client resets cancel Responses work; orderly half-close stays valid."""
import http.client
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import struct
import os
import select
import sys
import threading
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError

C = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(C))
import openai_server as gw
from test_openai_server import ScriptedEngine, _spawn_test_server

spec = importlib.util.spec_from_file_location("responses_disconnect_shim", C.parent / "ui/responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


def reset_connection(connection):
    # Winsock's linger members are unsigned shorts; POSIX uses two ints.
    linger = struct.pack("HH" if os.name == "nt" else "ii", 1, 0)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
    connection.close()


class PrefillEngine(ScriptedEngine):
    def __init__(self):
        super().__init__(chunks=("answer",))
        self.entered = threading.Event()
        self.cancelled = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, *args, **kwargs):
        if not self.entered.is_set():
            self.entered.set()
            try:
                while not self.release.wait(0.01):
                    # The real handler supplies cancellation as positional argument 7.
                    cancelled = args[6] if len(args) > 6 else kwargs.get("cancelled")
                    if cancelled and cancelled():
                        self.cancelled.set()
                        raise gw.ClientCancelled()
                return super().generate(*args, **kwargs)
            finally:
                self.finished.set()
        return super().generate(*args, **kwargs)


class ResponsesDisconnectTest(unittest.TestCase):
    def setUp(self):
        for name, value in (("ARCH", "glm"), ("CHAT_FLAVOR", None)):
            patcher = patch.object(gw, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def harness(self, proxy, engine=None):
        engine = engine if engine is not None else PrefillEngine()
        if isinstance(engine, PrefillEngine):
            self.addCleanup(engine.release.set)
        backend = urlsplit(_spawn_test_server(self, engine))
        if not proxy:
            return engine, backend, "/v1/chat/completions"

        return engine, self.proxy(backend), "/v1/responses"

    def proxy(self, backend):
        self.proxy_errors = errors = []
        self.proxy_finished = finished = threading.Event()

        class Proxy(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    shim.serve(self, backend)
                except Exception as error:
                    errors.append(error)
                finally:
                    finished.set()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, timeout=2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return urlsplit("http://127.0.0.1:%d" % server.server_port)

    def disconnect(self, proxy, streaming):
        engine, endpoint, path = self.harness(proxy)
        body = {"model": "test-model", "stream": streaming}
        body.update({"input": "hello", "model": "test-model"} if proxy else {"messages": [{"role": "user", "content": "hello"}]})
        data = json.dumps(body).encode()
        connection = socket.create_connection((endpoint.hostname, endpoint.port), timeout=3)
        connection.sendall(("POST %s HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
                            "Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (path, len(data))).encode() + data)
        try:
            self.assertTrue(engine.entered.wait(2), "backend never entered prefill")
        finally:
            if proxy:
                reset_connection(connection)
            else:
                connection.close()
        try:
            self.assertTrue(engine.cancelled.wait(1), "reset client left prefill running")
            self.assertTrue(engine.finished.wait(1))
            # The cancelled request must release admission for the next request.
            next_body = {"model": "test-model", "stream": False}
            next_body.update({"input": "next"} if proxy else {"messages": [{"role": "user", "content": "next"}]})
            req = Request(endpoint.geturl() + path, data=json.dumps(next_body).encode(), headers={
                "Content-Type": "application/json", "Authorization": "Bearer secret"})
            with urlopen(req, timeout=3) as response:
                self.assertEqual(response.status, 200)
            self.assert_watchers_finished()
        finally:
            engine.release.set()

    def half_close(self, streaming):
        engine, endpoint, path = self.harness(True)
        data = json.dumps({"input": "hello", "model": "test-model", "stream": streaming}).encode()
        connection = socket.create_connection((endpoint.hostname, endpoint.port), timeout=3)
        try:
            connection.sendall(("POST %s HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
                                "Content-Type: application/json\r\nContent-Length: %d\r\n\r\n" % (path, len(data))).encode() + data)
            self.assertTrue(engine.entered.wait(2))
            connection.shutdown(socket.SHUT_WR)
            self.assertFalse(engine.cancelled.wait(0.2), "write-half-close abandoned a live response reader")
            engine.release.set()
            response = http.client.HTTPResponse(connection)
            try:
                response.begin()
                self.assertEqual(response.status, 200)
                body = response.read()
                if streaming:
                    self.assertIn(b"response.completed", body)
                else:
                    self.assertEqual(json.loads(body)["status"], "completed")
            finally:
                response.close()
            self.assertTrue(engine.finished.wait(1))
            req = Request(endpoint.geturl() + path, data=json.dumps({"input": "next", "model": "test-model"}).encode(), headers={
                "Content-Type": "application/json", "Authorization": "Bearer secret"})
            with urlopen(req, timeout=3) as next_response:
                self.assertEqual(json.load(next_response)["status"], "completed")
            self.assert_watchers_finished()
        finally:
            engine.release.set()
            connection.close()

    def test_streaming_half_close_keeps_response_reader_and_next_request_alive(self):
        self.half_close(True)

    def test_nonstreaming_half_close_keeps_response_reader_and_next_request_alive(self):
        self.half_close(False)

    def assert_watchers_finished(self):
        for thread in threading.enumerate():
            if thread.name == "responses-disconnect":
                thread.join(timeout=1)
                self.assertFalse(thread.is_alive(), "request monitor outlived its handler")

    def test_normal_completion_cleans_up_request_monitors(self):
        _, endpoint, path = self.harness(True, ScriptedEngine(chunks=("answer",)))
        req = Request(endpoint.geturl() + path, data=json.dumps({"input": "hello", "model": "test-model"}).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer secret"})
        responses = []
        getresponse = shim.http.client.HTTPConnection.getresponse

        def capture(connection):
            response = getresponse(connection)
            responses.append(response)  # Retain references: cleanup cannot rely on GC.
            return response

        with patch.object(shim.http.client.HTTPConnection, "getresponse", capture):
            with urlopen(req, timeout=3) as response:
                self.assertEqual(json.load(response)["status"], "completed")
            self.assert_watchers_finished()
            self.assertEqual(len(responses), 2)
            self.assertTrue(all(response.isclosed() for response in responses))

    def test_upstream_validation_error_cleans_up_request_monitor(self):
        engine, endpoint, path = self.harness(True, ScriptedEngine())
        req = Request(endpoint.geturl() + path, data=json.dumps({"input": "hello", "model": "unknown"}).encode(), headers={
            "Content-Type": "application/json", "Authorization": "Bearer secret"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 404)
        self.assertEqual(engine.calls, [])
        self.assert_watchers_finished()

    def test_pipelined_bytes_are_not_mistaken_for_a_disconnect(self):
        engine, endpoint, path = self.harness(True)
        data = json.dumps({"input": "hello", "model": "test-model"}).encode()
        connection = socket.create_connection((endpoint.hostname, endpoint.port), timeout=3)
        connection.sendall(("POST %s HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\n"
                            "Content-Length: %d\r\n\r\n" % (path, len(data))).encode() + data)
        try:
            self.assertTrue(engine.entered.wait(2))
            connection.sendall(b"GET /health HTTP/1.1\r\nX-Padding: " + b"x" * 20000 + b"\r\n\r\n")
            self.assertFalse(engine.cancelled.wait(0.2))
        finally:
            engine.release.set()
            connection.close()

    def response_owned_socket(self, status=200, chunked=False, abort_upstream=False):
        gone, release, started = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)

        class Backend(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1" if chunked else "HTTP/1.0"

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                chunk = {"choices": [{"delta": {"content": "hello"}}]}
                data = ("data: " + json.dumps(chunk) + "\n\n").encode()
                self.wfile.write((b"%x\r\n" % len(data) + data + b"\r\n") if chunked else data)
                self.wfile.flush()
                started.set()
                if abort_upstream:
                    self.close_connection = True
                    return
                while not release.wait(0.01):
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        gone.set()
                        return

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        self.addCleanup(thread.join, timeout=2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        endpoint = self.proxy(urlsplit("http://127.0.0.1:%d" % server.server_port))
        data = json.dumps({"input": "hello", "model": "test-model", "stream": True}).encode()
        connection = socket.create_connection((endpoint.hostname, endpoint.port), timeout=3)
        connection.sendall(("POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: %d\r\n\r\n" % len(data)).encode() + data)
        try:
            self.assertTrue(started.wait(2))
            if status == 200:
                received = b""
                while b"response.output_text.delta" not in received:
                    chunk = connection.recv(8192)
                    self.assertTrue(chunk)
                    received += chunk
            if abort_upstream:
                while True:
                    chunk = connection.recv(8192)
                    if not chunk:
                        break
                    received += chunk
                self.assertTrue(self.proxy_finished.wait(1))
                # the cut upstream reaches the live client as a failed response, not a handler error
                self.assertEqual(self.proxy_errors, [])
                self.assertIn(b"response.failed", received)
                self.assertIn(b"server_error", received)
                self.assert_watchers_finished()
                return
        finally:
            reset_connection(connection)
        try:
            self.assertTrue(gone.wait(1), "response-owned upstream socket stayed open")
            self.assert_watchers_finished()
            self.assertTrue(self.proxy_finished.wait(1))
            self.assertEqual(self.proxy_errors, [])
        finally:
            release.set()

    def test_disconnect_closes_response_owned_connection_close_socket(self):
        self.response_owned_socket()

    def test_disconnect_during_chunked_body_does_not_raise_a_handler_error(self):
        self.response_owned_socket(chunked=True)

    def test_live_client_still_observes_an_upstream_chunk_framing_failure(self):
        self.response_owned_socket(chunked=True, abort_upstream=True)

    def test_disconnect_closes_an_upstream_error_held_body(self):
        self.response_owned_socket(status=503)

    def test_direct_gateway_prefill_cancellation_control(self):
        self.disconnect(False, True)

    def test_streaming_responses_reset_cancels_prefill(self):
        self.disconnect(True, True)

    def test_nonstreaming_responses_reset_cancels_prefill(self):
        self.disconnect(True, False)


if __name__ == "__main__":
    unittest.main()
