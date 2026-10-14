"""The gateway must not execute a request with an ambiguous HTTP body boundary."""
import json
import socket
import threading
import unittest

from openai_server import APIServer
from test_openai_server import ScriptedEngine


class RequestFramingTest(unittest.TestCase):
    def setUp(self):
        self.engine = ScriptedEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model", "secret", 16)
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       args=(0.01,), daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, timeout=2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)
        self.payload = json.dumps({"model": "test-model", "prompt": "hello"}).encode()

    def exchange(self, headers, *, authenticated=True, followup=False):
        auth = b"Authorization: Bearer secret\r\n" if authenticated else b""
        wire = (b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n" + auth
                + headers + b"\r\n" + self.payload)
        if followup:
            wire += b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        with socket.create_connection(self.server.server_address, timeout=2) as client:
            client.sendall(wire)
            response = bytearray()
            while True:
                part = client.recv(65536)
                if not part:
                    return bytes(response)
                response.extend(part)

    def test_transfer_encoding_is_refused_before_generation(self):
        response = self.exchange(
            f"Content-Length: {len(self.payload)}\r\n".encode()
            + b"Transfer-Encoding: chunked\r\n", followup=True)
        self.assertTrue(response.startswith(b"HTTP/1.1 400"), response)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)
        self.assertEqual(self.engine.calls, [])

    def test_identical_duplicate_lengths_are_refused_consistently(self):
        response = self.exchange(
            (f"Content-Length: {len(self.payload)}\r\n" * 2).encode(), followup=True)
        self.assertTrue(response.startswith(b"HTTP/1.1 400"), response)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)

    def test_conflicting_lengths_are_refused_and_connection_closed(self):
        response = self.exchange(
            f"Content-Length: {len(self.payload)}\r\nContent-Length: 1\r\n".encode(),
            followup=True)
        self.assertTrue(response.startswith(b"HTTP/1.1 400"), response)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)
        self.assertEqual(self.engine.calls, [])

    def test_early_auth_error_does_not_reuse_conflicting_framing(self):
        response = self.exchange(
            f"Content-Length: {len(self.payload)}\r\nContent-Length: 1\r\n".encode(),
            authenticated=False, followup=True)
        self.assertTrue(response.startswith(b"HTTP/1.1 401"), response)
        self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)

    def test_normal_body_still_generates(self):
        response = self.exchange(
            f"Content-Length: {len(self.payload)}\r\nConnection: close\r\n".encode())
        self.assertTrue(response.startswith(b"HTTP/1.1 200"), response)
        self.assertEqual(len(self.engine.calls), 1)

    def test_non_decimal_lengths_are_refused(self):
        length = str(len(self.payload))
        for value in ("+" + length, "_".join(length)):
            with self.subTest(value=value):
                response = self.exchange(f"Content-Length: {value}\r\n".encode(), followup=True)
                self.assertTrue(response.startswith(b"HTTP/1.1 400"), response)
                self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)
        self.assertEqual(self.engine.calls, [])

    def test_early_auth_error_closes_non_decimal_lengths(self):
        length = str(len(self.payload))
        for value in ("+" + length, "_".join(length)):
            with self.subTest(value=value):
                response = self.exchange(f"Content-Length: {value}\r\n".encode(),
                                         authenticated=False, followup=True)
                self.assertTrue(response.startswith(b"HTTP/1.1 401"), response)
                self.assertEqual(response.count(b"HTTP/1.1 "), 1, response)

    def test_decimal_leading_zeroes_still_generate(self):
        response = self.exchange(
            f"Content-Length: 000{len(self.payload)}\r\nConnection: close\r\n".encode())
        self.assertTrue(response.startswith(b"HTTP/1.1 200"), response)
        self.assertEqual(len(self.engine.calls), 1)


if __name__ == "__main__":
    unittest.main()
