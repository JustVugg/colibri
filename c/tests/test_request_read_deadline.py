"""A continuously arriving body or header cannot renew the read deadline."""
import socket
import threading
import time
import unittest
from unittest.mock import patch

from openai_server import APIHandler, APIServer, _DeadlineReader
from test_openai_server import ScriptedEngine, _error_body, _post_completions, _spawn_test_server


class RequestReadDeadlineTest(unittest.TestCase):
    def stream(self):
        client, server = socket.socketpair()
        raw = server.makefile("rb")
        self.addCleanup(client.close)
        self.addCleanup(server.close)
        self.addCleanup(raw.close)
        return client, _DeadlineReader(raw, server, 1, 0.25)

    def assert_drip_expires(self, method):
        client, reader = self.stream()
        stopped = threading.Event()

        def drip():
            for _ in range(30):
                if stopped.wait(0.04):
                    break
                try:
                    client.sendall(b"x")
                except OSError:
                    break
            try:
                client.shutdown(socket.SHUT_WR)
            except OSError:
                pass

        writer = threading.Thread(target=drip)
        writer.start()
        try:
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                getattr(reader, method)(30)
            self.assertLess(time.monotonic() - started, 0.8)
        finally:
            stopped.set()
            writer.join(timeout=2)

    def test_body_read_deadline_expires_during_one_buffered_read(self):
        self.assert_drip_expires("read")

    def test_header_deadline_expires_during_one_buffered_readline(self):
        self.assert_drip_expires("readline")

    def test_readline_preserves_following_pipelined_bytes_and_limits(self):
        client, reader = self.stream()
        client.sendall(b"first\nsecond\nbody")
        client.shutdown(socket.SHUT_WR)
        self.assertEqual(reader.readline(3), b"fir")
        self.assertEqual(reader.readline(), b"st\n")
        self.assertEqual(reader.readline(), b"second\n")
        self.assertEqual(reader.read(4), b"body")
        self.assertEqual(reader.read(1), b"")

    def test_zero_length_reads_do_not_consume_input(self):
        client, reader = self.stream()
        client.sendall(b"ok\n")
        self.assertEqual(reader.read(0), b"")
        self.assertEqual(reader.readline(0), b"")
        self.assertEqual(reader.readline(), b"ok\n")

    def test_slow_http_body_closes_without_misreporting_an_engine_error(self):
        engine = ScriptedEngine()
        with patch.object(APIHandler, "READ_DEADLINE", 0.25):
            server = APIServer(("127.0.0.1", 0), engine, "test-model", None, 16)
            serving = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
            serving.start()
            stopped = threading.Event()
            try:
                with socket.create_connection(server.server_address, timeout=2) as client:
                    client.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: localhost\r\n"
                                   b"Content-Length: 30\r\n\r\n")

                    def drip():
                        while not stopped.wait(0.04):
                            try:
                                client.sendall(b"x")
                            except OSError:
                                break

                    writer = threading.Thread(target=drip)
                    writer.start()
                    try:
                        started = time.monotonic()
                        try:
                            data = client.recv(65536)
                        except ConnectionResetError:
                            # the server closed with a dripped byte still unread: the
                            # kernel resets instead of a clean close (seen on macOS);
                            # either way nothing was answered
                            data = b""
                        self.assertEqual(data, b"")
                        self.assertLess(time.monotonic() - started, 0.8)
                    finally:
                        stopped.set()
                        writer.join(timeout=2)
                self.assertEqual(engine.calls, [])
            finally:
                server.scheduler.close()
                server.shutdown()
                server.server_close()
                serving.join(timeout=2)

    def test_engine_timeout_keeps_the_existing_structured_error(self):
        class TimedOutEngine(ScriptedEngine):
            def generate(self, *args, **kwargs):
                raise TimeoutError("engine operation timed out")

        base = _spawn_test_server(self, TimedOutEngine())
        status, error = _error_body(self, lambda: _post_completions(
            base, {"model": "test-model", "prompt": "hello"}))
        self.assertEqual(status, 500)
        self.assertEqual(error["code"], "engine_error")


if __name__ == "__main__":
    unittest.main()
