"""Completed MCP calls are released while EOF still waits for active workers."""
import io
import json
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

C_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(C_DIR))
import mcp_server


def call(number):
    return {"jsonrpc": "2.0", "id": number, "method": "tools/call",
            "params": {"name": "status", "arguments": {}}}


class ThreadLifecycle(unittest.TestCase):
    def test_finished_calls_do_not_accumulate(self):
        output = io.StringIO()
        server = mcp_server.Server(out=output)
        with mock.patch.dict(mcp_server.HANDLERS, {"status": lambda args, notify: {"ok": True}}):
            for number in range(50):
                server.handle(call(number))
                for worker in server.threads:
                    worker.join(timeout=2)
                    self.assertFalse(worker.is_alive())
            self.assertLessEqual(len(server.threads), 1)
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual({reply['id'] for reply in replies}, set(range(50)))
        self.assertTrue(all(reply['result']['isError'] is False for reply in replies))

    def test_eof_preserves_and_waits_for_active_calls(self):
        output = io.StringIO()
        server = mcp_server.Server(out=output)
        entered = threading.Event()
        release = threading.Event()

        def handler(args, notify):
            entered.set()
            self.assertTrue(release.wait(timeout=3))
            return {"ok": True}

        with mock.patch.dict(mcp_server.HANDLERS, {"status": handler}):
            server.handle(call(1))
            active = server.threads[-1]
            self.assertTrue(entered.wait(timeout=2))
            with mock.patch.dict(mcp_server.HANDLERS, {"status": lambda args, notify: {"ok": True}}):
                server.handle(call(2))
                server.threads[-1].join(timeout=2)
            self.assertIn(active, server.threads)
            draining = threading.Thread(target=server.serve, args=(io.StringIO(""),))
            draining.start()
            try:
                draining.join(timeout=.1)
                self.assertTrue(draining.is_alive())
            finally:
                release.set()
                draining.join(timeout=3)
                active.join(timeout=3)
            self.assertFalse(draining.is_alive())
            self.assertFalse(active.is_alive())
        replies = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual({reply['id'] for reply in replies}, {1, 2})
        self.assertTrue(all(reply['result']['isError'] is False for reply in replies))


if __name__ == "__main__":
    unittest.main()
