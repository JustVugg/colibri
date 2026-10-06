"""Concurrent MCP calls cannot both pass the install admission check."""
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import mcp_server
import setup_flow


class InstallAdmissionTest(unittest.TestCase):
    def test_concurrent_tools_calls_spawn_only_one_install(self):
        with tempfile.TemporaryDirectory() as home, \
                patch.dict(os.environ, {"COLI_SETUP_HOME": home}):
            server = mcp_server.Server(out=io.StringIO())
            first_spawn = threading.Event()
            second_spawn = threading.Event()
            release = threading.Event()
            children = []
            lock = threading.Lock()

            def spawn(*args, **kwargs):
                with lock:
                    ordinal = len(children)
                    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"],
                                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                    children.append(child)
                if ordinal == 0:
                    first_spawn.set()
                    if not release.wait(3):
                        raise RuntimeError("test did not release install launch")
                else:
                    second_spawn.set()
                return child

            def call(msg_id):
                server.handle({"jsonrpc": "2.0", "id": msg_id, "method": "tools/call",
                               "params": {"name": "install", "arguments": {"start": False}}})

            try:
                with patch.object(setup_flow, "spawn_detached", side_effect=spawn):
                    call(1)
                    self.assertTrue(first_spawn.wait(2))
                    call(2)
                    # The first launch has not published its state yet. Without
                    # serialized admission the second call launches a second job.
                    second_spawn.wait(0.25)
                    release.set()
                    for thread in server.threads:
                        thread.join(timeout=3)
                    self.assertTrue(all(not thread.is_alive() for thread in server.threads))
                self.assertEqual(len(children), 1, "MCP started overlapping install processes")
                replies = {r["id"]: r["result"] for r in
                           map(json.loads, server.out.getvalue().splitlines())}
                self.assertFalse(replies[1]["isError"])
                self.assertTrue(replies[2]["isError"])
                self.assertIn("already running", replies[2]["content"][0]["text"])
                self.assertEqual(setup_flow.read_state()["pid"], children[0].pid)
            finally:
                release.set()
                for thread in server.threads:
                    thread.join(timeout=3)
                for child in children:
                    child.kill()
                    child.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
