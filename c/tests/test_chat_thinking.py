"""#979: `coli chat` must SHOW the model's thinking, and be able to hide it.

The server streams thinking as `reasoning_content` deltas; chat_attached's
loop used to drop every non-content delta (its comment listed `reasoning`
among the discards), so `coli run` showed reasoning while `coli chat`
silently hid it. These tests drive the real chat_attached under a pty
against a stdlib SSE server:

  1. default: the reasoning text is visible, inside the thinking box;
  2. COLI_SHOW_THINK=0: the reasoning is absent, the answer still arrives;
  3. both ways: the assistant history carries `content` only -- the second
     request's messages must not contain the scratchpad.
"""
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import unittest

try:
    import pty                      # POSIX-only: no termios/pty on Windows.
    HAVE_PTY = True
except ImportError:                 # The TUI path under test needs a real
    HAVE_PTY = False                # terminal; Linux and macOS runners cover it.
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

CLI = Path(__file__).resolve().parent.parent / "coli"

SEEN_MESSAGES = []

# The child is spawned with fork+EXEC, never bare fork: this test process is
# multi-threaded (the fake SSE server runs in a thread), and pty.fork() from a
# multi-threaded process inherits half-held runtime locks -- Python itself
# warns "use of forkpty() may lead to deadlocks in the child", and on the
# macOS runner it did exactly that, reliably, until the job timeout.
BOOTSTRAP = """
import importlib.machinery, importlib.util, os, sys
from types import SimpleNamespace
loader = importlib.machinery.SourceFileLoader("coli_t", os.environ["COLI_TEST_CLI"])
spec = importlib.util.spec_from_loader(loader.name, loader)
module = importlib.util.module_from_spec(spec)
sys.argv = ["coli"]
try:
    loader.exec_module(module)
except SystemExit:
    pass
module.chat_attached(SimpleNamespace(ngen=32, api_key=None),
                     os.environ["COLI_TEST_BASE"], "fake-model")
"""


class FakeSSE(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN_MESSAGES.append(body["messages"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def event(delta):
            payload = {"choices": [{"index": 0, "delta": delta,
                                    "finish_reason": None}]}
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")

        event({"role": "assistant"})
        event({"reasoning_content": "SCRATCHPAD-ALPHA "})
        event({"reasoning_content": "SCRATCHPAD-BETA"})
        event({"content": "ANSWER-GAMMA"})
        self.wfile.write(b"data: [DONE]\n\n")


def run_chat(base, extra_env):
    """Drive chat_attached through a pty slave: two prompts, then quit."""
    master, slave = pty.openpty()
    env = dict(os.environ, COLI_TEST_CLI=str(CLI), COLI_TEST_BASE=base,
               TERM="xterm", **extra_env)
    child = subprocess.Popen([sys.executable, "-c", BOOTSTRAP],
                             stdin=slave, stdout=slave, stderr=slave,
                             env=env, start_new_session=True, close_fds=True)
    os.close(slave)
    fd = master

    # Lockstep, not fire-and-forget: writing lines before the child reaches
    # input() is timing-dependent -- on macOS, libedit's initialization at the
    # first input() flushes the tty buffer and eats pre-written lines (the
    # readline activation from #928), while Linux's GNU readline preserves
    # them. Wait for each marker before typing, like a human would, with a
    # strictly-advancing cursor so an old prompt can never satisfy a new wait.
    output = b""
    pos = 0
    deadline = time.time() + 30

    def read_until(needle):
        nonlocal output, pos
        while time.time() < deadline:
            found = output.find(needle, pos)
            if found >= 0:
                pos = found + len(needle)
                return True
            try:
                chunk = os.read(fd, 65536)
            except OSError:
                return False
            if not chunk:
                return False
            output += chunk
        return False

    prompt = "\u203a".encode()           # the › of a fresh input box
    footer = "tok \u00b7".encode()       # the "~N tok ·" reply footer
    if read_until(prompt):
        os.write(fd, b"first question\n")
    read_until(footer)                    # reply finished
    if read_until(prompt):
        os.write(fd, b"second question\n")
    read_until(footer)
    if read_until(prompt):
        os.write(fd, b":q\n")
    while time.time() < deadline:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        output += chunk
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    os.close(master)
    return output.decode("utf-8", "replace")


@unittest.skipUnless(HAVE_PTY, "pty is POSIX-only; covered on Linux/macOS")
class ChatThinkingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeSSE)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        SEEN_MESSAGES.clear()

    def test_thinking_is_shown_by_default(self):
        out = run_chat(self.base, {"COLI_SHOW_THINK": "1"})
        self.assertIn("SCRATCHPAD-ALPHA", out, "reasoning deltas must render")
        self.assertIn("thinking", out, "the thinking box must announce itself")
        self.assertIn("ANSWER-GAMMA", out)

    def test_toggle_hides_thinking(self):
        out = run_chat(self.base, {"COLI_SHOW_THINK": "0"})
        self.assertNotIn("SCRATCHPAD-ALPHA", out,
                         "COLI_SHOW_THINK=0 must suppress the scratchpad")
        self.assertIn("ANSWER-GAMMA", out, "the answer must still stream")

    def test_history_carries_content_only(self):
        run_chat(self.base, {"COLI_SHOW_THINK": "1"})
        self.assertGreaterEqual(len(SEEN_MESSAGES), 2, "need the second request")
        second = SEEN_MESSAGES[1]
        assistant = [m for m in second if m["role"] == "assistant"]
        self.assertTrue(assistant)
        self.assertEqual(assistant[0]["content"], "ANSWER-GAMMA")
        self.assertNotIn("SCRATCHPAD", json.dumps(second),
                         "the scratchpad must never re-enter the context")


if __name__ == "__main__":
    unittest.main()
