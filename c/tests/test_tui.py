"""colibri-tui (tui.c), the terminal interface: it builds without a warning, its demo draws and
quits cleanly, and against a server it streams the answer, sends the conversation and the API key,
and shows /health's numbers (the GPU's among them) in its monitor."""
import json
import os
import select
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
WINDOWS = os.name == "nt"


def build(out_dir):
    exe = os.path.join(out_dir, "colibri-tui" + (".exe" if WINDOWS else ""))
    cmd = [CC, "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror", '-DCOLI_VERSION="test"',
           os.path.join(C_DIR, "tui.c"), "-o", exe, "-lm"] + (["-lws2_32"] if WINDOWS else [])
    return exe, subprocess.run(cmd, capture_output=True, text=True, timeout=300)


@unittest.skipUnless(CC, "no C compiler")
class TuiBuildTest(unittest.TestCase):
    def test_builds_without_a_warning(self):
        with tempfile.TemporaryDirectory() as d:
            exe, r = build(d)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(exe))


class Screen:
    """The TUI in a pseudo-terminal: what it wrote, the keys we type."""

    def __init__(self, exe, args, env=None):
        import pty
        import fcntl
        import struct
        import termios
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.execve(exe, [exe] + args, {**os.environ, "LANG": "C.UTF-8", **(env or {})})
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 140, 1400, 800))
        self.out = b""

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            r, _, _ = select.select([self.fd], [], [], 0.05)
            if r:
                try:
                    chunk = os.read(self.fd, 65536)
                except OSError:
                    return
                if not chunk:
                    return
                self.out += chunk

    def wait_for(self, text, seconds=10):
        end = time.time() + seconds
        while time.time() < end and text.encode() not in self.out:
            self.pump(0.1)
        return text.encode() in self.out

    def type(self, data):
        os.write(self.fd, data.encode() if isinstance(data, str) else data)

    def close(self, seconds=5):
        end = time.time() + seconds
        while time.time() < end:
            self.pump(0.1)
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                os.close(self.fd)
                return os.waitstatus_to_exitcode(status)
        os.kill(self.pid, 9)
        os.waitpid(self.pid, 0)
        os.close(self.fd)
        return None


class FakeServer(BaseHTTPRequestHandler):
    """A coli serve as far as the TUI looks: /health, /v1/models, /profile, /experts, and a chat
    completion streamed in SSE. Every request lands in `seen`."""
    seen = []

    def log_message(self, *args):
        pass

    def reply(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.seen.append(("GET", self.path, self.headers.get("Authorization"), None))
        if self.path == "/health":
            self.reply({"status": "ok", "arch": "tiny", "kv_slots": 1, "context_tokens": 4096, "live": [],
                        "scheduler": {"active": 0, "queued": 0, "capacity": 1, "max_queue": 8,
                                      "completed": 3, "failed": 0, "cancelled": 0},
                        "system": {"ram_total_gb": 8.0, "ram_avail_gb": 4.0, "load1": 0.5, "cores": 4,
                                   "engine_rss_gb": 1.5, "engine_cpu_pct": 50.0,
                                   "gpus": [{"name": "Test GPU", "util": 42.0, "mem_used_gb": 3.0,
                                             "mem_total_gb": 8.0, "temp": 50.0, "power_w": 30.0,
                                             "power_max_w": 70.0}]}})
        elif self.path == "/v1/models":
            self.reply({"object": "list", "data": [{"id": "tiny-model"}]})
        elif self.path == "/profile":
            self.reply({"seq": 1, "turns": [{"prompt_tokens": 10, "completion_tokens": 5, "forwards": 5,
                                             "wall_s": 1.0, "expert_wait_s": 0.1, "expert_matmul_s": 0.4,
                                             "attention_s": 0.2, "lm_head_s": 0.05, "expert_disk_s": 0.2}]})
        elif self.path.startswith("/experts"):
            self.reply({"rows": 2, "cols": 4, "map": "00" * 8, "hits": "ff", "live_seq": 0})
        else:
            self.send_error(404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
        self.seen.append(("POST", self.path, self.headers.get("Authorization"), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        deltas = [{"role": "assistant", "content": ""}, {"reasoning_content": "thinking it over"},
                  {"content": "\n\nHello"}, {"content": " from"}, {"content": " the fake server"}]
        for d in deltas:
            self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": d}]}).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.05)
        self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        self.close_connection = True


@unittest.skipUnless(CC and not WINDOWS, "needs a C compiler and a POSIX pseudo-terminal")
class TuiScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.exe, r = build(cls.dir.name)
        if r.returncode:
            raise AssertionError(r.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_the_demo_draws_and_quits_cleanly(self):
        s = Screen(self.exe, ["--demo", "--lang", "en", "--no-sixel"])
        self.assertTrue(s.wait_for("Ask anything"), s.out[-400:])
        self.assertIn(b"colibri", s.out)
        s.type("\x11")                                  # ctrl+q
        self.assertEqual(s.close(), 0)
        self.assertIn(b"\x1b[?1049l", s.out)            # the terminal's own screen is back

    def test_against_a_server_it_streams_sends_the_key_and_shows_health(self):
        FakeServer.seen = []
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)   # listening already: the TUI may connect
        url = "http://127.0.0.1:%d" % httpd.server_address[1]
        # the fork before the server's thread: forking a threaded process is what Python warns about
        s = Screen(self.exe, ["--server", url, "--lang", "en", "--no-sixel"], {"COLI_API_KEY": "s3cret"})
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            self.assertTrue(s.wait_for("tiny-model"), s.out[-400:])
            s.type("hi there")
            s.type("\r")
            for word in ("Hello", "from", "fake server"):
                self.assertTrue(s.wait_for(word), word)
            posts = [x for x in FakeServer.seen if x[0] == "POST"]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0][1], "/v1/chat/completions")
            self.assertEqual(posts[0][2], "Bearer s3cret")
            req = json.loads(posts[0][3])
            self.assertTrue(req["stream"])
            self.assertEqual(req["model"], "tiny-model")
            self.assertEqual(req["messages"], [{"role": "user", "content": "hi there"}])
            s.type("\x1bOQ")                            # F2: the monitor
            self.assertTrue(s.wait_for("42%"), s.out[-600:])          # the GPU's use, from /health
            self.assertTrue(s.wait_for("Test GPU"), s.out[-600:])
            self.assertTrue(any(x[1] == "/profile" for x in FakeServer.seen))
            self.assertTrue(all(x[2] == "Bearer s3cret" for x in FakeServer.seen))
            s.type("\x1b")
            s.pump(0.3)
            s.type("\x11")
            self.assertEqual(s.close(), 0)
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
