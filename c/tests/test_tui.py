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


    def test_dropped_files_go_with_the_message(self):
        # a terminal drops files as their paths, pasted: a text file read directly, a Word file through
        # the chat page's reader (COLI_DOCS), a picture as an image_url; one more path typed by hand
        import base64
        import sys
        import zipfile
        FakeServer.seen = []
        with tempfile.TemporaryDirectory() as d:
            note = os.path.join(d, "note.txt")
            with open(note, "w") as f:
                f.write("hello from the file")
            pic = os.path.join(d, "my pic.png")
            png = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
            with open(pic, "wb") as f:
                f.write(png)
            word = os.path.join(d, "report.docx")
            with zipfile.ZipFile(word, "w") as z:
                z.writestr("word/document.xml",
                           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                           "<w:body><w:p><w:r><w:t>Words of the report</w:t></w:r></w:p></w:body></w:document>")
            later = os.path.join(d, "later.md")
            with open(later, "w") as f:
                f.write("# typed later")
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
            url = "http://127.0.0.1:%d" % httpd.server_address[1]
            docs = os.path.join(os.path.dirname(C_DIR), "ui", "docs.py")
            s = Screen(self.exe, ["--server", url, "--lang", "en", "--no-sixel"],
                       {"COLI_DOCS": docs, "COLI_PYTHON": sys.executable})
            threading.Thread(target=httpd.serve_forever, daemon=True).start()
            try:
                self.assertTrue(s.wait_for("tiny-model"), s.out[-400:])
                s.type("\x1b[200~%s '%s' %s\x1b[201~" % (note, pic, word.replace(" ", "\\ ")))
                self.assertTrue(s.wait_for("Attached: note.txt"), s.out[-600:])
                s.type("what are these? " + later)
                s.type("\r")
                self.assertTrue(s.wait_for("fake server"), s.out[-600:])
                posts = [x for x in FakeServer.seen if x[0] == "POST"]
                self.assertEqual(len(posts), 1)
                content = json.loads(posts[0][3])["messages"][0]["content"]
                self.assertEqual(content[0]["type"], "text")
                text = content[0]["text"]
                for part in ('Attached document "note.txt"', "hello from the file", 'Attached document "report.docx"',
                             "Words of the report", 'Attached document "later.md"', "# typed later"):
                    self.assertIn(part, text)
                self.assertTrue(text.endswith("---\n\nwhat are these?"), text[-80:])
                self.assertNotIn(d, text)                       # the paths left the message
                self.assertEqual(len(content), 2)
                self.assertEqual(content[1]["type"], "image_url")
                head, data = content[1]["image_url"]["url"].split(",", 1)
                self.assertEqual(head, "data:image/png;base64")
                self.assertEqual(base64.b64decode(data), png)
                s.type("\x11")
                self.assertEqual(s.close(), 0)
            finally:
                httpd.shutdown()
                httpd.server_close()

    def test_a_drag_copies_the_text(self):
        # the terminal's selection does not reach a program that reads the mouse: a drag selects here,
        # the release copies (OSC 52 on a terminal without a desktop's clipboard tool)
        import base64
        s = Screen(self.exe, ["--demo", "--lang", "en", "--no-sixel"], {"DISPLAY": "", "WAYLAND_DISPLAY": ""})
        self.assertTrue(s.wait_for("Ask anything"), s.out[-400:])
        s.type("\x1b[<0;1;1M\x1b[<32;60;3M\x1b[<32;140;40M\x1b[<0;140;40m")
        self.assertTrue(s.wait_for("characters copied"), s.out[-600:])
        start = s.out.index(b"\x1b]52;c;") + 7
        copied = base64.b64decode(s.out[start:s.out.index(b"\x07", start)]).decode()
        self.assertIn("colibri", copied)
        self.assertIn("Ask anything", copied)
        self.assertFalse(any(line != line.rstrip() for line in copied.split("\n")))   # no trailing blanks
        s.type("\x11")
        self.assertEqual(s.close(), 0)


LONG_CODE = ("def greet(name):\n    message = 'hello ' + name + ', this line is long on purpose so that the screen wraps it "
             "at the edge of the window, and the copy has to give it back whole'\n    return message")
LONG_ANSWER = ("Here is the code:\n\n```python\n" + LONG_CODE + "\n```\n\n" +
               "\n".join("Line %d of the explanation, past the bottom of the screen." % i for i in range(1, 61)) +
               "\n\n```sh\necho done\n```\nThe end.")


def tiny_png(w=8, h=6):
    import struct
    import zlib
    raw = b"".join(b"\x00" + bytes(v for x in range(w) for v in (x * 30, y * 40, 200)) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


class LongAnswerServer(FakeServer):
    """A chat model whose answer is longer than the screen and holds two code blocks."""
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
        self.seen.append(("POST", self.path, self.headers.get("Authorization"), body))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for i in range(0, len(LONG_ANSWER), 50):
            d = {"content": LONG_ANSWER[i:i + 50]}
            self.wfile.write(b"data: " + json.dumps({"choices": [{"index": 0, "delta": d}]}).encode() + b"\n\n")
        self.wfile.write(b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n')
        self.close_connection = True


class ImageServer(FakeServer):
    """An image model (Qwen-Image): /v1/models says image_generation, chat is refused as coli serve refuses it."""
    def do_GET(self):
        if self.path == "/v1/models":
            self.seen.append(("GET", self.path, None, None))
            self.reply({"object": "list", "data": [{"id": "qwen-image", "capabilities": ["image_generation"]}]})
        else:
            super().do_GET()

    def do_POST(self):
        import base64
        body = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode()
        self.seen.append(("POST", self.path, self.headers.get("Authorization"), body))
        if self.path != "/v1/images/generations":
            data = json.dumps({"error": {"message": "`qwen-image` is an image generation model and does not chat"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.reply({"created": 1, "data": [{"b64_json": base64.b64encode(tiny_png()).decode(), "revised_prompt": None}],
                    "colibri": {"width": 8, "height": 6, "seed": 77, "steps": 4, "timings": {}}})


def osc52(s, seconds=5):
    """the text of the next OSC 52 the TUI writes (the terminal's clipboard), from s.out as it is now. The
    screen itself is not read for this: each frame writes only the cells that changed, so a word on it
    can arrive cut by cursor moves."""
    import base64
    end = time.time() + seconds
    while time.time() < end and not (b"\x1b]52;c;" in s.out and b"\x07" in s.out[s.out.find(b"\x1b]52;c;"):]):
        s.pump(0.1)
    start = s.out.index(b"\x1b]52;c;") + 7
    return base64.b64decode(s.out[start:s.out.index(b"\x07", start)]).decode()


def wait_until(cond, s, seconds=10):
    end = time.time() + seconds
    while time.time() < end and not cond():
        s.pump(0.1)
    return cond()


@unittest.skipUnless(CC and not WINDOWS, "needs a C compiler and a POSIX pseudo-terminal")
class TuiCopyAndPictureTest(unittest.TestCase):
    """A drag copies the cells on the screen: an answer longer than the screen comes out cut, a code line
    the screen wrapped comes out as two. ctrl+y and ctrl+k (and each answer's and code block's copy
    button) copy what the model wrote. An image model's turn goes to /v1/images/generations and the
    picture is drawn in half blocks and saved."""
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.exe, r = build(cls.dir.name)
        if r.returncode:
            raise AssertionError(r.stderr)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def serve(self, handler, env=None):
        handler.seen = []
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        url = "http://127.0.0.1:%d" % httpd.server_address[1]
        s = Screen(self.exe, ["--server", url, "--lang", "en", "--no-sixel"],
                   {"DISPLAY": "", "WAYLAND_DISPLAY": "", **(env or {})})
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        return s, httpd

    def test_the_whole_answer_and_its_last_code_block(self):
        s, httpd = self.serve(LongAnswerServer)
        try:
            self.assertTrue(s.wait_for("tiny-model"), s.out[-400:])
            s.type("write a function\r")
            self.assertTrue(wait_until(lambda: any(x[0] == "POST" for x in LongAnswerServer.seen), s))
            s.pump(2.0)                                     # the stream's end
            s.out = b""
            s.type("\x19")                                  # ctrl+y: the last answer
            self.assertEqual(osc52(s), LONG_ANSWER)
            s.out = b""
            s.type("\x0b")                                  # ctrl+k: its last code block
            self.assertEqual(osc52(s), "echo done")
            s.type("\x11")
            self.assertEqual(s.close(), 0)
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_an_image_model_draws_and_saves_the_picture(self):
        with tempfile.TemporaryDirectory() as home:
            s, httpd = self.serve(ImageServer, {"HOME": home})
            try:
                self.assertTrue(s.wait_for("qwen-image"), s.out[-400:])
                s.type("a red hummingbird\r")
                pics = os.path.join(home, "colibri-images")
                self.assertTrue(wait_until(lambda: os.path.isdir(pics) and os.listdir(pics), s), s.out[-600:])
                s.pump(1.0)
                posts = [x for x in ImageServer.seen if x[0] == "POST"]
                self.assertEqual([p[1] for p in posts], ["/v1/images/generations"])
                req = json.loads(posts[0][3])
                self.assertEqual((req["model"], req["prompt"], req["response_format"], req["n"]),
                                 ("qwen-image", "a red hummingbird", "b64_json", 1))
                self.assertIn("▀".encode(), s.out)     # the picture, in half blocks
                saved = os.listdir(os.path.join(home, "colibri-images"))
                self.assertEqual(len(saved), 1)
                path = os.path.join(home, "colibri-images", saved[0])
                with open(path, "rb") as f:
                    self.assertEqual(f.read(), tiny_png())
                s.out = b""
                s.type("\x19")                              # ctrl+y on a picture: its file
                self.assertEqual(osc52(s), path)
                s.type("\x11")
                self.assertEqual(s.close(), 0)
            finally:
                httpd.shutdown()
                httpd.server_close()

if __name__ == "__main__":
    unittest.main()
