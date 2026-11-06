"""Exercise real HTTP/SSE translation when the backend fails or closes early."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

spec = importlib.util.spec_from_file_location("responses_shim", Path(__file__).resolve().parents[2] / "ui/responses_shim.py")
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)


def frames(*chunks, done=False):
    data = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks)
    return data + (b"data: [DONE]\n\n" if done else b"")


TEXT = {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
FINISH = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class ResponsesUpstreamFailureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Backend(BaseHTTPRequestHandler):
            body = b""

            def do_POST(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(type(self).body)

            def log_message(self, *args):
                pass

        cls.backend_handler = Backend
        cls.backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        backend = urlsplit("http://127.0.0.1:%d" % cls.backend.server_port)

        class Proxy(BaseHTTPRequestHandler):
            def do_POST(self):
                shim.serve(self, backend)

            def log_message(self, *args):
                pass

        cls.proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
        cls.threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (cls.backend, cls.proxy)]
        for thread in cls.threads:
            thread.start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.proxy, cls.backend):
            server.shutdown()
            server.server_close()
        for thread in cls.threads:
            thread.join(timeout=2)

    def request(self, data, stream=True):
        self.backend_handler.body = data
        req = Request("http://127.0.0.1:%d/v1/responses" % self.proxy.server_port,
                      data=json.dumps({"input": "hello", "stream": stream}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=5) as response:
            body = response.read().decode()
        if not stream:
            return json.loads(body)
        return [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]

    def test_upstream_error_is_a_failed_response_with_partial_text(self):
        events = self.request(frames(TEXT, {"error": {"message": "engine unavailable"}}, done=True))
        last = events[-1]
        self.assertEqual(last["type"], "response.failed")
        self.assertIn("engine unavailable", last["response"]["error"]["message"])
        item = last["response"]["output"][0]
        self.assertEqual(item["content"][0]["text"], "partial")
        self.assertEqual(item["status"], "incomplete")
        self.assertNotIn("response.completed", [e["type"] for e in events])

    def test_eof_without_done_is_not_completion(self):
        events = self.request(frames(TEXT, FINISH))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_done_without_finish_reason_is_not_completion(self):
        events = self.request(frames(TEXT, done=True))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_malformed_frame_is_not_silently_skipped(self):
        malformed = (b"data: {bad json}\n\n", frames({"choices": {"index": 0}}),
                     frames({"choices": [{"index": 0, "delta": ["bad"]}]}))
        for data in malformed:
            with self.subTest(data=data):
                events = self.request(frames(TEXT) + data + frames(FINISH, done=True))
                self.assertEqual(events[-1]["type"], "response.failed")

    def test_unknown_finish_reason_is_not_success(self):
        events = self.request(frames(TEXT, {"choices": [{"delta": {}, "finish_reason": "error"}]}, done=True))
        self.assertEqual(events[-1]["type"], "response.failed")

    def test_nonstream_reply_also_reports_failure(self):
        response = self.request(frames({"error": {"message": "engine unavailable"}}), stream=False)
        self.assertEqual(response["status"], "failed")
        self.assertIn("engine unavailable", response["error"]["message"])

    def test_malformed_usage_reports_failure_instead_of_crashing_finalization(self):
        for usage in ({"prompt_tokens": "bad"}, {"completion_tokens": float("inf")},
                      {"completion_tokens": -1}, {"prompt_tokens": True}, [1]):
            with self.subTest(usage=usage):
                data = frames(TEXT, FINISH, {"choices": [], "usage": usage}, done=True)
                events = self.request(data)
                self.assertEqual(events[-1]["type"], "response.failed")
                self.assertEqual(self.request(data, stream=False)["status"], "failed")

    def test_normal_completion_and_multiline_sse_still_work(self):
        # SSE joins data lines before JSON decoding. A legal multiline JSON object
        # must not be mistaken for two malformed chunks.
        data = b'data: {"choices":\ndata: [{"index":0,"delta":{"content":"ok"}}]}\n\n'
        events = self.request(data + frames(FINISH, done=True))
        self.assertEqual(events[-1]["type"], "response.completed")
        self.assertEqual(events[-1]["response"]["output"][0]["content"][0]["text"], "ok")


if __name__ == "__main__":
    unittest.main()
