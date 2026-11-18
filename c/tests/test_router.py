"""Static router dispatch and public-surface tests."""

import http.client
import json
import pathlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from router import Router, RouterHandler  # noqa: E402


class BackendHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path == "/health":
            self.reply(200, {"status": "ok"})
        else:
            self.reply(404, {})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if self.path == "/v1/embeddings":
            count = len(body["input"]) if isinstance(body["input"], list) else 1
            self.reply(200, {"object": "list", "data": [
                {"object": "embedding", "index": i, "embedding": [float(i)]}
                for i in range(count)
            ], "model": body["model"], "usage": {"prompt_tokens": 1}})
        else:
            self.reply(200, {"backend": "chat", "model": body["model"]})

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class RouterTest(unittest.TestCase):
    def setUp(self):
        self.backend = ThreadingHTTPServer(("127.0.0.1", 0), BackendHandler)
        self.backend_thread = threading.Thread(target=self.backend.serve_forever, daemon=True)
        self.backend_thread.start()
        config = {"backends": [{
            "url": f"http://127.0.0.1:{self.backend.server_port}",
            "models": ["nomic-embed-text-v1.5"],
            "capabilities": ["embeddings"],
        }]}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), RouterHandler)
        self.server.router = Router(config)
        self.server.api_key = "front-secret"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.backend.shutdown)
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend_thread.join, 2)

    def request(self, method, path, body=None, auth="front-secret"):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        headers = {"Authorization": "Bearer " + auth} if auth else {}
        if body is not None:
            headers["Content-Type"] = "application/json"
            body = json.dumps(body)
        connection.request(method, path, body, headers)
        response = connection.getresponse()
        result = json.loads(response.read())
        connection.close()
        return response.status, result

    def test_embeddings_dispatch_to_configured_backend(self):
        status, body = self.request("POST", "/v1/embeddings", {
            "model": "nomic-embed-text-v1.5", "input": ["a", "b"]})
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][1]["embedding"], [1.0])

    def test_models_and_health_are_aggregated(self):
        status, models = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(models["data"][0]["id"], "nomic-embed-text-v1.5")
        status, health = self.request("GET", "/health", auth=None)
        self.assertEqual(status, 200)
        self.assertEqual(health["status"], "ok")

    def test_unknown_model_is_not_forwarded(self):
        status, body = self.request("POST", "/v1/embeddings", {
            "model": "missing", "input": "a"})
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "model_not_found")

    def test_duplicate_model_capability_is_rejected(self):
        with self.assertRaises(ValueError):
            Router({"backends": [
                {"url": "http://a", "models": ["m"], "capabilities": ["chat"]},
                {"url": "http://b", "models": ["m"], "capabilities": ["chat"]},
            ]})


if __name__ == "__main__":
    unittest.main()
