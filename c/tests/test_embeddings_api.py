"""Permanent HTTP contract tests for the embedding-only gateway surface."""

import http.client
import json
import pathlib
import sys
import threading
import unittest

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))

from openai_server import APIServer  # noqa: E402


class FakeEmbeddingEngine:
    embeds = True
    chats = False

    def __init__(self):
        self.calls = []

    def embed(self, texts, cache_slot=0, cancelled=None):
        self.calls.append((list(texts), cache_slot))
        return {
            "embedding": [[float(i), float(i + 1)] for i in range(len(texts))],
            "input_tokens": sum(len(text.split()) for text in texts),
        }


class EmbeddingsHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEmbeddingEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine,
                                "nomic-embed-text-v1.5", "secret", 1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.thread.join, 2)

    def post(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        self.addCleanup(connection.close)
        connection.request("POST", "/v1/embeddings", json.dumps(body),
                           {"Authorization": "Bearer secret",
                            "Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())

    def test_single_input_uses_openai_shape_and_usage(self):
        status, body = self.post({"model": "nomic-embed-text-v1.5",
                                  "input": "hello world"})
        self.assertEqual(status, 200)
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "nomic-embed-text-v1.5")
        self.assertEqual(body["data"], [{"object": "embedding", "index": 0,
                                         "embedding": [0.0, 1.0]}])
        self.assertEqual(body["usage"], {"prompt_tokens": 2, "total_tokens": 2})
        self.assertEqual(self.engine.calls, [(["hello world"], 0)])

    def test_batch_input_preserves_order(self):
        status, body = self.post({"model": "nomic-embed-text-v1.5",
                                  "input": ["one", "two words"]})
        self.assertEqual(status, 200)
        self.assertEqual([item["index"] for item in body["data"]], [0, 1])
        self.assertEqual([item["embedding"] for item in body["data"]],
                         [[0.0, 1.0], [1.0, 2.0]])
        self.assertEqual(body["usage"]["prompt_tokens"], 3)

    def test_empty_batch_is_rejected(self):
        status, body = self.post({"model": "nomic-embed-text-v1.5", "input": []})
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["param"], "input")


if __name__ == "__main__":
    unittest.main()
