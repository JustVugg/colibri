"""POSTs from a browser origin the gateway does not know are refused (CSRF)."""
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import APIServer
from test_openai_server import ScriptedEngine


class OriginGuardTest(unittest.TestCase):
    def setUp(self):
        # no API key: the case where a foreign page could otherwise drive the engine
        self.server = APIServer(("127.0.0.1", 0), ScriptedEngine(), "test-model", None, 16,
                                cors_origins=("http://localhost:5173",))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       args=(0.01,), daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, timeout=2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)
        self.host = f"127.0.0.1:{self.server.server_port}"

    def post(self, origin=None, content_type="text/plain"):
        """A body naming another model: past the guards it ends in 404, never in the engine."""
        headers = {"Content-Type": content_type}
        if origin is not None:
            headers["Origin"] = origin
        body = json.dumps({"model": "other", "messages": [{"role": "user", "content": "hi"}]})
        request = Request(f"http://{self.host}/v1/chat/completions", data=body.encode(),
                          headers=headers, method="POST")
        try:
            with urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def test_foreign_origin_is_refused(self):
        status, body = self.post("https://evil.invalid")
        self.assertEqual(status, 403)
        self.assertEqual(body["error"]["code"], "origin_not_allowed")

    def test_null_origin_is_refused(self):
        self.assertEqual(self.post("null")[0], 403)   # sandboxed iframe, file:// page

    def test_no_origin_passes(self):
        self.assertEqual(self.post(None)[0], 404)

    def test_listed_origin_passes(self):
        self.assertEqual(self.post("http://localhost:5173", "application/json")[0], 404)

    def test_own_origin_passes(self):
        self.assertEqual(self.post(f"http://{self.host}")[0], 404)

    def test_wildcard_keeps_the_old_behaviour(self):
        self.server.cors_origins = ("*",)
        self.assertEqual(self.post("https://evil.invalid")[0], 404)


if __name__ == "__main__":
    unittest.main()
