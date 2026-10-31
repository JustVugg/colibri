"""The metrics responder honors the gateway's existing browser-origin policy."""
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import APIServer
from test_openai_server import ScriptedEngine


class MetricsCorsTest(unittest.TestCase):
    def setUp(self):
        self.server = APIServer(("127.0.0.1", 0), ScriptedEngine(),
                                "test-model", "fixture-secret", 16,
                                cors_origins=("http://localhost:5173",))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       args=(0.01,), daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, timeout=2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)
        self.url = f"http://127.0.0.1:{self.server.server_port}/metrics"

    def request(self, origin=None, key="fixture-secret"):
        headers = {"Authorization": f"Bearer {key}"}
        if origin is not None:
            headers["Origin"] = origin
        return urlopen(Request(self.url, headers=headers), timeout=2)

    def test_allowed_origin_receives_the_metrics_policy(self):
        origin = "http://localhost:5173"
        with self.request(origin) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], origin)
            self.assertEqual(response.headers["Vary"], "Origin")
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(response.headers["Content-Type"],
                             "text/plain; version=0.0.4; charset=utf-8")
            self.assertIn("colibri_scheduler_capacity 1\n", response.read().decode())

    def test_wildcard_origin_uses_the_configured_wildcard(self):
        self.server.cors_origins = ("*",)
        with self.request("https://configured-ui.invalid") as response:
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")
            self.assertIsNone(response.headers["Vary"])

    def test_unlisted_or_absent_origin_receives_no_permission(self):
        for origin in ("https://unlisted-ui.invalid", None):
            with self.subTest(origin=origin), self.request(origin) as response:
                self.assertEqual(response.status, 200)
                self.assertIsNone(response.headers["Access-Control-Allow-Origin"])
                self.assertIsNone(response.headers["Vary"])

    def test_origin_permission_does_not_replace_authentication(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("http://localhost:5173", key="wrong")
        with caught.exception as response:
            self.assertEqual(response.code, 401)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"],
                             "http://localhost:5173")


if __name__ == "__main__":
    unittest.main()
