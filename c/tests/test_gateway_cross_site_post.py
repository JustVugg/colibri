"""The gateway does not run a POST that a page on another site sent.

A page can POST here without a preflight (a text/plain body is a "simple" request, and the
gateway reads the body as JSON whatever its content type). CORS only decides whether that page may
read the reply, so the request would still run on the engine. A request that names an Origin now
runs only for this server's own, or one allowed with --cors-origin.
"""
import http.client
import json
import threading
import unittest

from openai_server import APIServer


class FakeEngine:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0, cancelled=None,
                 grammar=None, stopped=None, on_accept=None, on_tool=None, image=None):
        self.prompts.append(prompt)
        on_text("ok")
        return {"prompt_tokens": 3, "completion_tokens": 1, "length_limited": False}


BODY = json.dumps({"model": "test-model", "max_tokens": 1,
                   "messages": [{"role": "user", "content": "hi"}]})


class CrossSitePostTest(unittest.TestCase):
    def start(self, api_key=None, cors=None):
        engine = FakeEngine()
        kwargs = {"kv_slots": 2}
        if cors is not None:
            kwargs["cors_origins"] = cors
        server = APIServer(("127.0.0.1", 0), engine, "test-model", api_key, 64, **kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop():
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
        self.addCleanup(stop)
        return server, engine

    def post(self, server, origin=None, ctype="text/plain", key=None, host=None, path="/v1/chat/completions"):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.putrequest("POST", path, skip_host=True)
            conn.putheader("Host", host or "127.0.0.1:%d" % server.server_port)
            conn.putheader("Content-Type", ctype)
            if origin is not None:
                conn.putheader("Origin", origin)
            if key:
                conn.putheader("Authorization", "Bearer " + key)
            conn.putheader("Content-Length", str(len(BODY)))
            conn.endheaders(BODY.encode())
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    def test_a_foreign_origin_does_not_reach_the_engine(self):
        server, engine = self.start()
        for ctype in ("text/plain", "application/x-www-form-urlencoded", "application/json"):
            for origin in ("https://evil.example", "http://evil.example:8000", "null"):
                with self.subTest(ctype=ctype, origin=origin):
                    status, body = self.post(server, origin, ctype)
                    self.assertEqual(status, 403)
                    self.assertIn(b"--cors-origin", body)
        self.assertEqual(engine.prompts, [])

    def test_a_foreign_origin_is_refused_with_a_key_too(self):
        server, engine = self.start(api_key="secret")
        self.assertEqual(self.post(server, "https://evil.example", key="secret")[0], 403)
        self.assertEqual(engine.prompts, [])

    def test_no_origin_is_a_script_or_an_sdk_and_runs(self):
        server, engine = self.start()
        self.assertEqual(self.post(server, None, "application/json")[0], 200)
        self.assertEqual(self.post(server, None, "text/plain")[0], 200)   # nothing here asks for a content type
        self.assertEqual(len(engine.prompts), 2)

    def test_the_origin_of_this_server_runs(self):
        server, engine = self.start()
        own = "http://127.0.0.1:%d" % server.server_port
        self.assertEqual(self.post(server, own)[0], 200)
        self.assertEqual(len(engine.prompts), 1)

    def test_an_origin_the_operator_allows_runs(self):
        server, engine = self.start(cors=("https://app.example",))
        self.assertEqual(self.post(server, "https://app.example")[0], 200)
        self.assertEqual(self.post(server, "https://evil.example")[0], 403)
        self.assertEqual(len(engine.prompts), 1)

    def test_the_default_origins_still_run(self):
        server, engine = self.start()
        for origin in ("http://localhost:5173", "http://127.0.0.1:5173", "tauri://localhost"):
            with self.subTest(origin=origin):
                self.assertEqual(self.post(server, origin)[0], 200)

    def test_a_wildcard_accepts_any_origin(self):
        server, engine = self.start(cors=("*",))
        self.assertEqual(self.post(server, "https://anywhere.example")[0], 200)

    def test_a_get_with_a_foreign_origin_is_unchanged(self):
        server, _ = self.start()
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        try:
            conn.request("GET", "/v1/models", headers={"Origin": "https://evil.example"})
            response = conn.getresponse()
            response.read()
            self.assertEqual(response.status, 200)
            self.assertIsNone(response.getheader("Access-Control-Allow-Origin"))
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
