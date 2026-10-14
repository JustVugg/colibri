"""POST /v1/images/generations, end to end against the stub engine.

A real ImageEngine drives tools/qwenimage_stub.py over its serve protocol, and
a real APIServer answers HTTP on a free port: nothing between the socket and
the engine's stdin is mocked. The stub draws a synthetic picture that is a
pure function of (prompt, seed, size), which is what lets these tests pin
determinism, the size actually drawn and the PNG framing.
"""
import base64
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from qwen_image_fixture import STUB, make_fake_pipeline, stub_env  # noqa: E402

import image_engine  # noqa: E402
from openai_server import APIServer  # noqa: E402

MODEL_ID = "qwen-image-2.1-colibri"
KEY = "secret"


class GatewayHarness:
    """One stub engine behind one API server, torn down in reverse order."""

    def __init__(self, delay=0.0, max_queue=8, api_key=KEY, **stub):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = make_fake_pipeline(Path(self.tmp.name) / "qwen-image")
        self.engine = image_engine.ImageEngine(STUB, self.model, env=stub_env(delay, **stub),
                                               on_log=lambda _m: None)
        self.server = APIServer(("127.0.0.1", 0), self.engine, MODEL_ID, api_key, 16,
                                max_queue=max_queue, queue_timeout=30)
        self.thread = threading.Thread(target=self.server.serve_forever, args=(0.01,),
                                       daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def close(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.engine.close(timeout=5)
        self.tmp.cleanup()

    def request(self, path, body=None, headers=None, method=None, key=KEY):
        all_headers = {"Content-Type": "application/json"}
        if key:
            all_headers["Authorization"] = f"Bearer {key}"
        all_headers.update(headers or {})
        data = None if body is None else json.dumps(body).encode()
        req = Request(f"http://127.0.0.1:{self.port}{path}", data=data, headers=all_headers,
                      method=method)
        try:
            with urlopen(req, timeout=30) as response:
                return response.status, dict(response.headers), response.read()
        except HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def generate(self, **body):
        body.setdefault("model", MODEL_ID)
        return self.request("/v1/images/generations", body)

    def stream(self, **body):
        body.setdefault("model", MODEL_ID)
        body["stream"] = True
        status, headers, raw = self.request("/v1/images/generations", body)
        return status, headers, parse_sse(raw.decode("utf-8"))


def parse_sse(text):
    """[(event, data)] in order; `data: [DONE]` appears as (None, "[DONE]")."""
    events = []
    for block in text.split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
        if data:
            payload = "\n".join(data)
            events.append((name, payload if payload == "[DONE]" else json.loads(payload)))
    return events


def png_pixels(b64):
    return image_engine.decode_png(base64.b64decode(b64))


class ImagesEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gw = GatewayHarness()

    @classmethod
    def tearDownClass(cls):
        cls.gw.close()

    def error(self, status, raw):
        body = json.loads(raw)
        self.assertIn("error", body)
        return status, body["error"]

    def test_non_stream_returns_a_png_of_the_requested_size(self):
        status, headers, raw = self.gw.generate(prompt="a red fox", size="512x288", seed=7,
                                                steps=3)
        self.assertEqual(status, 200, raw)
        body = json.loads(raw)
        self.assertEqual(set(body), {"created", "data", "colibri"})
        self.assertIsInstance(body["created"], int)
        self.assertEqual(len(body["data"]), 1)
        self.assertIsNone(body["data"][0]["revised_prompt"])
        width, height, channels, pixels = png_pixels(body["data"][0]["b64_json"])
        self.assertEqual((width, height, channels), (512, 288, 4))
        self.assertEqual(len(pixels), 512 * 288 * 4)
        meta = body["colibri"]
        self.assertEqual((meta["width"], meta["height"], meta["seed"], meta["steps"]),
                         (512, 288, 7, 3))
        self.assertEqual(set(meta["timings"]), {"encode", "denoise", "decode"})
        self.assertIn("x-colibri-queue-wait-ms", {k.lower() for k in headers})

    def test_same_seed_same_image_other_seed_other_image(self):
        def pixels(seed):
            _s, _h, raw = self.gw.generate(prompt="a lighthouse", size="256x256", seed=seed,
                                           steps=2)
            return png_pixels(json.loads(raw)["data"][0]["b64_json"])[3]
        self.assertEqual(pixels(11), pixels(11))
        self.assertNotEqual(pixels(11), pixels(12))

    def test_width_and_height_fields_and_defaults(self):
        status, _h, raw = self.gw.generate(prompt="x", width=256, height=512, steps=2)
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["colibri"]["height"], 512)
        status, _h, raw = self.gw.generate(prompt="x", steps=2)
        meta = json.loads(raw)["colibri"]
        info = self.gw.engine.info
        self.assertEqual((meta["width"], meta["height"]),
                         (info["default_width"], info["default_height"]))
        # No seed given: the server draws one and reports it, so the image can
        # be reproduced.
        self.assertIsInstance(meta["seed"], int)

    def test_stream_event_order(self):
        status, headers, events = self.gw.stream(prompt="a red fox", size="512x256", seed=3,
                                                 steps=4)
        self.assertEqual(status, 200)
        self.assertTrue(headers.get("Content-Type", "").startswith("text/event-stream"))
        names = [name for name, _ in events]
        self.assertEqual(events[-1], (None, "[DONE]"))
        self.assertEqual(names[-2], "image_generation.completed")
        self.assertEqual(names[0], "image_generation.progress")
        progress = [data for name, data in events if name == "image_generation.progress"]
        for frame in progress:
            self.assertEqual(set(frame), {"stage", "step", "steps", "elapsed"})
        stages = [frame["stage"] for frame in progress]
        self.assertEqual(stages[0], "encode")
        self.assertEqual(stages[-1], "decode")
        denoise = [frame["step"] for frame in progress if frame["stage"] == "denoise"]
        self.assertEqual(denoise, sorted(denoise))
        self.assertEqual(denoise[0], 0)              # steps COMPLETED: 0 as the first starts
        self.assertEqual(progress[-1]["step"], 4)    # all of them at decode
        partial = [data for name, data in events if name == "image_generation.partial_image"]
        self.assertEqual([p["partial_image_index"] for p in partial], list(range(len(partial))))
        self.assertTrue(partial)
        for p in partial:
            self.assertEqual(set(p), {"b64_json", "partial_image_index"})
            w, h, ch, _px = png_pixels(p["b64_json"])
            self.assertEqual((w, h, ch), (512 // 16, 256 // 16, 3))
        completed = events[-2][1]
        self.assertEqual(set(completed), {"created", "data", "colibri"})
        self.assertEqual(completed["colibri"]["seed"], 3)
        self.assertEqual(png_pixels(completed["data"][0]["b64_json"])[:2], (512, 256))

    def test_stream_without_partial_images(self):
        _s, _h, events = self.gw.stream(prompt="x", size="256x256", steps=3, partial_images=0)
        self.assertNotIn("image_generation.partial_image", [name for name, _ in events])
        _s, _h, events = self.gw.stream(prompt="x", size="256x256", steps=6, partial_images=2)
        partial = [d for name, d in events if name == "image_generation.partial_image"]
        self.assertEqual(len(partial), 2)

    def test_bad_sizes_are_400(self):
        for body in ({"size": "500x256"}, {"size": "768x432"}, {"size": "128x128"},
                     {"size": "4096x256"}, {"size": "big"}, {"width": 250},
                     {"size": "256x256", "width": 256}):
            with self.subTest(body=body):
                status, _h, raw = self.gw.generate(prompt="x", **body)
                status, error = self.error(status, raw)
                self.assertEqual(status, 400)
                self.assertIn(error["param"], ("size", "width", "height"))

    def test_empty_prompt_is_400(self):
        for prompt in ("", "   ", None, 7):
            with self.subTest(prompt=prompt):
                status, _h, raw = self.gw.generate(prompt=prompt)
                status, error = self.error(status, raw)
                self.assertEqual((status, error["param"]), (400, "prompt"))

    def test_n_other_than_one_is_refused(self):
        for n in (2, 0, True):
            with self.subTest(n=n):
                status, _h, raw = self.gw.generate(prompt="x", n=n)
                status, error = self.error(status, raw)
                self.assertEqual((status, error["param"]), (400, "n"))

    def test_other_bad_parameters(self):
        # A prompt past MAX_PROMPT_BYTES is refused before it reaches the engine: a
        # 72 KB one used to be written as one GEN line the engine never answered.
        status, _h, raw = self.gw.generate(prompt="x" * 8193, size="256x256")
        status, error = self.error(status, raw)
        self.assertEqual((status, error["param"]), (400, "prompt"))
        for key, value in (("steps", 0), ("steps", 1), ("steps", 10**6), ("seed", -1), ("seed", "x"),
                           ("response_format", "url"), ("output_format", "jpeg"),
                           ("stream", "yes"), ("partial_images", -1)):
            with self.subTest(key=key, value=value):
                status, _h, raw = self.gw.generate(prompt="x", size="256x256", **{key: value})
                status, error = self.error(status, raw)
                self.assertEqual((status, error["param"]), (400, key))

    def test_chat_endpoints_are_refused_with_a_pointer(self):
        for path, body in (("/v1/chat/completions",
                            {"messages": [{"role": "user", "content": "hi"}]}),
                           ("/v1/completions", {"prompt": "hi"})):
            with self.subTest(path=path):
                status, _h, raw = self.gw.request(path, dict(body, model=MODEL_ID))
                status, error = self.error(status, raw)
                self.assertEqual(status, 400)
                self.assertIn("/v1/images/generations", error["message"])

    def test_models_advertise_image_generation(self):
        status, _h, raw = self.gw.request("/v1/models")
        self.assertEqual(status, 200)
        entry = json.loads(raw)["data"][0]
        self.assertEqual(entry["id"], MODEL_ID)
        self.assertEqual(entry["capabilities"], ["image_generation"])
        self.assertEqual(entry["image"]["multiple"], 32)
        self.assertEqual(entry["image"]["default_height"], 512)
        status, _h, raw = self.gw.request(f"/v1/models/{MODEL_ID}")
        self.assertEqual(json.loads(raw)["capabilities"], ["image_generation"])

    def test_auth_is_required(self):
        for key in (None, "wrong"):
            with self.subTest(key=key):
                status, _h, raw = self.gw.request(
                    "/v1/images/generations", {"model": MODEL_ID, "prompt": "x"}, key=key)
                self.assertEqual(status, 401, raw)
        status, _h, _raw = self.gw.request("/v1/models", key=None)
        self.assertEqual(status, 401)

    def test_host_guard_and_cors(self):
        status, _h, _raw = self.gw.request("/v1/images/generations",
                                           {"model": MODEL_ID, "prompt": "x"},
                                           headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        status, headers, _raw = self.gw.request(
            "/v1/images/generations",
            {"model": MODEL_ID, "prompt": "x", "size": "256x256", "steps": 2},
            headers={"Origin": "http://localhost:5173"})
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")

    def test_wrong_model_is_404(self):
        status, _h, _raw = self.gw.generate(prompt="x", model="gpt-image-1")
        self.assertEqual(status, 404)

    def test_engine_error_before_any_frame_is_an_http_error(self):
        status, _h, raw = self.gw.generate(prompt="__stub_error__", size="256x256")
        status, error = self.error(status, raw)
        self.assertEqual(status, 500)
        self.assertIn("stub failure", error["message"])
        # Streaming too: the 200 is not committed before the engine's first frame.
        status, _h, raw = self.gw.request("/v1/images/generations",
                                          {"model": MODEL_ID, "prompt": "__stub_error__",
                                           "stream": True})
        self.assertEqual(status, 500)
        # And the engine is still serving.
        status, _h, _raw = self.gw.generate(prompt="fine", size="256x256", steps=2)
        self.assertEqual(status, 200)

    def test_stray_engine_output_does_not_break_the_protocol(self):
        status, _h, raw = self.gw.generate(prompt="__stub_garbage__ a boat", size="256x256",
                                           steps=2)
        self.assertEqual(status, 200, raw)


class ImagesFailureTest(unittest.TestCase):
    def test_engine_death_mid_stream_is_an_error_event_then_done(self):
        gw = GatewayHarness()
        try:
            status, _h, events = gw.stream(prompt="__stub_crash__", size="256x256", steps=4)
            self.assertEqual(status, 200)
            self.assertEqual(events[-1], (None, "[DONE]"))
            name, data = events[-2]
            self.assertEqual(name, "error")
            self.assertIn("message", data["error"])
            self.assertIn("type", data["error"])
            # The engine is gone: the next request is told so, not queued forever.
            status, _h, raw = gw.generate(prompt="x", size="256x256")
            self.assertEqual(status, 503, raw)
        finally:
            gw.close()

    def test_busy_past_the_queue_is_503(self):
        gw = GatewayHarness(delay=0.4, max_queue=0)
        try:
            first = threading.Thread(target=gw.generate,
                                     kwargs={"prompt": "slow", "size": "256x256", "steps": 4})
            first.start()
            deadline = time.time() + 10
            while gw.server.scheduler.snapshot()["active"] == 0 and time.time() < deadline:
                time.sleep(0.02)
            status, headers, raw = gw.generate(prompt="second", size="256x256", steps=2)
            self.assertEqual(status, 503, raw)
            self.assertEqual(json.loads(raw)["error"]["code"], "queue_full")
            first.join(timeout=20)
        finally:
            gw.close()

    def test_client_hangup_cancels_the_generation(self):
        gw = GatewayHarness(delay=0.3)
        try:
            body = json.dumps({"model": MODEL_ID, "prompt": "long", "size": "256x256",
                               "steps": 20, "stream": True}).encode()
            sock = socket.create_connection(("127.0.0.1", gw.port), timeout=10)
            sock.sendall(b"POST /v1/images/generations HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                         b"Authorization: Bearer secret\r\nContent-Type: application/json\r\n"
                         b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
            received = b""
            while b"image_generation.progress" not in received:
                chunk = sock.recv(4096)
                self.assertTrue(chunk, "stream closed before any progress")
                received += chunk
            sock.close()
            deadline = time.time() + 15
            while gw.server.scheduler.snapshot()["cancelled"] == 0 and time.time() < deadline:
                time.sleep(0.05)
            snapshot = gw.server.scheduler.snapshot()
            self.assertEqual(snapshot["cancelled"], 1, snapshot)
            self.assertEqual(snapshot["completed"], 0, snapshot)
            self.assertTrue(gw.engine.alive)
            # 20 steps at 0.3 s would be 6 s; the next request starts right away.
            started = time.time()
            status, _h, _raw = gw.generate(prompt="next", size="256x256", steps=2)
            self.assertEqual(status, 200)
            self.assertLess(time.time() - started, 5)
        finally:
            gw.close()


class TextModelImagesTest(unittest.TestCase):
    def test_a_chat_model_refuses_image_generation(self):
        class Chat:
            def generate(self, *args, **kwargs):
                raise AssertionError("never called")

        server = APIServer(("127.0.0.1", 0), Chat(), "text-model", None, 16)
        thread = threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            conn.request("POST", "/v1/images/generations",
                         json.dumps({"model": "text-model", "prompt": "x"}),
                         {"Content-Type": "application/json"})
            response = conn.getresponse()
            body = json.loads(response.read())
            self.assertEqual(response.status, 400)
            self.assertIn("does not generate images", body["error"]["message"])
            conn.request("GET", "/v1/models")
            entry = json.loads(conn.getresponse().read())["data"][0]
            self.assertNotIn("image_generation", entry["capabilities"])
            self.assertEqual(entry["capabilities"], ["chat", "systemone"])
            conn.close()
        finally:
            server.scheduler.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
