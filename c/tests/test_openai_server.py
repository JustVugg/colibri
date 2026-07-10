import io
import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import (APIError, APIServer, ClientCancelled, END, GenerationScheduler,
                           generation_options, read_engine_turn, render_chat)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, maximum, temperature, top_p, on_text):
        self.calls.append((prompt, maximum, temperature, top_p))
        on_text("Hé")
        on_text("llo")
        return {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text)


class TemplateTest(unittest.TestCase):
    def test_renders_text_subset_of_official_template(self):
        prompt = render_chat([
            {"role": "system", "content": "System"},
            {"role": "developer", "content": "Developer"},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": " Hello "},
            {"role": "user", "content": "Again"},
        ])
        self.assertEqual(
            prompt,
            "[gMASK]<sop><|system|>System<|system|>Developer<|user|>Hi"
            "<|assistant|><think></think>Hello<|user|>Again"
            "<|assistant|><think></think>",
        )

    def test_rejects_non_text_content(self):
        with self.assertRaisesRegex(APIError, "text message content only"):
            render_chat([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "x"}}
            ]}])

    def test_renders_thinking_prefix(self):
        self.assertEqual(
            render_chat([{"role": "user", "content": "Hi"}], True, "high"),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>Hi<|assistant|><think>",
        )

    def test_validates_generation_limits(self):
        self.assertEqual(generation_options({"max_tokens": 4, "temperature": 0, "top_p": 1}, 8),
                         (4, 0.0, 1.0))
        with self.assertRaises(APIError):
            generation_options({"max_tokens": 9}, 8)
        self.assertEqual(generation_options({"temperature": None, "top_p": None}, 8),
                         (8, 0.7, 0.9))


class ProtocolTest(unittest.TestCase):
    def test_reads_payload_and_extended_status(self):
        stream = io.BytesIO(b"hello" + END + b"STAT 2 3.5 44 1.2 7 1\n")
        chunks = []
        stats = read_engine_turn(stream, END, chunks.append)
        self.assertEqual(b"".join(chunks), b"hello")
        self.assertEqual(stats["prompt_tokens"], 7)
        self.assertTrue(stats["length_limited"])


class SchedulerTest(unittest.TestCase):
    def test_rejects_when_waiting_queue_is_full(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1)
        with scheduler.admit():
            with self.assertRaises(APIError) as caught:
                with scheduler.admit():
                    pass
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.code, "queue_full")
        self.assertEqual(scheduler.snapshot()["rejected"], 1)

    def test_times_out_and_cancels_queued_requests(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=0.02)
        with scheduler.admit():
            with self.assertRaises(APIError) as timed_out:
                with scheduler.admit():
                    pass
            with self.assertRaises(ClientCancelled):
                with scheduler.admit(lambda: True):
                    pass
        stats = scheduler.snapshot()
        self.assertEqual(timed_out.exception.code, "queue_timeout")
        self.assertEqual(stats["timed_out"], 1)
        self.assertEqual(stats["cancelled"], 1)

    def test_admits_waiters_in_fifo_order(self):
        scheduler = GenerationScheduler(max_queue=2, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        order = []

        def run(name, block=False):
            with scheduler.admit():
                order.append(name)
                if block:
                    entered.set()
                    release.wait(1)

        first = threading.Thread(target=run, args=("first", True))
        second = threading.Thread(target=run, args=("second",))
        third = threading.Thread(target=run, args=("third",))
        first.start(); entered.wait(1)
        second.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 1: break
            threading.Event().wait(0.005)
        third.start()
        for _ in range(100):
            if scheduler.snapshot()["queued"] == 2: break
            threading.Event().wait(0.005)
        release.set()
        first.join(1); second.join(1); third.join(1)
        self.assertEqual(order, ["first", "second", "third"])
        self.assertEqual(scheduler.snapshot()["completed"], 3)

    def test_close_rejects_waiters(self):
        scheduler = GenerationScheduler(max_queue=1, queue_timeout=1)
        entered = threading.Event()
        release = threading.Event()
        errors = []

        def active():
            with scheduler.admit():
                entered.set(); release.wait(1)

        def waiting():
            try:
                with scheduler.admit(): pass
            except APIError as error:
                errors.append(error.code)

        first = threading.Thread(target=active); first.start(); entered.wait(1)
        second = threading.Thread(target=waiting); second.start()
        scheduler.close(); release.set(); first.join(1); second.join(1)
        self.assertEqual(errors, ["scheduler_closed"])


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0), cls.engine, "test-model", "secret", 16)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.scheduler.close()
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, body=None, key="secret"):
        headers = {"Authorization": f"Bearer {key}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        return urlopen(Request(self.base + path, data=data, headers=headers), timeout=2)

    def test_lists_models_and_checks_auth(self):
        with self.request("/v1/models") as response:
            self.assertEqual(json.load(response)["data"][0]["id"], "test-model")
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/models", key="wrong")
        self.assertEqual(caught.exception.code, 401)

    def test_health_reports_scheduler(self):
        with self.request("/health") as response:
            scheduler = json.load(response)["scheduler"]
        self.assertEqual(scheduler["max_queue"], 8)
        self.assertIn("queued", scheduler)

    def test_browser_preflight(self):
        request = Request(self.base + "/v1/chat/completions", method="OPTIONS", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        })
        with urlopen(request, timeout=2) as response:
            self.assertEqual(response.status, 204)
            self.assertEqual(response.headers["Access-Control-Allow-Origin"], "http://localhost:5173")
            self.assertIn("Authorization", response.headers["Access-Control-Allow-Headers"])

    def test_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        }) as response:
            body = json.load(response)
            queue_wait = response.headers.get("x-colibri-queue-wait-ms")
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(body["usage"], {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})
        self.assertIsNotNone(queue_wait)
        self.assertIn("<|user|>Hi<|assistant|><think></think>", self.engine.calls[-1][0])

    def test_streaming_chat_completion(self):
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "stream_options": {"include_usage": True},
        }) as response:
            stream = response.read().decode()
        self.assertIn('\"delta\":{\"role\":\"assistant\",\"content\":\"\"}', stream)
        self.assertIn('\"object\":\"chat.completion.chunk\"', stream)
        self.assertIn('\"content\":\"Hé\"', stream)
        self.assertIn('\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":2,\"total_tokens\":9}', stream)
        self.assertTrue(stream.endswith("data: [DONE]\n\n"))

    def test_legacy_completion(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Complete me", "temperature": 0,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "Héllo")
        self.assertEqual(self.engine.calls[-1][0], "Complete me")

    def test_rejects_invalid_stream_options(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stream": True, "stream_options": "usage",
            })
        self.assertEqual(caught.exception.code, 400)


class SchedulerHTTPTest(unittest.TestCase):
    def setUp(self):
        self.engine = BlockingEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                max_tokens=16, max_queue=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"

    def tearDown(self):
        self.engine.release.set()
        self.server.scheduler.close()
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self):
        body = json.dumps({"model": "test-model", "messages": [
            {"role": "user", "content": "Hi"}]}).encode()
        return urlopen(Request(self.url, data=body, headers={"Content-Type": "application/json"}), timeout=2)

    def test_queue_full_returns_429_before_generation(self):
        first_errors = []

        def first_request():
            try:
                with self.request() as response: response.read()
            except Exception as error:
                first_errors.append(error)

        first = threading.Thread(target=first_request); first.start()
        self.assertTrue(self.engine.entered.wait(1))
        with self.assertRaises(HTTPError) as caught:
            self.request()
        error = json.loads(caught.exception.read())["error"]
        self.assertEqual(caught.exception.code, 429)
        self.assertEqual(caught.exception.headers["Retry-After"], "1")
        self.assertEqual(error["code"], "queue_full")
        self.engine.release.set(); first.join(2)
        self.assertEqual(first_errors, [])


if __name__ == "__main__":
    unittest.main()
