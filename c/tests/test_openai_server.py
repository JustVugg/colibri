import io
import json
import math
import socket
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from openai_server import (APIError, APIServer, ClientCancelled, END, GenerationScheduler,
                           READY, Engine, generation_options, read_engine_turn, render_chat,
                           serve)


class FakeEngine:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot))
        on_text("Hé")
        on_text("llo")
        return {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text, cache_slot,
                                cancelled)


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
        with self.assertRaises(APIError):
            generation_options({"temperature": math.nan}, 8)
        with self.assertRaises(APIError):
            generation_options({"top_p": math.inf}, 8)
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

    def test_rejects_invalid_kv_pool_before_engine_start(self):
        with self.assertRaisesRegex(ValueError, "kv_slots"):
            serve("/missing", kv_slots=0)

    def test_occupied_port_fails_before_engine_start(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        try:
            with patch("openai_server.subprocess.Popen") as popen:
                with self.assertRaises(OSError):
                    serve("/missing", port=listener.getsockname()[1])
            popen.assert_not_called()
        finally:
            listener.close()


class SchedulerTest(unittest.TestCase):
    def test_admits_up_to_capacity_without_serializing(self):
        scheduler = GenerationScheduler(max_queue=0, queue_timeout=1, capacity=2)
        with scheduler.admit() as first:
            with scheduler.admit() as second:
                self.assertEqual({first[1], second[1]}, {0, 1})
                self.assertEqual(scheduler.snapshot()["active"], 2)

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


class BlockingStream:
    def __init__(self, initial=b""):
        self.buffer = bytearray(initial)
        self.closed = False
        self.condition = threading.Condition()

    def feed(self, data):
        with self.condition:
            self.buffer.extend(data)
            self.condition.notify_all()

    def read(self, size=1):
        with self.condition:
            while len(self.buffer) < size and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            size = min(size, len(self.buffer))
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def readline(self):
        with self.condition:
            while b"\n" not in self.buffer and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            end = self.buffer.find(b"\n")
            size = len(self.buffer) if end < 0 else end + 1
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class FakeProcess:
    def __init__(self, on_write):
        self.stdout = BlockingStream(READY + b"STAT 0 0 0 0\n")
        self.stdin = self
        self.on_write = on_write
        self.writes = []
        self.returncode = None

    def write(self, data):
        self.writes.append(data)
        self.on_write(self, data)
        return len(data)

    def flush(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.stdout.close()

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminate()


class DispatcherTest(unittest.TestCase):
    def test_dispatches_interleaved_requests_by_id(self):
        submitted = []

        def respond(process, frame):
            fields = frame.split(b"\n", 1)[0].split()
            self.assertEqual(fields[0], b"SUBMIT")
            submitted.append(fields[1])
            if len(submitted) == 2:
                first, second = submitted
                process.stdout.feed(b"DATA " + second + b" 3\nB-2\n")
                process.stdout.feed(b"DATA " + first + b" 3\nA-1\n")
                process.stdout.feed(b"DONE " + second + b" STAT 1 2.5 0 1.0 4 0\n")
                process.stdout.feed(b"DATA " + first + b" 3\nA-2\n")
                process.stdout.feed(b"DONE " + first + b" STAT 2 3.5 0 1.0 5 1\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model", kv_slots=2)
        results = {}

        def generate(name, prompt, slot):
            chunks = []
            stats = engine.generate(prompt, 8, 0.7, 0.9, chunks.append, slot)
            results[name] = ("".join(chunks), stats)

        threads = [threading.Thread(target=generate, args=("a", "alpha", 0)),
                   threading.Thread(target=generate, args=("b", "beta", 1))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
        engine.close()

        self.assertEqual(results["a"][0], "A-1A-2")
        self.assertTrue(results["a"][1]["length_limited"])
        self.assertEqual(results["b"][0], "B-2")
        headers = [frame.split(b"\n", 1)[0].split() for frame in process.writes]
        self.assertEqual({int(header[2]) for header in headers}, {0, 1})
        self.assertEqual({header[3] for header in headers}, {b"4", b"5"})

    def test_routes_engine_error_to_request(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"ERROR " + request_id + b" slot is busy\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "slot is busy"):
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        engine.close()

    def test_close_wakes_pending_generation_and_is_idempotent(self):
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        errors = []

        def generate():
            try:
                engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
            except RuntimeError as error:
                errors.append(str(error))

        thread = threading.Thread(target=generate)
        thread.start()
        for _ in range(100):
            with engine.pending_lock:
                if engine.pending:
                    break
            threading.Event().wait(0.01)
        engine.close()
        engine.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ["colibri engine is shutting down"])
        self.assertFalse(engine.dispatcher.is_alive())
        with engine.pending_lock:
            self.assertFalse(engine.pending)
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            engine.generate("again", 4, 0.7, 0.9, lambda _: None)

    def test_protocol_corruption_fails_request_and_stops_dispatcher(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" -1\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        with self.assertRaisesRegex(RuntimeError, "DATA size"):
            engine.generate("hello", 4, 0.7, 0.9, lambda _: None)
        with self.assertRaisesRegex(RuntimeError, "dispatcher stopped"):
            engine.generate("again", 4, 0.7, 0.9, lambda _: None)
        engine.close()

    def test_decodes_utf8_split_across_data_frames(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 1\n\xc3\n")
            process.stdout.feed(b"DATA " + request_id + b" 1\n\xa9\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 1 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        chunks = []
        engine.generate("hello", 4, 0.7, 0.9, chunks.append)
        engine.close()
        self.assertEqual(chunks, ["é"])

    def test_cancels_generation_after_consumer_disconnects(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"CANCEL":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        with self.assertRaises(ClientCancelled):
            engine.generate("hello", 8, 0.7, 0.9, output.append, cancelled=lambda: True)
        engine.close()
        self.assertEqual(output, ["x"])
        self.assertEqual(process.writes[-1].split(), [b"CANCEL", request_id])


class HTTPTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = FakeEngine()
        cls.server = APIServer(("127.0.0.1", 0),cls.engine,"test-model","secret",16,kv_slots=2)
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

    def test_health_reports_scheduler_and_kv_slots(self):
        with self.request("/health") as response:
            health = json.load(response)
            scheduler = health["scheduler"]
        self.assertEqual(scheduler["max_queue"], 8)
        self.assertIn("queued", scheduler)
        self.assertEqual(health["kv_slots"], 2)

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
            "max_tokens": 4, "cache_slot": 1,
        }) as response:
            body = json.load(response)
            queue_wait = response.headers.get("x-colibri-queue-wait-ms")
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(body["usage"], {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9})
        self.assertIsNotNone(queue_wait)
        self.assertIn("<|user|>Hi<|assistant|><think></think>", self.engine.calls[-1][0])
        self.assertEqual(self.engine.calls[-1][4], 1)

    def test_rejects_invalid_cache_slot(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "cache_slot": 2,
            })
        self.assertEqual(caught.exception.code, 400)

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
