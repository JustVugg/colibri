import http.client
import io
import json
import math
import socket
import tempfile
import threading
import sys
import time
import struct
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path

from openai_server import (APIError, APIHandler, APIServer, ClientCancelled,
                           DEFAULT_CHAT_STOP_SEQUENCES, END, GenerationScheduler,
                           READY, Engine, InklingStreamSplit, StopFilter, ThinkingStreamSplit,
                           _engine_error, cap_for_arch, conversation_cache_slot, model_arch,
                           generation_options, parse_tool_calls, parse_dsv4_tool_calls,
                           read_engine_turn, render_chat, render_chat_kimi, render_chat_olmoe,
                           render_chat_v4, _dsv4_tool_calls, serve, split_thinking_reply,
                           stop_policy, tune_child_env)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.stop_requests = 0

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        if on_accept is not None:                 # simulate the engine's ACCEPT frame (#597)
            on_accept({"prompt_tokens": 7})
        for chunk in ("Hé", "llo"):
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        return {"prompt_tokens": 7, "completion_tokens": 2, "length_limited": False}


class BlockingEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.entered.set()
        self.release.wait(2)
        return super().generate(prompt, maximum, temperature, top_p, on_text, cache_slot,
                                cancelled, grammar, stopped, on_accept)


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

    def test_kimi_payload_preserves_utf8_lengths_and_turns(self):
        prompt = render_chat_kimi([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "你好\nKimi"},
            {"role": "assistant", "content": "你好。"},
            {"role": "user", "content": "Continue"},
        ], enable_thinking=True)
        self.assertEqual(
            prompt,
            "K3CHAT1\n"
            "M system 11\nBe precise."
            "M user 11\n你好\nKimi"
            "A 0 9\n你好。"
            "M user 8\nContinue"
            "G 1\n",
        )

    def test_kimi_rejects_tools_and_unknown_roles(self):
        with self.assertRaisesRegex(APIError, "Tool use"):
            render_chat_kimi([{"role": "user", "content": "Hi"}],
                             tools=[{"type": "function"}])
        with self.assertRaisesRegex(APIError, "Unsupported role"):
            render_chat_kimi([{"role": "tool", "content": "result"}])

    def test_kimi_preserves_prior_reasoning_channel(self):
        self.assertEqual(
            render_chat_kimi([{"role": "assistant", "reasoning_content": "why",
                               "content": "answer"}], enable_thinking=True),
            "K3CHAT1\nA 3 6\nwhyanswerG 1\n",
        )

    def test_olmoe_renders_native_chat_template(self):
        # Matches allenai/OLMoE-1B-7B-0125-Instruct's tokenizer_config.json
        # chat_template exactly: one leading bos_token, per-role turns closed
        # by a trailing newline, a prior (non-final) assistant turn also closed
        # by eos_token before that newline (bos_token == eos_token ==
        # "|||IP_ADDRESS|||" in this tokenizer), and a trailing
        # "<|assistant|>\n" generation prompt.
        prompt = render_chat_olmoe([
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Continue"},
        ])
        self.assertEqual(
            prompt,
            "|||IP_ADDRESS|||<|system|>\nBe terse.\n<|user|>\nHi\n"
            "<|assistant|>\nHello|||IP_ADDRESS|||\n<|user|>\nContinue\n"
            "<|assistant|>\n",
        )

    def test_olmoe_rejects_tools_and_unknown_roles(self):
        with self.assertRaisesRegex(APIError, "Tool use"):
            render_chat_olmoe([{"role": "user", "content": "Hi"}],
                              tools=[{"type": "function"}])
        with self.assertRaisesRegex(APIError, "Unsupported role"):
            render_chat_olmoe([{"role": "tool", "content": "result"}])

    def test_olmoe_rejects_empty_messages(self):
        with self.assertRaisesRegex(APIError, "non-empty array"):
            render_chat_olmoe([])

    def test_validates_generation_limits(self):
        self.assertEqual(generation_options({"max_tokens": 4, "temperature": 0, "top_p": 1}, 8),
                         (4, 0.0, 1.0, None, ()))
        # max_tokens above the server cap is clamped, not rejected (#260): OpenAI
        # clients default to large values; erroring breaks them.
        self.assertEqual(generation_options({"max_tokens": 9, "temperature": 0, "top_p": 1}, 8),
                         (8, 0.0, 1.0, None, ()))
        # non-positive / non-int max_tokens is still a hard error
        with self.assertRaises(APIError):
            generation_options({"max_tokens": 0}, 8)
        with self.assertRaises(APIError):
            generation_options({"temperature": math.nan}, 8)
        with self.assertRaises(APIError):
            generation_options({"top_p": math.inf}, 8)
        self.assertEqual(generation_options({"temperature": None, "top_p": None}, 8),
                         (8, 0.7, 0.9, None, ()))
        # response_format -> grammar plumbing (draft source, never a constraint)
        opts = generation_options({"max_tokens": 4, "response_format": {"type": "json_object"}}, 8)
        self.assertIn("root ::=", opts[3])
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        opts = generation_options({"max_tokens": 4, "response_format":
                                   {"type": "json_schema", "json_schema": {"schema": schema}}}, 8)
        self.assertEqual(json.loads(opts[3]), schema)
        opts = generation_options({"max_tokens": 4, "response_format":
                                   {"type": "gbnf", "grammar": 'root ::= "x"'}}, 8)
        self.assertEqual(opts[3], 'root ::= "x"')
        with self.assertRaises(APIError):
            generation_options({"response_format": {"type": "yaml"}}, 8)
        with self.assertRaises(APIError):
            generation_options({"response_format": {"type": "json_schema", "json_schema": {}}}, 8)
        with self.assertRaises(APIError):   # non-dict response_format
            generation_options({"response_format": "json"}, 8)
        with self.assertRaises(APIError):   # empty gbnf
            generation_options({"response_format": {"type": "gbnf", "grammar": "  "}}, 8)
        with self.assertRaises(APIError):   # oversized grammar (> 1 MiB pre-check)
            generation_options({"response_format": {"type": "gbnf", "grammar": "x" * ((1 << 20) + 1)}}, 8)
        # malformed GBNF passes the gateway by design: the ENGINE fail-softs it
        # (draft source only — bad grammar costs the speedup, never the request)
        opts = generation_options({"response_format": {"type": "gbnf", "grammar": "not a grammar ::="}}, 8)
        self.assertEqual(opts[3], "not a grammar ::=")

    def test_validates_stop_sequences(self):
        self.assertEqual(generation_options({"stop": "END"}, 8)[4], ("END",))
        self.assertEqual(generation_options({"stop": ["ONE", "TWO"]}, 8)[4],
                         ("ONE", "TWO"))
        for value in ("", [], [""], ["1", "2", "3", "4", "5"], 7, ["ok", 7]):
            with self.subTest(value=value), self.assertRaises(APIError):
                generation_options({"stop": value}, 8)

    def test_glm_chat_defaults_role_stops_without_changing_other_policies(self):
        with patch("openai_server.ARCH", "glm"):
            self.assertEqual(stop_policy({}, True), (DEFAULT_CHAT_STOP_SEQUENCES, True))
            self.assertEqual(stop_policy({}, False), ((), False))
            self.assertEqual(stop_policy({"stop": "END"}, True), (("END",), False))
            self.assertEqual(stop_policy({
                "stop": "END", "x_colibri_ignore_leading_stop": True,
            }, True), (("END",), True))
        with patch("openai_server.ARCH", "inkling"):
            self.assertEqual(stop_policy({}, True), ((), False))
            self.assertEqual(stop_policy({"stop": "END"}, True), (("END",), False))
        with self.assertRaises(APIError):
            stop_policy({"x_colibri_ignore_leading_stop": "yes"}, True)


class StopFilterTest(unittest.TestCase):
    def test_explicit_stop_composes_with_inkling_stream_split(self):
        content = []
        reasoning = []
        splitter = InklingStreamSplit(content.append, reasoning.append)
        stop_filter = StopFilter(("END",), splitter.feed)
        for chunk in ("<|content_thinking|>why<|content_text|>answer EN", "Dignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        splitter.close()
        self.assertEqual("".join(reasoning), "why")
        self.assertEqual("".join(content), "answer ")
        self.assertEqual(stop_filter.matched, "END")

    def test_hides_match_split_across_chunks(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        for chunk in ("answer S", "TO", "Pignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        self.assertEqual("".join(output), "answer ")
        self.assertEqual(stop_filter.matched, "STOP")

    def test_flushes_partial_prefix_when_generation_finishes(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        stop_filter.feed("answer ST")
        stop_filter.finish()
        self.assertEqual("".join(output), "answer ST")

    def test_optional_patient_mode_ignores_only_leading_matches(self):
        output = []
        stop_filter = StopFilter(("<|user|>",), output.append, ignore_leading=True)
        for chunk in ("<|us", "er|>answer", "<|user|>ignored"):
            stop_filter.feed(chunk)
        stop_filter.finish()
        self.assertEqual("".join(output), "answer")
        self.assertEqual(stop_filter.matched, "<|user|>")
        self.assertEqual(stop_filter.leading_matches_ignored, 1)

    def test_patient_mode_preserves_remainder_after_same_chunk_leading_match(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append, ignore_leading=True)
        stop_filter.feed("STOPuseful STOPdiscarded")
        stop_filter.finish()
        self.assertEqual("".join(output), "useful ")
        self.assertEqual(stop_filter.matched, "STOP")

    def test_strict_mode_still_stops_on_a_leading_match(self):
        output = []
        stop_filter = StopFilter(("STOP",), output.append)
        stop_filter.feed("STOPignored")
        stop_filter.finish()
        self.assertEqual(output, [])
        self.assertEqual(stop_filter.matched, "STOP")


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

    def test_records_profile_snapshots_from_prof_lines(self):
        def respond(process, frame):
            request_id = frame.split()[1]
            process.stdout.feed(b"DATA " + request_id + b" 2\nok\n")
            process.stdout.feed(b"PROF 2.500 7 12 0.400 0.100 0.900 0.600 0.200 15\n")
            process.stdout.feed(b"DONE " + request_id + b" STAT 12 4.8 0 1.0 7 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        engine.generate("hello", 16, 0.7, 0.9, lambda _: None)
        engine.close()
        self.assertEqual(engine.profile_seq, 1)
        self.assertEqual(list(engine.profile), [{
            "wall_s": 2.5, "prompt_tokens": 7, "completion_tokens": 12,
            "expert_disk_s": 0.4, "expert_wait_s": 0.1, "expert_matmul_s": 0.9,
            "attention_s": 0.6, "lm_head_s": 0.2, "forwards": 15,
        }])

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

    def test_cancels_generation_before_first_frame(self):
        # #908: a client that disconnects while the engine is still prefilling
        # (no DATA frame has arrived) must cancel too. cancelled() used to be
        # polled only in the "data" branch, so the CANCEL never went out and
        # the turn ran to its token limit while this thread stayed blocked.
        # The fake engine emits nothing until it sees CANCEL -- exactly the
        # pre-first-frame regime -- and must still get one.
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
            elif fields[0] == b"CANCEL":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"ERROR " + request_id + b" CANCELLED\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        flag = {"cancelled": False}
        outcome = []

        def generate():
            try:
                engine.generate("hello", 8, 0.7, 0.9, lambda _: None,
                                cancelled=lambda: flag["cancelled"])
            except ClientCancelled:
                outcome.append("cancelled")

        thread = threading.Thread(target=generate)
        thread.start()
        for _ in range(200):
            if any(frame.startswith(b"SUBMIT") for frame in process.writes):
                break
            time.sleep(0.01)
        flag["cancelled"] = True
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        engine.close()
        self.assertEqual(outcome, ["cancelled"])
        self.assertEqual(process.writes[-1].split(), [b"CANCEL", request_id])

    def test_stops_generation_through_successful_done_path(self):
        request_id = None

        def respond(process, frame):
            nonlocal request_id
            fields = frame.split()
            if fields[0] == b"SUBMIT":
                request_id = fields[1]
                process.stdout.feed(b"DATA " + request_id + b" 1\nx\n")
            elif fields[0] == b"STOP":
                self.assertEqual(fields[1], request_id)
                process.stdout.feed(b"DONE " + request_id + b" STAT 1 1 0 1 2 0\n")

        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            engine = Engine("glm", "model")
        output = []
        stats = engine.generate("hello", 8, 0.7, 0.9, output.append,
                                stopped=lambda: output == ["x"])
        engine.close()
        self.assertEqual(output, ["x"])
        self.assertEqual(stats["completion_tokens"], 1)
        self.assertEqual(process.writes[-1].split(), [b"STOP", request_id])


class CapSentinelShimTest(unittest.TestCase):
    # #379 cap-sentinel shim, arch-keyed (#386 r2, F3): an absent cap is
    # "platform-auto" only for the glm engine (colibri.c coli_resolve_cap);
    # inkling reads cap <= 0 as "fit the expert LRU to all available RAM"
    # (inkling.c), so leaking the sentinel to a non-glm arch silently changes
    # its memory behavior. The key is the MODEL's config.json model_type, not
    # the engine binary's file name -- COLI_ENGINE users package the glm
    # engine as glm52/colibri-1.2/glm-metal, and basename keying disabled the
    # platform default for exactly them (and an inkling binary someone names
    # `glm` would get the leak back). Engine() is the one funnel every launch
    # passes through, so the translation is pinned at the argv it emits, over
    # the full matrix: arch x arbitrary executable name x cap absent/explicit.
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _model(self, model_type):
        model = Path(self.tmp.name) / f"model-{model_type}"
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text(json.dumps({"model_type": model_type}))
        return str(model)

    def _spawn_argv(self, executable, model, **kwargs):
        process = FakeProcess(lambda _process, _frame: None)
        with patch("openai_server.subprocess.Popen", return_value=process) as popen:
            engine = Engine(executable, model, **kwargs)
            engine.close()
        return popen.call_args[0][0]

    def test_matrix_arch_times_engine_name_times_cap(self):
        # COLI_ENGINE axis: the executable name carries no information; only
        # the model arch and the explicitness of cap may matter.
        executables = ("colibri", "glm", "glm52", "colibri-1.2", "glm-metal",
                       "/opt/custom/inkling", "kimi_k3.exe")
        cases = (  # (model_type, cap kwargs, expected argv cap)
            ("glm_moe_dsa", {}, "0"),          # absent -> glm sentinel
            ("inkling", {}, "8"),              # absent -> legacy 8
            ("kimi_k3", {}, "8"),              # absent -> legacy 8
            ("glm_moe_dsa", {"cap": 5}, "5"),  # explicit -> verbatim
            ("inkling", {"cap": 5}, "5"),
            ("kimi_k3", {"cap": 5}, "5"),
            ("glm_moe_dsa", {"cap": 0}, "0"),  # explicit 0 -> verbatim
            ("inkling", {"cap": 0}, "0"),      # upstream RAM-auto, by request
            ("kimi_k3", {"cap": 0}, "0"),
        )
        for model_type, kwargs, want in cases:
            model = self._model(model_type)
            for executable in executables:
                self.assertEqual(
                    self._spawn_argv(executable, model, **kwargs),
                    [executable, want],
                    f"arch={model_type} exe={executable} kwargs={kwargs}")

    def test_missing_or_unreadable_config_is_glm(self):
        # historic default: anything that cannot be classified is glm
        self.assertEqual(self._spawn_argv("engine", "/nonexistent/model"),
                         ["engine", "0"])
        model = Path(self.tmp.name) / "model-broken"
        model.mkdir()
        (model / "config.json").write_text("{not json")
        self.assertEqual(self._spawn_argv("engine", str(model)), ["engine", "0"])

    def test_cap_for_arch_is_the_single_translation_point(self):
        self.assertEqual(cap_for_arch("glm", None), 0)
        self.assertEqual(cap_for_arch("inkling", None), 8)
        self.assertEqual(cap_for_arch("kimi", None), 8)
        self.assertEqual(cap_for_arch("olmoe", None), 8)
        self.assertEqual(cap_for_arch("glm", 3), 3)
        self.assertEqual(cap_for_arch("inkling", 3), 3)
        self.assertEqual(cap_for_arch("inkling", 0), 0)   # explicit 0 is explicit
        self.assertEqual(cap_for_arch("glm", 0), 0)

    def test_model_arch_reads_model_type(self):
        self.assertEqual(model_arch(self._model("glm_moe_dsa")), "glm")
        self.assertEqual(model_arch(self._model("inkling")), "inkling")
        self.assertEqual(model_arch(self._model("kimi_k3")), "kimi")
        self.assertEqual(model_arch(self._model("deepseek_v4")), "deepseek_v4")
        self.assertEqual(model_arch(self._model("olmoe")), "olmoe")
        self.assertEqual(model_arch("/nonexistent"), "glm")

    def test_direct_v4_server_gets_bounded_dspark_defaults(self):
        env = {"V4_MTP_CONF": "0.7"}
        with patch("resource_plan.physical_cpu_count", return_value=6), \
             patch("openai_server.sys.platform", "linux"):
            tune_child_env(env, "deepseek_v4")
        self.assertEqual(env["OMP_NUM_THREADS"], "6")
        self.assertEqual(env["OMP_PROC_BIND"], "close")
        self.assertEqual(env["V4_DRAFT"], "0")
        self.assertEqual(env["V4_MTP"], "0")
        self.assertEqual(env["V4_MTP_DRAFT"], "3")
        self.assertEqual(env["V4_MTP_GB"], "0.45")
        self.assertEqual(env["V4_MTP_CONF"], "0.7")  # explicit override wins


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

    def test_profile_requires_auth(self):
        """/profile is served before require_auth(), so it needs its own gate.

        This test previously asserted the opposite -- that the telemetry was
        readable without a key. That was not a client requirement: the web
        dashboard sends `Authorization: Bearer` to /profile exactly as it does
        to /health and /experts (web/src/lib/api.ts). The turns carry prompt and
        completion token counts and per-phase timings for the last 120 requests,
        which describes what the operator is running and how much of it, so an
        anonymous caller now gets the same empty shape those two endpoints give.
        """
        turn = {"wall_s": 2.5, "prompt_tokens": 7, "completion_tokens": 12,
                "expert_disk_s": 0.4, "expert_wait_s": 0.1, "expert_matmul_s": 0.9,
                "attention_s": 0.6, "lm_head_s": 0.2, "forwards": 15}
        self.engine.profile = [turn]
        self.engine.profile_seq = 1
        try:
            with urlopen(self.base + "/profile", timeout=2) as response:
                self.assertEqual(json.load(response), {"seq": 0, "turns": []},
                                 "unauthenticated caller received telemetry")
            with self.request("/profile") as response:
                self.assertEqual(json.load(response), {"seq": 1, "turns": [turn]},
                                 "authenticated caller lost access")
        finally:
            del self.engine.profile, self.engine.profile_seq

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

    def test_kimi_chat_completion_uses_multiturn_wire_payload(self):
        with patch("openai_server.ARCH", "kimi"):
            with self.request("/v1/chat/completions", {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "content": "你好。"},
                    {"role": "user", "content": "Continue"},
                ],
                "enable_thinking": False,
            }) as response:
                body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "Héllo")
        self.assertEqual(
            self.engine.calls[-1][0],
            "K3CHAT1\n"
            "M user 6\n你好"
            "M assistant 9\n你好。"
            "M user 8\nContinue"
            "G 0\n",
        )

    def test_chat_completion_stops_across_engine_chunks(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stop": "éll",
        }) as response:
            body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "H")
        self.assertEqual(body["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.stop_requests, before + 1)

    def test_patient_stop_extension_ignores_a_leading_match(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stop": "H", "x_colibri_ignore_leading_stop": True,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["choices"][0]["message"]["content"], "éllo")
        self.assertEqual(self.engine.stop_requests, before)

    def test_patient_stop_extension_requires_a_boolean(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stop": "H", "x_colibri_ignore_leading_stop": "yes",
            })
        self.assertEqual(caught.exception.code, 400)

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

    def test_streaming_stop_never_exposes_partial_sequence(self):
        before = self.engine.stop_requests
        with self.request("/v1/chat/completions", {
            "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
            "stream": True, "stop": "éll",
        }) as response:
            raw = response.read().decode()
        payloads = [json.loads(line[6:]) for line in raw.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"]
        content = "".join((choice.get("delta") or {}).get("content", "")
                          for payload in payloads for choice in payload["choices"])
        self.assertEqual(content, "H")
        self.assertEqual(payloads[-1]["choices"][0]["finish_reason"], "stop")
        self.assertEqual(self.engine.stop_requests, before + 1)

    def test_legacy_completion(self):
        with self.request("/v1/completions", {
            "model": "test-model", "prompt": "Complete me", "temperature": 0,
        }) as response:
            body = json.load(response)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["choices"][0]["text"], "Héllo")
        self.assertEqual(self.engine.calls[-1][0], "Complete me")

    def test_rejects_empty_legacy_completion(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/completions", {"model": "test-model", "prompt": ""})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(json.load(caught.exception)["error"]["param"], "prompt")

    def test_rejects_invalid_stream_options(self):
        with self.assertRaises(HTTPError) as caught:
            self.request("/v1/chat/completions", {
                "model": "test-model", "messages": [{"role": "user", "content": "Hi"}],
                "stream": True, "stream_options": "usage",
            })
        self.assertEqual(caught.exception.code, 400)


class ClientHangupTest(unittest.TestCase):
    """A client that disconnects mid-response must not print a traceback.

    `coli chat` polls /health on a 2 s timeout while the model loads and drops
    each connection the moment it has an answer; Ctrl-C during a stream closes
    the socket by design -- the banner tells the user to do exactly that. Both
    reach the handler as BrokenPipeError, and socketserver logs an unhandled
    exception per occurrence, so a normal DeepSeek V4 start buried the loading
    spinner under stack traces and every cancelled answer looked like a crash.
    """

    def setUp(self):
        self.engine = FakeEngine()
        self.server = APIServer(("127.0.0.1", 0), self.engine, "test-model",
                                None, 16, kv_slots=1)
        self.errors = []
        # socketserver routes an escaped exception here; the base class prints
        # it to stderr. Recording instead of printing is what lets the test
        # assert on it rather than on captured output.
        self.server.handle_error = lambda request, address: self.errors.append(
            sys.exc_info()[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.thread.join, 2)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)

    def _hang_up_after_request(self, path):
        """Send a request, then close without reading the response."""
        sock = socket.create_connection(("127.0.0.1", self.server.server_port), 2)
        sock.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                     f"Connection: close\r\n\r\n".encode())
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                        struct.pack("ii", 1, 0))   # RST rather than a clean FIN
        sock.close()
        time.sleep(0.3)

    def test_hangup_on_health_is_not_an_error(self):
        for _ in range(5):
            self._hang_up_after_request("/health")
        self.assertEqual(self.errors, [],
                         "client disconnect surfaced as a server error")

    def test_the_server_still_answers_afterwards(self):
        """The real damage would be a handler thread lost to the exception."""
        self._hang_up_after_request("/health")
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health",
                     timeout=2) as response:
            self.assertEqual(json.load(response)["status"], "ok")

    def _abort(self, *args):
        """Windows' spelling of the same disconnect, raised where it really lands.

        In #854's log the traceback runs do_GET -> send_json -> end_headers ->
        flush_headers -> wfile.write -> sendall, so raising from end_headers
        reproduces the exact shape on any platform.
        """
        raise ConnectionAbortedError(
            10053, "An established connection was aborted by the software in your "
                   "host machine")

    def test_windows_aborted_connection_is_not_an_error(self):
        """ConnectionAbortedError is a SIBLING of BrokenPipeError and
        ConnectionResetError under ConnectionError, not a subclass of either --
        so catching the pair caught the POSIX spellings and let the Windows one
        escape. #854 is pages of WinError 10053 tracebacks from a healthy start.
        """
        self.assertFalse(
            issubclass(ConnectionAbortedError, (BrokenPipeError, ConnectionResetError)),
            "the old except clause would have covered this; the test proves nothing")
        with patch.object(APIHandler, "end_headers", self._abort):
            try:
                urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=2)
            except Exception:
                pass                      # the client sees a broken response; that is fine
            time.sleep(0.3)
        self.assertEqual(self.errors, [],
                         "WinError 10053 surfaced as a server error (#854)")

    def test_the_server_survives_an_aborted_connection(self):
        """Same as the hangup case: the damage is a lost handler, not the log."""
        with patch.object(APIHandler, "end_headers", self._abort):
            try:
                urlopen(f"http://127.0.0.1:{self.server.server_port}/health", timeout=2)
            except Exception:
                pass
            time.sleep(0.3)
        with urlopen(f"http://127.0.0.1:{self.server.server_port}/health",
                     timeout=2) as response:
            self.assertEqual(json.load(response)["status"], "ok")


class StaticServingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        dist = root / "dist"
        dist.mkdir()
        (dist / "index.html").write_text("dashboard", encoding="utf-8")
        sibling = root / "dist-private"
        sibling.mkdir()
        (sibling / "secret.txt").write_text("private", encoding="utf-8")
        self.web_dist = patch.object(APIHandler, "WEB_DIST", dist)
        self.web_dist.start()
        self.server = APIServer(("127.0.0.1", 0), FakeEngine(), "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.web_dist.stop()
        self.tmp.cleanup()

    def test_static_root_stays_inside_dist_directory(self):
        with urlopen(self.base + "/", timeout=2) as response:
            self.assertEqual(response.read(), b"dashboard")
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + "/%2e%2e/dist-private/secret.txt", timeout=2)
        self.assertEqual(caught.exception.code, 404)


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



ORDER_TOOL = [{"type": "function", "function": {
    "name": "lookup_order",
    "parameters": {"type": "object", "properties": {
        "order_id": {"type": "string"},
        "qty": {"type": "integer"},
        "express": {"type": "boolean"},
    }, "required": ["order_id"]}}}]


class ToolArgumentTypeTest(unittest.TestCase):
    """The model emits every argument as text. Without the schema, a string-typed value that
    happens to look numeric is json.loads()'d into an int and the tool gets the wrong type."""

    def _args(self, reply, tools=ORDER_TOOL):
        _, calls = parse_tool_calls(reply, tools)
        self.assertEqual(len(calls), 1)
        return json.loads(calls[0]["function"]["arguments"])

    def test_string_parameter_holding_digits_stays_a_string(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>order_id</arg_key><arg_value>12345</arg_value></tool_call>")
        self.assertEqual(args["order_id"], "12345")
        self.assertIsInstance(args["order_id"], str)

    def test_declared_numeric_and_boolean_parameters_are_decoded(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>"
                          "<arg_key>qty</arg_key><arg_value>2</arg_value>"
                          "<arg_key>express</arg_key><arg_value>true</arg_value></tool_call>")
        self.assertEqual(args, {"order_id": "A-1", "qty": 2, "express": True})
        self.assertIsInstance(args["qty"], int)
        self.assertIs(args["express"], True)

    def test_unknown_parameter_keeps_permissive_decoding(self):
        args = self._args("<tool_call>lookup_order"
                          "<arg_key>extra</arg_key><arg_value>7</arg_value></tool_call>")
        self.assertEqual(args["extra"], 7)


class DeepSeekV4ToolCallTest(unittest.TestCase):
    """DeepSeek V4 (#916): DSML tool blocks. Schemas render into the first system/developer
    message, assistant tool_calls render as <｜DSML｜invoke> blocks, tool results merge into
    user turns as <tool_result>, and model output parses back into OpenAI tool_calls."""

    DSML = "｜DSML｜"
    WEATHER = [{"type": "function", "function": {
        "name": "get_weather", "description": "current weather",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"]}}}]
    CALL = [{"id": "call_1", "type": "function", "function": {
        "name": "get_weather",
        "arguments": json.dumps({"city": "Paris", "days": 3})}}]

    def test_tools_declared_on_first_system_message(self):
        prompt = render_chat_v4([{"role": "system", "content": "Be brief."},
                                 {"role": "user", "content": "Weather?"}], tools=self.WEATHER)
        self.assertLess(prompt.index("Be brief."), prompt.index("## Tools"))
        self.assertIn(f'"{self.WEATHER[0]["function"]["name"]}"', prompt)
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>"))

    def test_tools_prepended_when_no_system_message(self):
        prompt = render_chat_v4([{"role": "user", "content": "hi"}], tools=self.WEATHER)
        # The official encoder renders tools on an empty system message: bos + "\n\n" + tools.
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>\n\n## Tools"))

    def test_tool_choice_none_suppresses_declaration(self):
        prompt = render_chat_v4([{"role": "user", "content": "hi"}],
                                tools=self.WEATHER, tool_choice="none")
        self.assertNotIn("## Tools", prompt)

    def test_developer_message_is_wrapped_in_user_token(self):
        # encoding_dsv4.py wraps developer content in <｜User｜>; system stays bare.
        prompt = render_chat_v4([{"role": "developer", "content": "Be terse."},
                                 {"role": "user", "content": "hi"}], tools=self.WEATHER)
        self.assertIn("<｜User｜>Be terse.", prompt)
        self.assertNotIn("<｜User｜>## Tools", prompt)

    def test_thinking_mode_prepends_effort_prompt(self):
        # encoding_dsv4.py prepends the level prompt after BOS in thinking mode.
        prompt = render_chat_v4([{"role": "user", "content": "hi"}], enable_thinking=True,
                                reasoning_effort="high")
        self.assertTrue(prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: High."))
        self.assertTrue(prompt.endswith("<｜Assistant｜><think>"))
        # V4-native level names work too; low adds nothing.
        self.assertTrue(render_chat_v4([{"role": "user", "content": "hi"}],
                                       enable_thinking=True, reasoning_effort="max").startswith(
            "<｜begin▁of▁sentence｜>Reasoning Effort: Maximum."))
        self.assertFalse(render_chat_v4([{"role": "user", "content": "hi"}],
                                        enable_thinking=True, reasoning_effort="low").startswith(
            "<｜begin▁of▁sentence｜>Reasoning Effort:"))

    def test_assistant_tool_calls_render_as_dsml(self):
        prompt = render_chat_v4([{"role": "user", "content": "Paris?"},
                                 {"role": "assistant", "content": None,
                                  "tool_calls": self.CALL}])
        self.assertIn(f'<{self.DSML}invoke name="get_weather">', prompt)
        self.assertIn(f'<{self.DSML}parameter name="city" string="true">Paris</{self.DSML}parameter>',
                      prompt)
        self.assertIn(f'<{self.DSML}parameter name="days" string="false">3</{self.DSML}parameter>',
                      prompt)
        self.assertTrue(prompt.endswith("</think>"))

    def test_tool_results_merge_into_one_user_turn(self):
        prompt = render_chat_v4([{"role": "user", "content": "Paris?"},
                                 {"role": "assistant", "content": None, "tool_calls": self.CALL},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "21c"},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                                 {"role": "user", "content": "And London?"}])
        self.assertEqual(prompt.count("<｜User｜>"), 2)
        self.assertIn("<tool_result>21c</tool_result>", prompt)
        self.assertIn("<tool_result>sunny</tool_result>", prompt)
        self.assertIn("And London?", prompt)

    def test_parse_dsml_reply_to_openai_tool_calls(self):
        raw = (f"Here you go.\n\n<{self.DSML}tool_calls>\n"
               f"<{self.DSML}invoke name=\"get_weather\">\n"
               f"<{self.DSML}parameter name=\"city\" string=\"true\">Paris</{self.DSML}parameter>\n"
               f"<{self.DSML}parameter name=\"days\" string=\"false\">3</{self.DSML}parameter>\n"
               f"</{self.DSML}invoke>\n</{self.DSML}tool_calls><｜end▁of▁sentence｜>")
        content, calls = parse_dsv4_tool_calls(raw)
        self.assertEqual(content, "Here you go.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]),
                         {"city": "Paris", "days": 3})

    def test_parse_truncated_block_leaks_no_markers(self):
        content, calls = parse_dsv4_tool_calls("Almost.\n\n<｜DSML｜tool_calls><｜DSML｜invoke na")
        self.assertEqual(calls, [])
        self.assertEqual(content, "Almost.")
        self.assertNotIn("DSML", content)


class EngineErrorFrameTest(unittest.TestCase):
    """#401: an over-long prompt used to be silently truncated to the first CTX-2 tokens, so the
    model answered from a mutilated prompt and the client got HTTP 200 with junk. The engine now
    refuses, and the refusal has to reach the client as a 400 it can act on -- not a 500."""

    def test_context_exceeded_becomes_a_400_the_client_can_act_on(self):
        err = _engine_error(["CONTEXT_EXCEEDED", "8321", "4094"], "CONTEXT_EXCEEDED 8321 4094")
        self.assertIsInstance(err, APIError)
        self.assertEqual(err.status, 400)
        self.assertEqual(err.code, "context_length_exceeded")
        self.assertEqual(err.param, "messages")
        self.assertIn("4094", err.message)
        self.assertIn("8321", err.message)

    def test_other_engine_errors_stay_runtime_errors(self):
        for frame in (["SLOT_BUSY"], ["BAD_REQUEST"], []):
            err = _engine_error(frame, " ".join(frame) or "engine request failed")
            self.assertIsInstance(err, RuntimeError)
            self.assertNotIsInstance(err, APIError)

    def test_malformed_context_frame_does_not_crash_the_dispatcher(self):
        err = _engine_error(["CONTEXT_EXCEEDED"], "CONTEXT_EXCEEDED")
        self.assertIsInstance(err, APIError)
        self.assertEqual(err.status, 400)
class UnclosedToolCallTest(unittest.TestCase):
    """#401: the model opens <tool_call>, emits a well-formed call, then stops without the
    closing tag (budget ran out, or quantization mangled it). The strict regex needs both tags,
    so the client used to get zero tool_calls -- a total failure from a recoverable output."""

    NO_ARG_TOOL = ORDER_TOOL + [{"type": "function",
                                 "function": {"name": "list_orders", "parameters": {}}}]

    def _calls(self, reply, tools=ORDER_TOOL):
        return parse_tool_calls(reply, tools)

    def test_unclosed_box_is_recovered(self):
        content, calls = self._calls("<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"order_id": "A-1"})
        self.assertEqual(content, "")

    def test_mangled_closing_tag_is_recovered(self):
        _, calls = self._calls("<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>A-1</arg_value></tool_cal")
        self.assertEqual(len(calls), 1)
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"order_id": "A-1"})

    def test_leading_prose_is_kept_as_content(self):
        content, calls = self._calls("Let me check.\n<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Let me check.")

    def test_closed_call_followed_by_an_unclosed_one(self):
        _, calls = self._calls("<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>A-1</arg_value></tool_call>"
                               "<tool_call>lookup_order"
                               "<arg_key>order_id</arg_key><arg_value>B-2</arg_value>")
        self.assertEqual([json.loads(c["function"]["arguments"])["order_id"] for c in calls],
                         ["A-1", "B-2"])

    def test_bare_declared_name_recovers_a_zero_argument_call(self):
        _, calls = self._calls("<tool_call>list_orders", self.NO_ARG_TOOL)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "list_orders")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {})

    def test_prose_mentioning_the_marker_does_not_fabricate_a_call(self):
        content, calls = self._calls("To call a tool, write <tool_call> and then the name.")
        self.assertEqual(calls, [])
        self.assertIn("<tool_call>", content)

    def test_undeclared_name_without_arguments_is_not_recovered(self):
        _, calls = self._calls("<tool_call>drop_all_tables")
        self.assertEqual(calls, [])

    def test_well_formed_output_is_untouched(self):
        content, calls = self._calls("Done.<tool_call>lookup_order"
                                     "<arg_key>order_id</arg_key><arg_value>A-1</arg_value>"
                                     "</tool_call>")
        self.assertEqual(len(calls), 1)
        self.assertEqual(content, "Done.")


class ToolChoiceTest(unittest.TestCase):
    def test_none_does_not_offer_the_tools(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="none")
        self.assertNotIn("<tools>", prompt)

    def test_auto_offers_the_tools(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="auto")
        self.assertIn("<tools>", prompt)

    def test_required_instructs_the_model_to_call_one(self):
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=ORDER_TOOL,
                             tool_choice="required")
        self.assertIn("<tools>", prompt)
        self.assertIn("must call one of the functions", prompt)

    def test_named_function_restricts_to_that_function(self):
        tools = ORDER_TOOL + [{"type": "function", "function": {"name": "other", "parameters": {}}}]
        prompt = render_chat([{"role": "user", "content": "hi"}], tools=tools,
                             tool_choice={"type": "function", "function": {"name": "lookup_order"}})
        self.assertIn("must call the function `lookup_order`", prompt)
        self.assertNotIn('"other"', prompt)

    def test_rejects_unknown_string_and_unknown_function(self):
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tools": ORDER_TOOL, "tool_choice": "maybe"}, 128)
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tools": ORDER_TOOL,
                                "tool_choice": {"type": "function",
                                                "function": {"name": "nope"}}}, 128)

    def test_rejects_tool_choice_without_tools(self):
        with self.assertRaises(APIError):
            generation_options({"messages": [], "tool_choice": "required"}, 128)


class AllowedHostsTest(unittest.TestCase):
    """#597: the DNS-rebinding guard must accept operator-trusted reverse-proxy
    Host values, while still rejecting everything else by default."""

    def _make_server(self, allowed_hosts=()):
        server = APIServer(("127.0.0.1", 0), FakeEngine(), "test-model",
                           allowed_hosts=allowed_hosts)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def _get_models(self, port, host_header):
        # http.client lets us set an arbitrary Host header (urlopen forces its own).
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            conn.putrequest("GET", "/v1/models", skip_host=True)
            conn.putheader("Host", host_header)
            conn.endheaders()
            return conn.getresponse().status
        finally:
            conn.close()

    def test_allowlist_wiring_normalises_and_filters(self):
        server = self._make_server(allowed_hosts=("  Proxy.Example.TS.net ", "", "   "))
        self.assertEqual(server.allowed_hosts, ("proxy.example.ts.net",))

    def test_trusted_reverse_proxy_host_is_accepted(self):
        server = self._make_server(allowed_hosts=("proxy.example.ts.net",))
        port = server.server_port
        # trusted name (with a port suffix, case-insensitive) passes the guard
        self.assertEqual(self._get_models(port, "Proxy.Example.TS.net:8000"), 200)
        # loopback still works, unaffected by the allowlist
        self.assertEqual(self._get_models(port, "localhost"), 200)

    def test_untrusted_host_is_rejected_by_default(self):
        server = self._make_server()               # no allowlist: loopback/bind only
        self.assertEqual(self._get_models(server.server_port, "evil.example.com"), 403)

    def test_untrusted_host_still_rejected_with_allowlist(self):
        server = self._make_server(allowed_hosts=("proxy.example.ts.net",))
        self.assertEqual(self._get_models(server.server_port, "evil.example.com"), 403)


class ThinkingSplitUnitTest(unittest.TestCase):
    """#597 item 4: the GLM reasoning splitter, incl. mkelcb's cross-chunk cases."""

    def test_single_chunk(self):
        self.assertEqual(split_thinking_reply("abc</think>def"), ("abc", "def"))

    def test_close_tag_split_across_chunks(self):
        thinking, answer = [], []
        s = ThinkingStreamSplit(thinking.append, answer.append)
        s.feed("abc</thi"); s.feed("nk>def"); s.finish()
        self.assertEqual(("".join(thinking), "".join(answer)), ("abc", "def"))

    def test_open_tag_split_and_stray_open_marker(self):
        thinking, answer = [], []
        s = ThinkingStreamSplit(thinking.append, answer.append)
        s.feed("abc<thi"); s.feed("nk>def</think>ghi"); s.finish()
        self.assertEqual(("".join(thinking), "".join(answer)), ("abcdef", "ghi"))

    def test_thinking_disabled_is_all_answer(self):
        # initial_thinking=False: a pure answer with no markers must not be filed as reasoning
        self.assertEqual(split_thinking_reply("plain answer", enable_thinking=False),
                         ("", "plain answer"))

    def test_missing_close_tag_surfaces_reasoning(self):
        self.assertEqual(split_thinking_reply("thought with no end"),
                         ("thought with no end", ""))


class _ChunkEngine(FakeEngine):
    """Engine that emits a caller-supplied chunk sequence, to exercise the streaming
    reasoning splitter across arbitrary chunk boundaries."""
    def __init__(self, chunks):
        super().__init__()
        self.chunks = list(chunks)

    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        self.calls.append((prompt, maximum, temperature, top_p, cache_slot, grammar))
        if on_accept is not None:                 # simulate the engine's ACCEPT frame (#597)
            on_accept({"prompt_tokens": 7})
        for chunk in self.chunks:
            on_text(chunk)
            if stopped and stopped():
                self.stop_requests += 1
                break
        return {"prompt_tokens": 7, "completion_tokens": len(self.chunks), "length_limited": False}


class GlmReasoningStreamTest(unittest.TestCase):
    """#597 item 4 end-to-end: GLM reasoning streams as reasoning_content, the answer as
    content, no <think>/</think> leaks, cross-chunk-safe, and reasoning never contaminates
    the tool-call buffer."""

    def _server(self, chunks):
        server = APIServer(("127.0.0.1", 0), _ChunkEngine(chunks), "test-model")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return f"http://127.0.0.1:{server.server_port}"

    def _post(self, base, body):
        req = Request(base + "/v1/chat/completions",
                      data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            return response.read().decode()

    def _deltas(self, raw):
        reasoning, content, tool_calls = [], [], []
        for line in raw.splitlines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            for choice in json.loads(line[6:])["choices"]:
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_calls.extend(delta["tool_calls"])
        return "".join(reasoning), "".join(content), tool_calls

    def test_streaming_splits_reasoning_from_answer(self):
        base = self._server(["I think ", "42", "</think>", "The answer ", "is 42"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "2+2?"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "I think 42")
        self.assertEqual(content, "The answer is 42")
        self.assertNotIn("<think>", raw)
        self.assertNotIn("</think>", raw)

    def test_streaming_close_tag_split_across_chunks(self):
        base = self._server(["reason</thi", "nk>ans", "wer"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "x"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "reason")
        self.assertEqual(content, "answer")
        self.assertNotIn("think>", raw)

    def test_streaming_thinking_off_is_all_content(self):
        base = self._server(["Just ", "the answer"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": False,
                                "messages": [{"role": "user", "content": "x"}]})
        reasoning, content, _ = self._deltas(raw)
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "Just the answer")

    def test_streaming_reasoning_stays_out_of_tool_call(self):
        base = self._server(["deciding to call", "</think>",
                             "<tool_call>get_weather<arg_key>city</arg_key>"
                             "<arg_value>Paris</arg_value></tool_call>"])
        raw = self._post(base, {"model": "test-model", "stream": True, "enable_thinking": True,
                                "messages": [{"role": "user", "content": "weather?"}],
                                "tools": [{"type": "function", "function": {
                                    "name": "get_weather", "parameters": {"type": "object",
                                    "properties": {"city": {"type": "string"}}}}}]})
        reasoning, content, tool_calls = self._deltas(raw)
        self.assertEqual(reasoning, "deciding to call")
        self.assertTrue(tool_calls, "expected a parsed tool call")
        args = tool_calls[0]["function"]["arguments"]
        self.assertIn("Paris", args)
        self.assertNotIn("deciding", args)     # reasoning must not leak into the tool arguments
        self.assertNotIn("deciding", content)  # nor into the visible answer

    def test_nonstreaming_splits_reasoning(self):
        base = self._server(["mulling ", "it over", "</think>", "final ", "answer"])
        req = Request(base + "/v1/chat/completions",
                      data=json.dumps({"model": "test-model", "enable_thinking": True,
                        "messages": [{"role": "user", "content": "x"}]}).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as response:
            body = json.load(response)
        message = body["choices"][0]["message"]
        self.assertEqual(message["reasoning_content"], "mulling it over")
        self.assertEqual(message["content"], "final answer")


class AcceptFrameTest(unittest.TestCase):
    """#597 item 6: the engine's ACCEPT frame gates the HTTP commit, and its invariants."""

    def _engine(self, respond):
        process = FakeProcess(respond)
        with patch("openai_server.subprocess.Popen", return_value=process):
            return Engine("glm", "model")

    def test_accept_fires_before_any_data(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + rid + b" 42\n")
            process.stdout.feed(b"DATA " + rid + b" 3\nHi!\n")
            process.stdout.feed(b"DONE " + rid + b" STAT 1 2.5 0 1.0 4 0\n")
        engine = self._engine(respond)
        seen = []
        engine.generate("hi", 8, 0.7, 0.9, lambda t: seen.append(("text", t)),
                        on_accept=lambda info: seen.append(("accept", info)))
        engine.close()
        self.assertEqual(seen[0], ("accept", {"prompt_tokens": 42}))
        self.assertEqual("".join(t for k, t in seen if k == "text"), "Hi!")

    def test_error_before_accept_never_commits(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ERROR " + rid + b" CONTEXT_EXCEEDED 5000 4094\n")
        engine = self._engine(respond)
        accepts = []
        with self.assertRaises(APIError) as caught:
            engine.generate("hi", 8, 0.7, 0.9, lambda _: None,
                            on_accept=lambda info: accepts.append(info))
        engine.close()
        self.assertEqual(caught.exception.status, 400)
        self.assertEqual(caught.exception.code, "context_length_exceeded")
        self.assertEqual(accepts, [])          # nothing committed -> HTTP layer can send a clean 400

    def test_data_before_accept_still_commits_for_old_engine(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"DATA " + rid + b" 3\nHey\n")
            process.stdout.feed(b"DONE " + rid + b" STAT 1 2.5 0 1.0 4 0\n")
        engine = self._engine(respond)
        accepts, chunks = [], []
        engine.generate("hi", 8, 0.7, 0.9, chunks.append,
                        on_accept=lambda info: accepts.append(info))
        engine.close()
        self.assertEqual("".join(chunks), "Hey")
        self.assertEqual(len(accepts), 1)      # first DATA implies acceptance (no ACCEPT frame)
        self.assertIsNone(accepts[0]["prompt_tokens"])

    def test_duplicate_accept_is_a_protocol_error(self):
        def respond(process, frame):
            rid = frame.split()[1]
            process.stdout.feed(b"ACCEPT " + rid + b" 10\n")
            process.stdout.feed(b"ACCEPT " + rid + b" 10\n")
        engine = self._engine(respond)
        with self.assertRaisesRegex(RuntimeError, "duplicate ACCEPT"):
            engine.generate("hi", 8, 0.7, 0.9, lambda _: None, on_accept=lambda _: None)
        engine.close()


class _ContextExceededEngine(FakeEngine):
    """Engine that rejects the prompt before ACCEPT — on_accept is never called."""
    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        raise APIError(400, "This model's maximum context length is 4094 tokens.",
                       "messages", "context_length_exceeded")


class StreamingContextRejectTest(unittest.TestCase):
    """#597 item 6: an oversized prompt on a *streaming* request must return a clean HTTP 400,
    not a committed 200 SSE stream that only later discovers the overflow."""

    def setUp(self):
        self.server = APIServer(("127.0.0.1", 0), _ContextExceededEngine(), "test-model")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.scheduler.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_streaming_context_exceeded_is_clean_400(self):
        req = Request(self.base + "/v1/chat/completions",
                      data=json.dumps({"model": "test-model", "stream": True,
                        "messages": [{"role": "user", "content": "x" * 100}]}).encode(),
                      headers={"Content-Type": "application/json"})
        with self.assertRaises(HTTPError) as caught:
            urlopen(req, timeout=3)
        self.assertEqual(caught.exception.code, 400)          # a real 400, not a 200 stream
        body = json.load(caught.exception)
        self.assertEqual(body["error"]["code"], "context_length_exceeded")
        self.assertEqual(body["error"]["param"], "messages")


class _ExplodingEngine(FakeEngine):
    """ACCEPTs the prompt (committing the streaming 200), then dies mid-generation."""
    def generate(self, prompt, maximum, temperature, top_p, on_text, cache_slot=0,
                 cancelled=None, grammar=None, stopped=None, on_accept=None):
        if on_accept is not None:
            on_accept({"prompt_tokens": 7})
        on_text("partial")
        raise RuntimeError("engine died mid-stream")


class KeepAliveFramingTest(unittest.TestCase):
    """#597 item 3: HTTP/1.1 persistence must not desynchronise.

    The report was `Bad request syntax ('{...json body...}POST /v1/chat/completions HTTP/1.1')`
    -- a previous body being parsed as the next request line. Two independent causes: an early
    rejection returning before the body is read, and a streaming 200 that neither announced
    close-framing nor stopped offering the socket for reuse when generation failed."""

    CHAT = {"model": "test-model", "messages": [{"role": "user", "content": "x"}]}

    def _server(self, engine=None, **kw):
        server = APIServer(("127.0.0.1", 0), engine or FakeEngine(), "test-model", **kw)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.scheduler.close)
        return server

    def _conn(self, server):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=3)
        self.addCleanup(conn.close)
        return conn

    def _post(self, conn, body=None, headers=None, path="/v1/chat/completions"):
        payload = json.dumps(self.CHAT if body is None else body)
        head = {"Content-Type": "application/json"}
        head.update(headers or {})
        conn.request("POST", path, body=payload, headers=head)
        response = conn.getresponse()
        return response.status, response.read()

    def _raw(self, server, request_bytes, read=4096):
        """Byte-level exchange, for assertions about framing that a client library hides."""
        sock = socket.create_connection(("127.0.0.1", server.server_port), timeout=3)
        self.addCleanup(sock.close)
        sock.sendall(request_bytes)
        chunks = []
        try:
            while True:
                chunk = sock.recv(read)
                if not chunk:
                    break                      # server closed: the SSE message boundary
                chunks.append(chunk)
        except socket.timeout:
            chunks.append(b"<STILL-OPEN>")     # no EOF: the connection was left reusable
        return b"".join(chunks).decode("utf-8", "replace")

    def _request_bytes(self, body, host="127.0.0.1"):
        payload = json.dumps(body)
        return (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}\r\n"
                f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                f"\r\n{payload}").encode()

    # --- an early rejection must not leave its body in the socket -------------------------

    def test_rejected_host_does_not_desync_the_next_request(self):
        server = self._server()
        conn = self._conn(server)
        status, _ = self._post(conn, headers={"Host": "evil.example.com"})
        self.assertEqual(status, 403)
        status, payload = self._post(conn)                  # same connection, valid request
        self.assertEqual(status, 200, "the 403's unread body desynchronised the connection")
        self.assertEqual(json.loads(payload)["object"], "chat.completion")

    def test_rejected_auth_does_not_desync_the_next_request(self):
        server = self._server(api_key="secret")
        conn = self._conn(server)
        status, _ = self._post(conn)                        # no Authorization header
        self.assertEqual(status, 401)
        status, payload = self._post(conn, headers={"Authorization": "Bearer secret"})
        self.assertEqual(status, 200, "the 401's unread body desynchronised the connection")
        self.assertEqual(json.loads(payload)["object"], "chat.completion")

    def test_unknown_model_does_not_desync_the_next_request(self):
        server = self._server()
        conn = self._conn(server)
        status, _ = self._post(conn, body=dict(self.CHAT, model="nope"))
        self.assertEqual(status, 404)
        status, _ = self._post(conn)
        self.assertEqual(status, 200)

    def test_each_body_is_consumed_exactly_once_across_reused_requests(self):
        """The engine sees one prompt per request, with no body bytes bleeding between them."""
        engine = FakeEngine()
        server = self._server(engine)
        conn = self._conn(server)
        for index in range(4):
            body = {"model": "test-model",
                    "messages": [{"role": "user", "content": f"question-{index}"}]}
            status, _ = self._post(conn, body=body)
            self.assertEqual(status, 200)
        self.assertEqual(len(engine.calls), 4)
        for index, call in enumerate(engine.calls):
            self.assertIn(f"question-{index}", call[0])
            self.assertNotIn("question-", call[0].split(f"question-{index}")[1],
                             "a later body leaked into an earlier prompt")

    def test_interleaved_rejections_and_successes_stay_in_sync(self):
        server = self._server(api_key="secret")
        conn = self._conn(server)
        good = {"Authorization": "Bearer secret"}
        for _ in range(3):
            self.assertEqual(self._post(conn)[0], 401)
            self.assertEqual(self._post(conn, headers={"Host": "evil.example.com", **good})[0], 403)
            self.assertEqual(self._post(conn, headers=good)[0], 200)

    # --- bodies we refuse to swallow must close rather than desynchronise -----------------

    def test_oversized_content_length_closes_instead_of_desyncing(self):
        server = self._server()
        raw = self._raw(server, (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 b"Content-Type: application/json\r\n"
                                 b"Content-Length: 99999999\r\n\r\n{}"))
        self.assertIn(" 400 ", raw.splitlines()[0])
        self.assertNotIn("<STILL-OPEN>", raw,
                         "an over-limit body must close the connection, not keep it alive")

    def test_unparseable_content_length_closes_instead_of_desyncing(self):
        server = self._server()
        raw = self._raw(server, (b"POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 b"Content-Type: application/json\r\n"
                                 b"Content-Length: abc\r\n\r\n{}"))
        self.assertIn("400", raw.splitlines()[0])
        self.assertNotIn("<STILL-OPEN>", raw)

    # --- a streaming 200 is close-framed, and says so ------------------------------------

    def test_streaming_response_announces_close_framing(self):
        server = self._server()
        raw = self._raw(server, self._request_bytes(dict(self.CHAT, stream=True)))
        headers = raw.split("\r\n\r\n", 1)[0].lower()
        self.assertIn("content-type: text/event-stream", headers)
        self.assertIn("connection: close", headers,
                      "SSE has no Content-Length, so the close IS the boundary and must be declared")
        self.assertIn("data: [DONE]", raw)
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_anthropic_stream_announces_close_framing(self):
        server = self._server()
        payload = json.dumps({"model": "test-model", "stream": True, "max_tokens": 16,
                              "messages": [{"role": "user", "content": "x"}]})
        raw = self._raw(server, (f"POST /v1/messages HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                 f"Content-Type: application/json\r\n"
                                 f"Content-Length: {len(payload)}\r\n\r\n{payload}").encode())
        self.assertIn("connection: close", raw.split("\r\n\r\n", 1)[0].lower())
        self.assertIn("event: message_stop", raw)
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_engine_failure_after_commit_does_not_splice_a_second_response(self):
        """Once the 200 is out, a 500 status line would land inside the event stream."""
        server = self._server(_ExplodingEngine())
        raw = self._raw(server, self._request_bytes(dict(self.CHAT, stream=True)))
        self.assertEqual(raw.count("HTTP/1."), 1,
                         "a second HTTP response was spliced into the committed SSE stream")
        self.assertIn("partial", raw)            # the events sent before the failure survive
        self.assertNotIn("<STILL-OPEN>", raw)

    def test_non_streaming_response_still_reuses_the_connection(self):
        """The fix must not turn every response into a close: plain JSON stays persistent."""
        server = self._server()
        conn = self._conn(server)
        self.assertEqual(self._post(conn)[0], 200)
        self.assertEqual(self._post(conn)[0], 200)
        self.assertIsNotNone(conn.sock, "the JSON path should keep the connection open")


class ConversationCacheSlotTest(unittest.TestCase):
    """#634 Defect 1: a conversation must map to one stable KV slot across its turns."""

    def _conv(self, *user_and_assistant_turns, system="you are a helpful assistant"):
        messages = [{"role": "system", "content": system}]
        for i, text in enumerate(user_and_assistant_turns):
            messages.append({"role": "user" if i % 2 == 0 else "assistant", "content": text})
        return messages

    def test_single_slot_is_always_zero(self):
        self.assertEqual(conversation_cache_slot(self._conv("hi"), 1), 0)

    def test_empty_or_bad_input_is_zero(self):
        self.assertEqual(conversation_cache_slot(None, 8), 0)
        self.assertEqual(conversation_cache_slot([], 8), 0)

    def test_slot_is_in_range(self):
        for kv in (2, 3, 8, 16):
            slot = conversation_cache_slot(self._conv("solve x"), kv)
            self.assertTrue(0 <= slot < kv, f"slot {slot} out of range for kv={kv}")

    def test_stable_across_turns_of_one_conversation(self):
        # The engine caches the prefix; every turn of the same conversation must return
        # the same slot so it lands on its warm KV instead of re-prefilling.
        first_turn = self._conv("1+1=")
        second_turn = self._conv("1+1=", "2", "and 2+2?")
        third_turn = self._conv("1+1=", "2", "and 2+2?", "4", "thanks")
        base = conversation_cache_slot(first_turn, 8)
        self.assertEqual(conversation_cache_slot(second_turn, 8), base)
        self.assertEqual(conversation_cache_slot(third_turn, 8), base)

    def test_distinct_conversations_can_differ(self):
        # Not a guarantee for any single pair (hashing collides sometimes), but across a
        # spread of openings we must see more than one slot used, i.e. not everything on 0.
        slots = {conversation_cache_slot(self._conv(f"task number {i}"), 8) for i in range(40)}
        self.assertGreater(len(slots), 1)

    def test_deterministic(self):
        conv = self._conv("same question", "same answer", "again")
        self.assertEqual(conversation_cache_slot(conv, 8), conversation_cache_slot(conv, 8))


class ConnectionLimitTest(unittest.TestCase):
    """Bounds on the accept loop, which nothing bounded before.

    ThreadingHTTPServer spawns a thread per connection with no ceiling, and
    `timeout` is per socket operation, so it restarts on every byte: a client
    dripping one byte kept a thread and a slot forever. Threads at 8 MiB of
    stack each made that a memory-exhaustion DoS, reachable before any Host
    check or auth.
    """

    def setUp(self):
        self.engine = FakeEngine()
        APIServer.MAX_CONNECTIONS = 6
        APIServer.MAX_CONNECTIONS_PER_IP = 3
        APIHandler.READ_DEADLINE = 2
        self.addCleanup(setattr, APIServer, "MAX_CONNECTIONS", 64)
        self.addCleanup(setattr, APIServer, "MAX_CONNECTIONS_PER_IP", 8)
        self.addCleanup(setattr, APIHandler, "READ_DEADLINE", 30)
        self.server = APIServer(("127.0.0.1", 0), self.engine, "m", None, 16, kv_slots=1)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.server.scheduler.close)
        self.port = self.server.server_port
        self.held = []
        self.addCleanup(self._drop_all)

    def _drop_all(self):
        for sock in self.held:
            try:
                sock.close()
            except OSError:
                pass

    def _dribble(self, count):
        """Open connections that send a partial request line and never finish."""
        for _ in range(count):
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), 2)
                sock.settimeout(2)
                sock.sendall(b"GET /health HTTP/1.1\r\n")
                self.held.append(sock)
            except OSError:
                pass
        time.sleep(0.4)

    def test_slowloris_cannot_grow_threads_without_bound(self):
        self._dribble(40)
        self.assertLessEqual(self.server._conn_live, APIServer.MAX_CONNECTIONS)

    def test_one_address_cannot_take_every_slot(self):
        """A global cap alone only turns exhaustion into starvation."""
        self._dribble(40)
        self.assertLessEqual(self.server._conn_live,
                             APIServer.MAX_CONNECTIONS_PER_IP)
        self.assertLess(self.server._conn_live, APIServer.MAX_CONNECTIONS,
                        "one source filled the server cap")

    def test_dripping_connections_are_reclaimed(self):
        """The cumulative deadline, which the per-operation timeout is not."""
        self._dribble(10)
        self.assertGreater(self.server._conn_live, 0)
        time.sleep(APIHandler.READ_DEADLINE + 1.5)
        self.assertEqual(self.server._conn_live, 0,
                         "dripping connections were never reclaimed")


class ReasoningEffortTest(unittest.TestCase):
    """render_chat mapped every level except "high" onto Max (#809).

    The endpoint accepts none/minimal/low/medium/high/xhigh. Four of the five
    that enable thinking rendered Max, so a client asking for `minimal` got
    more reasoning than one asking for `high` -- and on a single machine
    unrequested reasoning spends the token budget before the answer starts.
    """

    MESSAGES = [{"role": "user", "content": "hi"}]

    def effort(self, level):
        import re
        text = render_chat(self.MESSAGES, enable_thinking=True,
                           reasoning_effort=level)
        found = re.search(r"Reasoning Effort: (\w+)", text)
        return found.group(1) if found else None

    def test_levels_are_distinct_and_ordered(self):
        rendered = [self.effort(l) for l in
                    ("minimal", "low", "medium", "high", "xhigh")]
        rank = {"Low": 0, "Medium": 1, "High": 2, "Max": 3}
        scores = [rank[r] for r in rendered]
        self.assertEqual(scores, sorted(scores), rendered)
        self.assertLess(scores[0], scores[-1],
                        "minimal and xhigh render the same effort")

    def test_minimal_is_not_max(self):
        """The reported symptom, pinned on its own."""
        self.assertNotEqual(self.effort("minimal"), "Max")
        self.assertNotEqual(self.effort("low"), "Max")
        self.assertNotEqual(self.effort("medium"), "Max")

    def test_thinking_off_emits_no_effort_line(self):
        text = render_chat(self.MESSAGES, enable_thinking=False,
                           reasoning_effort="xhigh")
        self.assertNotIn("Reasoning Effort", text)


if __name__ == "__main__":
    unittest.main()
