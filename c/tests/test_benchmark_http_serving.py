"""HTTP benchmark contract tests; no model, GPU or third-party packages."""
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from tools import benchmark_http_serving as bench
from openai_server import APIServer


def event(value):
    return "data: " + (value if isinstance(value, str) else json.dumps(value)) + "\n\n"


def stream(delta=None, usage=3, finish=True, done=True):
    text = event({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    text += event({"choices": [{"index": 0, "delta": delta or {}}]})
    if finish:
        text += event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    if usage is not None:
        text += event({"choices": [], "usage": {"completion_tokens": usage}})
    if done:
        text += event("[DONE]")
    return text.encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock:
            self.server.payloads.append(body)
            self.server.active += 1
            self.server.peak = max(self.server.peak, self.server.active)
        try:
            if self.server.barrier:
                self.server.barrier.wait(timeout=5)
            time.sleep(self.server.delay)
            self.send_response(self.server.status)
            self.send_header("Content-Type", self.server.content_type)
            if self.server.status == 302:
                self.send_header("Location", "/must-not-follow")
            self.end_headers()
            self.wfile.write(self.server.body)
        finally:
            with self.server.lock:
                self.server.active -= 1


class BenchmarkTest(unittest.TestCase):
    def setUp(self):
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.lock = threading.Lock()
        self.server.payloads = []
        self.server.active = self.server.peak = 0
        self.server.status = 200
        self.server.content_type = "text/event-stream"
        self.server.body = stream({"content": "hello"})
        self.server.barrier = None
        self.server.delay = 0
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01})
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions"
        self.workload = [{"messages": [{"role": "user", "content": "hello"}]}]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def request(self):
        return bench.request_one(self.url, self.workload[0], "", 2, 0, time.perf_counter())

    def test_success_and_first_output(self):
        for delta in ({"content": "a"}, {"reasoning_content": "b"},
                      {"reasoning": "c"}, {"tool_calls": [{"function": {"arguments": "{}"}}]}):
            with self.subTest(delta=delta):
                self.server.body = stream(delta)
                row = self.request()
                self.assertTrue(row["success"])
                self.assertEqual(row["completion_tokens"], 3)
                self.assertLessEqual(row["first_output_seconds"], row["duration_seconds"])

    def test_legacy_function_call_output_and_validation(self):
        for function, valid, output in (({"name": "lookup"}, True, True),
                                        ({"arguments": "{}"}, True, True),
                                        ({"arguments": ""}, True, False),
                                        ("lookup", False, False),
                                        ({"arguments": 42}, False, False)):
            with self.subTest(function=function):
                chunk = {"choices": [{"index": 0, "delta": {"function_call": function},
                                      "finish_reason": "function_call"}]}
                self.server.body = (event(chunk) + event("[DONE]")).encode()
                row = self.request()
                self.assertEqual(row["success"], valid)
                if valid:
                    self.assertEqual(row["first_output_seconds"] is not None, output)
                    summary = bench.summarize([row], 1, slo_first_output=5)
                    self.assertEqual(summary["latency_slo"]["requests_met"], int(output))

    def test_empty_output_is_not_first_output(self):
        self.server.body = stream({"content": "", "tool_calls": [{"id": "id", "function": {}}]}, usage=0)
        row = self.request()
        self.assertTrue(row["success"])
        self.assertIsNone(row["first_output_seconds"])
        self.assertEqual(row["completion_tokens"], 0)

    def test_failures_are_not_successes(self):
        bodies = [stream(done=False), stream(finish=False), b"data: {bad}\n\n",
                  event({"error": {"message": "private diagnostic"}}).encode(),
                  stream(usage=-1), stream(usage=True), b"data: [DONE]"]
        for body in bodies:
            with self.subTest(body=body):
                self.server.body = body
                row = self.request()
                self.assertFalse(row["success"])
                self.assertIsNotNone(row["error"])
                self.assertNotIn("private diagnostic", json.dumps(row))

    def test_invalid_choices_do_not_count_as_success(self):
        choices = [
            [{"index": 0, "delta": {"content": 123}, "finish_reason": "stop"}],
            [{"index": 0, "delta": {"content": "ok", "reasoning": []}, "finish_reason": "stop"}],
            [{"index": 0, "delta": {}, "finish_reason": False}],
            [{"index": 0, "delta": {}, "finish_reason": ""}],
            [{"index": 0, "delta": {}, "finish_reason": "error"}],
            [{"index": False, "delta": {}, "finish_reason": "stop"}],
            [{"delta": {}, "finish_reason": "stop"}],
            [{"index": 0, "delta": {}, "finish_reason": "stop"}] * 2,
            [{"index": 0, "delta": {"tool_calls": [{"function": {"arguments": 7}}]}, "finish_reason": "tool_calls"}],
        ]
        for malformed in choices:
            with self.subTest(choices=malformed):
                self.server.body = (event({"choices": malformed}) + event("[DONE]")).encode()
                row = self.request()
                self.assertFalse(row["success"], row)
                self.assertEqual(bench.summarize([row], 1, slo_duration=1)["latency_slo"]["requests_met"], 0)

    def test_choice_after_finish_is_not_successful(self):
        terminal = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        for delta, finish in (({"content": "late output"}, None), ({}, "length")):
            with self.subTest(delta=delta, finish=finish):
                extra = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.server.body = (event(terminal) + event(extra)
                                    + event({"choices": [], "usage": {"completion_tokens": 999}})
                                    + event("[DONE]")).encode()
                row = self.request()
                self.assertFalse(row["success"])
                self.assertEqual(row["error"], "choice_after_finish")
                self.assertEqual(row["finish_reason"], "stop")
                summary = bench.summarize([row], 1, slo_duration=10)
                self.assertEqual(summary["reported_successful_completion_tokens"], 0)
                self.assertEqual(summary["latency_slo"]["requests_met"], 0)

    def test_supported_finish_reasons_and_null_delta(self):
        for finish in ("stop", "length", "tool_calls", "function_call", "content_filter"):
            with self.subTest(finish=finish):
                self.server.body = (event({"choices": [{"index": 0, "delta": None,
                                                       "finish_reason": finish}]}) + event("[DONE]")).encode()
                row = self.request()
                self.assertTrue(row["success"], row)
                self.assertIsNone(row["first_output_seconds"])

    def test_http_errors_and_redirects(self):
        for status in (429, 500, 302):
            self.server.status = status
            row = self.request()
            self.assertFalse(row["success"])
            self.assertEqual(row["http_status"], status)
        self.assertEqual(len(self.server.payloads), 3)

    def test_content_type(self):
        self.server.content_type = "application/json"
        self.assertFalse(self.request()["success"])

    def test_missing_usage_disables_token_rate(self):
        self.server.body = stream({"content": "a"}, usage=None)
        row = self.request()
        summary = bench.summarize([row], 1)
        self.assertEqual(summary["succeeded"], 1)
        self.assertEqual(summary["successful_requests_with_usage"], 0)
        self.assertIsNone(summary["successful_completion_tokens_per_second"])

    def test_partial_failure_accounting(self):
        success = self.request()
        failure = dict(success, success=False, completion_tokens=999)
        summary = bench.summarize([success, failure], 2)
        self.assertEqual(summary["failure_rate"], .5)
        self.assertEqual(summary["successful_completion_tokens_per_second"], 1.5)
        self.assertEqual(summary["successful_duration_seconds"]["count"], 1)
        self.assertIsNone(bench.summarize([failure], 1)["successful_completion_tokens_per_second"])

    def test_concurrency_and_payload(self):
        self.server.barrier = threading.Barrier(2)
        rows, summary = bench.run(self.url, self.workload, "test-model", 2, 4, 8, 0, "", 2)
        self.assertEqual(summary["succeeded"], 4)
        self.assertEqual(self.server.peak, 2)
        self.assertEqual([r["index"] for r in rows], list(range(4)))
        for body in self.server.payloads:
            self.assertEqual(body, dict(self.workload[0], model="test-model", stream=True,
                                       stream_options={"include_usage": True}, max_tokens=8,
                                       temperature=0, n=1))

    def test_fixed_rate_keeps_client_wait_visible(self):
        self.server.delay = .04
        rows, summary = bench.run(self.url, self.workload, "fixture", 1, 3, 8, 0, "", 2,
                                  request_rate=1000)
        self.assertEqual(summary["succeeded"], 3)
        self.assertEqual(self.server.peak, 1)
        self.assertEqual([r["scheduled_seconds"] for r in rows], [0, .001, .002])
        self.assertGreater(rows[-1]["dispatch_delay_seconds"], .05)
        for row in rows:
            self.assertAlmostEqual(row["arrival_duration_seconds"],
                                   row["duration_seconds"] + row["dispatch_delay_seconds"])
            self.assertAlmostEqual(row["arrival_first_output_seconds"],
                                   row["first_output_seconds"] + row["dispatch_delay_seconds"])
        self.assertEqual(summary["arrival_timing"]["dispatch_delay_seconds"]["count"], 3)

    def test_cli_warmup_is_separate_and_failure_skips_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "prompts.jsonl"
            output = Path(directory) / "report.json"
            prompts = self.workload + [{"messages": [{"role": "user", "content": "second"}]}]
            workload.write_text("".join(json.dumps(p) + "\n" for p in prompts))
            command = [sys.executable, bench.__file__, "--base-url", self.url.rsplit("/", 1)[0],
                       "--model", "fixture", "--workload", str(workload), "--output", str(output),
                       "--warmup-requests", "3", "--concurrency", "2", "--repeats", "2",
                       "--request-rate", "100", "--slo-duration", "5"]
            for status, expected_exit in ((200, 0), (503, 1)):
                with self.subTest(status=status):
                    self.server.status = status
                    self.server.payloads.clear()
                    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                    self.assertEqual(completed.returncode, expected_exit, completed.stderr)
                    report = json.loads(output.read_text())
                    warmup = report["warmup"]
                    self.assertEqual(warmup["summary"]["requests"], 3)
                    self.assertIsNone(warmup["summary"]["arrival_timing"])
                    self.assertIsNone(warmup["summary"]["latency_slo"])
                    self.assertLessEqual(self.server.peak, 2)
                    self.assertCountEqual([p["messages"] for p in self.server.payloads[:3]],
                                          [prompts[i % 2]["messages"] for i in range(3)])
                    if expected_exit:
                        self.assertEqual(report["status"], "warmup_failed")
                        self.assertIsNone(report["summary"])
                        self.assertEqual(report["requests"], [])
                        self.assertEqual(len(self.server.payloads), 3)
                    else:
                        self.assertEqual(report["status"], "measured")
                        self.assertEqual(len(self.server.payloads), 7)
                        self.assertEqual(report["summary"]["requests"], 4)
                        self.assertEqual(report["summary"]["reported_successful_completion_tokens"], 12)
                        self.assertEqual(report["summary"]["latency_slo"]["requests_met"], 4)
                        self.assertEqual([r["scheduled_seconds"] for r in report["requests"]],
                                         [0, .01, .02, .03])
            completed = subprocess.run(command + ["--warmup-requests", "-1"],
                                       capture_output=True, text=True, timeout=10)
            self.assertEqual(completed.returncode, 2)

    def test_cli_report_and_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "prompts.jsonl"
            output = Path(directory) / "report.json"
            workload.write_text(json.dumps(self.workload[0]) + "\n", encoding="utf-8")
            command = [sys.executable, bench.__file__, "--base-url", self.url.rsplit("/", 1)[0],
                       "--model", "fixture", "--workload", str(workload), "--output", str(output),
                       "--slo-first-output", "5", "--slo-duration", "5", "--request-rate", "10"]
            for status, exit_code in ((200, 0), (503, 1)):
                self.server.status = status
                completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
                self.assertEqual(completed.returncode, exit_code, completed.stderr)
                report = json.loads(output.read_text())
                self.assertEqual(report["summary"]["succeeded"], 1 - exit_code)
                self.assertEqual(len(report["config"]["workload_sha256"]), 64)
                self.assertNotIn("messages", report["config"])
                self.assertEqual(report["config"]["slo_duration_seconds"], 5)
                self.assertEqual(report["config"]["load_model"], "fixed_rate")
                self.assertEqual(report["config"]["request_rate"], 10)
                self.assertEqual(report["summary"]["latency_slo"]["timing_basis"], "scheduled_arrival")
                self.assertEqual(report["summary"]["latency_slo"]["requests_met"], 1 - exit_code)


    def test_poisson_cli_reports_seed_and_requires_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            output = Path(directory) / "report.json"
            workload.write_text('{"messages":[{"role":"user","content":"hello"}]}\n')
            argv = [bench.__file__, "--base-url", self.url.removesuffix("/chat/completions"),
                    "--model", "fixture", "--workload", str(workload), "--output", str(output),
                    "--arrival-distribution", "poisson", "--seed", "42"]
            with patch.object(sys, "argv", argv), patch.object(sys, "stderr", io.StringIO()), \
                 self.assertRaises(SystemExit) as error:
                bench.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())
            with patch.object(sys, "argv", argv + ["--request-rate", "1000"]), \
                 patch.object(sys, "stdout", io.StringIO()):
                self.assertEqual(bench.main(), 0)
            report = json.loads(output.read_text())
            self.assertEqual(report["config"]["load_model"], "poisson")
            self.assertEqual(report["config"]["arrival_distribution"], "poisson")
            self.assertEqual(report["config"]["arrival_seed"], 42)
            self.assertEqual(report["requests"][0]["scheduled_seconds"], 0)


class ColibriIntegrationTest(unittest.TestCase):
    def test_real_gateway_with_fake_engine(self):
        class Engine:
            fail = False

            def generate(self, prompt, maximum, temperature, top_p, on_text,
                         cache_slot=0, cancelled=None, **kwargs):
                kwargs["on_accept"]({"prompt_tokens": 7})
                on_text("Hello")
                if self.fail:
                    raise RuntimeError("private engine diagnostic")
                return {"prompt_tokens": 7, "completion_tokens": 1, "length_limited": False}

        server = APIServer(("127.0.0.1", 0), Engine(), "fixture", api_key="test-secret")
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            rows, summary = bench.run(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                [{"messages": [{"role": "user", "content": "Hi"}]}],
                "fixture", 1, 2, 8, 0, "test-secret", 2)
            self.assertEqual(summary["succeeded"], 2, rows)
            self.assertEqual(summary["reported_successful_completion_tokens"], 2)
            self.assertEqual(summary["successful_first_output_seconds"]["count"], 2)
            self.assertNotIn("test-secret", json.dumps(rows))
            server.engine.fail = True
            failed = bench.request_one(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                dict(messages=[{"role": "user", "content": "Hi"}], model="fixture", stream=True),
                "test-secret", 2, 2, time.perf_counter())
            self.assertFalse(failed["success"], failed)
            self.assertIsNotNone(failed["first_output_seconds"])
            self.assertIsNone(failed["finish_reason"])
            self.assertNotIn("private engine diagnostic", json.dumps(failed))
            summary = bench.summarize(rows + [failed], 1, slo_duration=5)
            self.assertEqual((summary["succeeded"], summary["failed"]), (2, 1))
            self.assertEqual(summary["reported_successful_completion_tokens"], 2)
            self.assertEqual(summary["latency_slo"]["requests_met"], 2)
            server.engine.fail = False
            recovered, _ = bench.run(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                [{"messages": [{"role": "user", "content": "Hi again"}]}],
                "fixture", 1, 1, 8, 0, "test-secret", 2)
            self.assertTrue(recovered[0]["success"], recovered)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()



class ParsingTest(unittest.TestCase):
    def test_sse_multiline_comments_crlf_and_partial_eof(self):
        raw = b': ping\r\nevent: message\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: truncated'
        self.assertEqual(list(bench.sse_events(io.BytesIO(raw))), ['{"choices":\n[]}'])

    def test_endpoint(self):
        self.assertEqual(bench.endpoint("http://localhost:8000/v1/"), "http://localhost:8000/v1/chat/completions")
        for url in ("file:///v1", "http://key@host/v1", "http://host/v1?key=secret", "http://host/v1#x"):
            with self.assertRaises(ValueError):
                bench.endpoint(url)

    def test_workload_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.jsonl"
            for text in ('', '{}', '{"messages":[]}', '{"messages":[{"role":"user","content":3}]}',
                         '{"messages":[{"role":"user","content":"ok"}],"temperature":1}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    bench.load_workload(path)

    def test_latency_slo_counts_all_attempts_in_denominator(self):
        def row(first, duration, success=True):
            return {"success": success, "first_output_seconds": first,
                    "duration_seconds": duration, "completion_tokens": None}
        rows = [row(.5, 2), row(.6, 1), row(.2, 3), row(None, 1), row(.1, 1, False)]
        summary = bench.summarize(rows, 10, slo_first_output=.5, slo_duration=2)
        slo = summary["latency_slo"]
        self.assertEqual(slo["requests_met"], 1)
        self.assertEqual(slo["fraction_of_attempts"], .2)
        self.assertEqual(slo["goodput_requests_per_second"], .1)
        self.assertIsNone(summary["successful_completion_tokens_per_second"])
        self.assertEqual(bench.summarize(rows, 10, slo_duration=2)["latency_slo"]["requests_met"], 3)
        self.assertEqual(bench.summarize(rows, 10, slo_first_output=.5)["latency_slo"]["requests_met"], 2)
        self.assertIsNone(bench.summarize(rows, 10)["latency_slo"])

    def test_latency_slo_no_successes(self):
        row = {"success": False, "first_output_seconds": .1,
               "duration_seconds": .2, "completion_tokens": 10}
        slo = bench.summarize([row], 1, slo_duration=1)["latency_slo"]
        self.assertEqual(slo["requests_met"], 0)
        self.assertEqual(slo["goodput_requests_per_second"], 0)

    def test_fixed_rate_slo_includes_client_backlog(self):
        row = {"success": True, "first_output_seconds": .1, "duration_seconds": .2,
               "completion_tokens": None, "scheduled_seconds": 0,
               "dispatch_delay_seconds": 2, "arrival_first_output_seconds": 2.1,
               "arrival_duration_seconds": 2.2}
        summary = bench.summarize([row], 3, slo_first_output=1, slo_duration=1)
        self.assertEqual(summary["latency_slo"]["requests_met"], 0)
        self.assertEqual(summary["latency_slo"]["timing_basis"], "scheduled_arrival")
        row.update(success=False, arrival_first_output_seconds=None)
        self.assertEqual(bench.summarize([row], 3)["arrival_timing"]["successful_first_output_seconds"]["count"], 0)

    def test_arrival_schedule_uses_absolute_deadlines(self):
        now = [100.0]
        sleeps = []
        def sleep(delay):
            sleeps.append(delay)
            now[0] += delay + .01  # deterministic late wakeup
        def submit(_fn, _url, _payload, _key, _timeout, index, origin):
            future = Future()
            future.set_result({"index": index, "start_seconds": now[0] - origin,
                               "success": True, "first_output_seconds": .01,
                               "duration_seconds": .02, "completion_tokens": 1})
            now[0] += .02
            return future
        with patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]), \
             patch.object(bench.time, "sleep", side_effect=sleep), \
             patch.object(bench.concurrent.futures, "ThreadPoolExecutor") as executor:
            executor.return_value.__enter__.return_value.submit.side_effect = submit
            rows, _ = bench.run("unused", [{"messages": []}], "fixture", 1, 3, 8, 0, "", 2,
                                request_rate=10)
        self.assertEqual(len(sleeps), 2)
        self.assertAlmostEqual(sleeps[0], .08)
        self.assertAlmostEqual(sleeps[1], .07)
        for row, expected in zip(rows, [0, .11, .21]):
            self.assertAlmostEqual(row["start_seconds"], expected)

    def test_poisson_schedule_is_seeded_and_keeps_backlog_in_slo(self):
        def measure(seed, service_time):
            now = [100.0]
            def submit(_fn, _url, _payload, _key, _timeout, index, origin):
                future = Future()
                future.set_result({"index": index, "start_seconds": now[0] - origin,
                                   "success": True, "first_output_seconds": .01,
                                   "duration_seconds": .02, "completion_tokens": 1})
                now[0] += service_time
                return future
            def sleep(delay):
                now[0] += delay
            with patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]), \
                 patch.object(bench.time, "sleep", side_effect=sleep), \
                 patch.object(bench.concurrent.futures, "ThreadPoolExecutor") as executor:
                executor.return_value.__enter__.return_value.submit.side_effect = submit
                return bench.run("unused", [{"messages": []}], "fixture", 1, 5, 8, 0, "", 2,
                                 request_rate=10, arrival_distribution="poisson", seed=seed,
                                 slo_duration=.1)
        fast, fast_summary = measure(42, .001)
        slow, slow_summary = measure(42, 1)
        other, _ = measure(43, .001)
        expected = [0, .1020060287274801, .104538912631754, .1367013190392506,
                    .161959937606262]
        for row, deadline in zip(fast, expected):
            self.assertAlmostEqual(row["scheduled_seconds"], deadline)
        self.assertEqual([r["scheduled_seconds"] for r in fast],
                         [r["scheduled_seconds"] for r in slow])
        self.assertNotEqual([r["scheduled_seconds"] for r in fast],
                            [r["scheduled_seconds"] for r in other])
        self.assertEqual(fast_summary["latency_slo"]["requests_met"], 5)
        self.assertEqual(slow_summary["latency_slo"]["requests_met"], 1)
        self.assertAlmostEqual(slow[1]["arrival_duration_seconds"], 1 - expected[1] + .02)

    def test_warmup_time_is_excluded_from_measured_throughput(self):
        now = [0.0]
        calls = []
        def request(_url, _payload, _key, _timeout, index, origin):
            start = now[0]
            duration = 100 if not calls else 1
            calls.append(origin)
            now[0] += duration
            return {"index": index, "start_seconds": start - origin, "success": True,
                    "first_output_seconds": duration, "duration_seconds": duration,
                    "completion_tokens": 3}
        with tempfile.TemporaryDirectory() as directory:
            workload = Path(directory) / "workload.jsonl"
            output = Path(directory) / "report.json"
            workload.write_text('{"messages":[{"role":"user","content":"hello"}]}\n')
            argv = [bench.__file__, "--base-url", "http://unused/v1", "--model", "fixture",
                    "--workload", str(workload), "--output", str(output),
                    "--warmup-requests", "1", "--repeats", "2", "--slo-duration", "2"]
            with patch.object(sys, "argv", argv), patch.object(sys, "stdout", io.StringIO()), \
                 patch.object(bench.time, "perf_counter", side_effect=lambda: now[0]), \
                 patch.object(bench, "request_one", side_effect=request):
                self.assertEqual(bench.main(), 0)
            report = json.loads(output.read_text())
        self.assertEqual(calls, [0, 100, 100])
        self.assertEqual(report["warmup"]["summary"]["wall_seconds"], 100)
        self.assertEqual(report["summary"]["wall_seconds"], 2)
        self.assertEqual(report["summary"]["successful_completion_tokens_per_second"], 3)
        self.assertEqual(report["summary"]["latency_slo"]["goodput_requests_per_second"], 1)

    def test_request_rate_must_be_positive_and_finite(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(bench.argparse.ArgumentTypeError):
                bench.positive_float(value)

    def test_nearest_rank_distribution(self):
        self.assertEqual(bench.distribution(list(range(1, 101)))["p95"], 95)
        self.assertIsNone(bench.distribution([])["p95"])


if __name__ == "__main__":
    unittest.main()
