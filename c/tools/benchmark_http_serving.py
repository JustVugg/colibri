#!/usr/bin/env python3
"""OpenAI chat streaming benchmark with bounded concurrency (stdlib only)."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request


class StreamError(ValueError):
    pass


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def endpoint(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("base URL must be HTTP(S), without credentials, query or fragment")
    return base_url.rstrip("/") + "/chat/completions"


def load_workload(path):
    raw = Path(path).read_bytes()
    workload = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item, dict) or set(item) != {"messages"}:
            raise ValueError("each JSONL row must contain only 'messages'")
        messages = item["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] not in ("system", "user", "assistant")
                    or not isinstance(message["content"], str)):
                raise ValueError("messages require role and string content")
        workload.append(item)
    if not workload:
        raise ValueError("workload is empty")
    return workload, hashlib.sha256(raw).hexdigest()


def sse_events(response):
    """Dispatch only complete SSE events; EOF is not a successful terminator."""
    data = []
    for raw in response:
        line = raw.decode("utf-8").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data = []
            continue
        field, _, value = line.partition(":")
        if field == "data":
            data.append(value[1:] if value.startswith(" ") else value)


def nonempty_text(value):
    if value is not None and not isinstance(value, str):
        raise StreamError("invalid_output_text")
    return bool(value)


def has_output(delta):
    if not isinstance(delta, dict):
        raise StreamError("invalid_delta")
    output = False
    for field in ("content", "reasoning_content", "reasoning"):
        output |= nonempty_text(delta.get(field))
    calls = delta.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise StreamError("invalid_tool_calls")
    legacy = delta.get("function_call")
    if legacy is not None:
        calls = calls + [{"function": legacy}]
    for tool in calls:
        if not isinstance(tool, dict):
            raise StreamError("invalid_tool_call")
        function = tool.get("function")
        if function is None:
            continue
        if not isinstance(function, dict):
            raise StreamError("invalid_tool_function")
        for field in ("name", "arguments"):
            output |= nonempty_text(function.get(field))
    return output


def request_one(url, payload, key, timeout, index, origin):
    start = time.perf_counter()
    result = {"index": index, "start_seconds": start - origin,
              "success": False, "http_status": None, "error": None,
              "first_output_seconds": None, "completion_tokens": None,
              "finish_reason": None}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    try:
        opener = urllib.request.build_opener(NoRedirect)
        with opener.open(request, timeout=timeout) as response:
            result["http_status"] = response.status
            if response.headers.get_content_type() != "text/event-stream":
                raise StreamError("unexpected_content_type")
            done = False
            for event in sse_events(response):
                if event == "[DONE]":
                    done = True
                    break
                chunk = json.loads(event)
                if not isinstance(chunk, dict):
                    raise StreamError("invalid_chunk")
                if "error" in chunk:
                    raise StreamError("stream_error")
                choices = chunk.get("choices", [])
                if not isinstance(choices, list) or len(choices) > 1:
                    raise StreamError("invalid_choices")
                if choices and result["finish_reason"] is not None:
                    raise StreamError("choice_after_finish")
                for choice in choices:
                    if (not isinstance(choice, dict) or type(choice.get("index")) is not int
                            or choice["index"] != 0):
                        raise StreamError("unexpected_choice")
                    delta = choice.get("delta")
                    if has_output({} if delta is None else delta):
                        if result["first_output_seconds"] is None:
                            result["first_output_seconds"] = time.perf_counter() - start
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        if finish not in ("stop", "length", "tool_calls", "function_call", "content_filter"):
                            raise StreamError("unsuccessful_finish_reason")
                        result["finish_reason"] = finish
                usage = chunk.get("usage")
                if usage is not None:
                    if not isinstance(usage, dict):
                        raise StreamError("invalid_usage")
                    tokens = usage.get("completion_tokens")
                    if type(tokens) is not int or tokens < 0:
                        raise StreamError("invalid_completion_tokens")
                    result["completion_tokens"] = tokens
            if not done or result["finish_reason"] is None:
                raise StreamError("incomplete_stream")
            result["success"] = True
    except StreamError as exc:
        result["error"] = str(exc)
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = "HTTPError"
        exc.close()
    except Exception as exc:
        # Do not copy server bodies, request contents or credentials into reports.
        result["error"] = type(exc).__name__
    result["duration_seconds"] = time.perf_counter() - start
    return result


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}
    ordered = sorted(values)
    result = {"count": len(values), "mean": sum(values) / len(values)}
    for percentile in (50, 95, 99):
        result[f"p{percentile}"] = ordered[math.ceil(len(ordered) * percentile / 100) - 1]
    return result


def summarize(results, elapsed, slo_first_output=None, slo_duration=None):
    successful = [row for row in results if row["success"]]
    counted = [row for row in successful if row["completion_tokens"] is not None]
    tokens = sum(row["completion_tokens"] for row in counted)
    complete = bool(successful) and len(counted) == len(successful)
    summary = {"requests": len(results), "succeeded": len(successful),
            "failed": len(results) - len(successful),
            "failure_rate": (len(results) - len(successful)) / len(results),
            "wall_seconds": elapsed,
            "successful_requests_per_second": len(successful) / elapsed,
            "successful_requests_with_usage": len(counted),
            "reported_successful_completion_tokens": tokens,
            "successful_completion_tokens_per_second": tokens / elapsed if complete else None,
            "successful_duration_seconds": distribution([r["duration_seconds"] for r in successful]),
            "successful_first_output_seconds": distribution([
                r["first_output_seconds"] for r in successful if r["first_output_seconds"] is not None])}

    paced = all(row.get("scheduled_seconds") is not None for row in results)
    summary["arrival_timing"] = None
    if paced:
        summary["arrival_timing"] = {
            "dispatch_delay_seconds": distribution([r["dispatch_delay_seconds"] for r in results]),
            "successful_duration_seconds": distribution([r["arrival_duration_seconds"] for r in successful]),
            "successful_first_output_seconds": distribution([
                r["arrival_first_output_seconds"] for r in successful
                if r["arrival_first_output_seconds"] is not None])}
    prefix = "arrival_" if paced else ""
    summary["latency_slo"] = None
    if slo_first_output is not None or slo_duration is not None:
        met = sum(
            (slo_duration is None or row[prefix + "duration_seconds"] <= slo_duration)
            and (slo_first_output is None or (
                row[prefix + "first_output_seconds"] is not None
                and row[prefix + "first_output_seconds"] <= slo_first_output))
            for row in successful)
        summary["latency_slo"] = {
            "timing_basis": "scheduled_arrival" if paced else "request_start",
            "first_output_seconds": slo_first_output, "duration_seconds": slo_duration,
            "requests_met": met, "fraction_of_attempts": met / len(results),
            "goodput_requests_per_second": met / elapsed}
    return summary


def run(url, workload, model, concurrency, repeats, max_tokens, temperature, key, timeout,
        slo_first_output=None, slo_duration=None, request_rate=None,
        arrival_distribution="periodic", seed=0):
    count = len(workload) * repeats
    scheduled = [0.0] * count
    if request_rate is not None:
        rng = random.Random(seed)
        for index in range(1, count):
            scheduled[index] = (index / request_rate if arrival_distribution == "periodic"
                                else scheduled[index - 1] + rng.expovariate(request_rate))
    origin = time.perf_counter()
    # Without a rate, workers immediately take the next request. With a rate,
    # absolute arrival deadlines keep submission independent of response time.
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = []
        for index in range(count):
            if request_rate is not None:
                delay = origin + scheduled[index] - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
            payload = dict(workload[index % len(workload)], model=model, stream=True,
                           stream_options={"include_usage": True}, max_tokens=max_tokens,
                           temperature=temperature, n=1)
            futures.append(pool.submit(request_one, url, payload, key, timeout, index, origin))
        results = [future.result() for future in futures]
    if request_rate is not None:
        for row in results:
            row["scheduled_seconds"] = scheduled[row["index"]]
            wait = max(0.0, row["start_seconds"] - row["scheduled_seconds"])
            row["dispatch_delay_seconds"] = wait
            row["arrival_duration_seconds"] = wait + row["duration_seconds"]
            first = row["first_output_seconds"]
            row["arrival_first_output_seconds"] = None if first is None else wait + first
    return results, summarize(results, time.perf_counter() - origin, slo_first_output, slo_duration)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API root including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", required=True, help="JSONL rows containing messages")
    parser.add_argument("--output", required=True, help="JSON report path")
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--warmup-requests", type=nonnegative_int, default=0,
                        help="unmeasured requests before the timed phase (default: 0)")
    parser.add_argument("--request-rate", type=positive_float, help="mean scheduled arrivals per second")
    parser.add_argument("--arrival-distribution", choices=("periodic", "poisson"),
                        default="periodic", help="arrival intervals when --request-rate is set")
    parser.add_argument("--seed", type=int, default=0, help="Poisson arrival seed (default: 0)")
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=positive_float, default=60, help="socket operation timeout in seconds")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--slo-first-output", type=positive_float, help="first output latency target, seconds")
    parser.add_argument("--slo-duration", type=positive_float, help="completed request latency target, seconds")
    args = parser.parse_args()
    if args.arrival_distribution == "poisson" and args.request_rate is None:
        parser.error("--arrival-distribution poisson requires --request-rate")
    try:
        url = endpoint(args.base_url)
        workload, digest = load_workload(args.workload)
        if not math.isfinite(args.temperature) or args.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if Path(args.output).resolve() == Path(args.workload).resolve():
            raise ValueError("output must differ from workload")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    key = os.environ.get(args.api_key_env, "")
    warmup = None
    if args.warmup_requests:
        prompts = [workload[i % len(workload)] for i in range(args.warmup_requests)]
        rows, stats = run(url, prompts, args.model, args.concurrency, 1,
                          args.max_tokens, args.temperature, key, args.timeout)
        warmup = {"summary": stats, "requests": rows}
    results, summary = [], None
    if warmup is None or warmup["summary"]["failed"] == 0:
        results, summary = run(url, workload, args.model, args.concurrency, args.repeats,
                               args.max_tokens, args.temperature, key, args.timeout,
                               args.slo_first_output, args.slo_duration, args.request_rate,
                               args.arrival_distribution, args.seed)
    report = {"schema_version": 1, "config": {
        "endpoint": url, "model": args.model, "workload_sha256": digest,
        "workload_rows": len(workload), "concurrency": args.concurrency,
        "load_model": ("closed_loop" if args.request_rate is None else
                       "poisson" if args.arrival_distribution == "poisson" else "fixed_rate"),
        "arrival_distribution": args.arrival_distribution if args.request_rate is not None else None,
        "arrival_seed": args.seed if args.arrival_distribution == "poisson" else None,
        "request_rate": args.request_rate, "warmup_requests": args.warmup_requests,
        "repeats": args.repeats, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "socket_timeout_seconds": args.timeout,
        "slo_first_output_seconds": args.slo_first_output, "slo_duration_seconds": args.slo_duration,
        "stream": True, "include_usage": True, "n": 1},
        "status": "warmup_failed" if summary is None else "measured",
        "warmup": warmup, "summary": summary, "requests": results}
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary if summary is not None else warmup, indent=2))
    return 1 if summary is None or summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
