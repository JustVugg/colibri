#!/usr/bin/env python3
"""Replay this experiment against a separately started local engine; no lifecycle changes.

The measure function is the implementation used for the reported HTTP timing.
Telemetry/lifecycle guards belong to the local session runner and are not included.
"""
import argparse
import datetime
import http.client
import json
import math
from pathlib import Path
import socket
import statistics
import threading
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    i = int(x)
    return values[i] + (values[math.ceil(x)] - values[i]) * (x - i)

def measure(cfg, name, prompt, cap, rep, warmup=False):
    result = dict(workload=name, repetition=rep, warmup=warmup, status="error", error="",
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), max_tokens=cap,
                  prompt=prompt, output="", reasoning="", usage={}, events_s=[], chunk_gaps_ms=[],
                  finish_reason=None, http_status=None, start_ts=time.time())
    url = urlsplit(cfg["url"])
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = connection_type(url.hostname, url.port, timeout=cfg["timeout"])
    sockets = []
    expired = threading.Event()
    response = None

    def abort():
        expired.set()
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        conn.close()

    body = dict(model=cfg["model"], messages=[dict(role="user", content=prompt)], max_tokens=cap, temperature=0, stream=True)
    if cfg.get("reasoning_effort"):
        body["reasoning_effort"] = cfg["reasoning_effort"]
    if cfg["usage"]:
        body["stream_options"] = {"include_usage": True}
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    started = time.perf_counter()
    timer = threading.Timer(cfg["timeout"], abort)
    timer.daemon = True
    timer.start()

    def consume_event(lines):
        raw = "\n".join(lines)
        if raw.strip() == "[DONE]":
            return True
        data = json.loads(raw)
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        if data.get("usage"):
            result["usage"] = data["usage"]
        for choice in data.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            text = delta.get("content") or ""
            thought = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if not isinstance(text, str) or not isinstance(thought, str):
                raise ValueError("Expected text streaming deltas")
            if text or thought:
                result["events_s"].append(time.perf_counter() - started)
            result["output"] += text
            result["reasoning"] += thought
            if choice.get("finish_reason") is not None:
                result["finish_reason"] = choice["finish_reason"]
        return False

    try:
        conn.connect()
        sockets.append(conn.sock)
        if expired.is_set():
            raise TimeoutError("Request deadline exceeded")
        path = url.path + ("?" + url.query if url.query else "")
        conn.request("POST", path, json.dumps(body), headers)
        response = conn.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            detail = response.read(4096).decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {response.status}: {detail}")
        lines = []
        done = False
        while not done:
            if expired.is_set():
                raise TimeoutError("Request deadline exceeded")
            raw = response.readline()
            if not raw:
                break
            line = raw.decode("utf-8").rstrip("\r\n")
            if not line and lines:
                done = consume_event(lines)
                lines = []
            elif line.startswith("data:"):
                lines.append(line[5:].lstrip(" "))
        if lines and not done:
            done = consume_event(lines)
        if expired.is_set():
            raise TimeoutError("Request deadline exceeded")
        if not done:
            raise RuntimeError("Incomplete stream: missing [DONE]")
        result["status"] = "ok"
    except Exception as exc:
        result["error"] = ("Timeout: " if expired.is_set() else "") + str(exc)
    finally:
        result["wall_s"] = time.perf_counter() - started
        result["end_ts"] = time.time()
        timer.cancel()
        if response:
            response.close()
        conn.close()

    events = result["events_s"]
    gaps = [(b - a) * 1000 for a, b in zip(events, events[1:])]
    result["chunk_gaps_ms"] = gaps
    result["ttft_s"] = events[0] if events else None
    result["chunk_mean_ms"] = statistics.mean(gaps) if gaps else None
    for label, p in (("chunk_p50_ms", .5), ("chunk_p95_ms", .95), ("chunk_p99_ms", .99)):
        result[label] = quantile(gaps, p) if gaps else None
    result["chunk_max_ms"] = max(gaps) if gaps else None
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = result["usage"].get(key)
        result[key] = value if type(value) == int and value >= 0 else None
    pt, ct = result["prompt_tokens"], result["completion_tokens"]
    span = events[-1] - events[0] if len(events) > 1 else 0
    result["prefill_proxy_tps"] = pt / events[0] if pt and events and events[0] > 0 else None
    result["decode_proxy_tps"] = (ct - 1) / span if ct and ct > 1 and span > 0 else None
    result["tpot_proxy_ms"] = 1000 * span / (ct - 1) if ct and ct > 1 and span > 0 else None
    result["e2e_output_tps"] = ct / result["wall_s"] if ct is not None else None
    result["output_chars"] = len(result["output"])
    result["reasoning_chars"] = len(result["reasoning"])
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8002/v1/chat/completions')
    parser.add_argument('--model', default='deepseek-v4-flash-reap')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve the existing output; choose a new path')
    here = Path(__file__).resolve().parent
    cfg = dict(url=args.url, model=args.model, timeout=1800, reasoning_effort='none', usage=True)
    # Reproduce the smoke/reference traffic before the measured workload.
    smoke = measure(cfg, 'smoke', 'What is 17 * 23? Reply with the number only.', 32, 1)
    if smoke['status'] != 'ok' or smoke['output'].strip() != '391':
        raise RuntimeError(smoke)
    ref = json.loads((here/'reference.json').read_text())
    answers = []
    for prompt, expected in zip(ref['prompts'], ref['answers']):
        body = dict(model=args.model, messages=[dict(role='user', content=prompt)],
                    max_tokens=160, temperature=0, stream=False, reasoning_effort='none')
        req = Request(args.url, json.dumps(body).encode(), {'Content-Type':'application/json'})
        with urlopen(req, timeout=900) as response:
            message = json.load(response)['choices'][0]['message']
        answer = (message.get('reasoning_content') or message.get('reasoning') or '')+'\n---\n'+(message.get('content') or '')
        answers.append(answer)
        if answer != expected:
            raise RuntimeError('Reference answer differs; do not label this an exact comparison')
    rows = []
    expected_rows = json.loads((here/'end-to-end-candidate.json').read_text())['runs']
    for task, expected in zip(json.loads((here/'prompts.json').read_text()), expected_rows):
        row = measure(cfg, task['workload'], task['prompt'], task['max_tokens'], task['repetition'], task['warmup'])
        rows.append(row)
        args.output.write_text(json.dumps(dict(configuration=cfg, smoke=smoke, answers=answers, runs=rows), indent=2)+'\n')
        if row['status'] != 'ok':
            raise RuntimeError(row['error'])
        for key in ('output','reasoning','completion_tokens','finish_reason'):
            if row[key] != expected[key]:
                raise RuntimeError(f'{task["workload"]}: {key} differs')
        print(task['workload'], task['repetition'], row['ttft_s'], row['decode_proxy_tps'], flush=True)

if __name__ == '__main__':
    main()
