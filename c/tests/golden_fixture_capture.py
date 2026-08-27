#!/usr/bin/env python3
"""Golden-fixture invariant harness (U7a acceptance test 1 / packet test-plan I1).

Captures full HTTP responses for a battery of NON-logprobs requests against a
RUNNING colibri server, normalizes the volatile fields (ids, timestamps), and
byte-diffs two capture directories. The engine-channel change (U7a) is opt-in
and no server request path opts in, so a pre-U7a capture and a post-U7a capture
of the same battery on the same box/backend/model must be byte-identical --
this script is the mechanism proving that, not reviewer inspection.

Usage:
    # 1. start the server on the CURRENT (pre-change) build, then:
    python3 tests/golden_fixture_capture.py capture \
        --base-url http://127.0.0.1:8000 --model <model-id> --out fixtures_pre

    # 2. rebuild/restart on the post-change build (same box, same backend,
    #    same container, same CTX/KV_SLOTS env), then:
    python3 tests/golden_fixture_capture.py capture \
        --base-url http://127.0.0.1:8000 --model <model-id> --out fixtures_post

    # 3. compare (exit 0 = byte-identical modulo normalized fields):
    python3 tests/golden_fixture_capture.py diff fixtures_pre fixtures_post

Battery: chat, chat+tools, chat streaming, completions without logprobs, and
the error cases whose behavior must not move (seed 400, array-prompt 400,
logprobs 400, out-of-range temperature 400) plus /v1/models.

Normalization: every "id"/"created" field (recursively, and per SSE event) is
replaced with a constant; nothing else is touched. Generation-bearing requests
run at temperature 0. If a token-level diff still appears pre-vs-post on the
SAME backend, triage it against the open engine-determinism finding
(CUDA_VS_CPU_ARM_B_2026-08-17) before treating it as a U7a regression.
"""
import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

NORMALIZED_KEYS = ("id", "created")


def battery(model):
    chat_messages = [
        {"role": "system", "content": "You are a terse assistant."},
        {"role": "user", "content": "Say the word 'fixture' and stop."},
    ]
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    return [
        ("chat_basic", "POST", "/v1/chat/completions",
         {"model": model, "messages": chat_messages,
          "max_tokens": 16, "temperature": 0}),
        ("chat_tools", "POST", "/v1/chat/completions",
         {"model": model, "messages": [
             {"role": "user", "content": "What is the weather in Rome?"}],
          "tools": tools, "tool_choice": "auto",
          "max_tokens": 48, "temperature": 0}),
        ("chat_stream", "POST", "/v1/chat/completions",
         {"model": model, "messages": chat_messages,
          "max_tokens": 16, "temperature": 0, "stream": True}),
        ("completions_plain", "POST", "/v1/completions",
         {"model": model, "prompt": "The capital of France is",
          "max_tokens": 8, "temperature": 0}),
        ("completions_stop", "POST", "/v1/completions",
         {"model": model, "prompt": "Count: one, two,",
          "max_tokens": 16, "temperature": 0, "stop": ["five"]}),
        ("err_seed", "POST", "/v1/completions",
         {"model": model, "prompt": "hello", "max_tokens": 1, "seed": 1234}),
        ("err_array_prompt", "POST", "/v1/completions",
         {"model": model, "prompt": [1, 2, 3], "max_tokens": 1}),
        ("err_logprobs", "POST", "/v1/completions",
         {"model": model, "prompt": "hello", "max_tokens": 1, "logprobs": 1}),
        ("err_bad_temperature", "POST", "/v1/chat/completions",
         {"model": model, "messages": chat_messages,
          "max_tokens": 1, "temperature": 9}),
        ("models", "GET", "/v1/models", None),
    ]


def normalize(obj):
    if isinstance(obj, dict):
        return {k: ("<normalized>" if k in NORMALIZED_KEYS else normalize(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    return obj


def normalize_body(raw, content_type):
    text = raw.decode("utf-8", "replace")
    if "text/event-stream" in (content_type or ""):
        events = []
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("data:"):
                payload = line[5:].strip()
                if payload == "[DONE]":
                    events.append("[DONE]")
                else:
                    try:
                        events.append(normalize(json.loads(payload)))
                    except ValueError:
                        events.append({"unparsed": payload})
            else:
                events.append({"raw_line": line})
        return {"sse_events": events}
    try:
        return normalize(json.loads(text))
    except ValueError:
        return {"raw_text": text}


def run_case(base_url, name, method, path, body):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + path, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            status = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = response.read()
    except urllib.error.HTTPError as error:
        status = error.code
        content_type = error.headers.get("Content-Type", "")
        raw = error.read()
    return {"case": name, "request": {"method": method, "path": path,
                                      "body": body},
            "status": status,
            "body": normalize_body(raw, content_type)}


def capture(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, method, path, body in battery(args.model):
        record = run_case(args.base_url, name, method, path, body)
        target = out / (name + ".json")
        target.write_text(json.dumps(record, indent=2, sort_keys=True,
                                     ensure_ascii=False) + "\n")
        print(f"[capture] {name}: HTTP {record['status']} -> {target}")
    print(f"[capture] done: {out}")
    return 0


def diff(args):
    a, b = pathlib.Path(args.dir_a), pathlib.Path(args.dir_b)
    names = sorted({p.name for p in a.glob("*.json")} |
                   {p.name for p in b.glob("*.json")})
    failures = 0
    for name in names:
        fa, fb = a / name, b / name
        if not fa.exists() or not fb.exists():
            print(f"[diff] {name}: MISSING on one side")
            failures += 1
            continue
        if fa.read_bytes() != fb.read_bytes():
            print(f"[diff] {name}: DIFFERS")
            failures += 1
        else:
            print(f"[diff] {name}: identical")
    if failures:
        print(f"[diff] FAIL: {failures}/{len(names)} cases differ")
        return 1
    print(f"[diff] PASS: {len(names)} cases byte-identical "
          "(modulo normalized ids/timestamps)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_capture = sub.add_parser("capture")
    p_capture.add_argument("--base-url", default="http://127.0.0.1:8000")
    p_capture.add_argument("--model", required=True)
    p_capture.add_argument("--out", required=True)
    p_capture.set_defaults(func=capture)
    p_diff = sub.add_parser("diff")
    p_diff.add_argument("dir_a")
    p_diff.add_argument("dir_b")
    p_diff.set_defaults(func=diff)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
