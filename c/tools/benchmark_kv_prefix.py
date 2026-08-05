#!/usr/bin/env python3
"""Measure cross-slot KV prefix adoption through Colibri's mux protocol."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time


def stderr_drain(pipe, lines):
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        lines.append(line)
        print(line, file=sys.stderr)


def wait_ready(pipe):
    while True:
        raw = pipe.readline()
        if not raw:
            raise RuntimeError("engine exited before READY")
        if b"READY" in raw:
            print("[bench] READY", file=sys.stderr, flush=True)
            return


def request(proc, request_id, slot, payload):
    body = payload.encode()
    frame = (
        f"SUBMIT {request_id} {slot} {len(body)} 1 0 1\n".encode()
        + body
        + b"\n"
    )
    started = time.perf_counter()
    proc.stdin.write(frame)
    proc.stdin.flush()
    print(f"[bench] submitted id={request_id} slot={slot} bytes={len(body)}", file=sys.stderr, flush=True)
    accepted = first_data = done = None
    done_fields = None
    data = bytearray()
    while done is None:
        raw = proc.stdout.readline()
        if not raw:
            raise RuntimeError(f"engine exited during request {request_id}")
        line = raw.decode("utf-8", "replace").rstrip()
        if line.startswith(f"ACCEPT {request_id} "):
            accepted = time.perf_counter()
            print(f"[bench] accepted id={request_id}", file=sys.stderr, flush=True)
        elif line.startswith(f"DATA {request_id} "):
            if first_data is None:
                first_data = time.perf_counter()
                print(f"[bench] first data id={request_id}", file=sys.stderr, flush=True)
            size = int(line.rsplit(" ", 1)[1])
            data.extend(proc.stdout.read(size))
            proc.stdout.read(1)
        elif line.startswith(f"DONE {request_id} "):
            done = time.perf_counter()
            done_fields = line
            print(f"[bench] done id={request_id}", file=sys.stderr, flush=True)
        elif line.startswith("ERROR "):
            raise RuntimeError(line)
    return {
        "submit_to_accept_s": accepted - started if accepted else None,
        "ttft_s": first_data - started if first_data else None,
        "total_s": done - started,
        "done": done_fields,
        "data_hex": data.hex(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--share", type=int, choices=(0, 1), required=True)
    ap.add_argument("--repeat", type=int, default=80)
    ap.add_argument("--rounds", type=int, default=1)
    args = ap.parse_args()

    env = os.environ.copy()
    env.update(
        SNAP=args.model,
        SERVE="1",
        SERVE_BATCH="1",
        KV_SLOTS="2",
        COLI_KV_SHARE=str(args.share),
        DRAFT="0",
        COLI_TEMP="0",
    )
    proc = subprocess.Popen(
        [args.engine, "8", "4", "4"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    stderr_lines = []
    thread = threading.Thread(target=stderr_drain, args=(proc.stderr, stderr_lines), daemon=True)
    thread.start()
    results = []
    try:
        wait_ready(proc.stdout)
        for round_id in range(args.rounds):
            base = (
                f"Experiment round {round_id}: "
                + (
                    "A hummingbird can hover by reversing lift through each wingbeat. "
                    "Its diet combines flower nectar with small insects for protein. "
                )
                * args.repeat
            )
            prompts = (
                base + "\nQuestion A: Summarize the passage.",
                base + "\nQuestion B: State the main facts.",
            )
            first = request(proc, round_id * 2 + 1, 0, prompts[0])
            print(json.dumps({"event": "first", "round": round_id, **first}), flush=True)
            second = request(proc, round_id * 2 + 2, 1, prompts[1])
            print(json.dumps({"event": "second", "round": round_id, **second}), flush=True)
            results.append(
                {
                    "round": round_id,
                    "payload_bytes": [len(p.encode()) for p in prompts],
                    "first": first,
                    "second": second,
                }
            )
    finally:
        proc.stdin.close()
        proc.wait(timeout=30)
        thread.join(timeout=1)

    prefix_lines = [line for line in stderr_lines if "[API] KV " in line]
    print(
        json.dumps(
            {
                "share": args.share,
                "repeat": args.repeat,
                "rounds": results,
                "kv_events": prefix_lines,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
