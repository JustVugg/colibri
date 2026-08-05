#!/usr/bin/env python3
"""Measure pure mux decode throughput as the number of active slots grows."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time


def frame(request_id, slot, prompt, max_tokens):
    body = prompt.encode()
    return (
        f"SUBMIT {request_id} {slot} {len(body)} {max_tokens} 0 1\n".encode()
        + body
        + b"\n"
    )


def drain_stderr(pipe, lines):
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        lines.append(line)
        print(line, file=sys.stderr)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_phase(proc, batch, tokens, phase_id):
    prompts = [
        f"Request {slot}: explain one practical property of hash tables. "
        f"Use complete sentences. Measurement phase {phase_id}."
        for slot in range(batch)
    ]
    base = phase_id * 1000
    for slot, prompt in enumerate(prompts):
        proc.stdin.write(frame(base + slot + 1, slot, prompt, tokens))
    proc.stdin.flush()

    data = {base + slot + 1: [] for slot in range(batch)}
    done = {}
    start = None
    while len(done) < batch:
        raw = proc.stdout.readline()
        if not raw:
            raise RuntimeError("engine exited during batch phase")
        now = time.perf_counter()
        line = raw.decode("utf-8", "replace").rstrip()
        if line.startswith("DATA "):
            _, sid, size = line.split()
            request_id = int(sid)
            proc.stdout.read(int(size))
            proc.stdout.read(1)
            if request_id in data:
                data[request_id].append(now)
                # SUBMIT is serial. The last request's first token marks the
                # point after every prompt has completed prefill.
                if request_id == base + batch and len(data[request_id]) == 1:
                    start = now
        elif line.startswith("DONE "):
            request_id = int(line.split()[1])
            if request_id in data:
                done[request_id] = (now, line)
        elif line.startswith("ERROR "):
            raise RuntimeError(line)

    if start is None:
        raise RuntimeError("last request produced no token")
    end = max(item[0] for item in done.values())
    measured_tokens = sum(
        sum(timestamp >= start for timestamp in timestamps)
        for timestamps in data.values()
    )
    gaps = []
    for timestamps in data.values():
        measured = [timestamp for timestamp in timestamps if timestamp >= start]
        gaps.extend(b - a for a, b in zip(measured, measured[1:]))
    wall = end - start
    return {
        "batch": batch,
        "requested_tokens": tokens,
        "measured_tokens": measured_tokens,
        "decode_wall_s": wall,
        "aggregate_tok_s": measured_tokens / wall,
        "per_session_tok_s": measured_tokens / wall / batch,
        "median_tbt_s": statistics.median(gaps) if gaps else None,
        "p95_tbt_s": percentile(gaps, 0.95) if gaps else None,
        "done": [done[key][1] for key in sorted(done)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", default="1,2,4,8,4,2,1")
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--kv-slots", type=int, default=8)
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",")]
    if not batches or min(batches) < 1 or max(batches) > args.kv_slots:
        parser.error("batches must be between 1 and --kv-slots")

    env = os.environ.copy()
    env.update(
        SNAP=args.model,
        SERVE="1",
        SERVE_BATCH="1",
        KV_SLOTS=str(args.kv_slots),
        COLI_KV_SHARE="0",
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
    thread = threading.Thread(
        target=drain_stderr, args=(proc.stderr, stderr_lines), daemon=True
    )
    thread.start()
    results = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited before READY")
            if b"READY" in line:
                break
        for phase_id, batch in enumerate(batches, 1):
            results.append(run_phase(proc, batch, args.tokens, phase_id))
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=60)
        thread.join(timeout=1)

    grouped = {}
    for result in results:
        grouped.setdefault(result["batch"], []).append(result["aggregate_tok_s"])
    summary = {
        "runs": results,
        "median_aggregate_tok_s": {
            str(batch): statistics.median(values)
            for batch, values in sorted(grouped.items())
        },
        "stderr_tail": stderr_lines[-40:],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
