#!/usr/bin/env python3
"""Measure decode stalls caused by serial prefill in Colibri's mux scheduler."""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time


def submit(proc, request_id, slot, prompt, max_tokens):
    body = prompt.encode()
    frame = (
        f"SUBMIT {request_id} {slot} {len(body)} {max_tokens} 0 1\n".encode()
        + body
        + b"\n"
    )
    proc.stdin.write(frame)
    proc.stdin.flush()
    return time.perf_counter()


def drain_stderr(pipe, lines):
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", "replace").rstrip()
        lines.append(line)
        print(line, file=sys.stderr)


def gaps(times):
    return [b - a for a, b in zip(times, times[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--long-repeat", type=int, default=20)
    ap.add_argument("--decode-tokens", type=int, default=32)
    args = ap.parse_args()

    env = os.environ.copy()
    env.update(
        SNAP=args.model,
        SERVE="1",
        SERVE_BATCH="1",
        KV_SLOTS="2",
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
    requests = {}
    phase = "ready"
    long_submitted = False
    try:
        while True:
            raw = proc.stdout.readline()
            if not raw:
                raise RuntimeError("engine exited before experiment completed")
            now = time.perf_counter()
            line = raw.decode("utf-8", "replace").rstrip()
            if "READY" in line and phase == "ready":
                requests[1] = {
                    "submitted": submit(
                        proc,
                        1,
                        0,
                        "Explain why a Python dictionary lookup is usually fast, "
                        "then give several implementation details.",
                        args.decode_tokens,
                    ),
                    "data": [],
                }
                phase = "control"
            elif line.startswith("DATA "):
                _, sid, size = line.split()
                request_id = int(sid)
                proc.stdout.read(int(size))
                proc.stdout.read(1)
                requests[request_id]["data"].append(now)
                if phase == "treatment" and request_id == 2 and not long_submitted:
                    long_prompt = (
                        "A hummingbird can hover by reversing lift through each wingbeat. "
                        "Its diet combines flower nectar with small insects for protein. "
                    ) * args.long_repeat
                    requests[3] = {
                        "submitted": submit(proc, 3, 1, long_prompt, 1),
                        "data": [],
                    }
                    long_submitted = True
            elif line.startswith("ACCEPT "):
                request_id = int(line.split()[1])
                if request_id in requests:
                    requests[request_id]["accepted"] = now
            elif line.startswith("DONE "):
                request_id = int(line.split()[1])
                if request_id not in requests:
                    continue
                requests[request_id]["done"] = now
                requests[request_id]["done_line"] = line
                if request_id == 1:
                    requests[2] = {
                        "submitted": submit(
                            proc,
                            2,
                            0,
                            "Explain how virtual memory works in a modern operating "
                            "system, including page faults and replacement.",
                            args.decode_tokens,
                        ),
                        "data": [],
                    }
                    phase = "treatment"
                elif request_id == 2 and 3 in requests and "done" in requests[3]:
                    break
                elif request_id == 3 and 2 in requests and "done" in requests[2]:
                    break
            elif line.startswith("ERROR "):
                raise RuntimeError(line)
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=30)
        thread.join(timeout=1)

    control_gaps = gaps(requests[1]["data"])
    treatment_gaps = gaps(requests[2]["data"])
    summary = {
        "control": {
            "tokens": len(requests[1]["data"]),
            "median_tbt_s": statistics.median(control_gaps),
            "max_tbt_s": max(control_gaps),
            "total_s": requests[1]["done"] - requests[1]["submitted"],
        },
        "treatment": {
            "tokens": len(requests[2]["data"]),
            "median_tbt_s": statistics.median(treatment_gaps),
            "max_tbt_s": max(treatment_gaps),
            "total_s": requests[2]["done"] - requests[2]["submitted"],
        },
        "interfering_prefill": {
            "submit_to_accept_s": requests[3]["accepted"] - requests[3]["submitted"],
            "ttft_s": requests[3]["data"][0] - requests[3]["submitted"],
            "total_s": requests[3]["done"] - requests[3]["submitted"],
        },
        "prefill_events": [
            line for line in stderr_lines if "[API] KV slot" in line
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
