#!/usr/bin/env python3
"""Benchmark MiMo through its local serve pipe; no HTTP server or model changes.

One persistent engine, one warmup, >=3 measured requests, fixed greedy prompt.
Records TTFT, request throughput, steady decode throughput, raw replies, PROF,
GPU telemetry and hashes. Model startup and warmup are excluded. Prefix reuse
is disabled so every request does the same prompt work. Run arms sequentially.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import statistics
import subprocess
import threading
import time

PROMPT = "<|im_start|>user\nExplain briefly what binary search is.<|im_end|><|im_start|>assistant\n<think></think>"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_cuda(stderr):
    """Require actual nonzero GPU work, not just a discovered device."""
    match = re.search(r"\[CUDA\] mimo dense: (\d+) matrices, [\d.]+ GiB resident, (\d+) matmuls,", stderr)
    if not match or min(map(int, match.groups())) <= 0 or "device failed" in stderr:
        raise RuntimeError("CUDA work was not confirmed; inspect engine.log")


def read_frames(pipe, events):
    try:
        while line := pipe.readline():
            fields = line.decode("utf-8", errors="replace").strip().split()
            payload = b""
            if fields and fields[0] in ("DATA", "ECHO", "TOOL", "DECISION"):
                n = int(fields[2])
                chunks = []
                while n:
                    b = pipe.read(n)
                    if not b:
                        raise EOFError("incomplete payload")
                    chunks.append(b)
                    n -= len(b)
                payload = b"".join(chunks)
                if pipe.read(1) != b"\n":
                    raise ValueError("missing payload newline")
            events.put((time.perf_counter(), fields, payload))
    except Exception as exc:
        events.put((time.perf_counter(), ["ERROR", "reader", str(exc)], b""))
    finally:
        events.put((time.perf_counter(), ["EOF"], b""))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    p.add_argument("--shards", default="")
    p.add_argument("--cap", type=int, default=88)
    p.add_argument("--threads", type=int, default=24)
    p.add_argument("--read-threads", type=int, default=16)
    p.add_argument("--context", type=int, default=16384)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--dense-gb", type=float, default=20)
    p.add_argument("--release-host", action="store_true")
    p.add_argument("--chain-layers", type=int, default=32)
    p.add_argument("--timeout", type=float, default=1800)
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra engine variable, recorded in the config (repeatable)")
    args = p.parse_args()
    if args.runs < 3 or args.tokens < 2 or args.timeout <= 0:
        p.error("use at least three measured runs, two tokens and a positive timeout")
    args.engine = args.engine.resolve()
    args.model = args.model.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    # Do not inherit another model's GPU, cache, trace or precision options.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("COLI_", "MIMO_", "CUDA_"))}
    config = {"SNAP": str(args.model), "SERVE": "1", "KV_SLOTS": "1",
              "CTX": str(args.context), "COLI_NO_OMP_TUNE": "1", "COLI_KV_PREFIX": "0",
              "OMP_NUM_THREADS": str(args.threads), "MIMO_READ_THREADS": str(args.read_threads),
              "MIMO_DENSE_BITS": "0", "MIMO_IDOT": "0", "MIMO_DIRECT": "1", "MIMO_CHUNK": "64",
              "MIMO_DIRS": args.shards, "MIMO_CUDA": "1" if args.backend == "cuda" else "0",
              "MIMO_CUDA_DENSE_GB": str(args.dense_gb), "MIMO_CUDA_RESERVE_GB": "2",
              "MIMO_CUDA_RELEASE_HOST": "1" if args.release_host else "0",
              "COLI_VULKAN": "1" if args.backend == "vulkan" else "0",
              "COLI_VK_CHAIN": "1", "COLI_VK_DENSE": "1", "COLI_VK_TIER_RESERVE_GB": "2",
              "COLI_VK_CHAIN_LAYERS": str(args.chain_layers)}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            p.error(f"--env wants KEY=VALUE, got {item!r}")
        config[key] = value
    env.update(config)
    events = queue.Queue()
    monitor = None
    log = (args.output / "engine.log").open("wb")
    proc = None
    samples = []
    try:
        proc = subprocess.Popen([str(args.engine), str(args.cap)], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, bufsize=0)
        threading.Thread(target=read_frames, args=(proc.stdout, events), daemon=True).start()
        deadline = time.perf_counter() + args.timeout

        def take():
            try:
                stamp, fields, payload = events.get(timeout=max(0.001, deadline - time.perf_counter()))
            except queue.Empty as exc:
                raise TimeoutError("engine timed out; see engine.log") from exc
            if fields and fields[0] in ("EOF", "ERROR"):
                raise RuntimeError("engine failed: " + " ".join(fields))
            return stamp, fields, payload

        while True:
            _, fields, _ = take()
            if fields and "READY" in fields[0]:
                break
        print(f"{args.backend}: ready, warmup + {args.runs} runs x {args.tokens} tokens", flush=True)
        try:
            monitor = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                                        "--format=csv", "-lms", "500", "-f", str(args.output / "gpu.csv")],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
        prompt = PROMPT.encode()
        for run in range(args.runs + 1):
            begin = time.perf_counter()
            deadline = begin + args.timeout
            proc.stdin.write(f"SUBMIT {run+1} 0 {len(prompt)} {args.tokens} 0 1\n".encode() + prompt + b"\n")
            proc.stdin.flush()
            first = None
            chunks = []
            done = None
            while True:
                stamp, fields, payload = take()
                if not fields:
                    continue
                if fields[0] == "DATA":
                    if first is None:
                        first = stamp
                    chunks.append(payload)
                if fields[0] == "DONE":
                    done = fields
                    elapsed = stamp - begin
                    ended = stamp
                if fields[0] == "PROF" and done:
                    prof = fields
                    break
            n = int(done[3])
            if n != args.tokens or first is None:
                raise RuntimeError(f"expected fixed {args.tokens} tokens, got {n}: {done}")
            answer = b"".join(chunks)
            (args.output / f"reply-{run}.txt").write_bytes(answer)
            item = {"run": run, "warmup": run == 0, "completion_tokens": n,
                    "seconds": elapsed, "ttft_seconds": first - begin,
                    "tok_s": n / elapsed, "decode_tok_s": (n - 1) / (ended - first),
                    "reply_sha256": hashlib.sha256(answer).hexdigest(), "done": done, "prof": prof}
            samples.append(item)
            (args.output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
            print(f"run {run}: {item['tok_s']:.4f} tok/s overall, {item['decode_tok_s']:.4f} decode, TTFT {item['ttft_seconds']:.3f}s", flush=True)
        proc.stdin.close()
        proc.wait(timeout=60)
        if proc.returncode:
            raise RuntimeError(f"engine exit {proc.returncode}")
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)
        if monitor:
            monitor.terminate()
            monitor.wait(timeout=15)
        log.close()
    stderr = (args.output / "engine.log").read_text(errors="replace")
    if args.backend == "cuda":
        verify_cuda(stderr)
    measured = samples[1:]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    record = {"backend": args.backend, "commit": commit, "engine_sha256": sha(args.engine),
              "config": config, "cap": args.cap, "warmup_runs": 1, "samples": samples,
              "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
              "median_tok_s": statistics.median(x["tok_s"] for x in measured),
              "median_decode_tok_s": statistics.median(x["decode_tok_s"] for x in measured),
              "evidence_sha256": sha(args.output / "samples.json")}
    dll = args.engine.parent / "coli_cuda_mimo.dll"
    if args.backend == "cuda" and dll.exists():
        record["cuda_dll_sha256"] = sha(dll)
    (args.output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"median {record['median_tok_s']:.4f} tok/s; {args.output}", flush=True)


if __name__ == "__main__":
    main()
