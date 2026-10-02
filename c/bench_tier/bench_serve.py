#!/usr/bin/env python3
"""Serve-mode latency benchmark for the OLMoE tier-cache/PILOT work.

Drives ./olmoe in SERVE=1 mode over the real wire protocol, times every DATA
frame (per decode step), and collects the engine's own telemetry:
DONE STAT (tok/s, hit%), PROF (disk/matmul/attn/head seconds), HITS.

Usage:
  python3 bench_tier/bench_serve.py <label> [config_filter] [reps]

Configs come from CONFIGS below; env overrides are applied on top of the
current environment. Measurement only: the driver never modifies the engine.
"""
import json
import os
import re
import select
import subprocess
import sys
import time

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.environ.get("SNAP", os.path.expanduser("~/models/olmoe_merged"))
BIN = os.path.join(C_DIR, "olmoe")
PROMPT_PATH = os.path.join(C_DIR, "bench_tier", "prompt.txt")
PIN_FILE = os.path.join(SNAP, "hot_pinned.bin")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "200"))
CTX = int(os.environ.get("CTX", "2048"))

CONFIGS = [
    ("base",            {"PILOT": "0", "HOT": "0"}),
    ("pilot1",          {"PILOT": "1", "HOT": "0"}),
    ("pilot2",          {"PILOT": "2", "HOT": "0"}),
    ("pilot3",          {"PILOT": "3", "HOT": "0"}),
    ("hot_dyn",         {"PILOT": "0", "HOT": "100"}),
    ("pilot2_hot_dyn",  {"PILOT": "2", "HOT": "100"}),
    ("pilot3_hot_dyn",  {"PILOT": "3", "HOT": "100"}),
]


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def run_once(env_overrides: dict) -> dict:
    if os.path.exists(PIN_FILE):
        os.remove(PIN_FILE)
    env = dict(os.environ)
    env.update(env_overrides)
    env["SNAP"] = SNAP
    env["SERVE"] = "1"
    env["CTX"] = str(CTX)
    prompt = open(PROMPT_PATH, "rb").read()

    cap = env.get("CAP", "8")
    p = subprocess.Popen([BIN, cap, "8"], cwd=C_DIR, env=env,
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, bufsize=0)

    rec = {"config": None, "env": env_overrides}
    step_times = []
    data_bytes = bytearray()
    prof = None
    done = None
    hits = None
    t_last = None
    ready = False

    def read_line():
        line = b""
        while True:
            ch = p.stdout.read(1)
            if not ch:
                return None
            if ch == b"\n":
                return line
            line += ch

    # handshake
    t_start = time.time()
    while time.time() - t_start < 300:
        line = read_line()
        if line is None:
            break
        if b"READY" in line:
            ready = True
            break
    if not ready:
        p.kill()
        raise RuntimeError("engine never sent READY")

    hdr = f"SUBMIT bench 0 {len(prompt)} {MAX_TOKENS} 0.0 1.0\n".encode()
    p.stdin.write(hdr + prompt + b"\n")
    p.stdin.flush()

    t0 = time.time()
    while True:
        line = read_line()
        if line is None:
            break
        if line.startswith(b"DATA "):
            parts = line.split()
            size = int(parts[2])
            payload = p.stdout.read(size)
            p.stdout.read(1)  # trailing \n
            now = time.time()
            if t_last is not None:
                step_times.append((now - t_last) * 1000.0)
            t_last = now
            data_bytes += payload
        elif line.startswith(b"DONE "):
            m = re.match(rb"DONE (\S+) STAT (\d+) ([\d.]+) ([\d.]+) ([\d.]+) (\d+) (\d+)", line)
            if m:
                done = {"tokens": int(m.group(2)), "tok_s": float(m.group(3)),
                        "hit_pct": float(m.group(4)), "rss_gb": float(m.group(5)),
                        "prompt_tokens": int(m.group(6)), "limited": int(m.group(7))}
        elif line.startswith(b"PROF "):
            f = line.split()
            prof = {"dt_s": float(f[1]), "np": int(f[2]), "gen": int(f[3]),
                    "disk_s": float(f[4]), "wait_s": float(f[5]),
                    "matmul_s": float(f[6]), "attn_s": float(f[7]),
                    "head_s": float(f[8]), "forwards": int(f[9])}
        elif line.startswith(b"HITS "):
            hits = {"rows": int(line.split()[1]), "cols": int(line.split()[2])}
        # keep reading until DONE+PROF+HITS all seen or timeout
        if done is not None and prof is not None and hits is not None:
            break

    wall = time.time() - t0
    p.stdin.close()
    p.wait(timeout=10)
    rec.update({"ready": ready, "wall_s": round(wall, 2), "done": done, "prof": prof,
                "hits": hits, "n_steps": len(step_times),
                "out_sha": __import__("hashlib").sha256(bytes(data_bytes)).hexdigest()[:16],
                "out_len": len(data_bytes)})
    st = sorted(step_times)
    if st:
        rec["p50_ms"] = round(percentile(st, 50), 2)
        rec["p90_ms"] = round(percentile(st, 90), 2)
        rec["p99_ms"] = round(percentile(st, 99), 2)
        rec["max_ms"] = round(st[-1], 2)
        rec["mean_ms"] = round(sum(st) / len(st), 2)
        if prof and prof["disk_s"] > 0:
            rec["disk_ms_per_step"] = round(prof["disk_s"] * 1000.0 / max(1, len(st)), 2)
    return rec


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    filt = sys.argv[2] if len(sys.argv) > 2 else None
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    out_path = os.path.join(C_DIR, "bench_tier", f"serve_{label}.jsonl")
    with open(out_path, "a") as fout:
        for name, env in CONFIGS:
            if filt and filt not in name:
                continue
            for rep in range(reps):
                rec = run_once(env)
                rec["config"] = name
                rec["rep"] = rep
                rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(json.dumps(rec))


if __name__ == "__main__":
    main()
