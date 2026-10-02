#!/usr/bin/env python3
"""Standalone A/B runner for the OLMoE tier-cache/PILOT benchmark.

Runs ./olmoe <cap> 8 bench_tier/ref200.json under a set of env configs and
captures: tok/s, engine cache hit rate, matching tokens (token-exactness),
peak RSS, load time. Measurement only; no engine code involved.

Usage: python3 bench_tier/bench_standalone.py [run_label] [config_filter]
"""
import json
import os
import re
import subprocess
import sys
import time

C_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.environ.get("SNAP", os.path.expanduser("~/models/olmoe_merged"))
REF = os.path.join(C_DIR, "bench_tier", "ref200.json")
BIN = os.path.join(C_DIR, "olmoe")
PIN_FILE = os.path.join(SNAP, "hot_pinned.bin")

CONFIGS = [
    ("base",            {"PILOT": "0", "HOT": "0"}),
    ("pilot1",          {"PILOT": "1", "HOT": "0"}),
    ("pilot2",          {"PILOT": "2", "HOT": "0"}),
    ("pilot3",          {"PILOT": "3", "HOT": "0"}),
    ("hot4",            {"PILOT": "0", "HOT": "4"}),
    ("hot_dyn",         {"PILOT": "0", "HOT": "100"}),
    ("pilot2_hot_dyn",  {"PILOT": "2", "HOT": "100"}),
    ("pilot3_hot_dyn",  {"PILOT": "3", "HOT": "100"}),
    ("pilot2_wide2",    {"PILOT": "2", "HOT": "0", "WIDE": "2"}),
]


def parse(out: str) -> dict:
    m = {}
    r = re.search(r"Matching tokens: (\d+)/(\d+)", out)
    if r:
        m["match"], m["n_new"] = int(r.group(1)), int(r.group(2))
    r = re.search(r"Expert cache hit rate: ([\d.]+)%\s+\(hit=(\d+) miss=(\d+)\)", out)
    if r:
        m["hit_pct"] = float(r.group(1))
        m["hits"], m["misses"] = int(r.group(2)), int(r.group(3))
    r = re.search(r"Speed: ([\d.]+) tok/s \(([\d.]+)s for (\d+) tokens\)", out)
    if r:
        m["tok_s"] = float(r.group(1))
        m["seconds"] = float(r.group(2))
    r = re.search(r"PEAK RSS: ([\d.]+) GB", out)
    if r:
        m["peak_rss_gb"] = float(r.group(1))
    r = re.search(r"resident weights loaded in ([\d.]+)s", out)
    if r:
        m["load_s"] = float(r.group(1))
    return m


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    filt = sys.argv[2] if len(sys.argv) > 2 else None
    cap = os.environ.get("CAP", "8")
    out_path = os.path.join(C_DIR, "bench_tier", f"results_{label}.jsonl")
    with open(out_path, "a") as fout:
        for name, env in CONFIGS:
            if filt and filt not in name:
                continue
            # A stale persistent pin file would leak hub pinning into later
            # configs; each config starts clean unless it creates its own.
            if os.path.exists(PIN_FILE):
                os.remove(PIN_FILE)
            e = dict(os.environ)
            e.update(env)
            e["SNAP"] = SNAP
            t0 = time.time()
            p = subprocess.run([BIN, cap, "8", REF], cwd=C_DIR, env=e,
                               capture_output=True, text=True)
            dt = time.time() - t0
            rec = {"config": name, "cap": cap, "env": env, "wall_s": round(dt, 2),
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
            rec.update(parse(p.stdout + p.stderr))
            if p.returncode != 0:
                rec["error"] = p.returncode
                rec["tail"] = (p.stdout + p.stderr)[-500:]
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            print(json.dumps(rec))


if __name__ == "__main__":
    main()
