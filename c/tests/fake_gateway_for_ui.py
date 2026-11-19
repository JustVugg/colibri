#!/usr/bin/env python3
"""A stand-in `coli serve` for looking at the chat page's Monitor without an engine or a model.

    python3 c/tests/fake_gateway_for_ui.py --port 8799 [--metal on|off|none]
    cd c && python3 coli ui --backend http://127.0.0.1:8799 --no-browser --port 8798
    open http://127.0.0.1:8798/#monitor

Canned /health (macOS-shaped `system`, a turn being written, and with --metal on|off the
`metal` block of a qwen38 METAL=1 engine whose chain is on or off), /profile (a few finished
turns), /experts and /v1/models. CPU, GPU and speed move with the clock, so the live chart
draws lines. Any API key is accepted. Standard library only; not a test by itself.
"""
import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

T0 = time.monotonic()


def wave(period, lo, hi, phase=0.0):
    x = (time.monotonic() - T0) / period * 2 * math.pi + phase
    return round(lo + (hi - lo) * (0.5 + 0.5 * math.sin(x)), 2)


def health(metal):
    h = {
        "status": "ok", "arch": "qwen38", "engine_restarts": 0, "kv_slots": 1,
        "scheduler": {"active": 1, "queued": 0, "capacity": 1, "max_queue": 8, "completed": 12,
                      "failed": 0, "cancelled": 1},
        "context_compaction": True, "context_tokens": 32768, "max_tokens": 8192,
        "live": [{"state": "writing", "elapsed_s": 14.2, "read_s": 2.1, "prompt_tokens": 2412,
                  "cached_tokens": 1800, "tokens": 187, "forwards": 121,
                  "tok_s": 15.4, "tok_s_now": wave(9, 11, 19)}],
        "system": {"cores": 12, "load1": wave(20, 3, 6), "ram_total_gb": 24.0,
                   "ram_avail_gb": wave(30, 4.5, 6.5), "engine_rss_gb": 14.82,
                   "engine_cpu_pct": wave(7, 180, 520),
                   "gpus": [{"name": "Apple M4 Pro", "util_pct": int(wave(5, 35, 92, 1.3)),
                             "mem_used_gb": wave(25, 5.8, 6.4)}]},
    }
    if metal != "none":
        on = metal == "on"
        h["metal"] = {"chain": 1 if on else 0, "reason": None if on else "COLI_METAL=0",
                      "forwards": 4821 if on else 0, "declined": 3, "experts_gpu": 1 if on else 0,
                      "cache_gb": 6.0 if on else 0.0, "slots": 1536 if on else 0,
                      "loads": 812, "hits": 3410, "wait_s": 6.8, "ahead": 1 if on else 0,
                      "guesses": 2210, "right": 1655, "kv_regrow": 2,
                      "totals": {"loads": 9120, "hits": 40112, "wait_s": 71.5, "guesses": 25010,
                                 "right": 18800}}
    return h


def profile():
    turns = []
    for i in range(8):
        w = 9.0 + 1.7 * i
        turns.append({"wall_s": w, "prompt_tokens": 900 + 210 * i, "completion_tokens": 120 + 15 * i,
                      "forwards": 80 + 9 * i, "expert_wait_s": 0.9 * w * 0.12, "expert_disk_s": 0.3 * w,
                      "expert_matmul_s": 0.35 * w, "attention_s": 0.18 * w, "lm_head_s": 0.04 * w})
    turns[-1]["wall_s"] = 12.4
    return {"seq": len(turns), "turns": turns}


class Handler(BaseHTTPRequestHandler):
    metal = "on"

    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self.send(health(self.metal))
        elif path == "/profile":
            self.send(profile())
        elif path == "/experts":
            self.send({"rows": 0, "cols": 0, "map": "", "hits": "", "seq": 0})
        elif path == "/v1/models":
            self.send({"object": "list", "data": [{"id": "qwen3.8-flash-next", "object": "model",
                                                   "owned_by": "colibri"}]})
        else:
            self.send({"error": {"message": "not here"}}, 404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8799)
    ap.add_argument("--metal", choices=("on", "off", "none"), default="on")
    a = ap.parse_args()
    Handler.metal = a.metal
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
