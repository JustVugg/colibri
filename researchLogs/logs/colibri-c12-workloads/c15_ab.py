#!/usr/bin/env python3
"""C15 phase-1 A/B: row-blocked vs baseline attention prefill.

Run against a server with COLI_PROMPT_CACHE=0 so every request is a full
prefill (the no-rewind rule declines exact repeats), making the SAME prompt a
cold on both configs — output shas must match across configs, which is the
bit-exactness gate on real weights. The launcher alternates the two engine
binaries across process restarts (drift rule: paired, alternating runs).
"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/logs/logs/colibri-c12-workloads")
import c13_matrix as M

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
LABEL = sys.argv[2] if len(sys.argv) > 2 else "ab"

PHASE_KEYS = ("prefill_s", "prefill_attention_s", "prefill_deltanet_s",
              "prefill_moe_s", "prefill_chunk_count", "prompt_tokens",
              "engine_ttft_s", "decode_tokens_per_sec")

def send_cold(msgs, label):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": 128, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        M.BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {"label": label, "client_total_s": round(time.time() - t0, 4)}
    for k in PHASE_KEYS:
        row[k] = t.get(k)
    row["output_sha12"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(json.dumps(row), flush=True)

def main():
    # same tagged prompts every run of the series: shas comparable across runs
    q4 = "[c15-ab] " + M.make_q(310, 114)
    q12 = "[c15-ab] " + M.make_q(320, 349)
    send_cold(M.messages(("user", q4)), f"{LABEL}_4k_a")
    send_cold(M.messages(("user", q4)), f"{LABEL}_4k_b")
    send_cold(M.messages(("user", q12)), f"{LABEL}_12k")

if __name__ == "__main__":
    main()
