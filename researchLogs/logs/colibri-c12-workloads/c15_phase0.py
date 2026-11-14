#!/usr/bin/env python3
"""C15 phase 0: per-phase prefill accumulators on the unblocked kernel.

One cold prefill per size (1K/4K/12K), fresh pass tag so each diverges and
full-prefills. Records the engine's per-request phase timers
(prefill_attention_s, prefill_deltanet_s, prefill_moe_s, shared/router/lm_head,
embedding, chunk count/max) — the unblocked baseline the row-blocking A/B
compares against, plus the chunk-count confirmation (whole prompt = one chunk).
"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/logs/logs/colibri-c12-workloads")
import c13_matrix as M

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
PASS = sys.argv[2] if len(sys.argv) > 2 else "p0"

PHASE_KEYS = ("prefill_s", "prefill_embedding_s", "prefill_chunk_count",
              "prefill_chunk_avg_s", "prefill_chunk_max_s",
              "prefill_deltanet_s", "prefill_attention_s", "prefill_moe_s",
              "prefill_shared_s", "prefill_router_s", "prefill_lm_head_s",
              "prefill_moe_hold_s", "prefill_moe_mm_s", "prefill_moe_act_s",
              "prefill_moe_bytes", "prefill_moe_experts", "prefill_moe_pairs",
              "prefill_moe_calls",
              "prompt_tokens", "engine_ttft_s")

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
    row = {"label": label,
           "client_total_s": round(time.time() - t0, 4)}
    for k in PHASE_KEYS:
        row[k] = t.get(k)
    row["output_sha12"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(json.dumps(row), flush=True)

def main():
    for name, pads, i in M.SIZES:
        q = f"[c15-p0 {PASS} {name}] " + M.make_q(i, pads)
        send_cold(M.messages(("user", q)), f"p0_{name}_cold")

if __name__ == "__main__":
    main()
