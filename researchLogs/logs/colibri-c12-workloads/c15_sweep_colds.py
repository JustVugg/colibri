#!/usr/bin/env python3
"""C15 adoption sweep: the long-cold arm. Sends one 16K and one 24K cold
prompt (COLI_PROMPT_CACHE=0 makes every request a full prefill of the SAME
tagged prompt on both configs — the sha-agreement + timing evidence for sizes
the C15 A/B did not cover). Run after the agent-ext arm in each round.
"""
import hashlib
import json
import sys
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads")
import c13_matrix as M

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
LABEL = sys.argv[2] if len(sys.argv) > 2 else "sweep"

# pads ≈ (target - filler)/34.5 tokens-per-pad; the recorded prompt_tokens is
# what counts, the targets only need to be far past the 12K A/B point.
COLD_SIZES = [("16k", 330, 470), ("24k", 340, 707)]

def send_cold(msgs, label):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": 128, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        M.BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    import time
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {
        "label": label, "client_total_s": round(time.time() - t0, 1),
        "prefill_s": t.get("prefill_s"),
        "prefill_attention_s": t.get("prefill_attention_s"),
        "prefill_deltanet_s": t.get("prefill_deltanet_s"),
        "prefill_moe_s": t.get("prefill_moe_s"),
        "prefill_chunk_count": t.get("prefill_chunk_count"),
        "prompt_tokens": t.get("prompt_tokens"),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "decode_tokens_per_sec": t.get("decode_tokens_per_sec"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)

def main():
    for name, i, pads in COLD_SIZES:
        q = "[c15-sweep] " + M.make_q(i, pads)
        send_cold(M.messages(("user", q)), f"{LABEL}_{name}")

if __name__ == "__main__":
    main()
