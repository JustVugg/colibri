#!/usr/bin/env python3
"""C14 pressure-arm workload: one 4-turn agent-session conversation per run,
actual engine outputs replayed as the next turn's assistant messages (the
2026-11-12 finding: scripted paraphrases silently skip live reuse).

Metric of interest (per the C14 re-scope): per-turn decode_tokens_per_sec
STABILITY under co-tenant memory pressure, and whether Q36_HYBRID_GB=2
insulates decode. TTFT deltas under pressure are secondary (prefill may
legitimately slow as pages re-fault; decode must not).

Runs against whatever server is on :8001 — the launcher sets the arm's env
and starts/stops the hog around the run.
"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads")
import c13_matrix as M

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
LABEL = sys.argv[2] if len(sys.argv) > 2 else "arm"

FILLER = (
    "Study these operation notes and keep them for later turns. "
    "Module {i}: the scheduler balances latency against fairness when the queue "
    "is deep, the compressor prefers short windows under memory pressure, the "
    "router selects eight experts per token and keeps a shared subset active, "
    "the writer flushes the snapshot only after the forward pass completes, "
    "the auditor reconciles the ledger after every checkpoint. "
)
PAD = ("the pipeline stays stable while tokens per second drift with context "
       "length, and the budget is a ceiling rather than a reservation, so "
       "observation beats assumption at every step of the rollout. ")

TURN_KEYS = ("decode_tokens_per_sec", "prompt_tokens", "reused_prompt_tokens",
             "reuse_source", "prefill_s", "prefill_attention_s", "prefill_moe_s",
             "engine_ttft_s")

def send(msgs, label, max_tokens=128):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        M.BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {"label": label, "client_total_s": round(time.time() - t0, 4)}
    for k in TURN_KEYS:
        row[k] = t.get(k)
    row["output_sha12"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(json.dumps(row), flush=True)
    return row, text

def main():
    tag = f"[c14-press {LABEL}] "
    hist = [("system", "You are a careful operations assistant. Answer briefly."),
            ("user", tag + FILLER.format(i=1) + PAD * 52 +
             " Question: in module 1, which component flushes the snapshot, and when?")]
    for turn in range(1, 5):
        row, text = send(M.messages(*hist), f"{LABEL}_t{turn}")
        if turn < 4:
            hist = hist + [("assistant", text),
                           ("user", tag + FILLER.format(i=turn + 1) + PAD * 6 +
                            f" Question: in module {turn + 1}, which component reconciles the ledger, and when?")]

if __name__ == "__main__":
    main()
