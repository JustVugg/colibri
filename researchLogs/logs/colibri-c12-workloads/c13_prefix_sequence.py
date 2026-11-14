#!/usr/bin/env python3
"""C13 prefix-reuse validation sequence against a running Qwen3.6 server.

Sends a deterministic multi-turn sequence and records, per request:
prompt/reused tokens, reuse source/miss, snapshot restore/store, prefill and
TTFT, plus the generated text for determinism comparison. No prompt text is
printed; only hashes and counters.
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MAX_TOKENS = 48

FILLER = (
    "Review the following revision notes and keep them available for later turns. "
    "Module {i}: the scheduler trades latency for fairness when the queue is deep, "
    "the compressor prefers short windows under memory pressure, the router selects "
    "eight experts per token and keeps a shared subset always active, the writer "
    "flushes the snapshot only after the forward pass completes. "
)
PAD = ("the pipeline stays stable while tokens per second drift with context length, "
       "and the budget is a ceiling rather than a reservation, so observation beats "
       "assumption at every step of the rollout. ")

def make_q(i):
    return FILLER.format(i=i) + PAD * 12 + (
        f" Question: in module {i}, which component flushes the snapshot, and when?")

def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]

def send(messages, label):
    body = {
        "model": "qwen3.6-colibri",
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
    }
    request = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=14400) as response:
        payload = json.loads(response.read())
    timing = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {
        "label": label,
        "prompt_tokens": timing.get("prompt_tokens"),
        "reused_prompt_tokens": timing.get("reused_prompt_tokens"),
        "reuse_source": timing.get("prefix_reuse_source"),
        "reuse_miss": timing.get("prefix_reuse_miss"),
        "snapshot_restore_s": timing.get("snapshot_restore_s"),
        "snapshot_store_s": timing.get("snapshot_store_s"),
        "prefill_s": timing.get("prefill_s"),
        "engine_ttft_s": timing.get("engine_ttft_s"),
        "decode_tokens_per_sec": timing.get("decode_tokens_per_sec"),
        "output_sha12": hash_text(text),
    }
    return row, text

def messages(*pairs):
    out = []
    for role, content in pairs:
        out.append({"role": role, "content": content})
    return out

def main():
    q1, q2, q3 = make_q(1), make_q(2), make_q(3)
    q1_div = make_q(1).replace("Review the following", "Please skim these")  # divergence near start
    a1 = "Answer: the writer flushes the snapshot after the forward pass completes."
    rows = []
    texts = {}

    seq = [
        ("r1_cold_first", messages(("user", q1))),
        ("r2_exact_repeat", messages(("user", q1))),
        ("r3_turn2_append", messages(("user", q1), ("assistant", a1), ("user", q2))),
        ("r4_turn2_repeat", messages(("user", q1), ("assistant", a1), ("user", q2))),
        ("r5_turn3_append", messages(("user", q1), ("assistant", a1), ("user", q2),
                                     ("assistant", a1), ("user", q3))),
        ("r6_divergent_start", messages(("user", q1_div))),
        ("r7_mid_divergence", messages(("user", q1), ("assistant", "Answer: the router "
                                       "selects eight experts per token."), ("user", q2))),
    ]
    for label, msgs in seq:
        row, text = send(msgs, label)
        texts[label] = text
        rows.append(row)
        print(json.dumps(row), flush=True)

    checks = [
        ("r2_vs_r1_determinism", texts["r2_exact_repeat"] == texts["r1_cold_first"]),
        ("r4_vs_r3_determinism", texts["r4_turn2_repeat"] == texts["r3_turn2_append"]),
        ("r6_differs_from_r1", texts["r6_divergent_start"] != texts["r1_cold_first"]),
        ("r7_differs_from_r3", texts["r7_mid_divergence"] != texts["r3_turn2_append"]),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))

if __name__ == "__main__":
    main()
