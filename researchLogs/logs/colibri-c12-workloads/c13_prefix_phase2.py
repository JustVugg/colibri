#!/usr/bin/env python3
"""C13 phase 2: restored/reused-state output correctness vs cold-prefill references.

Runs on a server with COLI_PROMPT_CACHE=0 (no disk snapshots, empty live state):
 s1: cold prefill of the turn-3 prompt; output must equal phase-1 r5 (disk-restored).
 s2: appended turn-4; live state should reuse s1's tokens, prefill the tail.
 s3: longer-than-state prompt diverging early; expect divergent_prefix, full prefill.
 s2c: exact repeat of s2 after s3 (shorter than s3's state) -> forced full cold
     prefill; output must equal s2's reused-state output.
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

A1 = "Answer: the writer flushes the snapshot after the forward pass completes."
A_WRONG = "Answer: the router selects eight experts per token and keeps them forever."
Q1, Q2, Q3, Q4 = make_q(1), make_q(2), make_q(3), make_q(4)

def messages(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]

def send(msgs, label):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": MAX_TOKENS, "temperature": 0, "stream": False}
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
        "prefill_s": timing.get("prefill_s"),
        "engine_ttft_s": timing.get("engine_ttft_s"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row, text

def main():
    texts = {}
    rows = {}
    rows["s1_cold_t3_ref"], texts["s1"] = send(
        messages(("user", Q1), ("assistant", A1), ("user", Q2), ("assistant", A1), ("user", Q3)),
        "s1_cold_t3_ref")
    rows["s2_append_live"], texts["s2"] = send(
        messages(("user", Q1), ("assistant", A1), ("user", Q2), ("assistant", A1),
                 ("user", Q3), ("assistant", A1), ("user", Q4)),
        "s2_append_live")
    rows["s3_divergent_longer"], texts["s3"] = send(
        messages(("user", Q1), ("assistant", A_WRONG), ("user", Q2), ("assistant", A1),
                 ("user", Q3), ("assistant", A1), ("user", Q4)),
        "s3_divergent_longer")
    rows["s2c_cold_t4_ref"], texts["s2c"] = send(
        messages(("user", Q1), ("assistant", A1), ("user", Q2), ("assistant", A1),
                 ("user", Q3), ("assistant", A1), ("user", Q4)),
        "s2c_cold_t4_ref")

    # 49ad13729105 was phase-1 r5 (disk-snapshot-restored run of the same prompt).
    checks = [
        ("s1_matches_phase1_r5_disk_restore", rows["s1_cold_t3_ref"]["output_sha12"] == "49ad13729105"),
        ("s2c_matches_s2_reused_live_state", texts["s2c"] == texts["s2"]),
        ("s3_differs_from_s2", texts["s3"] != texts["s2"]),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))

if __name__ == "__main__":
    main()
