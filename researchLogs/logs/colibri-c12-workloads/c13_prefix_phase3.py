#!/usr/bin/env python3
"""C13 phase 3: validation of the three C13 fixes (exact-repeat resume, miss
labels, snapshot threshold).

Runs against the user's serving config (disk prompt cache on, new default
COLI_PROMPT_CACHE_MIN=256):
 p1: unique ~330-token prompt, cold. Snapshot stored (>= new min). Expect miss
     no_matching_disk_snapshot (no live state, disk found nothing).
 p2: exact repeat of p1. Expect FULL disk restore (reused == prompt tokens,
     prefill ~0, TTFT ~ restore time) and identical output.  <- fix 1
 p3: different ~270-token prompt. Live state (300) is longer, so expect the
     informative prompt_not_longer label to SURVIVE the disk branch
     (masking fix).                                             <- fix 2
 p4: exact repeat of p1 while live state holds p3: live diverges, disk saves.
     Expect full disk restore, output == p1.
 p5: exact repeat of the phase-1/2 t3 prompt (1,580 tokens). Output must equal
     the twice-verified cold reference 49ad13729105.
 p6: appended turn on the p1 conversation: live-state reuse (300 of ~630) with
     a tail prefill, the classic extension path unchanged.
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
T3_REF = "49ad13729105"   # cold reference, verified twice (phase 2 s1 + phase 1 r5)

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

def make_q(i, pads):
    return FILLER.format(i=i) + PAD * pads + (
        f" Question: in module {i}, which component flushes the snapshot, and when?")

A1 = "Answer: the writer flushes the snapshot after the forward pass completes."

def messages(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]

def send(msgs, label):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": 48, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {
        "label": label,
        "prompt_tokens": t.get("prompt_tokens"),
        "reused_prompt_tokens": t.get("reused_prompt_tokens"),
        "reuse_source": t.get("prefix_reuse_source"),
        "reuse_miss": t.get("prefix_reuse_miss"),
        "snapshot_restore_s": t.get("snapshot_restore_s"),
        "prefill_s": round(t.get("prefill_s") or 0.0, 4),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row

def main():
    p1q = make_q(7, 7)     # fresh module number: no snapshot from earlier phases
    p3q = make_q(8, 5)
    rows = {}

    rows["p1_cold_short"] = send(messages(("user", p1q)), "p1_cold_short")
    rows["p2_exact_repeat"] = send(messages(("user", p1q)), "p2_exact_repeat")
    rows["p3_shorter_new"] = send(messages(("user", p3q)), "p3_shorter_new")
    rows["p4_repeat_after_other"] = send(messages(("user", p1q)), "p4_repeat_after_other")
    rows["p5_t3_exact_repeat"] = send(messages(
        ("user", make_q(1, 12)), ("assistant", A1), ("user", make_q(2, 12)),
        ("assistant", A1), ("user", make_q(3, 12))), "p5_t3_exact_repeat")
    rows["p6_append_live"] = send(messages(
        ("user", p1q), ("assistant", A1), ("user", make_q(9, 5))), "p6_append_live")

    checks = [
        ("p1_stored_no_reuse", rows["p1_cold_short"]["reuse_source"] == "none"),
        ("p2_full_disk_restore",
         rows["p2_exact_repeat"]["reuse_source"] == "disk_snapshot"
         and rows["p2_exact_repeat"]["reused_prompt_tokens"]
             == rows["p2_exact_repeat"]["prompt_tokens"]
         and rows["p2_exact_repeat"]["prefill_s"] < 1.0),
        ("p2_output_equals_p1",
         rows["p2_exact_repeat"]["output_sha12"] == rows["p1_cold_short"]["output_sha12"]),
        ("p3_label_prompt_not_longer_survives",
         rows["p3_shorter_new"]["reuse_miss"] == "prompt_not_longer"),
        ("p4_full_restore_after_divergent_live",
         rows["p4_repeat_after_other"]["reuse_source"] == "disk_snapshot"
         and rows["p4_repeat_after_other"]["output_sha12"]
             == rows["p1_cold_short"]["output_sha12"]),
        ("p5_t3_output_matches_cold_ref",
         rows["p5_t3_exact_repeat"]["output_sha12"] == T3_REF),
        ("p6_live_extension_reuse",
         (rows["p6_append_live"]["reused_prompt_tokens"] or 0) > 0),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))

if __name__ == "__main__":
    main()
