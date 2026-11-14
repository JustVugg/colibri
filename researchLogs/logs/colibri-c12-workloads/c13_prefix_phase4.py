#!/usr/bin/env python3
"""C13 phase 4: truncation and reset probes, plus the larger append comparison.

Mode A (default, snapshots on) runs, in one process:
  t1: two-turn conversation -> live state ~700 tokens.
  t2: truncation - drop the assistant turn, send only the first user turn
      (shorter than state). Expect miss prompt_not_longer, 0 reuse, full
      prefill, output recorded.
  t3: reset - new system prompt plus a fresh question (longer than state,
      diverges at position 0). Expect miss divergent_prefix, 0 reuse, full
      prefill, and output different from t2's answer.
  p1: cold ~4K single-turn prompt. Snapshot stored (>= 256 minimum).
  p2: exact repeat of p1. Expect full disk restore, prefill ~0, TTFT ~
      restore time, output == p1.
  p3: appended turn on the p1 conversation (~4.5K). Expect live-state reuse
      of the ~4K prefix with a short tail prefill.
Mode B (env COLI_PROMPT_CACHE=0, fresh process, run by the caller):
  c1: exact cold prefill of p3's prompt. Output must equal p3 (correctness)
      and its TTFT is the controlled same-prompt cold reference for p3.
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""

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
B1 = "Answer: the compressor prefers short windows under memory pressure."

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
    p1q = make_q(20, 96)   # ~4K tokens, fresh module number
    p3q = make_q(21, 8)
    rows = {}

    rows["t1_state"] = send(messages(
        ("user", make_q(11, 8)), ("assistant", B1), ("user", make_q(12, 3))), "t1_state")
    rows["t2_truncated"] = send(messages(
        ("user", make_q(11, 8))), "t2_truncated")
    rows["t3_reset_system"] = send(messages(
        ("system", "You are a terse inline critic. Answer in one short sentence."),
        ("user", make_q(13, 8))), "t3_reset_system")

    rows["p1_cold_4k"] = send(messages(("user", p1q)), "p1_cold_4k")
    rows["p2_exact_repeat_4k"] = send(messages(("user", p1q)), "p2_exact_repeat_4k")
    rows["p3_append_4k"] = send(messages(
        ("user", p1q), ("assistant", A1), ("user", p3q)), "p3_append_4k")

    checks = [
        ("t2_truncation_no_reuse_labeled",
         rows["t2_truncated"]["reuse_source"] == "none"
         and rows["t2_truncated"]["reuse_miss"] == "prompt_not_longer"),
        ("t3_reset_no_reuse_divergent",
         rows["t3_reset_system"]["reuse_source"] == "none"
         and rows["t3_reset_system"]["reuse_miss"] == "divergent_prefix"
         and rows["t3_reset_system"]["output_sha12"]
             != rows["t2_truncated"]["output_sha12"]),
        ("p1_4k_cold_full_prefill",
         rows["p1_cold_4k"]["reuse_source"] == "none"
         and (rows["p1_cold_4k"]["prompt_tokens"] or 0) > 3900),
        ("p2_4k_full_restore",
         rows["p2_exact_repeat_4k"]["reuse_source"] == "disk_snapshot"
         and rows["p2_exact_repeat_4k"]["reused_prompt_tokens"]
             == rows["p2_exact_repeat_4k"]["prompt_tokens"]
         and rows["p2_exact_repeat_4k"]["prefill_s"] < 1.0),
        ("p2_output_equals_p1",
         rows["p2_exact_repeat_4k"]["output_sha12"] == rows["p1_cold_4k"]["output_sha12"]),
        ("p3_live_reuse_most_of_4k",
         rows["p3_append_4k"]["reuse_source"] == "live_state"
         and (rows["p3_append_4k"]["reused_prompt_tokens"] or 0)
             >= rows["p1_cold_4k"]["prompt_tokens"]),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))

if __name__ == "__main__":
    main()
