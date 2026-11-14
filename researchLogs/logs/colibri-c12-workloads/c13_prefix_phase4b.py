#!/usr/bin/env python3
"""C13 phase 4b: fixed reset probe and the cold reference for phase-4 p3.

Modes:
  reset  - fresh process, snapshots on: system-prompt change + a question long
           enough to exceed the state held by prior turns in THIS process
           (t1 two-turn conversation first, then the longer reset prompt).
           Expect: t1 stores state; reset prompt is longer but diverges at
           position 0 -> miss divergent_prefix, 0 reuse, full prefill, and an
           output different from the non-system answer.
  coldref- fresh process, COLI_PROMPT_CACHE=0: exact cold prefill of phase-4
           p3's appended prompt. Output must equal p3's sha (a85ee064470b);
           TTFT is the controlled same-prompt cold reference (p3 reused 3,458
           tokens and prefilled only a tail).
"""
import hashlib
import json
import sys
import urllib.request

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "reset"
P3_SHA = "a85ee064470b"

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
        "prefill_s": round(t.get("prefill_s") or 0.0, 4),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row

def main():
    if MODE == "reset":
        state = send(messages(
            ("user", make_q(14, 8)), ("assistant", B1), ("user", make_q(15, 3))),
            "s1_state")
        reset = send(messages(
            ("system", "You are a terse inline critic. Answer in one short sentence."),
            ("user", make_q(16, 20))), "s2_reset_system")
        longer = reset["prompt_tokens"] > state["prompt_tokens"]
        checks = [
            ("reset_prompt_longer_than_state", longer),
            ("reset_divergent_no_reuse",
             reset["reuse_source"] == "none"
             and reset["reuse_miss"] == "divergent_prefix"
             and reset["reused_prompt_tokens"] == 0),
        ]
        print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))
    elif MODE == "coldref":
        row = send(messages(
            ("user", make_q(20, 96)), ("assistant", A1), ("user", make_q(21, 8))),
            "c1_cold_append_ref")
        checks = [
            ("cold_ref_full_prefill", row["reuse_source"] == "none"),
            ("cold_ref_output_equals_reused",
             row["output_sha12"] == P3_SHA),
        ]
        print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}))
    else:
        raise SystemExit(f"unknown mode {MODE}")

if __name__ == "__main__":
    main()
