#!/usr/bin/env python3
"""C13/C12 repeated representative matrix: 1K/4K/12K prompts, cold vs warm vs
restored separation, speculation on vs off, n repeats per cell.

Server configurations the caller launches per mode:
  cold        - COLI_PROMPT_CACHE=0 (and optionally COLI_LOOKUP=0 for the
                speculation-off variant). Every request is a FULL prefill: the
                live state always holds prompt + generated tokens, so an exact
                repeat is declined by the no-rewind rule and prefills from
                scratch. Repeats within one process are therefore cold-STATE
                prefills (the first request of the process is a true fresh
                cold); the sequence index is recorded so the summarizer can
                separate them.
  reuse       - snapshots on, fresh process. Per size: one true cold prefill
                (also stores the snapshot), an exact repeat (full disk
                restore, zero prefill), an appended turn (prefix reuse + tail
                prefill), a second exact repeat. All in one process.
  restart     - snapshots on, fresh process that has seen none of the prompts:
                exact repeats only, so every row is a pure cross-process disk
                restore.

Prompt sizes are calibrated on this model + template from phases 4-5:
tokens ~= 98 + 35*pads for filler A. pads 26 -> ~1,008; 114 -> ~4,088;
349 -> ~12,313. max_tokens 128 everywhere, greedy, concurrency 1.
"""
import hashlib
import json
import os
import statistics
import sys
import urllib.request
from pathlib import Path

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "cold"
HERE = Path(__file__).resolve().parent

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

SIZES = [("1k", 26, 300), ("4k", 114, 310), ("12k", 349, 320)]  # (name, pads, module i)

def make_q(i, pads):
    return FILLER.format(i=i) + PAD * pads + (
        f" Question: in module {i}, which component flushes the snapshot, and when?")

A1 = "Answer: the writer flushes the snapshot after the forward pass completes."  # unused since the actual-output replay rule; kept for reference

def messages(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]

def send(msgs, label, cell):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": 128, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {
        "mode": MODE, "cell": cell, "label": label,
        "prompt_tokens": t.get("prompt_tokens"),
        "reused_prompt_tokens": t.get("reused_prompt_tokens"),
        "reuse_source": t.get("prefix_reuse_source"),
        "reuse_miss": t.get("prefix_reuse_miss"),
        "snapshot_restore_s": t.get("snapshot_restore_s"),
        "snapshot_store_s": t.get("snapshot_store_s"),
        "prefill_s": round(t.get("prefill_s") or 0.0, 4),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "decode_tokens_per_s": t.get("decode_tokens_per_sec"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row, text

def main():
    if MODE == "cold":
        for name, pads, i in SIZES:
            q = make_q(i, pads)
            for r in range(5 if name == "1k" else 3):
                send(messages(("user", q)), f"cold_{name}_r{r}", f"cold_{name}")
    elif MODE == "reuse":
        for name, pads, i in SIZES:
            q = make_q(i, pads)
            qt = make_q(i + 1, 4)
            # The append turn replays the engine's ACTUAL cold-run output (the
            # 2026-11-12 finding: a scripted assistant turn diverges from the
            # held generated ids and silently skips the live-reuse path).
            cold, a1_text = send(messages(("user", q)), f"reuse_{name}_cold", f"reuse_{name}_cold")
            send(messages(("user", q)), f"reuse_{name}_repeat1", f"reuse_{name}_restore")
            send(messages(("user", q), ("assistant", a1_text), ("user", qt)),
                 f"reuse_{name}_append", f"reuse_{name}_append")
            send(messages(("user", q)), f"reuse_{name}_repeat2", f"reuse_{name}_restore")
    elif MODE == "restart":
        for name, pads, i in SIZES:
            q = make_q(i, pads)
            for r in range(3):
                send(messages(("user", q)), f"restart_{name}_r{r}", f"restart_{name}_restore")
    elif MODE == "decode":
        # Decode-rate probe: the speculation toggle changes DECODE (prompt
        # lookup drafts), not prefill. Small prompt, long generation, n=3.
        q = make_q(400, 26)
        for r in range(3):
            send(messages(("user", q)), f"decode_r{r}", "decode_probe")
    else:
        raise SystemExit(f"unknown mode {MODE}")

if __name__ == "__main__":
    main()
