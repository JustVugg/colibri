#!/usr/bin/env python3
"""C13 phase 2b: exercise divergent_prefix classification on the live path.

Server is in COLI_PROMPT_CACHE=0 mode and holds s3's 2,112-token state.
d1 diverges early (wrong assistant turn) and is strictly longer, ending with a
genuinely different question, so the output should also differ.
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

def make_q(i, question=None):
    return FILLER.format(i=i) + PAD * 12 + (
        question or f" Question: in module {i}, which component flushes the snapshot, and when?")

A1 = "Answer: the writer flushes the snapshot after the forward pass completes."
A_WRONG = "Answer: the router selects eight experts per token and keeps them forever."

def messages(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]

body = {
    "model": "qwen3.6-colibri",
    "messages": messages(
        ("user", make_q(1)), ("assistant", A_WRONG),          # divergence ~520
        ("user", make_q(2)), ("assistant", A1),
        ("user", make_q(3)), ("assistant", A1),
        ("user", make_q(4)), ("assistant", A1),
        ("user", make_q(5, " Question: in module 5, how many experts does the router "
                          "select per token, and what stays always active?"))),
    "max_tokens": 48, "temperature": 0, "stream": False,
}
request = urllib.request.Request(
    BASE, data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=14400) as response:
    payload = json.loads(response.read())
timing = payload.get("timings", {})
text = payload["choices"][0]["message"]["content"]
print(json.dumps({
    "label": "d1_divergent_prefix",
    "prompt_tokens": timing.get("prompt_tokens"),
    "reused_prompt_tokens": timing.get("reused_prompt_tokens"),
    "reuse_source": timing.get("prefix_reuse_source"),
    "reuse_miss": timing.get("prefix_reuse_miss"),
    "prefill_s": timing.get("prefill_s"),
    "engine_ttft_s": timing.get("engine_ttft_s"),
    "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    "output_head": text[:80],
}, ensure_ascii=False))
