#!/usr/bin/env python3
"""Capture the matrix 4K prompt's output text (and timings) once, for the
C15 flip-text record. Run against :8001 in whatever config is up; the label
goes in the output filename, not here."""
import json, sys, urllib.request
sys.path.insert(0, "/Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads")
from c13_matrix import make_q, messages

KEY = "-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk"
q = make_q(310, 114)  # the matrix 4K prompt
body = {"model": "qwen3.6-colibri", "messages": messages(("user", q)),
        "max_tokens": 128, "temperature": 0, "stream": False}
req = urllib.request.Request(
    "http://localhost:8001/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
payload = json.loads(urllib.request.urlopen(req, timeout=14400).read())
t = payload.get("timings", {})
print(json.dumps({
    "text": payload["choices"][0]["message"]["content"],
    "prompt_tokens": t.get("prompt_tokens"),
    "prefill_s": t.get("prefill_s"),
    "engine_ttft_s": t.get("engine_ttft_s"),
}, indent=1))
