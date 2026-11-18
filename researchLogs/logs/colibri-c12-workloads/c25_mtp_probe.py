#!/usr/bin/env python3
"""One fixed greedy decode request for a single C25 MTP probe arm."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "phaseC25_mtp_probe.jsonl"
KEY = "-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk"
PROMPT = (
    "Write a detailed Python tutorial explaining how to implement a queue using "
    "two stacks. Explain the invariant and give a worked example. Be thorough."
)
body = {
    "model": "qwen3.6-colibri",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": 256,
    "temperature": 0,
    "stream": False,
}
req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
)
start = time.monotonic()
with urllib.request.urlopen(req, timeout=14400) as response:
    payload = json.loads(response.read())
elapsed = time.monotonic() - start
text = payload["choices"][0]["message"]["content"]
row = {
    "tag": sys.argv[1],
    "arm": sys.argv[2],
    "sha256": hashlib.sha256(text.encode()).hexdigest(),
    "text": text,
    "elapsed_client_s": elapsed,
    "timings": payload.get("timings", {}),
    "usage": payload.get("usage", {}),
}
with OUT.open("a") as f:
    f.write(json.dumps(row) + "\n")
print(json.dumps({k: row[k] for k in ("tag", "arm", "sha256", "elapsed_client_s", "timings", "usage")}))
