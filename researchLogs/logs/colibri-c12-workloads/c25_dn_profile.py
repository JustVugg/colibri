#!/usr/bin/env python3
"""One cold ~11K-token prefill for the C25 DeltaNet CPU profile."""
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
doc = (HERE / "c23_eval_doc.txt").read_text()
prompt = (doc + "\n\n---\n\n") * 2 + (
    "\n\nUsing the document above, answer only with the ledger total."
)
body = {
    "model": "qwen3.6-colibri",
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "temperature": 0,
    "stream": False,
}
key = "-4Jc6egKcHmI-1okTj4_MUWOU4etmppT8LLG-wxzUBk"
req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=14400) as response:
    payload = json.loads(response.read())
row = {
    "prompt_tokens": payload.get("timings", {}).get("prompt_tokens"),
    "completion_tokens": payload.get("usage", {}).get("completion_tokens"),
    "timings": payload.get("timings", {}),
    "usage": payload.get("usage", {}),
    "answer": payload["choices"][0]["message"]["content"],
}
(HERE / "phaseC25_dn_profile_12k.json").write_text(json.dumps(row, indent=2) + "\n")
print(json.dumps(row, indent=2))
