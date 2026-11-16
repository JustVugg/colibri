#!/usr/bin/env python3
"""C16: Q4_K_M vs IQ2_XXS quantization comparison — one round of cold cells
plus the quality probe. The launcher runs this once per arm (4 alternating
restarts: iq2_r1, q4_r1, iq2_r2, q4_r2), cache off, single slot.

Cells: the calibrated matrix prompts at 1K/4K/12K (cold prefills, greedy,
128 tokens) — where the +66% expert bytes of Q4_K_M should land. Then a short
quality probe: three distinct questions, 256 tokens each, whose full text is
recorded so the two quants' answers can be compared side by side.

stdout is the result stream (the launcher tees it per round).
"""
import hashlib, json, sys, urllib.request
from pathlib import Path

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c13_matrix import make_q, messages

PROBE = [
    ("q1", "A train leaves at 3:45 PM and travels for 2 hours 50 minutes. "
           "What time does it arrive? Show the reasoning briefly."),
    ("q2", "Write a one-line Python function that returns the n-th Fibonacci "
           "number iteratively, then state its time complexity."),
    ("q3", "Explain in two sentences why importance-matrix quantization "
           "usually preserves quality better than naive round-to-nearest "
           "at the same bit width."),
]

def send(prompt, label, max_tokens):
    body = {"model": "qwen3.6-colibri", "messages": messages(("user", prompt)),
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=14400) as resp:
            payload = json.loads(resp.read())
    except Exception as e:
        print(json.dumps({"label": label, "error": f"{type(e).__name__}: {e}"}), flush=True)
        return
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {"label": label,
           "prompt_tokens": t.get("prompt_tokens"),
           "reused_prompt_tokens": t.get("reused_prompt_tokens"),
           "prefill_s": t.get("prefill_s"),
           "prefill_attention_s": t.get("prefill_attention_s"),
           "prefill_deltanet_s": t.get("prefill_deltanet_s"),
           "prefill_moe_s": t.get("prefill_moe_s"),
           "engine_ttft_s": t.get("engine_ttft_s"),
           "decode_tokens_per_sec": t.get("decode_tokens_per_sec"),
           "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12]}
    if label.startswith("q"):
        row["text"] = text
    print(json.dumps(row), flush=True)

def main():
    for pads, i, name in ((26, 300, "1k"), (114, 310, "4k"), (349, 320, "12k")):
        send(make_q(i, pads), f"cold_{name}", 128)
    for label, prompt in PROBE:
        send(prompt, label, 256)
    print("C16 round done", flush=True)

if __name__ == "__main__":
    main()
