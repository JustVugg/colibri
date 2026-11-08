#!/usr/bin/env python3
"""Turn-timing harness for docs/experiments/dsv41-trunk-first-vram.json.
Usage: bench <tag> <warmup> <measured> [max_tokens] against a coli serve on 127.0.0.1:8010.
Prints one JSON line per turn (see the raw evidence file for the recorded arms)."""
import http.client, json, sys, time, hashlib

PORT = 8010
PROMPT = "Describe the nature of a transformer."

def run_turn(tag, name, max_tokens=96):
    body = json.dumps({
        "model": "deepseek-v4.1-flash-colibri",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "stream": True,
    })
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"[{tag}] HTTP {resp.status}: {resp.read()[:200]}", file=sys.stderr); sys.exit(1)
    first = None; last = t0; ntok = 0; usage = None; text = []
    for line in resp:
        line = line.strip()
        if not line.startswith(b"data: "): continue
        payload = line[6:]
        if payload == b"[DONE]": break
        try: d = json.loads(payload)
        except ValueError: continue
        now = time.perf_counter()
        if d.get("usage"): usage = d["usage"]
        delta = d.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            if first is None: first = now
            last = now; ntok += 1; text.append(delta["content"])
    conn.close()
    total = last - t0
    ttft = (first - t0) if first else float("nan")
    gen = (last - first) if first else float("nan")
    comp = (usage or {}).get("completion_tokens", ntok)
    full = "".join(text)
    rec = {"turn": name, "completion_tokens": comp, "streamed": ntok,
           "ttft_s": round(ttft, 2), "decode_tok_s": round((ntok - 1) / gen, 3),
           "end_to_end_tok_s": round(comp / max(total, 1e-9), 3), "wall_s": round(total, 2),
           "text_sha": hashlib.sha256(full.encode()).hexdigest()[:16], "text_len": len(full)}
    print(json.dumps(rec)); sys.stdout.flush()

tag = sys.argv[1]
warmup = int(sys.argv[2]); measured = int(sys.argv[3])
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 96
for i in range(warmup):
    run_turn(tag, f"warmup{i+1}", max_tokens)
for i in range(measured):
    run_turn(tag, f"meas{i+1}", max_tokens)
