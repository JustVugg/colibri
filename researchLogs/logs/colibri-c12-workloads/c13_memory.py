#!/usr/bin/env python3
"""C13 memory + TTFT + sustained-decode correlation run.

Companion to c13_matrix.py: same prompt families, but every request is
streaming so the client's own first-token time is measured, and it is meant to
run under c13_mem_sampler.py so per-request peak/baseline engine RSS can be
joined to each window afterwards (c13_mem_join.py).

Per size (1k/4k/12k):
  cold      - divergent-tagged prompt (full prefill, snapshot stored)
  restore1  - exact repeat (disk restore + 128-token decode)
  restore2  - exact repeat again (restore + decode, second observation)
The restore rows double as the sustained-decode rows: the window after the
~0.1-0.5 s restore is decode over 128 tokens, and the sampler timeline covers
it. Client-instrumentation overhead = client_ttft_s - engine_ttft_s.
"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/logs/logs/colibri-c12-workloads")
import c13_matrix as M   # FILLER, PAD, SIZES, make_q, messages (no side effects)

BASE = M.BASE
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "mem"
PASS = sys.argv[3] if len(sys.argv) > 3 else "1"   # unique tag per pass: forces a true cold

def send_stream(msgs, label, cell):
    body = {"model": "qwen3.6-colibri", "messages": msgs,
            "max_tokens": 128, "temperature": 0, "stream": True}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    text_parts = []
    first_chunk_at = None
    timings = None
    with urllib.request.urlopen(req, timeout=14400) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            payload = json.loads(line[6:])
            choice = (payload.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or (choice.get("message") or {}).get("content")
            if piece:
                if first_chunk_at is None:
                    first_chunk_at = time.time()
                text_parts.append(piece)
            if choice.get("timings"):
                timings = choice["timings"]
    t_end = time.time()
    text = "".join(text_parts)
    t = timings or {}
    row = {
        "mode": MODE, "cell": cell, "label": label,
        "t_start": t0, "t_first": first_chunk_at, "t_end": t_end,
        "client_ttft_s": round((first_chunk_at - t0) if first_chunk_at else None, 4),
        "client_total_s": round(t_end - t0, 4),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "instr_overhead_s": (round((first_chunk_at - t0) - (t.get("engine_ttft_s") or 0), 4)
                             if first_chunk_at and t.get("engine_ttft_s") else None),
        "prompt_tokens": t.get("prompt_tokens"),
        "reused_prompt_tokens": t.get("reused_prompt_tokens"),
        "reuse_source": t.get("prefix_reuse_source"),
        "reuse_miss": t.get("prefix_reuse_miss"),
        "snapshot_restore_s": t.get("snapshot_restore_s"),
        "prefill_s": t.get("prefill_s"),
        "decode_tokens_per_sec": t.get("decode_tokens_per_sec"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row

def main():
    for name, pads, i in M.SIZES:
        q = M.make_q(i, pads)
        # Self-contained size cell: the tagged prompt diverges (cold prefill,
        # stores its snapshot), then the SAME tagged prompt repeats twice —
        # those are genuine disk restores of the snapshot the cold just
        # stored. (Untagged repeats would depend on older snapshots that a
        # third-party session may have evicted under the retention budget.)
        tagged = f"[mem-run p{PASS} tag {name}] " + q
        send_stream(M.messages(("user", tagged)), f"mem_{name}_cold", f"mem_{name}_cold")
        send_stream(M.messages(("user", tagged)), f"mem_{name}_restore1", f"mem_{name}_restore")
        send_stream(M.messages(("user", tagged)), f"mem_{name}_restore2", f"mem_{name}_restore")

if __name__ == "__main__":
    main()
