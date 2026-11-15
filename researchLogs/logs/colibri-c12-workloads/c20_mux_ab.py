#!/usr/bin/env python3
"""C20 five-point validation: the disk prompt cache under KV_SLOTS=2.

Two conversations (A on cache_slot 0, B on cache_slot 1), 4K-class prompts,
greedy, 128 output tokens, per the C20 scoping doc's protocol:

  round 1 (one process): A_cold + B_cold concurrently (two slots live),
                         then A_rep + B_rep concurrently (exact repeats)
  restart
  round 2 (fresh process): A_again + B_again concurrently -- the cross-restart
                         disk restore (cache-on arm) vs full colds (cache-off)
  corruption: truncate every snapshot payload on disk, then A_corr + B_corr --
                         both must fall through to a full prefill and match the
                         cold output; the slot must stay usable.

Expected sha behaviour on one build (sgemm is deterministic per build): every
run of the same prompt produces the same sha in both arms and both rounds.
The arms differ only in the reuse_source census and the timings.

stdout is the result stream (the launcher tees it to the arm's .jsonl); rows
record errors instead of raising so one failed request cannot silence its
concurrent partner.
"""
import hashlib, json, sys, threading, urllib.request
from pathlib import Path

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c13_matrix import make_q, messages

PC_DIR = HERE / "c20_pcache"

def send(prompt, slot, label):
    body = {"model": "qwen3.6-colibri", "messages": messages(("user", prompt)),
            "max_tokens": 128, "temperature": 0, "stream": False, "cache_slot": slot}
    req = urllib.request.Request(
        BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=14400) as resp:
            payload = json.loads(resp.read())
    except Exception as e:                       # record, never kill the partner thread
        row = {"label": label, "slot": slot, "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(row), flush=True)
        return row
    t = payload.get("timings", {})
    text = payload["choices"][0]["message"]["content"]
    row = {"label": label, "slot": slot,
           "prompt_tokens": t.get("prompt_tokens"),
           "reused_prompt_tokens": t.get("reused_prompt_tokens"),
           "reuse_source": t.get("prefix_reuse_source"),
           "reuse_miss": t.get("prefix_reuse_miss"),
           "prefill_s": t.get("prefill_s"),
           "engine_ttft_s": t.get("engine_ttft_s"),
           "decode_tokens_per_sec": t.get("decode_tokens_per_sec"),
           "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12]}
    print(json.dumps(row), flush=True)
    return row

def pair_concurrent(jobs):
    """jobs: [(label, prompt, slot)] -- send concurrently, mux both slots."""
    rows = [None] * len(jobs)
    def work(i):
        label, prompt, slot = jobs[i]
        rows[i] = send(prompt, slot, label)
    ts = [threading.Thread(target=work, args=(i,)) for i in range(len(jobs))]
    for t in ts: t.start()
    for t in ts: t.join()
    return rows

def corrupt_snapshots():
    """Point 4: truncate every snapshot payload (hash will fail at apply)."""
    n = 0
    for f in sorted(PC_DIR.glob("pfx_*.bin")):
        b = f.read_bytes()
        f.write_bytes(b[: int(len(b) * 0.7)])
        n += 1
    print(json.dumps({"corrupted_files": n}), flush=True)

def main():
    QA, QB = make_q(500, 114), make_q(600, 114)   # distinct 4K-class prompts
    round1 = [("A_cold", QA, 0), ("B_cold", QB, 1),
              ("A_rep", QA, 0), ("B_rep", QB, 1)]
    round2 = [("A_again", QA, 0), ("B_again", QB, 1),
              ("A_corr", QA, 0), ("B_corr", QB, 1)]
    pair_concurrent(round1[0:2])
    pair_concurrent(round1[2:4])
    if len(sys.argv) > 3 and sys.argv[3] == "round2":
        pair_concurrent(round2[0:2])
        corrupt_snapshots()
        pair_concurrent(round2[2:4])
    print("C20 arm stage done", flush=True)

if __name__ == "__main__":
    main()
