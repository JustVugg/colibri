#!/usr/bin/env python3
"""C13 phase 5: multiplexed serving (KV_SLOTS=2) and the retention budget on a
real multi-turn session.

Modes:
  mux       - server running with --kv-slots 2 and a SCRATCH cache dir.
              Conversations A and B (distinct fillers) interleave: both t1s
              concurrently, then both t2s concurrently. Each t2 replays the
              ACTUAL engine output of that conversation's t1 as its assistant
              turn (the 2026-11-12 finding: scripted paraphrases silently skip
              live reuse). Expected: t2 reuse_source=live_state with the whole
              held state reused; NO disk snapshots written in mux mode (the
              pcache is gated to KV_SLOTS==1); no reuse source ever
              disk_snapshot. Outputs are saved to phase5_mux_outputs.json for
              the cold-reference comparison (mode muxcoldref).
  muxcoldref- server running single-slot (KV_SLOTS=1) with COLI_PROMPT_CACHE=0.
              Replays the mux run's assistant texts (phase5_mux_outputs.json)
              and prefills each conversation alone; outputs must match the
              mux run's shas exactly (interleaved decoding must not change
              greedy results).
  retain    - server single-slot, snapshots on, SCRATCH cache dir, and
              COLI_PROMPT_CACHE_GB=0.5. Three 3-turn agent sessions (A, B, C;
              assistant turns replay actual outputs), then an exact repeat of
              session C's last prompt (newest snapshot: expect a full
              disk_snapshot restore) and of session A's first prompt (oldest:
              expect eviction; reuse 0 with an informative miss and an output
              identical to the first occurrence). The driver prints the cache
              directory's file count and total MB after every turn so the
              eviction order and the byte budget can be read off the server
              log's [PCACHE] lines alongside.
"""
import hashlib
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path

BASE = "http://localhost:8001/v1/chat/completions"
KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "mux"
HERE = Path(__file__).resolve().parent
MUX_OUT = HERE / "phase5_mux_outputs.json"
CACHE_DIR = os.environ.get("PHASE5_CACHE_DIR", "")

FILLER_A = (
    "Review the following revision notes and keep them available for later turns. "
    "Module {i}: the scheduler trades latency for fairness when the queue is deep, "
    "the compressor prefers short windows under memory pressure, the router selects "
    "eight experts per token and keeps a shared subset always active, the writer "
    "flushes the snapshot only after the forward pass completes. "
)
FILLER_B = (
    "Consider these deployment guidelines for the discussion below. "
    "Stage {i}: the dispatcher batches small requests until the window closes, "
    "the inspector rejects any frame whose checksum drifts from the header, "
    "the courier retries with an exponential backoff capped at one minute, "
    "the archivist writes the ledger only once the quorum acknowledges. "
)
PAD = ("the pipeline stays stable while tokens per second drift with context length, "
       "and the budget is a ceiling rather than a reservation, so observation beats "
       "assumption at every step of the rollout. ")

def qa(i, pads):
    return FILLER_A.format(i=i) + PAD * pads + (
        f" Question: in module {i}, which component flushes the snapshot, and when?")

def qb(i, pads):
    return FILLER_B.format(i=i) + PAD * pads + (
        f" Question: in stage {i}, which component writes the ledger, and when?")

# pads_a=9/pads_b=8: the gateway's conversation->slot hash (system + first user
# message, sha1 mod kv_slots) puts A on slot 0 and B on slot 1, so the two
# conversations do not collide and degrade to re-prefill (#634's fallback).

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
        "snapshot_restore_s": t.get("snapshot_restore_s"),
        "prefill_s": round(t.get("prefill_s") or 0.0, 4),
        "engine_ttft_s": t.get("engine_ttft_s"),
        "output_sha12": hashlib.sha256(text.encode()).hexdigest()[:12],
    }
    print(json.dumps(row), flush=True)
    return row, text

def cache_state():
    if not CACHE_DIR:
        return None
    files = sorted(Path(CACHE_DIR).glob("pfx_*.bin"))
    return {"files": len(files),
            "total_mb": round(sum(f.stat().st_size for f in files) / 1048576.0, 1)}

def note_cache():
    st = cache_state()
    if st:
        print(json.dumps({"cache": st}), flush=True)

def run_pair(a_msgs, a_label, b_msgs, b_label):
    """Two requests in flight at once; returns (a_row, a_text, b_row, b_text)."""
    out = {}
    def worker(key, msgs, label):
        out[key] = send(msgs, label)
    ta = threading.Thread(target=worker, args=("a", a_msgs, a_label))
    tb = threading.Thread(target=worker, args=("b", b_msgs, b_label))
    ta.start(); tb.start(); ta.join(); tb.join()
    return (*out["a"], *out["b"])

def mux():
    # Prompts fixed once so the run, the checks and the saved JSON all use the
    # same text (pads chosen so the gateway's slot hash separates them:
    # A -> slot 0, B -> slot 1, no collision, no degrade to re-prefill).
    a1p, a2p = qa(20, 9), qa(21, 4)
    b1p, b2p = qb(30, 8), qb(31, 4)
    # Turn 1 of both conversations, in flight together.
    a1, a1_text, b1, b1_text = run_pair(
        messages(("user", a1p)), "mux_a_t1",
        messages(("user", b1p)), "mux_b_t1")
    note_cache()
    # Turn 2 of both conversations, in flight together; each replays its own
    # conversation's ACTUAL t1 output as the assistant turn.
    a2, a2_text, b2, b2_text = run_pair(
        messages(("user", a1p), ("assistant", a1_text), ("user", a2p)),
        "mux_a_t2",
        messages(("user", b1p), ("assistant", b1_text), ("user", b2p)),
        "mux_b_t2")
    note_cache()
    checks = [
        ("mux_a_t2_live_reuse",
         a2["reuse_source"] == "live_state"
         and (a2["reused_prompt_tokens"] or 0) >= (a1["prompt_tokens"] or 0)),
        ("mux_b_t2_live_reuse",
         b2["reuse_source"] == "live_state"
         and (b2["reused_prompt_tokens"] or 0) >= (b1["prompt_tokens"] or 0)),
        ("mux_never_disk_snapshot",
         all(r["reuse_source"] != "disk_snapshot" for r in (a1, a2, b1, b2))),
        ("mux_no_snapshots_written", cache_state() == {"files": 0, "total_mb": 0}),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}), flush=True)
    MUX_OUT.write_text(json.dumps({
        "a_t1_prompt": a1p, "a_t1_reply": a1_text, "a_t1_sha": a1["output_sha12"],
        "a_t2_prompt_tail": a2p, "a_t2_sha": a2["output_sha12"],
        "b_t1_prompt": b1p, "b_t1_reply": b1_text, "b_t1_sha": b1["output_sha12"],
        "b_t2_prompt_tail": b2p, "b_t2_sha": b2["output_sha12"],
    }, indent=1))
    print(json.dumps({"saved": str(MUX_OUT)}), flush=True)

def muxcoldref():
    saved = json.loads(MUX_OUT.read_text())
    a1, _ = send(messages(("user", saved["a_t1_prompt"])), "cold_a_t1")
    a2, _ = send(messages(("user", saved["a_t1_prompt"]), ("assistant", saved["a_t1_reply"]),
                  ("user", saved["a_t2_prompt_tail"])), "cold_a_t2")
    b1, _ = send(messages(("user", saved["b_t1_prompt"])), "cold_b_t1")
    b2, _ = send(messages(("user", saved["b_t1_prompt"]), ("assistant", saved["b_t1_reply"]),
                  ("user", saved["b_t2_prompt_tail"])), "cold_b_t2")
    checks = [
        ("cold_a_t1_output_matches_mux", a1["output_sha12"] == saved["a_t1_sha"]),
        ("cold_a_t2_output_matches_mux", a2["output_sha12"] == saved["a_t2_sha"]),
        ("cold_b_t1_output_matches_mux", b1["output_sha12"] == saved["b_t1_sha"]),
        ("cold_b_t2_output_matches_mux", b2["output_sha12"] == saved["b_t2_sha"]),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}), flush=True)

def retain():
    sessions = {}
    # Session A: three turns, replaying actual outputs.
    a1, a1_text = send(messages(("user", qa(40, 8))), "retain_a_t1")
    note_cache()
    a2, a2_text = send(messages(("user", qa(40, 8)), ("assistant", a1_text),
                                ("user", qa(41, 10))), "retain_a_t2")
    note_cache()
    a3, a3_text = send(messages(("user", qa(40, 8)), ("assistant", a1_text),
                                ("user", qa(41, 10)), ("assistant", a2_text),
                                ("user", qa(42, 12))), "retain_a_t3")
    note_cache()
    sessions["a"] = (a1, a2, a3)
    # Session B: same shape, distinct filler.
    b1, b1_text = send(messages(("user", qb(50, 8))), "retain_b_t1")
    note_cache()
    b2, b2_text = send(messages(("user", qb(50, 8)), ("assistant", b1_text),
                                ("user", qb(51, 10))), "retain_b_t2")
    note_cache()
    b3, b3_text = send(messages(("user", qb(50, 8)), ("assistant", b1_text),
                                ("user", qb(51, 10)), ("assistant", b2_text),
                                ("user", qb(52, 12))), "retain_b_t3")
    note_cache()
    sessions["b"] = (b1, b2, b3)
    # Session C: same shape again.
    c1, c1_text = send(messages(("user", qa(60, 8))), "retain_c_t1")
    note_cache()
    c2, c2_text = send(messages(("user", qa(60, 8)), ("assistant", c1_text),
                                ("user", qa(61, 10))), "retain_c_t2")
    note_cache()
    c3, c3_text = send(messages(("user", qa(60, 8)), ("assistant", c1_text),
                                ("user", qa(61, 10)), ("assistant", c2_text),
                                ("user", qa(62, 12))), "retain_c_t3")
    note_cache()
    sessions["c"] = (c1, c2, c3)
    print(json.dumps({"cache_final": cache_state()}), flush=True)
    # Exact repeat of C's last prompt: newest snapshot, expect a full restore.
    cr, _ = send(messages(("user", qa(60, 8)), ("assistant", c1_text),
                          ("user", qa(61, 10)), ("assistant", c2_text),
                          ("user", qa(62, 12))), "repeat_c_t3")
    note_cache()
    # Exact repeat of A's first prompt: oldest snapshot, likely evicted.
    ar, _ = send(messages(("user", qa(40, 8))), "repeat_a_t1")
    note_cache()
    checks = [
        ("retain_c_t3_newest_restored",
         cr["reuse_source"] == "disk_snapshot"
         and (cr["reused_prompt_tokens"] or 0) == (cr["prompt_tokens"] or -1)
         and cr["prefill_s"] < 1.0
         and cr["output_sha12"] == c3["output_sha12"]),
        ("retain_a_t1_repeat_output_identical",
         ar["output_sha12"] == a1["output_sha12"]),
        ("retain_budget_respected",
         (cache_state() or {}).get("total_mb", 1 << 30) <= 512.0),
    ]
    print(json.dumps({"checks": [{"name": n, "pass": p} for n, p in checks]}), flush=True)

def main():
    if MODE == "mux":
        mux()
    elif MODE == "muxcoldref":
        muxcoldref()
    elif MODE == "retain":
        retain()
    else:
        raise SystemExit(f"unknown mode {MODE}")

if __name__ == "__main__":
    main()
