#!/usr/bin/env python3
"""Extended deterministic agent-session workload (Section 5 extension): large
stable system prompt + declared tools + real tool-result turns.

Shape per session (the Section 5 list, now fully realised):
  1. a LARGE STABLE system prompt (~1.2K tokens, byte-identical every turn —
     the reuse anchor) plus the declared tool block the template folds into
     the same system turn;
  2. initial user request;
  3. assistant reply (the ENGINE'S ACTUAL output, replayed verbatim — the
     2026-11-12 finding: anything else silently skips live reuse);
  4. a scripted assistant tool_calls turn (deterministic, byte-identical
     whenever replayed — prefix consistency is what reuse needs);
  5. a large tool-result turn (~0.8K tokens, role "tool";
     consecutive tool results share one user turn in the template);
  6. follow-up user request; steps 3-5 repeat;
  7. a second session with deliberately changed fillers/tool payloads
     (cross-session divergence, the phase-5 pattern).

Recorded per turn: decode rate, ttft, prefill, reused tokens, reuse_source,
sha. Primary test for TTFT + prefix reuse + residency combined value.

Smoke test FIRST (mode "smoke", short fillers): roles/tool_calls must render
and the turn sequence must reuse live state before this replaces phase 5's
driver in future benchmarks.
"""
import hashlib
import json
import sys
import time
import urllib.request

sys.path.insert(0, "/Users/creemac/colibri/colibri/researchLogs/logs/colibri-c12-workloads")
import c13_matrix as M

KEY = sys.argv[1] if len(sys.argv) > 1 else ""
MODE = sys.argv[2] if len(sys.argv) > 2 else "full"

SYSTEM = (
    "You are the operations assistant for a streaming platform's ingest fleet. "
    "Study this standing brief and keep every line available for later turns. "
    "Ingest: every recording enters through the capture proxy, which stamps a "
    "capture id, validates the container header, and forwards the stream to the "
    "normalizer. The normalizer resamples audio to 48 kHz stereo, aligns subtitle "
    "tracks to the video timeline, and emits a normalized segment every six "
    "seconds. Segments land in the staging bucket, where the cataloguer extracts "
    "keyframes, computes perceptual hashes, and registers the asset in the "
    "metadata store. The transcoder watches the metadata store and picks a ladder "
    "based on the source resolution and the content class the classifier assigned. "
    "Delivery: the origin serves manifests from the edge cache, the edge cache "
    "honours the origin's cache-control headers, and the purger walks the "
    "manifest tree whenever a title is re-cut. Failure handling: the watchdog "
    "tracks segment freshness per capture id, declares a stalled stream after "
    "three missed segments, pages the on-call bridge, and marks the capture for "
    "recovery. The recovery worker replays from the last acknowledged segment, "
    "reconciles the manifest, and reopens the session only after the cataloguer "
    "re-registers the tail. Cost controls: the budgeter samples egress per title "
    "every five minutes, throttles the ladder when a title exceeds its daily "
    "allowance, and reports the throttle to the platform ledger. "
) + ("Stability clause {n}: the pipeline stays stable while tokens per second "
     "drift with context length, the budget is a ceiling rather than a "
     "reservation, observation beats assumption at every step, the ledger is "
     "written only after the quorum acknowledges, and no stage may reorder "
     "another stage's frames while the watchdog holds its lease. ") * 22

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_notes",
        "description": "Read the standing brief's notes for one pipeline module.",
        "parameters": {"type": "object", "properties": {
            "module": {"type": "string", "enum": ["ingest", "delivery", "failure", "cost"]},
        }, "required": ["module"]},
    },
}]

TOOL_RESULT = (
    "notes for module {m} (revision {r}): the capture proxy stamps ids before "
    "validation, the normalizer's six-second cadence is load-bearing for the "
    "watchdog's freshness math, the cataloguer's perceptual hashes key the "
    "dedup path, the transcoder's ladder choice is logged with the content "
    "class, the purger walks depth-first and never crosses title boundaries, "
    "the watchdog's three-miss rule counts only complete misses, the recovery "
    "worker replays from the last ACK and not from the manifest head, the "
    "budgeter's five-minute sample window is not aligned to the billing hour. "
) * 22

TURN_KEYS = ("decode_tokens_per_sec", "prompt_tokens", "reused_prompt_tokens",
             "reuse_source", "prefill_s", "engine_ttft_s")

def tool_call_msg(module):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"type": "function", "function": {
                "name": "read_notes",
                "arguments": json.dumps({"module": module})}}]}

def send(msgs, label, max_tokens=96):
    body = {"model": "qwen3.6-colibri", "messages": msgs, "tools": TOOLS,
            "max_tokens": max_tokens, "temperature": 0, "stream": False}
    req = urllib.request.Request(
        M.BASE, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=14400) as resp:
        payload = json.loads(resp.read())
    t = payload.get("timings", {})
    msg = payload["choices"][0]["message"]
    text = msg.get("content") or ""
    calls = [(c.get("function", {}).get("name"), c.get("function", {}).get("arguments"))
             for c in (msg.get("tool_calls") or [])]
    row = {"label": label, "client_total_s": round(time.time() - t0, 4)}
    for k in TURN_KEYS:
        row[k] = t.get(k)
    row["emitted_tool_calls"] = calls
    row["output_sha12"] = hashlib.sha256(text.encode()).hexdigest()[:12]
    print(json.dumps(row), flush=True)
    # the engine's own turn, replayed verbatim (role/content/tool_calls/
    # reasoning_content as received) — the reuse contract needs byte-identical
    # prefixes, and a tool-call turn must come back as one
    replay = {"role": "assistant", "content": text}
    if msg.get("tool_calls"):
        replay["tool_calls"] = msg["tool_calls"]
    if msg.get("reasoning_content"):
        replay["reasoning_content"] = msg["reasoning_content"]
    return row, replay

def replay_module(replay, fallback):
    """the module the engine's tool call asked for, for the tool-result turn"""
    try:
        calls = replay.get("tool_calls") or []
        args = json.loads(calls[0]["function"]["arguments"])
        m = args.get("module")
        if m in ("ingest", "delivery", "failure", "cost"):
            return m
    except (KeyError, ValueError, IndexError, TypeError):
        pass
    return fallback

def session(name, mods, pads_user, pads_result, tag):
    """One agent session: user -> assistant (actual output, possibly a real
    tool call) -> tool result for the module it asked about -> user -> ..."""
    hist = [{"role": "system", "content": SYSTEM}]
    for i, (mod, pu, pr) in enumerate(zip(mods, pads_user, pads_result), 1):
        hist.append({"role": "user",
                     "content": f"[agent-ext {tag} {name}] In module {mod}, how does the "
                                f"{'recovery worker' if i % 2 else 'budgeter'} behave, "
                                "and which clause covers it? " + "context padding. " * pu})
        row, replay = send(list(hist), f"{name}_t{i}")
        hist.append(replay)
        mod_r = replay_module(replay, mod)
        if not replay.get("tool_calls"):       # the engine stayed in prose: keep the
            hist.append(tool_call_msg(mod_r))  # scripted tool-call turn for the shape
        hist.append({"role": "tool",
                     "content": TOOL_RESULT.format(m=mod_r, r=i) + "detail padding. " * pr})

def main():
    if MODE == "smoke":
        session("S1", ["ingest", "cost"], [1, 1], [0, 0], "smoke")
    else:
        session("A", ["ingest", "failure", "cost"], [3, 2, 2], [1, 1, 1], "r1")
        session("B", ["delivery", "cost", "ingest"], [2, 3, 2], [1, 1, 1], "r1")

if __name__ == "__main__":
    main()
