#!/usr/bin/env python3
"""C14 pressure-arm verdict: per-turn decode rates by arm, paired deltas.

Reads phaseC14_pressure_{arm}.jsonl. Gates:
  1. idle arms: Q36_HYBRID_GB 0 vs 2 should differ ~0 (scoping expectation).
  2. pressure arms: does decode degrade under the hog, and does HYBRID_GB=2
     insulate it? Paired per-turn decode deltas press2 - press0.
  3. TTFT under pressure: expected to absorb re-faults (secondary metric).
All comparisons are within-round pairs (r1 idle0 vs r1 idle2, ...), per the
drift rule; the same arm order ran in both rounds.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def load(arm):
    p = HERE / f"phaseC14_pressure_{arm}.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return rows

def dec(rows):
    return {r["label"]: r.get("decode_tokens_per_sec") for r in rows}

def main():
    arms = ["idle0", "idle2", "press0", "press2"]
    have = {a: load(a) for a in arms}
    if not all(have.values()):
        print("missing result files:", [a for a in arms if not have[a]])
        return
    for a in arms:
        print(f"\n{a}: per-turn")
        for r in have[a]:
            print(f"  {r['label']}: decode {r.get('decode_tokens_per_sec'):.3f} tok/s"
                  f"  prefill {r.get('prefill_s'):.1f}s  ttft {r.get('engine_ttft_s'):.1f}s"
                  f"  reused {r.get('reused_prompt_tokens')}/{r.get('prompt_tokens')}"
                  f"  sha {r.get('output_sha12')}")

    def turn_dec(arm, r, t):
        rows = have[arm]
        lab = f"{arm}_r{r}_t{t}"
        row = next((x for x in rows if x["label"] == lab), None)
        return row.get("decode_tokens_per_sec") if row else None

    print("\npaired decode deltas (hybrid2 vs 0):")
    for cond in ("idle", "press"):
        ds = []
        for r in (1, 2):
            for t in range(1, 5):
                a, b = turn_dec(cond + "0", r, t), turn_dec(cond + "2", r, t)
                if a and b:
                    d = (b - a) / a * 100
                    ds.append(d)
                    print(f"  r{r} t{t} {cond}: {a:.3f} -> {b:.3f} tok/s ({d:+.1f}%)")
        if ds:
            print(f"  {cond} mean: {sum(ds)/len(ds):+.1f}%")
    print("\npaired decode deltas (pressure vs idle, hybrid 0):")
    ds = []
    for r in (1, 2):
        for t in range(1, 5):
            a, b = turn_dec("idle0", r, t), turn_dec("press0", r, t)
            if a and b:
                d = (b - a) / a * 100
                ds.append(d)
                print(f"  r{r} t{t}: {a:.3f} -> {b:.3f} tok/s ({d:+.1f}%)")
    if ds:
        print(f"  mean: {sum(ds)/len(ds):+.1f}%")

if __name__ == "__main__":
    main()
