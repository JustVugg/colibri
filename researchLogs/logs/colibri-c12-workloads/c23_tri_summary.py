#!/usr/bin/env python3
"""C23 three-container graded batch summary — paired-by-round comparison of
IQ2_XXS / UD-IQ3_XXS / Q4_K_M on the committed C23 set (the C24 build, 2026-11-18).

Reads phaseC23_eval_{iq2,iq3,q4km}_r{1,2}.jsonl and prints:
  - scores per round per container (category breakdown)
  - sha stability across rounds (greedy determinism) and cross-container sha overlaps
  - paired decode round-means per cell + ratios vs IQ2
  - TTFT / prefill_moe round-means from the rows' timings objects
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ["iq2", "iq3", "q4km"]
ROUNDS = ["r1", "r2"]

def load(arm, rnd):
    rows = {}
    p = HERE / f"phaseC23_eval_{arm}_{rnd}.jsonl"
    if not p.exists():
        return rows
    for ln in p.read_text().splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "task" in r:
            rows[r["task"]] = r
    return rows

def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None

def main():
    data = {(a, r): load(a, r) for a in ARMS for r in ROUNDS}
    if not any(data.values()):
        print("no phaseC23_eval_*.jsonl rows found")
        return 1

    tasks = sorted({t for d in data.values() for t in d})

    print("== scores per round ==")
    for a in ARMS:
        parts = []
        for r in ROUNDS:
            d = data[(a, r)]
            cats = {}
            for row in d.values():
                c = row.get("cat", "?")
                cats.setdefault(c, [0, 0])
                cats[c][0] += row.get("score", 0)
                cats[c][1] += 1
            total = sum(v[0] for v in cats.values())
            brk = " ".join(f"{k}{v[0]}/{v[1]}" for k, v in sorted(cats.items()))
            parts.append(f"{r}: {total}/25 ({brk})")
        print(f"  {a:5s} " + " | ".join(parts))

    print("\n== sha stability across rounds (per container) ==")
    for a in ARMS:
        same = sum(1 for t in tasks
                   if data[(a, "r1")].get(t, {}).get("sha12") ==
                      data[(a, "r2")].get(t, {}).get("sha12") and t in data[(a, "r1")] and t in data[(a, "r2")])
        print(f"  {a:5s} {same}/{len([t for t in tasks if t in data[(a,'r1')] and t in data[(a,'r2')]])} tasks sha-identical r1 vs r2")

    print("\n== failures by round (task:detail) ==")
    for a in ARMS:
        for r in ROUNDS:
            fails = [f"{row['task']}({row.get('detail','')[:36]})"
                     for row in data[(a, r)].values() if not row.get("score")]
            print(f"  {a}_{r}: {', '.join(fails) if fails else 'none'}")

    print("\n== paired decode tok/s (round means, ratio vs IQ2) ==")
    print(f"  {'task':18s} {'iq2':>12s} {'iq3':>12s} {'q4km':>12s}")
    for t in tasks:
        cells = []
        for a in ARMS:
            vals = [data[(a, r)].get(t, {}).get("decode_tokens_per_sec") for r in ROUNDS]
            m = mean(vals)
            cells.append(m)
        if all(c is not None for c in cells) and cells[0]:
            rat = [c / cells[0] for c in cells]
            print(f"  {t:18s} {cells[0]:6.2f} (1.00) {cells[1]:5.2f} ({rat[1]:.2f}) {cells[2]:5.2f} ({rat[2]:.2f})")

    print("\n== engine TTFT s (round means) ==")
    print(f"  {'task':18s} {'iq2':>10s} {'iq3':>10s} {'q4km':>10s}")
    for t in tasks:
        cells = []
        for a in ARMS:
            vals = [data[(a, r)].get(t, {}).get("ttft_s") or data[(a, r)].get(t, {}).get("engine_ttft_s") for r in ROUNDS]
            cells.append(mean(vals))
        if all(c is not None for c in cells):
            print(f"  {t:18s} {cells[0]:10.2f} {cells[1]:10.2f} {cells[2]:10.2f}")

    print("\n== prompt reuse on the longctx block (reused / prompt tokens, round means) ==")
    for t in tasks:
        if not t.startswith("ledger_"):
            continue
        cells = []
        for a in ARMS:
            vals = []
            for r in ROUNDS:
                row = data[(a, r)].get(t, {})
                if row.get("prompt_tokens") and row.get("reused_prompt_tokens") is not None:
                    vals.append(row["reused_prompt_tokens"] / row["prompt_tokens"])
            cells.append(mean(vals))
        if all(c is not None for c in cells):
            print(f"  {t:18s} iq2 {cells[0]*100:5.1f}%  iq3 {cells[1]*100:5.1f}%  q4km {cells[2]*100:5.1f}%")
    return 0

if __name__ == "__main__":
    sys.exit(main())
