#!/usr/bin/env python3
"""C24 A/B summary — paired-by-round comparison of the fuse1 vs fuse0 arms on
the Q4_K_M container (phaseC24_fuse_{arm}_r{1,2}.jsonl, the C16 driver's rows).

Reads: output shas per cell per round (correctness gate: identical across arms,
and vs the C16/C22 records), decode rates and TTFT per cell, round means, and
the RSS lines from the serve logs.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = [("fuse1", "fused (default)"), ("fuse0", "generic (FUSE=0)")]
ROUNDS = ["r1", "r2"]
CELLS = ["cold_1k", "cold_4k", "cold_12k", "q1", "q2", "q3"]


def load(arm, rnd):
    rows = {}
    p = HERE / f"phaseC24_fuse_{arm}_{rnd}.jsonl"
    if not p.exists():
        return rows
    for ln in p.read_text().splitlines():
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "label" in r:
            rows[r["label"]] = r
    return rows


def main():
    data = {(a, r): load(a, r) for a, _ in ARMS for r in ROUNDS}
    if not any(data.values()):
        print("no phaseC24_fuse_*.jsonl rows found yet")
        return 1

    print("== output shas per cell (fuse1 | fuse0) ==")
    for cell in CELLS:
        shas = [(data[(a, r)].get(cell, {}).get("output_sha12", "-") or "-")
                for a, _ in ARMS for r in ROUNDS]
        agree = "IDENTICAL" if len(set(shas)) == 1 and shas[0] != "-" else "DIFFER/MISSING"
        print(f"  {cell:8s} {shas[0]} {shas[1]} | {shas[2]} {shas[3]}  {agree}")

    print("\n== decode tokens/s per cell (fuse1 | fuse0) ==")
    for cell in CELLS:
        vals = [data[(a, r)].get(cell, {}).get("decode_tokens_per_sec") for a, _ in ARMS for r in ROUNDS]
        if all(v is not None for v in vals):
            f = [vals[0], vals[1]]
            g = [vals[2], vals[3]]
            fm = sum(f) / len(f)
            gm = sum(g) / len(g)
            print(f"  {cell:8s} {f[0]:7.2f}/{f[1]:7.2f} (mean {fm:7.2f}) | "
                  f"{g[0]:7.2f}/{g[1]:7.2f} (mean {gm:7.2f})  ratio {fm / gm:.2f}x")
        else:
            print(f"  {cell:8s} (missing)")

    print("\n== engine TTFT s per cell (fuse1 | fuse0) ==")
    for cell in CELLS:
        vals = [data[(a, r)].get(cell, {}).get("engine_ttft_s") for a, _ in ARMS for r in ROUNDS]
        if all(v is not None for v in vals):
            f = [vals[0], vals[1]]
            g = [vals[2], vals[3]]
            fm = sum(f) / len(f)
            gm = sum(g) / len(g)
            print(f"  {cell:8s} {f[0]:8.1f}/{f[1]:8.1f} (mean {fm:8.1f}) | "
                  f"{g[0]:8.1f}/{g[1]:8.1f} (mean {gm:8.1f})  ratio {fm / gm:.2f}x")
        else:
            print(f"  {cell:8s} (missing)")

    print("\n== prefill moe s per cell (fuse1 | fuse0) ==")
    for cell in CELLS:
        vals = [data[(a, r)].get(cell, {}).get("prefill_moe_s") for a, _ in ARMS for r in ROUNDS]
        if all(v is not None for v in vals):
            f = [vals[0], vals[1]]
            g = [vals[2], vals[3]]
            print(f"  {cell:8s} {f[0]:8.2f}/{f[1]:8.2f} | {g[0]:8.2f}/{g[1]:8.2f}")
        else:
            print(f"  {cell:8s} (missing)")

    print("\n== RSS after load per arm (from serve logs) ==")
    for a, _ in ARMS:
        for r in ROUNDS:
            log = HERE / f"serve-c24-{a}_{r}.log"
            if log.exists():
                for ln in log.read_text(errors="replace").splitlines():
                    if "RSS after load" in ln:
                        print(f"  {a}_{r}: {ln.strip()}")
                        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
