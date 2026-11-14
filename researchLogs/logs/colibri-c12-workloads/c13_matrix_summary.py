#!/usr/bin/env python3
"""Summarize c13_matrix.py result rows: p50/p95/p99 per cell.

Usage: python3 c13_matrix_summary.py <rows.jsonl> [<rows.jsonl> ...]
Reads JSON lines (skips non-JSON lines such as tracebacks), groups by
(mode, cell), and reports n, prompt tokens, prefill_s and engine_ttft_s
percentiles plus the reuse-source/miss census and decode rate percentiles.
"""
import json
import sys
from pathlib import Path

def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * p / 100.0
    f, c = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)

def rows(paths):
    for path in paths:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line)
                if "cell" in r:
                    yield r

def main():
    cells = {}
    for r in rows(sys.argv[1:]):
        cells.setdefault((r["mode"], r["cell"]), []).append(r)
    for (mode, cell), rs in sorted(cells.items()):
        rs.sort(key=lambda r: r["label"])
        pre = sorted(r["prefill_s"] for r in rs)
        ttft = sorted(r["engine_ttft_s"] for r in rs)
        dec = sorted(r["decode_tokens_per_s"] for r in rs if r.get("decode_tokens_per_s"))
        npts = sorted(set(r["prompt_tokens"] for r in rs))
        reu = sorted(set(r["reused_prompt_tokens"] for r in rs))
        src = {}
        for r in rs:
            key = f'{r["reuse_source"]}/{r["reuse_miss"] or "-"}'
            src[key] = src.get(key, 0) + 1
        out = {
            "mode": mode, "cell": cell, "n": len(rs),
            "prompt_tokens": npts, "reused": reu,
            "prefill_p50_s": round(pct(pre, 50), 3),
            "prefill_p95_s": round(pct(pre, 95), 3),
            "prefill_p99_s": round(pct(pre, 99), 3),
            "ttft_p50_s": round(pct(ttft, 50), 3),
            "ttft_p95_s": round(pct(ttft, 95), 3),
            "ttft_p99_s": round(pct(ttft, 99), 3),
            "decode_p50": round(pct(dec, 50), 2) if dec else None,
            "sources": src,
        }
        print(json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
