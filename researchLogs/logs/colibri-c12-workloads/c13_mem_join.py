#!/usr/bin/env python3
"""Join c13_memory.py request windows to c13_mem_sampler.py RSS samples.

Usage: python3 c13_mem_join.py <samples.jsonl> <requests.jsonl> [...]
For each request row: engine+gateway RSS at window start (baseline), peak
within the window, and peak-minus-baseline. Windows are [t_start, t_end].
"""
import json
import sys
from pathlib import Path

def load_rows(path, pred):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            r = json.loads(line)
            if pred(r):
                out.append(r)
    return out

def main():
    sample_path = sys.argv[1]
    req_paths = sys.argv[2:]
    samples = load_rows(sample_path, lambda r: "ts" in r)
    reqs = load_rows(req_paths[0], lambda r: "t_start" in r)
    for p in req_paths[1:]:
        reqs += load_rows(p, lambda r: "t_start" in r)
    # figure out which pid is the engine (larger RSS) from the last sample
    last = samples[-1]["pids"] if samples else {}
    eng_pid = max((k for k, v in last.items() if v is not None), key=lambda k: last[k])
    gwy_pid = [k for k in last if k != eng_pid][0]
    for r in sorted(reqs, key=lambda r: r["t_start"]):
        w = [s for s in samples if r["t_start"] <= s["ts"] <= r["t_end"]]
        if not w:
            print(json.dumps({**r, "mem": "no_samples"}), flush=True)
            continue
        eng = [s["pids"][eng_pid] for s in w if s["pids"].get(eng_pid) is not None]
        gwy = [s["pids"][gwy_pid] for s in w if s["pids"].get(gwy_pid) is not None]
        base = w[0]["pids"]
        peak_eng, peak_gwy = max(eng), max(gwy)
        out = {
            "label": r["label"], "cell": r["cell"],
            "prompt_tokens": r["prompt_tokens"],
            "reused_prompt_tokens": r["reused_prompt_tokens"],
            "reuse_source": r["reuse_source"],
            "prefill_s": r["prefill_s"], "restore_s": r["snapshot_restore_s"],
            "engine_ttft_s": r["engine_ttft_s"], "client_ttft_s": r["client_ttft_s"],
            "instr_overhead_s": r["instr_overhead_s"],
            "decode_tokens_per_sec": r["decode_tokens_per_sec"],
            "engine_rss_base_mb": round(base[eng_pid] / 1024, 1),
            "engine_rss_peak_mb": round(peak_eng / 1024, 1),
            "engine_rss_grow_mb": round((peak_eng - base[eng_pid]) / 1024, 1),
            "gateway_rss_peak_mb": round(peak_gwy / 1024, 1),
        }
        print(json.dumps(out), flush=True)

if __name__ == "__main__":
    main()
