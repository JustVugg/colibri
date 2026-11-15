#!/usr/bin/env python3
"""C15 adoption-sweep verdict: sha agreement (tolerance contract) + paired
timings across base vs sgemm for the agent-ext turns and the 16K/24K colds.

Reads phaseC15_sweep_{base,sgemm}_r{1,2}.jsonl. Gates:
  1. output sha agreement: for the same turn label with the round stripped
     (A_t1..B_t3, 16k, 24k), base and sgemm sha sets must intersect — a greedy
     argmax flip at the turn's first divergent token changes the whole tail.
     Agent-ext replay caveat: turn shas only match if all earlier turns matched
     (histories diverge otherwise) — the FIRST diverging turn is the signal.
  2. paired 16K/24K prefill_attention_s reduction (informational; the decision
     rule was already established at 12K in the C15 A/B).
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def load(kind, r):
    p = HERE / f"phaseC15_sweep_{kind}_r{r}.jsonl"
    if not p.exists():
        return None
    rows = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            row = json.loads(line)
            rows[row["label"]] = row
    return rows

def turn_key(label):
    # A_t1 -> A_t1 (strip only the base_r1_/sgemm_r2_ prefix)
    for k in ("base_r1_", "base_r2_", "sgemm_r1_", "sgemm_r2_"):
        if label.startswith(k):
            return label[len(k):]
    return label

def main():
    runs = {}
    for kind in ("base", "sgemm"):
        for r in (1, 2):
            d = load(kind, r)
            if d:
                runs[(kind, r)] = d
    if len(runs) < 2:
        print("not enough result files yet")
        return

    shas = {}
    for (kind, r), d in runs.items():
        for label, row in d.items():
            shas.setdefault(turn_key(label), {}).setdefault(kind, set()).add(row["output_sha12"])
    agree = diverge = 0
    print("output sha agreement (base vs sgemm, per turn):")
    for turn, m in sorted(shas.items()):
        b, s = m.get("base", set()), m.get("sgemm", set())
        match = bool(b & s)
        if match:
            agree += 1
            print(f"  {turn}: AGREE ({sorted(b & s)[0]})")
        else:
            diverge += 1
            print(f"  {turn}: DIVERGE base={sorted(b)} sgemm={sorted(s)}")
    print(f"turn agreement: {agree} agree / {diverge} diverge")

    print("\npaired cold timings (base -> sgemm):")
    for r in (1, 2):
        for size in ("16k", "24k"):
            b = runs.get(("base", r), {}).get(f"base_r{r}_{size}")
            k = runs.get(("sgemm", r), {}).get(f"sgemm_r{r}_{size}")
            if b and k:
                att = (b["prefill_attention_s"] - k["prefill_attention_s"]) / b["prefill_attention_s"] * 100
                ttft = (b["engine_ttft_s"] - k["engine_ttft_s"]) / b["engine_ttft_s"] * 100
                print(f"  r{r} {size} ({b['prompt_tokens']} tok): attn {b['prefill_attention_s']:.1f} -> "
                      f"{k['prefill_attention_s']:.1f} s ({att:+.1f}%) | ttft {b['engine_ttft_s']:.1f} -> "
                      f"{k['engine_ttft_s']:.1f} s ({ttft:+.1f}%)")

if __name__ == "__main__":
    main()
