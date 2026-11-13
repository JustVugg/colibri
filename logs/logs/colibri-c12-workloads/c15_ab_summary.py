#!/usr/bin/env python3
"""C15 phase-1 A/B verdict: pair base vs blk runs, gate on shas, report deltas.

Reads phaseC15_ab_{base,blk}_r{1,2}.jsonl (JSON lines). Gates:
  1. bit-exactness: every output_sha12 identical across base and blk for the
     same prompt (4k_a/4k_b/12k), across both alternations.
  2. decision rule: paired attention-timer delta at 12K >= 20% reduction.
  3. no decode regression: decode_tokens_per_sec not worse beyond noise.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def load(kind, r):
    p = HERE / f"phaseC15_ab_{kind}_r{r}.jsonl"
    if not p.exists():
        return None
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            rows.append(json.loads(line))
    return {row["label"]: row for row in rows}

def main():
    runs = {}
    for kind in ("base", "blk"):
        for r in (1, 2):
            d = load(kind, r)
            if d:
                runs[(kind, r)] = d
    if len(runs) < 2:
        print("not enough result files yet")
        return

    # 1. sha gate
    shas = {}
    for (kind, r), d in runs.items():
        for label, row in d.items():
            prompt = label.split("_", 1)[1]  # e.g. "r1_4k_a" -> "4k_a"
            shas.setdefault(prompt, {}).setdefault((kind, r), row["output_sha12"])
    ok = True
    for prompt, m in sorted(shas.items()):
        vals = set(m.values())
        status = "OK" if len(vals) == 1 else "MISMATCH"
        if len(vals) != 1:
            ok = False
        print(f"sha {prompt}: {sorted(vals)} {status}")
    print(f"sha gate: {'PASS' if ok else 'FAIL'}")

    # 2. paired attention deltas per prompt size
    print("\npaired prefill_attention_s (base -> blk):")
    for prompt in ("4k_a", "4k_b", "12k"):
        pairs = []
        for r in (1, 2):
            b = runs.get(("base", r), {}).get(f"base_r{r}_{prompt}")
            k = runs.get(("blk", r), {}).get(f"blk_r{r}_{prompt}")
            if b and k:
                d = (b["prefill_attention_s"] - k["prefill_attention_s"]) / b["prefill_attention_s"] * 100
                pairs.append(d)
                print(f"  r{r} {prompt}: {b['prefill_attention_s']:.1f} -> {k['prefill_attention_s']:.1f} s  ({d:+.1f}%)")
        if pairs:
            print(f"  {prompt} mean: {sum(pairs)/len(pairs):+.1f}%")

    # also full prefill and ttft
    print("\npaired prefill_s:")
    for prompt in ("4k_a", "4k_b", "12k"):
        for r in (1, 2):
            b = runs.get(("base", r), {}).get(f"base_r{r}_{prompt}")
            k = runs.get(("blk", r), {}).get(f"blk_r{r}_{prompt}")
            if b and k:
                print(f"  r{r} {prompt}: {b['prefill_s']:.1f} -> {k['prefill_s']:.1f} s")

    # 3. decode rates
    print("\ndecode_tokens_per_sec:")
    for (kind, r), d in sorted(runs.items()):
        for label, row in sorted(d.items()):
            print(f"  {label}: {row.get('decode_tokens_per_sec')}")

if __name__ == "__main__":
    main()
