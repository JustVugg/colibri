#!/usr/bin/env python3
"""C15 sgemm A/B verdict: pair base vs sgemm runs, tolerance-contract gates.

Reads phaseC15_sgemm_ab_{base,sgemm}_r{1,2}.jsonl (JSON lines). Gates (the NEW
decision rule from the C15 re-scope — NOT sha identity, the user signed off the
tolerance contract on 2026-11-15):
  1. output agreement: record the sha agreement rate across base/sgemm for the
     same tagged prompts across both alternations, and print every divergence
     (a greedy argmax flip changes the tail from the first divergent token).
  2. decision rule: paired 12K prefill_attention_s >= 20% reduction.
  3. decode no-regression: decode_tokens_per_sec within the recorded noise
     floor (the C14 finding: single-turn decode scatters +/-40-95% at identical
     config, so decode evidence is directional, not decisive).
In-tree real-activation evidence: Q36_ATTN_SGEMM_CHECK=1 per-layer maxdiff
(scores rel ~1.2-1.8e-06, ctx ~3.7e-06-1.1e-05 at 4K) in
serve-c15-sgemm-check.log.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent

def load(kind, r):
    p = HERE / f"phaseC15_sgemm_ab_{kind}_r{r}.jsonl"
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
    for kind in ("base", "sgemm"):
        for r in (1, 2):
            d = load(kind, r)
            if d:
                runs[(kind, r)] = d
    if len(runs) < 2:
        print("not enough result files yet")
        return

    # 1. sha agreement (tolerance contract: record the rate, list divergences)
    shas = {}
    for (kind, r), d in runs.items():
        for label, row in d.items():
            prompt = label.split("_", 1)[1]  # e.g. "r1_4k_a" -> "4k_a"
            shas.setdefault(prompt, {}).setdefault((kind, r), row["output_sha12"])
    agree = total = 0
    print("output sha agreement (base vs sgemm, per prompt):")
    for prompt, m in sorted(shas.items()):
        base_vals = {v for (kind, r), v in m.items() if kind == "base"}
        sg_vals = {v for (kind, r), v in m.items() if kind == "sgemm"}
        match = bool(base_vals & sg_vals)
        total += 1
        agree += 1 if match else 0
        flag = "AGREE" if match else "DIVERGE"
        print(f"  {prompt}: base={sorted(base_vals)} sgemm={sorted(sg_vals)} {flag}")
    print(f"sha agreement: {agree}/{total}")

    # 2. paired attention deltas per prompt size
    print("\npaired prefill_attention_s (base -> sgemm):")
    for prompt in ("4k_a", "4k_b", "12k"):
        pairs = []
        for r in (1, 2):
            b = runs.get(("base", r), {}).get(f"base_r{r}_{prompt}")
            k = runs.get(("sgemm", r), {}).get(f"sgemm_r{r}_{prompt}")
            if b and k:
                d = (b["prefill_attention_s"] - k["prefill_attention_s"]) / b["prefill_attention_s"] * 100
                pairs.append(d)
                print(f"  r{r} {prompt}: {b['prefill_attention_s']:.1f} -> {k['prefill_attention_s']:.1f} s  ({d:+.1f}%)")
        if pairs:
            m = sum(pairs) / len(pairs)
            print(f"  {prompt} mean: {m:+.1f}%  (decision rule at 12K: {'PASS' if m >= 20 else 'FAIL'})")

    # also full prefill and ttft
    print("\npaired prefill_s and engine_ttft_s (base -> sgemm):")
    for prompt in ("4k_a", "4k_b", "12k"):
        for r in (1, 2):
            b = runs.get(("base", r), {}).get(f"base_r{r}_{prompt}")
            k = runs.get(("sgemm", r), {}).get(f"sgemm_r{r}_{prompt}")
            if b and k:
                print(f"  r{r} {prompt}: prefill {b['prefill_s']:.1f} -> {k['prefill_s']:.1f} s | ttft {b['engine_ttft_s']:.1f} -> {k['engine_ttft_s']:.1f} s")

    # 3. decode rates (noise floor context)
    print("\ndecode_tokens_per_sec (single turns; scatter +-40-95% at identical config):")
    for (kind, r), d in sorted(runs.items()):
        for label, row in sorted(d.items()):
            print(f"  {label}: {row.get('decode_tokens_per_sec')}")

if __name__ == "__main__":
    main()
