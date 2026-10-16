#!/usr/bin/env python3
"""Maintainer-run oracle for the tiny Kolibri-1 fixture: writes kolibri1_tiny/ref.json.

The numbers come from the FP32 PyTorch reference implementation vendored and pinned in
tools/kolibri1_reference/ (PROVENANCE.json records where it came from and its sha256; this
script refuses to run when the file's hash differs). CI needs neither torch nor this
script: the harness (tests/kolibri1_tiny_harness.py) reads the committed ref.json.

ref.json holds, for 3 prompts that are each longer than the window (17) and generate
24 tokens each (so every case wraps the sliding ring):
  prompt, greedy continuation, the teacher-forced argmax at EVERY position of
  prompt+continuation, the top-1/top-2 logit margin at every position, the last-row
  top-5, and the experts the cases actually exercise per layer (the mutation matrix
  targets only those).
Usage: python tools/make_kolibri1_ref.py --fixture ./kolibri1_tiny --out ./kolibri1_tiny/ref.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REFDIR = HERE / "kolibri1_reference"
CASES = [("A", 40, 11), ("B", 57, 22), ("C", 33, 33)]    # name, prompt length, seed
NGEN = 24


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="./kolibri1_tiny")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    fx = Path(a.fixture)
    prov = json.loads((REFDIR / "PROVENANCE.json").read_text())
    got = sha(REFDIR / prov["file"])
    if got != prov["sha256"]:
        sys.exit(f"vendored reference sha256 {got} differs from PROVENANCE {prov['sha256']}: refusing")
    sys.path.insert(0, str(REFDIR))
    from kolibri1_ref import Kolibri1Ref

    cfg = json.loads((fx / "config.json").read_text())
    window, vocab, L, K = cfg["sliding_window"], cfg["vocab_size"], cfg["num_hidden_layers"], cfg["num_experts_per_tok"]
    ref = Kolibri1Ref(str(fx), dtype=torch.float32)
    exercised = {l: {} for l in range(L)}
    min_gap = {l: float("inf") for l in range(L)}
    cases = []
    for name, plen, seed in CASES:
        assert plen > window
        rng = np.random.default_rng(seed)
        ids = [int(t) for t in rng.integers(2, vocab, size=plen)]
        for _ in range(NGEN):
            lg = ref.run(ids, capture=False, logits=True)["logits"][-1]
            ids.append(int(torch.argmax(lg)))
        # teacher-forced pass over the whole sequence: argmax and margins at every position
        res = ref.run(ids, capture=True, logits=True)
        lg = res["logits"]
        top2 = torch.topk(lg, 2, dim=-1)
        argmax = [int(t) for t in top2.indices[:, 0]]
        margin = [float(m) for m in (top2.values[:, 0] - top2.values[:, 1])]
        assert argmax[plen - 1:plen - 1 + NGEN] == ids[plen:], "greedy continuation disagrees with the teacher-forced argmax"
        cap = res["cap"]
        for l in range(L):
            sel = cap[f"L{l}.sel_ids"]
            for e, n in zip(*np.unique(sel.numpy(), return_counts=True)):
                exercised[l][int(e)] = exercised[l].get(int(e), 0) + int(n)
            tb = cap[f"L{l}.top_biased"]
            min_gap[l] = min(min_gap[l], float((tb[:, K - 1] - tb[:, K]).min()))
        tk = torch.topk(lg[-1], 5)
        cases.append({"name": name, "prompt": ids[:plen], "generated": ids[plen:], "ngen": NGEN,
                      "argmax": argmax, "margin": margin, "min_margin": min(margin),
                      "last_top5": {"ids": [int(i) for i in tk.indices], "logits": [float(v) for v in tk.values]}})
    shards = json.loads((fx / "fixture.json").read_text())["shard_sha256"] if (fx / "fixture.json").exists() else {}
    out = {
        "schema": "kolibri1-tiny-ref/1",
        "oracle": "FP32 PyTorch reference implementation (tools/kolibri1_reference)", "generation_source_sha256": got, "generation_provenance": prov,
        "torch": torch.__version__, "numpy": np.__version__,
        "config_sha256": sha(fx / "config.json"), "fixture_shard_sha256": shards,
        "window": window, "vocab": vocab,
        "min_margin_all_cases": min(c["min_margin"] for c in cases),
        "min_router_gap_6_7_per_layer": min_gap,
        "exercised_experts": {str(l): {str(e): n for e, n in sorted(d.items())} for l, d in exercised.items()},
        "cases": cases,
    }
    outp = Path(a.out) if a.out else fx / "ref.json"
    outp.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{outp}: {len(cases)} cases, min margin {out['min_margin_all_cases']:.4g}, "
          f"min router gap {min(min_gap.values()):.4g}, exercised experts/layer "
          f"{[len(exercised[l]) for l in range(L)]}")


if __name__ == "__main__":
    main()
