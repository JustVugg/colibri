#!/usr/bin/env python3
"""Copula/Fréchet analysis of MoE routing traces (ROUTE_TRACE output).

Input lines: "<call> <pos> <layer> id:gate id:gate ..." — one line per (position, layer).

Analyses:
  1. Cross-layer dependence screen (consecutive layer pairs L -> L+1):
     observed pair co-activation counts vs (a) independence product and
     (b) the Fréchet–Hoeffding upper bound min(p_e, p_f). Reports how much
     dependence structure exists beyond marginals — the existence proof for
     coupling-aware prefetch.
  2. Prefetch simulation at equal budget (held-out split):
     predict layer-(L+1) experts from layer-L routing:
       marginal : top-B experts of L+1 by marginal frequency (today's heat logic)
       coupled  : top-B by sum of pair co-activation counts conditioned on the
                  observed L set (a mixture approximation of the pair-copula)
     Metric: recall of the true top-8 at budgets B = 8, 16, 32.
  3. Depth-2 (L -> L+2) repeat of (2) — the horizon where prefetch has a full
     disk round-trip to work in (LOOKA kind [2]).
"""
import sys
from collections import defaultdict
import numpy as np

def load(path):
    """group by (forward, position): the call counter increments per moe() CALL
    (i.e. per layer), so forwards are reconstructed from layer wrap-arounds"""
    rows = defaultdict(dict)          # (fwd,pos) -> {layer: [ids]}
    fwd, prev_layer = 0, -1
    for line in open(path):
        p = line.split()
        if len(p) < 4: continue
        pos, layer = int(p[1]), int(p[2])
        if layer < prev_layer: fwd += 1
        prev_layer = layer
        ids = [int(t.split(":")[0]) for t in p[3:]]
        rows[(fwd, pos)][layer] = ids
    return rows

def main():
    path = sys.argv[1]
    rows = load(path)
    if len(sys.argv) > 2:                 # transfer mode: train on file1, test on file2
        rows2 = load(sys.argv[2])
        keys = sorted(rows.keys()); keys2 = sorted(rows2.keys())
        base = 1 + max(k[0] for k in keys)
        rows.update({(f + base, p): v for (f, p), v in rows2.items()})
        train = keys
        test = [(f + base, p) for (f, p) in keys2]
        mode = f"TRANSFER train={sys.argv[1].split('/')[-1]} test={sys.argv[2].split('/')[-1]}"
    else:
        keys = sorted(rows.keys())
        split = int(len(keys) * 0.7)
        train, test = keys[:split], keys[split:]
        mode = "in-trace 70/30 split"
    layers = sorted({L for r in rows.values() for L in r})
    E = 1 + max(i for r in rows.values() for ids in r.values() for i in ids)
    print(f"trace: {len(rows)} positions, {len(layers)} routed layers, E={E} "
          f"(train {len(train)} / test {len(test)}) [{mode}]")

    # marginals + consecutive-pair co-occurrence from TRAIN
    marg = {L: np.zeros(E) for L in layers}
    pair = {}                          # (L, dL) -> sparse dict (e,f)->count
    for dL in (1, 2):
        for L in layers:
            if L + dL in layers: pair[(L, dL)] = defaultdict(int)
    for k in train:
        r = rows[k]
        for L, ids in r.items():
            for e in ids: marg[L][e] += 1
        for (L, dL), d in pair.items():
            if L in r and L + dL in r:
                for e in r[L]:
                    for f in r[L + dL]: d[(e, f)] += 1

    # 1. dependence screen on L->L+1
    lifts, bound_ratio = [], []
    N = len(train)
    for (L, dL), d in pair.items():
        if dL != 1: continue
        for (e, f), c in d.items():
            pe, pf = marg[L][e] / N, marg[L + 1][f] / N
            if pe * pf <= 0: continue
            lifts.append((c / N) / (pe * pf))
            bound_ratio.append((c / N) / min(pe, pf))
    lifts = np.array(lifts); bound_ratio = np.array(bound_ratio)
    print(f"\n[1] L->L+1 co-activation, {len(lifts)} observed pairs:")
    print(f"    lift vs independence: median {np.median(lifts):.2f}  "
          f"p90 {np.percentile(lifts,90):.2f}  p99 {np.percentile(lifts,99):.2f}")
    print(f"    fraction of pairs at >50% of the Fréchet upper bound: "
          f"{(bound_ratio>0.5).mean()*100:.1f}%  (>90%: {(bound_ratio>0.9).mean()*100:.1f}%)")

    # 2/3. prefetch recall on TEST
    for dL in (1, 2):
        print(f"\n[{1+dL}] prefetch L->L+{dL}, recall of true top-8 on held-out positions:")
        for B in (8, 16, 32):
            hit_m = tot = hit_c = 0
            for k in test:
                r = rows[k]
                for L in layers:
                    if L not in r or L + dL not in r or (L, dL) not in pair: continue
                    true = set(r[L + dL])
                    pm = np.argsort(marg[L + dL])[::-1][:B]
                    hit_m += len(true & set(pm.tolist()))
                    score = defaultdict(float)
                    d = pair[(L, dL)]
                    for (e, f), c in d.items():
                        if e in r[L]: score[f] += c
                    pc = sorted(score, key=score.get, reverse=True)[:B]
                    if len(pc) < B:                      # back-fill with marginals
                        for f in np.argsort(marg[L + dL])[::-1]:
                            if f not in pc: pc.append(int(f))
                            if len(pc) == B: break
                    hit_c += len(true & set(pc))
                    tot += len(true)
            print(f"    budget {B:3d}/layer: marginal {hit_m/tot*100:5.1f}%   "
                  f"coupled {hit_c/tot*100:5.1f}%   (+{(hit_c-hit_m)/tot*100:.1f}pp)")

if __name__ == "__main__":
    main()
