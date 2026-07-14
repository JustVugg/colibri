#!/usr/bin/env python3
"""Build a .coli_pairs cross-layer coupling table from ROUTE_TRACE dumps.

Input: one or more ROUTE_TRACE files ("call pos layer id:gate ...", one line per
(position, layer)). Output: a text table of the top-M layer-(L+dL) experts most
co-activated with each (layer L, expert e) conditioning event:

    COLIPAIRS 1 <n_lines>
    <L> <dL> <e> f1:c1 f2:c2 ... (up to M)

The engine (COUPLE=<file>) scores layer-(L+dL) candidates by summing counts over
the observed layer-L expert set — the mixture approximation of the pair copula
that measured +3.6..+9.4pp prefetch recall over marginal heat, in- and
cross-domain (see docs in the PR). Counts are raw co-occurrences; the consumer
only needs their ranking, so no normalization is stored.

Usage: python3 tools/route_pairs.py out.coli_pairs trace1.txt [trace2.txt ...]
"""
import sys
from collections import defaultdict

M = 16

def main():
    out_path, traces = sys.argv[1], sys.argv[2:]
    pair = defaultdict(lambda: defaultdict(int))   # (L, dL, e) -> {f: count}
    for path in traces:
        cur = {}                                   # layer -> ids (within one forward)
        prev_layer = -1
        def flush():
            layers = sorted(cur)
            for i, L in enumerate(layers):
                for dL in (1, 2):
                    if L + dL in cur:
                        for e in cur[L]:
                            d = pair[(L, dL, e)]
                            for f in cur[L + dL]: d[f] += 1
        # group lines by position within a forward: lines arrive layer-major
        # (all positions of layer L, then layer L+1, ...); regroup per position
        rows = defaultdict(dict)                   # (fwd,pos) -> {layer: ids}
        fwd = 0
        for line in open(path):
            p = line.split()
            if len(p) < 4: continue
            pos, layer = int(p[1]), int(p[2])
            if layer < prev_layer: fwd += 1
            prev_layer = layer
            rows[(fwd, pos)][layer] = [int(t.split(":")[0]) for t in p[3:]]
        for r in rows.values():
            cur = r; flush()
        print(f"{path}: {len(rows)} positions", file=sys.stderr)

    lines = []
    for (L, dL, e), d in sorted(pair.items()):
        top = sorted(d, key=d.get, reverse=True)[:M]
        lines.append(f"{L} {dL} {e} " + " ".join(f"{f}:{d[f]}" for f in top))
    with open(out_path, "w") as f:
        f.write(f"COLIPAIRS 1 {len(lines)}\n")
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}: {len(lines)} conditioning entries", file=sys.stderr)

if __name__ == "__main__":
    main()
