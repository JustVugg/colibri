#!/usr/bin/env python3
"""GLM-5.2 Expert Atlas — affinity with CROSS-PROMPT REPLICATION (#175).

Fixes a trap in the naive version. Routing selections inside one run are heavily correlated:
the same context routes to the same experts token after token. So an expert that fires 38
times during a single 'code' prompt is ONE effective observation, not 38 independent draws.
Treat them as independent and a chi-square will happily certify a single-prompt fluke as a
perfect specialist (spec=1.000, lift=10.0) on 38 selections. It is measuring one prompt.

So specialisation is only claimed when it REPLICATES across the independent prompts of a
category:
  - each category has R prompts (here 3), each run is one replicate
  - share[e][run] = selections of e in that run / total selections in that run
  - an expert is a candidate specialist for category c only if it fires in >= MIN_RUNS of
    c's runs (default 2/3) -> it is a property of the TOPIC, not of one prompt's wording
  - affinity uses the MEAN share across a category's runs, so one hot run cannot carry it
  - reliability = (runs in top category) / (runs in that category)

Also reports the generalist/specialist split by layer depth, which is an average over
thousands of experts and is robust to the above.
"""
import argparse, glob, json, math, os
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default="stats")
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--min-runs", type=int, default=2, help="must fire in >= this many of the top category's runs")
    ap.add_argument("--out", default="experts.json")
    ap.add_argument("--web", default="", help="also write the web-dashboard experts.json (Atlas/Brain hover)")
    a = ap.parse_args()

    # run[(cat,idx)][(layer,expert)] = count ; run_tot[(cat,idx)] = total
    run_counts, run_tot = defaultdict(dict), defaultdict(int)
    for path in sorted(glob.glob(os.path.join(a.stats, "*.txt"))):
        base = os.path.basename(path)[:-4]
        cat, idx = base.rsplit("_", 1)
        for line in open(path):
            p = line.split()
            if len(p) != 3:
                continue
            l, e, n = int(p[0]), int(p[1]), int(p[2])
            run_counts[(cat, idx)][(l, e)] = n
            run_tot[(cat, idx)] += n

    cats = sorted({c for c, _ in run_counts})
    runs_of = {c: sorted(i for cc, i in run_counts if cc == c) for c in cats}
    C = len(cats)
    print(f"categories ({C}): " + ", ".join(f"{c}[{len(runs_of[c])}]" for c in cats))

    experts = {k for d in run_counts.values() for k in d}
    print(f"experts seen: {len(experts):,}\n")

    atlas, dropped_sparse, dropped_unrepl = [], 0, 0
    for key in experts:
        total = sum(run_counts[r].get(key, 0) for r in run_counts)
        if total < a.min_count:
            dropped_sparse += 1
            continue
        # mean share per category across its runs (a single hot run cannot carry the category)
        mean_share, fired_runs = {}, {}
        for c in cats:
            shares, fired = [], 0
            for i in runs_of[c]:
                n = run_counts[(c, i)].get(key, 0)
                shares.append(n / max(1, run_tot[(c, i)]))
                if n > 0:
                    fired += 1
            mean_share[c] = sum(shares) / len(shares)
            fired_runs[c] = fired
        s = sum(mean_share.values())
        if s <= 0:
            continue
        p = {c: mean_share[c] / s for c in cats}
        top = max(cats, key=lambda c: p[c])
        # REPLICATION GATE: the affinity must show up in >= min-runs of that category's prompts
        if fired_runs[top] < a.min_runs:
            dropped_unrepl += 1
            continue
        H = -sum(v * math.log(v) for v in p.values() if v > 0)
        atlas.append({
            "layer": key[0], "expert": key[1], "total": total,
            "spec": round(1.0 - H / math.log(C), 4),
            "top_topic": top,
            "top_lift": round(p[top] * C, 2),
            "reliability": f"{fired_runs[top]}/{len(runs_of[top])}",
            "p": {c: round(p[c], 4) for c in cats},
        })
    atlas.sort(key=lambda r: (-r["spec"], -r["total"]))

    print(f"dropped {dropped_sparse:,} sparse (<{a.min_count} sel)")
    print(f"dropped {dropped_unrepl:,} UNREPLICATED (fired in <{a.min_runs} runs of their top topic)")
    print(f"kept    {len(atlas):,} experts\n")

    print("=== most specialised, replicated across prompts ===")
    print(f"{'layer':>5} {'exp':>4} {'sel':>6} {'spec':>6} {'lift':>6} {'repl':>5}  topic")
    for r in atlas[:20]:
        print(f"{r['layer']:>5} {r['expert']:>4} {r['total']:>6} {r['spec']:>6.3f} "
              f"{r['top_lift']:>6.2f} {r['reliability']:>5}  {r['top_topic']}")

    print("\n=== specialisation vs layer depth (mean over experts; robust to the above) ===")
    by_layer = defaultdict(list)
    for r in atlas:
        by_layer[r["layer"]].append(r["spec"])
    ls = sorted(by_layer)
    for L in ls[::max(1, len(ls)//13)]:
        v = by_layer[L]
        print(f"  layer {L:>3}  n={len(v):>4}  spec {sum(v)/len(v):.3f}  {'#'*int(60*sum(v)/len(v))}")

    print("\n=== experts owned per topic (replicated only) ===")
    own = defaultdict(int)
    for r in atlas:
        own[r["top_topic"]] += 1
    for c in sorted(own, key=lambda x: -own[x]):
        print(f"  {c:<14} {own[c]:>5}")

    strong = [r for r in atlas if r["spec"] >= 0.5]
    print(f"\nstrong specialists (spec >= 0.5, replicated): {len(strong):,} / {len(atlas):,} "
          f"({100*len(strong)/max(1,len(atlas)):.1f}%)")
    json.dump({"categories": cats, "experts": atlas}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")

    if a.web:
        # Same atlas, keyed "layer:expert" with per-expert affinity/entropy/top/label —
        # the shape the web dashboard consumes (Atlas galaxy, Brain hover).
        web = {}
        for r in atlas:
            aff = {c: v for c, v in r["p"].items() if v > 0}
            H = -sum(v * math.log2(v) for v in aff.values())
            web[f"{r['layer']}:{r['expert']}"] = {
                "affinity": aff, "entropy": round(H, 2), "top": r["top_topic"],
                "label": f"specialist: {r['top_topic']}" if r["spec"] >= 0.5 else "generalist"}
        json.dump({"categories": cats, "experts": web}, open(a.web, "w"))
        print(f"wrote {a.web} (dashboard format, {len(web):,} experts)")


if __name__ == "__main__":
    main()
