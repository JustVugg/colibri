#!/usr/bin/env python3
"""Demand-path replacement-policy replay for the OLMoE expert cache.

Measurement only: consumes a ROUTE_TRACE file produced by the streaming engine
(ROUTE_TRACE=<path> ./olmoe ...) and replays the routed-expert access stream
through per-layer bounded caches. It never loads the model and never touches
engine code.

ROUTE_TRACE lines have the form (one line per moe call, row, layer):

    <call> <row> <layer> <expert>:<gate> ...

Temporal order is file order. Within a line, experts are listed in the order
the engine's routing loop visits them (top-K order); repeats inside one
(call, row, layer) group are collapsed to one access.

Policies (per-layer independent caches, all misses admitted, no prefetch):

    lru    evict least-recently-used (engine demand path baseline, `used` clock)
    lfu    evict least-frequently-used, ties -> oldest last access
    lfru   evict min(score = freq << 8 | recency_255) exactly like the engine's
           lfru_score()/tier_lfru_score(): recent = age < 255 ? 255 - age : 0.
           Ties are broken by resident scan order (first minimum wins), the
           same way tier_pick_lfru() scans pinned slots.
    oracle Belady optimal: evict the resident whose next use is farthest away
           (cap 24 only, the decision point).

Assumptions: one shared global clock feeds recency (the engine has a single
m->clock); per-layer caches are independent; every miss is admitted; no
prefetch, pinning, or admission filtering.

Usage:
    python3 bench_tier/policy_replay.py bench_tier/route200.txt
    python3 bench_tier/policy_replay.py bench_tier/route200.txt --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict

EXPERT_FIELD = re.compile(r"^(\d+):([^:]+)$")


def parse_trace(path):
    """Parse ROUTE_TRACE into [(call, row, layer, [experts])] in file order.

    Experts repeated within a line are collapsed; a repeated (call, row, layer)
    group would also be collapsed to its first occurrence's access order.
    """
    rows = []
    seen_groups = set()
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            fields = raw.split()
            if not fields:
                continue
            if len(fields) < 4:
                raise ValueError(f"{path}:{lineno}: expected call row layer id:gate ...")
            try:
                call, row, layer = (int(f) for f in fields[:3])
            except ValueError as error:
                raise ValueError(f"{path}:{lineno}: invalid call/row/layer") from error
            group = (call, row, layer)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            experts = []
            known = set()
            for token in fields[3:]:
                match = EXPERT_FIELD.fullmatch(token)
                if not match:
                    raise ValueError(f"{path}:{lineno}: invalid expert field {token!r}")
                expert = int(match.group(1))
                if expert not in known:
                    known.add(expert)
                    experts.append(expert)
            rows.append((call, row, layer, experts))
    if not rows:
        raise ValueError(f"{path}: trace is empty")
    return rows


def per_layer_accesses(rows):
    """Per-layer access sequences in global time order: layer -> [(expert, t)]."""
    sequences = defaultdict(list)
    clock = 0
    for _call, _row, layer, experts in rows:
        for expert in experts:
            clock += 1
            sequences[layer].append((expert, clock))
    return sequences


class LRUCache:
    name = "lru"

    def __init__(self, cap):
        self.cap = cap
        self.order = {}
        self.hits = 0
        self.requests = 0

    def access(self, expert, clock):
        self.requests += 1
        if expert in self.order:
            self.hits += 1
            del self.order[expert]
            self.order[expert] = None
            return True
        if len(self.order) >= self.cap:
            oldest = next(iter(self.order))
            del self.order[oldest]
        self.order[expert] = None
        return False


class LFUCache:
    name = "lfu"

    def __init__(self, cap):
        self.cap = cap
        self.freq = defaultdict(int)
        self.last = {}
        self.resident = {}
        self.hits = 0
        self.requests = 0

    def access(self, expert, clock):
        self.requests += 1
        self.freq[expert] += 1
        self.last[expert] = clock
        if expert in self.resident:
            self.hits += 1
            return True
        if len(self.resident) >= self.cap:
            victim = min(self.resident, key=lambda e: (self.freq[e], self.last[e]))
            del self.resident[victim]
        self.resident[expert] = None
        return False


class LFRUCache:
    name = "lfru"

    def __init__(self, cap):
        self.cap = cap
        self.freq = defaultdict(int)
        self.last = {}
        self.resident = {}
        self.hits = 0
        self.requests = 0

    def _score(self, expert, clock):
        age = clock - self.last[expert]
        recent = 255 - age if age < 255 else 0
        return (self.freq[expert] << 8) | recent

    def access(self, expert, clock):
        self.requests += 1
        self.freq[expert] += 1
        self.last[expert] = clock
        if expert in self.resident:
            self.hits += 1
            return True
        if len(self.resident) >= self.cap:
            best = None
            best_score = None
            for candidate in self.resident:  # insertion order: first minimum wins
                score = self._score(candidate, clock)
                if best_score is None or score < best_score:
                    best = candidate
                    best_score = score
            del self.resident[best]
        self.resident[expert] = None
        return False


def belady_layer(cap, accesses):
    """Belady optimal for one layer: evict the resident whose next use is
    farthest away (never-used-again first). Returns (hits, requests)."""
    experts = [expert for expert, _clock in accesses]
    positions = defaultdict(list)
    for index, expert in enumerate(experts):
        positions[expert].append(index)
    cursor = {expert: 0 for expert in positions}
    resident = set()
    hits = 0
    for index, expert in enumerate(experts):
        cursor[expert] += 1  # next occurrence strictly after `index`
        if expert in resident:
            hits += 1
            continue
        if len(resident) >= cap:
            victim = None
            farthest = -1
            for candidate in resident:
                position = cursor[candidate]
                next_index = (positions[candidate][position]
                              if position < len(positions[candidate]) else None)
                if next_index is None:
                    victim = candidate
                    break
                if next_index > farthest:
                    farthest = next_index
                    victim = candidate
            resident.discard(victim)
        resident.add(expert)
    return hits, len(experts)


def replay_policy(name, cap, sequences):
    """Replay one policy at one cap; returns per-layer {hits, requests}."""
    stats = {}
    for layer, accesses in sequences.items():
        if name == "oracle":
            hits, requests = belady_layer(cap, accesses)
        else:
            factory = {"lru": LRUCache, "lfu": LFUCache, "lfru": LFRUCache}[name]
            cache = factory(cap)
            for expert, clock in accesses:
                cache.access(expert, clock)
            hits, requests = cache.hits, cache.requests
        stats[layer] = {"hits": hits, "requests": requests}
    return stats


def overall(stats):
    hits = sum(layer["hits"] for layer in stats.values())
    requests = sum(layer["requests"] for layer in stats.values())
    return hits, requests, (hits / requests if requests else 0.0)


def consecutive_overlap(rows):
    """Set overlap: fraction of each token's routed experts that were already
    routed by the previous token of the same layer. Tokens are rows (one moe
    position); consecutive rows of one layer are compared in file order.
    `pooled_decode` counts pairs where both rows belong to single-row calls.
    """
    rows_per = defaultdict(int)
    for call, row, layer, experts in rows:
        rows_per[(layer, call)] += 1
    decode_calls = {(layer, call) for (layer, call), n in rows_per.items() if n == 1}
    sequences = defaultdict(list)
    for call, row, layer, experts in rows:
        sequences[layer].append((call, set(experts)))

    all_hit = all_tot = decode_hit = decode_tot = 0
    for layer, sequence in sequences.items():
        for (call0, set0), (call1, set1) in zip(sequence, sequence[1:]):
            overlap = len(set1 & set0)
            all_hit += overlap
            all_tot += len(set1)
            if (layer, call0) in decode_calls and (layer, call1) in decode_calls:
                decode_hit += overlap
                decode_tot += len(set1)
    return {
        "pooled_all": all_hit / all_tot if all_tot else 0.0,
        "pooled_decode": decode_hit / decode_tot if decode_tot else 0.0,
    }


def cap_hit_provenance(rows, cap):
    """Ordered per-row LRU replay at `cap`, classifying each access by whether
    the expert was in the previous token's routed set. `all` covers every row;
    `decode` covers single-row calls only (the decode forwards). This is the
    bridge between the set-overlap statistic and the realized hit rate.
    """
    rows_per = defaultdict(int)
    for call, row, layer, experts in rows:
        rows_per[(layer, call)] += 1
    decode_calls = {(layer, call) for (layer, call), n in rows_per.items() if n == 1}
    totals = {name: dict(requests=0, hits=0, overlap=0, overlap_hits=0,
                         overlap_misses=0, hits_not_overlap=0)
              for name in ("all", "decode")}
    for layer in sorted({row[2] for row in rows}):
        cache = LRUCache(cap)
        previous = None
        for call, row, row_layer, experts in rows:
            if row_layer != layer:
                continue
            is_decode = (layer, call) in decode_calls
            for expert in experts:
                hit = cache.access(expert, 0)
                in_previous = previous is not None and expert in previous
                for name in ("all", "decode"):
                    if name == "decode" and (not is_decode or previous is None):
                        continue
                    stats = totals[name]
                    stats["requests"] += 1
                    stats["hits"] += hit
                    stats["overlap"] += in_previous
                    stats["overlap_hits"] += in_previous and hit
                    stats["overlap_misses"] += in_previous and not hit
                    stats["hits_not_overlap"] += hit and not in_previous
            previous = set(experts)
    for stats in totals.values():
        stats["hit_rate"] = (stats["hits"] / stats["requests"]
                             if stats["requests"] else 0.0)
        stats["overlap_rate"] = (stats["overlap"] / stats["requests"]
                                 if stats["requests"] else 0.0)
    return totals


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trace", help="ROUTE_TRACE file")
    parser.add_argument("--caps", type=int, nargs="+", default=[8, 16, 24, 32])
    parser.add_argument("--oracle-cap", type=int, default=24)
    parser.add_argument("--json", help="write full per-layer results as JSON")
    args = parser.parse_args()

    rows = parse_trace(args.trace)
    sequences = per_layer_accesses(rows)
    total_accesses = sum(len(v) for v in sequences.values())
    layers = sorted(sequences)
    print(f"trace: {args.trace}")
    print(f"lines={len(rows)} layers={len(layers)} accesses={total_accesses}")
    print("experts touched per layer: " +
          " ".join(f"L{l}={len(set(e for e, _ in sequences[l]))}" for l in layers))

    results = {}
    for cap in args.caps:
        for policy in ("lru", "lfu", "lfru"):
            results[(policy, cap)] = replay_policy(policy, cap, sequences)
        if cap == args.oracle_cap:
            results[("oracle", cap)] = replay_policy("oracle", cap, sequences)

    print()
    print(f"{'cap':>4} {'policy':>7} {'hit%':>8} {'hits':>7} {'requests':>9}")
    for cap in args.caps:
        policies = ("lru", "lfu", "lfru") + (("oracle",) if cap == args.oracle_cap else ())
        for policy in policies:
            hits, requests, rate = overall(results[(policy, cap)])
            print(f"{cap:>4} {policy:>7} {100*rate:>7.2f}% {hits:>7} {requests:>9}")

    print()
    print("deltas (percentage points)")
    for cap in args.caps:
        _h, _r, lru_rate = overall(results[("lru", cap)])
        _h, _r, lfu_rate = overall(results[("lfu", cap)])
        _h, _r, lfru_rate = overall(results[("lfru", cap)])
        line = (f"cap={cap:>2}: LRU->LFRU {100*(lfru_rate-lru_rate):+6.2f}pp  "
                f"LRU->LFU {100*(lfu_rate-lru_rate):+6.2f}pp")
        if cap == args.oracle_cap:
            _h, _r, oracle_rate = overall(results[("oracle", cap)])
            line += (f"  LFRU->oracle {100*(oracle_rate-lfru_rate):+6.2f}pp  "
                     f"LRU->oracle {100*(oracle_rate-lru_rate):+6.2f}pp")
        print(line)

    print()
    print(f"per-layer hit% at cap={args.oracle_cap} (layer: lru / lfu / lfru / oracle)")
    for layer in layers:
        cells = []
        for policy in ("lru", "lfu", "lfru", "oracle"):
            key = (policy, args.oracle_cap)
            if key not in results:
                cells.append("   -  ")
                continue
            stats = results[key][layer]
            rate = stats["hits"] / stats["requests"] if stats["requests"] else 0.0
            cells.append(f"{100*rate:6.1f}")
        print(f"  L{layer:>2}: " + " / ".join(cells))

    overlap = consecutive_overlap(rows)
    provenance = cap_hit_provenance(rows, 8)
    print()
    print("consecutive-token overlap + cap=8 realized hits (LRU)")
    print(f"  set overlap all rows:    {100*overlap['pooled_all']:.2f}%")
    print(f"  set overlap decode only: {100*overlap['pooled_decode']:.2f}%")
    for name in ("all", "decode"):
        stats = provenance[name]
        print(f"  realized cap=8 {name:<6}: hit%={100*stats['hit_rate']:.2f} "
              f"overlap%={100*stats['overlap_rate']:.2f} "
              f"overlap_hits={stats['overlap_hits']} "
              f"overlap_misses={stats['overlap_misses']} "
              f"hits_not_overlap={stats['hits_not_overlap']}")

    if args.json:
        document = {
            "trace": args.trace,
            "lines": len(rows),
            "accesses": total_accesses,
            "caps": args.caps,
            "oracle_cap": args.oracle_cap,
            "overlap": overlap,
            "cap8_provenance": provenance,
            "results": {
                f"{policy}@{cap}": {
                    "hits": sum(v["hits"] for v in stats.values()),
                    "requests": sum(v["requests"] for v in stats.values()),
                    "hit_rate": overall(stats)[2],
                    "per_layer": stats,
                }
                for (policy, cap), stats in results.items()
            },
        }
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(document, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    sys.exit(main())
