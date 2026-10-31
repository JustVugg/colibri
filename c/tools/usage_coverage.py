#!/usr/bin/env python3
"""Coverage curve of an expert history: what a pinned budget would have served.

Input is one or more `.coli_usage` / `STATS=` files (docs/routing-telemetry.md);
counts from several files are summed. Header records (negative layer) are read
for the dimensions and never counted as experts.

For a budget of B bytes the best pinned set maximises the selections it serves
subject to its size fitting in B (a 0/1 knapsack). Ranking experts by
selections per byte and taking the longest prefix that fits is optimal when
every expert has the same width, and at most one expert short of optimal when
widths differ (e.g. an int8 MTP row next to int4 routed rows). The engine's
pin_load takes a prefix the same way but ranks by raw count; `--row-mb` shows
what that costs.

Coverage is the share of the history's selections the pinned set would have
served. Two caveats, both printed: it is measured on the history it was built
from (pass `--holdout` to test the ranking on other files), and it is a lower
bound on the hit rate, because the LRU serves part of the remainder.

Reported:
  concentration  share of the top 1/10/30% of experts, Gini, effective experts
  curve          coverage at a list of budgets
  targets        the smallest budget that reaches each coverage target
  knee           the budget maximising coverage - share of bytes (normalised)
  lambda point   with --min-gain G: where one more GB adds less than G points;
                 the budget maximising coverage - G * GB
  plan           with --vram-gb/--ram-gb: VRAM prefix first (less --reserve-gb,
                 the CUDA_RESERVE_GB default), then RAM, and what pinning only
                 up to the knee would cost in coverage and free in memory
"""

import argparse
import bisect
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class History:
    counts: dict
    n_layers: int
    n_experts: int

    @property
    def total(self):
        return sum(self.counts.values())


def read_history(paths):
    files = []
    for path in map(Path, paths):
        files.extend(sorted(path.glob("*.txt")) if path.is_dir() else [path])
    if not files:
        raise ValueError("no history files")
    counts, dims = {}, None
    for path in files:
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 3:
                raise ValueError(f"{path}:{lineno}: expected layer expert count")
            layer, expert, count = map(int, fields)
            if layer == -1:
                if dims is not None and dims != (expert, count):
                    raise ValueError(f"{path}: dimensions {expert}x{count} differ from {dims[0]}x{dims[1]}")
                dims = (expert, count)
            if layer < 0:  # header record (route_trace.h), not an expert
                continue
            if expert < 0 or count < 0:
                raise ValueError(f"{path}:{lineno}: negative expert or count")
            counts[layer, expert] = counts.get((layer, expert), 0) + count
    counts = {key: n for key, n in counts.items() if n > 0}
    if not counts:
        raise ValueError("history has no selections")
    if dims is None:
        dims = (max(l for l, _ in counts) + 1, max(e for _, e in counts) + 1)
    n_layers, n_experts = dims
    for layer, expert in counts:
        # layer == n_layers is the MTP row (route_trace.h keeps it after the main layers)
        if layer > n_layers or expert >= n_experts:
            raise ValueError(f"expert {layer}:{expert} outside {n_layers}x{n_experts}")
    return History(counts, n_layers, n_experts)


class Curve:
    """Prefix sums over experts ranked by selections per byte (or by count)."""

    def __init__(self, history, width, by="density"):
        layers = sorted({layer for layer, _ in history.counts})
        items = [(layer, expert, history.counts.get((layer, expert), 0), width(layer))
                 for layer in layers for expert in range(history.n_experts)]
        if by == "density":
            items.sort(key=lambda t: (-t[2] / t[3], -t[2], t[0], t[1]))
        else:
            items.sort(key=lambda t: (-t[2], t[0], t[1]))
        self.items = items
        self.total = history.total
        self.bytes, self.covered = [0], [0]
        for _, _, count, size in items:
            self.bytes.append(self.bytes[-1] + size)
            self.covered.append(self.covered[-1] + count)

    @property
    def n(self):
        return len(self.items)

    @property
    def total_bytes(self):
        return self.bytes[-1]

    def coverage(self, k):
        return self.covered[k] / self.total

    def fit(self, budget):
        """Experts in the longest prefix whose size fits in `budget` bytes."""
        return bisect.bisect_right(self.bytes, budget) - 1

    def for_target(self, target):
        """Smallest prefix reaching `target` coverage."""
        need = target * self.total
        return min(bisect.bisect_left(self.covered, need), self.n)

    def knee(self):
        return max(range(self.n + 1),
                   key=lambda k: self.coverage(k) - self.bytes[k] / self.total_bytes)

    def lambda_point(self, min_gain, unit):
        """Prefix after which one more `unit` of bytes adds less than `min_gain` points."""
        for k, (_, _, count, size) in enumerate(self.items):
            if 100.0 * count / self.total / (size / unit) < min_gain:
                return k
        return self.n


def gini(values):
    values = sorted(values)
    n, total = len(values), sum(values)
    if n == 0 or total == 0:
        return 0.0
    weighted = sum((i + 1) * v for i, v in enumerate(values))
    return (2 * weighted) / (n * total) - (n + 1) / n


def concentration(curve):
    counts = sorted((c for _, _, c, _ in curve.items), reverse=True)
    total = curve.total
    top = {}
    for share in (0.01, 0.10, 0.30):
        k = max(1, round(share * len(counts)))
        top[f"top_{round(share * 100)}pct"] = sum(counts[:k]) / total
    return {
        "experts": len(counts),
        "experts_seen": sum(1 for c in counts if c),
        "selections": total,
        **top,
        "gini": gini(counts),
        "effective_experts": 1.0 / sum((c / total) ** 2 for c in counts),
    }


def describe(curve, k, unit, unit_name, holdout=None):
    row = {
        "experts": k,
        "experts_pct": 100.0 * k / curve.n,
        unit_name: curve.bytes[k] / unit,
        "coverage_pct": 100.0 * curve.coverage(k),
    }
    if holdout is not None:
        row["holdout_coverage_pct"] = 100.0 * holdout_coverage(curve, k, holdout)
    return row


def holdout_coverage(curve, k, history):
    served = sum(history.counts.get((layer, expert), 0) for layer, expert, _, _ in curve.items[:k])
    return served / history.total


def analyze(history, expert_mb=None, row_mb=None, targets=(0.7, 0.8, 0.9), budgets=None,
            min_gain=None, vram_gb=None, reserve_gb=2.0, ram_gb=None, holdout=None):
    row_mb = row_mb or {}
    if holdout is not None and (holdout.n_layers, holdout.n_experts) != (history.n_layers, history.n_experts):
        raise ValueError("held-out history comes from a model with other dimensions")
    if expert_mb is None and row_mb:
        raise ValueError("--row-mb needs --expert-mb")
    sized = expert_mb is not None
    if sized:
        width = lambda layer: row_mb.get(layer, expert_mb) * 1e6
        unit, unit_name = 1e9, "gb"
    else:
        width = lambda layer: 1.0
        unit_name = "pct_of_experts"
    curve = Curve(history, width)
    if not sized:
        unit = curve.n / 100.0
    if budgets is None:
        budgets = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256] if sized else [1, 5, 10, 20, 30, 50, 70, 100]
    show = lambda k: describe(curve, k, unit, unit_name, holdout)
    result = {
        "concentration": concentration(curve),
        "curve": [show(curve.fit(b * unit)) for b in budgets if b * unit <= curve.total_bytes * 1.0001],
        "targets": {f"{round(t * 100)}pct": show(curve.for_target(t)) for t in targets},
        "knee": show(curve.knee()),
    }
    if min_gain is not None:
        result["lambda_point"] = {"min_gain": min_gain, **show(curve.lambda_point(min_gain, unit))}
    if vram_gb is not None or ram_gb is not None:
        if not sized:
            raise ValueError("--vram-gb/--ram-gb need --expert-mb")
        vram = max(0.0, (vram_gb or 0.0) - reserve_gb) * 1e9
        k_vram = curve.fit(vram)
        k_all = curve.fit(curve.bytes[k_vram] + (ram_gb or 0.0) * 1e9)
        knee = curve.knee()
        plan = {
            "vram_budget_gb": vram / 1e9,
            "vram": show(k_vram),
            "vram_plus_ram": show(k_all),
        }
        if knee < k_vram:
            plan["vram_to_knee"] = {
                "freed_gb": (curve.bytes[k_vram] - curve.bytes[knee]) / 1e9,
                "coverage_cost_pct": 100.0 * (curve.coverage(k_vram) - curve.coverage(knee)),
            }
        result["plan"] = plan
    if row_mb:
        by_count = Curve(history, width, by="count")
        result["count_vs_density"] = [
            {"gb": b,
             "density_coverage_pct": 100.0 * curve.coverage(curve.fit(b * 1e9)),
             "count_coverage_pct": 100.0 * by_count.coverage(by_count.fit(b * 1e9))}
            for b in budgets if b * 1e9 <= curve.total_bytes * 1.0001]
    return result


def _fmt(row, unit_name):
    size = f"{row[unit_name]:8.2f} GB" if unit_name == "gb" else f"{row[unit_name]:6.1f}% of experts"
    text = f"{size}  {row['experts']:7d} experts ({row['experts_pct']:5.1f}%)  coverage {row['coverage_pct']:5.1f}%"
    if "holdout_coverage_pct" in row:
        text += f"  held-out {row['holdout_coverage_pct']:5.1f}%"
    return text


def print_report(result, unit_name):
    c = result["concentration"]
    print(f"experts {c['experts']} ({c['experts_seen']} seen)  selections {c['selections']}")
    print(f"top 1% {100 * c['top_1pct']:.1f}%  top 10% {100 * c['top_10pct']:.1f}%  "
          f"top 30% {100 * c['top_30pct']:.1f}%  gini {c['gini']:.3f}  "
          f"effective experts {c['effective_experts']:.0f} ({100 * c['effective_experts'] / c['experts']:.1f}%)")
    print("\ncurve")
    for row in result["curve"]:
        print("  " + _fmt(row, unit_name))
    print("\nsmallest budget for a target")
    for name, row in result["targets"].items():
        print(f"  {name:>6}  " + _fmt(row, unit_name))
    print("\nknee (max coverage - share of bytes)")
    print("  " + _fmt(result["knee"], unit_name))
    if "lambda_point" in result:
        lp = result["lambda_point"]
        per = "GB" if unit_name == "gb" else "1% of experts"
        print(f"\nlambda point (next {per} adds < {lp['min_gain']} points)")
        print("  " + _fmt(lp, unit_name))
    if "plan" in result:
        p = result["plan"]
        print(f"\nplan  VRAM budget {p['vram_budget_gb']:.2f} GB")
        print("  VRAM       " + _fmt(p["vram"], unit_name))
        print("  VRAM+RAM   " + _fmt(p["vram_plus_ram"], unit_name))
        if "vram_to_knee" in p:
            k = p["vram_to_knee"]
            print(f"  pinning VRAM only up to the knee frees {k['freed_gb']:.2f} GB "
                  f"for {k['coverage_cost_pct']:.1f} points of coverage")
    if "count_vs_density" in result:
        print("\nrank by count (engine) vs by selections per byte")
        for row in result["count_vs_density"]:
            print(f"  {row['gb']:8.2f} GB  count {row['count_coverage_pct']:5.1f}%  "
                  f"per byte {row['density_coverage_pct']:5.1f}%")
    print("\ncoverage is in-sample and a lower bound on the hit rate (the LRU serves part of the rest)")


def _row_mb(text):
    layer, _, mb = text.partition(":")
    return int(layer), float(mb)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("history", nargs="+", type=Path, help=".coli_usage / STATS files or directories")
    parser.add_argument("--expert-mb", type=float, help="size of one routed expert in MB")
    parser.add_argument("--row-mb", type=_row_mb, action="append", default=[], metavar="LAYER:MB",
                        help="a layer whose experts have another size (e.g. the MTP row)")
    parser.add_argument("--target", type=float, nargs="+", default=[0.7, 0.8, 0.9])
    parser.add_argument("--budget", type=float, nargs="+",
                        help="curve points: GB with --expert-mb, else percent of experts")
    parser.add_argument("--min-gain", type=float,
                        help="points of coverage one more GB (or 1%% of experts) must add")
    parser.add_argument("--vram-gb", type=float)
    parser.add_argument("--reserve-gb", type=float, default=2.0)
    parser.add_argument("--ram-gb", type=float)
    parser.add_argument("--holdout", nargs="+", type=Path, help="files to evaluate the ranking on")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for t in args.target:
        if not 0 < t <= 1:
            parser.error("--target values must be in (0, 1]")
    try:
        history = read_history(args.history)
        holdout = read_history(args.holdout) if args.holdout else None
        result = analyze(history, args.expert_mb, dict(args.row_mb), args.target, args.budget,
                         args.min_gain, args.vram_gb, args.reserve_gb, args.ram_gb, holdout)
    except ValueError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result, "gb" if args.expert_mb is not None else "pct_of_experts")


if __name__ == "__main__":
    main()
