#!/usr/bin/env python3
"""Choose a frequency-ranked GPU prefix that balances CPU and GPU tier time."""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Plan:
    slots: int
    gpu_share: float
    cpu_seconds: float
    gpu_seconds: float

    @property
    def critical_seconds(self) -> float:
        return max(self.cpu_seconds, self.gpu_seconds)


def read_usage(path: Path) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    paths = sorted(path.glob("*.txt")) if path.is_dir() else [path]
    if not paths:
        raise ValueError(f"{path}: no stats files")
    for stats in paths:
        run: dict[tuple[int, int], int] = {}
        for lineno, line in enumerate(stats.read_text().splitlines(), 1):
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{stats}:{lineno}: expected layer expert count")
            layer, expert, count = map(int, fields)
            key = (layer, expert)
            run[key] = max(run.get(key, 0), count)
        for key, count in run.items():
            counts[key] = counts.get(key, 0) + count
    return counts


def read_counts(path: Path) -> list[int]:
    return sorted(read_usage(path).values(), reverse=True)


def solve(
    counts: list[int],
    max_slots: int,
    cpu_seconds_per_selection: float,
    gpu_seconds_per_selection: float,
    min_slots: int = 0,
) -> Plan:
    if not counts or max_slots < 0 or min_slots < 0 or min_slots > max_slots:
        raise ValueError("counts must be non-empty and max_slots non-negative")
    if cpu_seconds_per_selection <= 0 or gpu_seconds_per_selection <= 0:
        raise ValueError("tier costs must be positive")
    total = sum(counts)
    gpu_count = 0
    best = None
    if min_slots == 0:
        best = Plan(0, 0.0, total * cpu_seconds_per_selection, 0.0)
    for slots, count in enumerate(counts[:max_slots], 1):
        gpu_count += count
        if slots < min_slots:
            continue
        plan = Plan(
            slots,
            gpu_count / total,
            (total - gpu_count) * cpu_seconds_per_selection,
            gpu_count * gpu_seconds_per_selection,
        )
        if best is None or plan.critical_seconds < best.critical_seconds:
            best = plan
    if best is None:
        raise ValueError("min_slots exceeds observed experts")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats", type=Path)
    parser.add_argument("--max-gpu-slots", type=int, required=True)
    parser.add_argument("--observed-cpu-seconds", type=float, required=True)
    parser.add_argument("--observed-cpu-selections", type=int, required=True)
    parser.add_argument("--observed-gpu-seconds", type=float, required=True)
    parser.add_argument("--observed-gpu-selections", type=int, required=True)
    parser.add_argument("--total-experts", type=int)
    parser.add_argument("--max-ram-slots", type=int)
    parser.add_argument("--write-ranked", type=Path)
    args = parser.parse_args()

    cpu_cost = args.observed_cpu_seconds / args.observed_cpu_selections
    gpu_cost = args.observed_gpu_seconds / args.observed_gpu_selections
    usage = read_usage(args.stats)
    counts = sorted(usage.values(), reverse=True)
    if (args.total_experts is None) != (args.max_ram_slots is None):
        parser.error("--total-experts and --max-ram-slots must be used together")
    min_slots = 0
    if args.total_experts is not None:
        min_slots = max(0, args.total_experts - args.max_ram_slots)
        if min_slots > args.max_gpu_slots:
            parser.error("GPU+RAM capacity cannot hold all experts")
    plan = solve(counts, args.max_gpu_slots, cpu_cost, gpu_cost, min_slots)
    full = Plan(
        args.max_gpu_slots,
        sum(counts[: args.max_gpu_slots]) / sum(counts),
        (sum(counts) - sum(counts[: args.max_gpu_slots])) * cpu_cost,
        sum(counts[: args.max_gpu_slots]) * gpu_cost,
    )

    print(f"experts={len(counts)} selections={sum(counts)}")
    if args.total_experts is not None:
        print(
            f"capacity total_experts={args.total_experts} ram_slots={args.max_ram_slots} "
            f"min_gpu_slots={min_slots}"
        )
    print(f"cost cpu={cpu_cost * 1e6:.3f} us/selection gpu={gpu_cost * 1e6:.3f} us/selection")
    print(
        f"fill slots={full.slots} gpu_share={full.gpu_share:.4%} "
        f"cpu={full.cpu_seconds:.3f}s gpu={full.gpu_seconds:.3f}s "
        f"critical={full.critical_seconds:.3f}s"
    )
    print(
        f"balance slots={plan.slots} gpu_share={plan.gpu_share:.4%} "
        f"cpu={plan.cpu_seconds:.3f}s gpu={plan.gpu_seconds:.3f}s "
        f"critical={plan.critical_seconds:.3f}s"
    )
    print(f"predicted_speedup={full.critical_seconds / plan.critical_seconds:.4f}x")
    if args.write_ranked:
        ranked = sorted(usage.items(), key=lambda item: (-item[1], item[0]))
        args.write_ranked.write_text(
            "".join(f"{layer} {expert} {count}\n" for (layer, expert), count in ranked)
        )
        print(f"ranked_stats={args.write_ranked}")


if __name__ == "__main__":
    main()
