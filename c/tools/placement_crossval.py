#!/usr/bin/env python3
"""Evaluate expert-placement rankings with category-held-out routing traces."""

import argparse
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PROMPT_RE = re.compile(r"^(?P<category>.+)_(?P<replicate>\d+)\.txt$")


def read_run(path: Path) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected layer expert count")
        layer, expert, count = map(int, fields)
        counts[layer, expert] = count
    if not counts:
        raise ValueError(f"{path}: empty stats")
    return counts


def read_categories(stats_dir: Path) -> dict[str, list[dict[tuple[int, int], int]]]:
    categories: dict[str, list[dict[tuple[int, int], int]]] = defaultdict(list)
    for path in sorted(stats_dir.glob("*.txt")):
        match = PROMPT_RE.match(path.name)
        if match:
            categories[match.group("category")].append(read_run(path))
    if len(categories) < 2:
        raise ValueError(f"{stats_dir}: need at least two categories")
    return dict(categories)


def category_shares(
    runs: list[dict[tuple[int, int], int]],
) -> dict[tuple[int, int], float]:
    shares: dict[tuple[int, int], float] = defaultdict(float)
    for run in runs:
        total = sum(run.values())
        for expert, count in run.items():
            shares[expert] += count / total
    return {expert: share / len(runs) for expert, share in shares.items()}


def rank_experts(
    categories: dict[str, list[dict[tuple[int, int], int]]],
    penalty: float,
) -> list[tuple[int, int]]:
    per_category = [category_shares(runs) for runs in categories.values()]
    experts = set().union(*(shares.keys() for shares in per_category))
    ranked = []
    for expert in experts:
        values = [shares.get(expert, 0.0) for shares in per_category]
        mean = statistics.fmean(values)
        deviation = statistics.pstdev(values)
        score = mean - penalty * deviation
        ranked.append((score, mean, expert))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    return [expert for _, _, expert in ranked]


def rank_pooled(
    categories: dict[str, list[dict[tuple[int, int], int]]],
) -> list[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for runs in categories.values():
        for run in runs:
            for expert, count in run.items():
                counts[expert] += count
    return [expert for expert, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def covered_share(run: dict[tuple[int, int], int], selected: set[tuple[int, int]]) -> float:
    return sum(count for expert, count in run.items() if expert in selected) / sum(run.values())


def cross_validate(
    categories: dict[str, list[dict[tuple[int, int], int]]],
    slots: int,
    penalty: float | None,
) -> tuple[float, float, list[tuple[str, float]]]:
    results = []
    for held_out in sorted(categories):
        train = {name: runs for name, runs in categories.items() if name != held_out}
        ranked = rank_pooled(train) if penalty is None else rank_experts(train, penalty)
        selected = set(ranked[:slots])
        shares = [covered_share(run, selected) for run in categories[held_out]]
        results.append((held_out, statistics.fmean(shares)))
    values = [share for _, share in results]
    return statistics.fmean(values), min(values), results


def write_ranked(path: Path, ranked: list[tuple[int, int]]) -> None:
    # pin_load only needs a deterministic descending integer score.
    size = len(ranked)
    path.write_text(
        "".join(f"{layer} {expert} {size - rank}\n" for rank, (layer, expert) in enumerate(ranked))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("--gpu-slots", type=int, default=9335)
    parser.add_argument("--penalties", default="0,0.25,0.5,1,2")
    parser.add_argument("--write-ranked", type=Path)
    parser.add_argument("--selected-penalty", type=float)
    args = parser.parse_args()

    if args.gpu_slots < 1:
        parser.error("--gpu-slots must be positive")
    penalties = [float(value) for value in args.penalties.split(",")]
    if any(not math.isfinite(value) or value < 0 for value in penalties):
        parser.error("--penalties must be finite and non-negative")

    categories = read_categories(args.stats_dir)
    print(
        f"categories={len(categories)} prompts={sum(map(len, categories.values()))} "
        f"gpu_slots={args.gpu_slots}"
    )
    pooled_mean, pooled_worst, pooled_detail = cross_validate(categories, args.gpu_slots, None)
    print(
        f"pooled heldout_mean={pooled_mean:.4%} heldout_worst={pooled_worst:.4%} "
        + " ".join(f"{name}={share:.2%}" for name, share in pooled_detail)
    )
    best = None
    for penalty in penalties:
        mean, worst, detail = cross_validate(categories, args.gpu_slots, penalty)
        print(
            f"penalty={penalty:g} heldout_mean={mean:.4%} heldout_worst={worst:.4%} "
            + " ".join(f"{name}={share:.2%}" for name, share in detail)
        )
        candidate = (worst, mean, -penalty)
        if best is None or candidate > best[0]:
            best = (candidate, penalty)

    selected = args.selected_penalty if args.selected_penalty is not None else best[1]
    print(f"selected_penalty={selected:g}")
    if args.write_ranked:
        ranked = rank_experts(categories, selected)
        write_ranked(args.write_ranked, ranked)
        print(f"ranked_experts={len(ranked)} ranked_stats={args.write_ranked}")


if __name__ == "__main__":
    main()
