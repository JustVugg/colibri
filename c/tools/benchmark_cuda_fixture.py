"""Reproducible CPU/CUDA A/B benchmark for tools/make_glm_bench_model.py output."""

import argparse
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


SPEED_RE = re.compile(r"REPLAY decode:.*\| ([0-9.]+) tok/s")
PROFILE_RE = re.compile(
    r"PROFILE: expert-disk ([0-9.]+)s \| expert-matmul ([0-9.]+)s "
    r"\| attention ([0-9.]+)s .* lm_head ([0-9.]+)s \| other ([0-9.-]+)s"
)
PROFILE_KEYS = ("disk", "expert_matmul", "attention", "lm_head", "other")


def parse_output(stdout: str, stderr: str = "") -> tuple[float, list[float]]:
    """Extract throughput and profile timings from one engine run."""
    speed = SPEED_RE.search(stdout)
    profile = PROFILE_RE.search(stdout)
    if not speed or not profile:
        raise RuntimeError(f"benchmark output missing\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return float(speed.group(1)), [float(value) for value in profile.groups()]


def execute(engine: str, env: dict[str, str]) -> tuple[float, list[float]]:
    run = subprocess.run(
        [engine, "4", "4", "4"], env=env, text=True, capture_output=True, check=True
    )
    return parse_output(run.stdout, run.stderr)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default="./glm")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--pin-gb", default="1")
    parser.add_argument("--cuda-expert-gb", default="2")
    args = parser.parse_args()

    model = Path(args.model).resolve()
    stats = model / "bench_stats.txt"
    base = os.environ.copy()
    for key in (
        "COLI_CUDA", "COLI_GPU", "COLI_GPUS", "CUDA_EXPERT_GB",
        "PIN", "PIN_GB", "STATS", "TF", "REPLAY", "CUDA_DENSE",
    ):
        base.pop(key, None)
    base.update(
        SNAP=str(model),
        REF=str(model / "ref_glm.json"),
        REPLAY="1",
        OMP_NUM_THREADS=str(args.threads),
        OMP_PROC_BIND="spread",
        OMP_PLACES="cores",
    )

    execute(args.engine, base | {"STATS": str(stats)})
    modes = {
        "cpu_stream": {},
        "dense_cuda": {"COLI_CUDA": "1", "COLI_GPU": args.gpu, "CUDA_DENSE": "1"},
        "cpu_pin": {"PIN": str(stats), "PIN_GB": args.pin_gb},
        "cuda_pin": {
            "COLI_CUDA": "1", "COLI_GPU": args.gpu,
            "PIN": str(stats), "PIN_GB": args.pin_gb,
            "CUDA_EXPERT_GB": args.cuda_expert_gb,
        },
        "cuda_pin_dense": {
            "COLI_CUDA": "1", "COLI_GPU": args.gpu, "CUDA_DENSE": "1",
            "PIN": str(stats), "PIN_GB": args.pin_gb,
            "CUDA_EXPERT_GB": args.cuda_expert_gb,
        },
    }

    for extra in modes.values():
        execute(args.engine, base | extra)  # warm-up
    speeds = {name: [] for name in modes}
    profiles = {name: [] for name in modes}
    names = list(modes)
    for run_index in range(args.runs):
        order = names[run_index % len(names):] + names[:run_index % len(names)]
        for name in order:
            speed, profile = execute(args.engine, base | modes[name])
            speeds[name].append(speed)
            profiles[name].append(profile)

    result = {}
    for name in names:
        result[name] = {
            "runs_tok_s": speeds[name],
            "median_tok_s": statistics.median(speeds[name]),
            "median_profile_s": {
                key: statistics.median(row[index] for row in profiles[name])
                for index, key in enumerate(PROFILE_KEYS)
            },
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
