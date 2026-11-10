#!/usr/bin/env python3
"""Run separate long-prompt HTTP benchmark reports from prepared JSONL workloads."""
import argparse
import math
from pathlib import Path
import subprocess
import sys


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def run_matrix(args, runner=subprocess.run):
    workloads = [
        ("32k", Path(args.workload_32k).resolve()),
        ("64k", Path(args.workload_64k).resolve()),
    ]
    if workloads[0][1] == workloads[1][1]:
        raise ValueError("the 32K and 64K workloads must be different files")
    for label, workload in workloads:
        if not workload.is_file():
            raise ValueError(f"{label} workload is not a file: {workload}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = [(label, workload, output_dir / f"prompt-{label}.json")
               for label, workload in workloads]
    existing = [report for _, _, report in reports if report.exists()]
    if existing and not args.overwrite:
        raise ValueError("report already exists; choose a new output directory or use --overwrite")

    benchmark = Path(__file__).with_name("benchmark_http_serving.py")
    failed = False
    for _, workload, report in reports:
        command = [
            sys.executable, str(benchmark),
            "--base-url", args.base_url,
            "--model", args.model,
            "--workload", str(workload),
            "--output", str(report),
            "--concurrency", str(args.concurrency),
            "--repeats", str(args.repeats),
            "--warmup-requests", str(args.warmup_requests),
            "--max-tokens", str(args.max_tokens),
            "--temperature", str(args.temperature),
            "--timeout", str(args.timeout),
            "--api-key-env", args.api_key_env,
        ]
        result = runner(command, check=False)
        failed |= result.returncode != 0
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="API root including /v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload-32k", required=True, help="prepared JSONL workload for 32K")
    parser.add_argument("--workload-64k", required=True, help="prepared JSONL workload for 64K")
    parser.add_argument("--output-dir", required=True, help="directory for separate JSON reports")
    parser.add_argument("--concurrency", type=positive_int, default=1)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--timeout", type=positive_float, default=14400,
                        help="socket operation timeout in seconds (default: 4 hours)")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--overwrite", action="store_true", help="replace existing report files")
    args = parser.parse_args(argv)
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be nonnegative")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    try:
        return run_matrix(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
