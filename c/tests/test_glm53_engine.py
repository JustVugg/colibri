#!/usr/bin/env python3
import argparse
import unittest
import json
import subprocess
from pathlib import Path

if __name__ != "__main__":
    raise unittest.SkipTest("command-line GLM-5.3 oracle driver")

parser = argparse.ArgumentParser()
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--cached", action="store_true")
args = parser.parse_args()
reference = json.loads((args.fixture / "ref.json").read_text())
prompt = ",".join(str(token) for token in reference["prompt_ids"])
command = [args.binary.resolve().as_posix(), args.fixture.resolve().as_posix(), "--ids", prompt,
           "--greedy", str(len(reference["greedy_new_ids"]))]
if args.cached:
    command.append("--cached")
result = subprocess.run(
    command,
    text=True, capture_output=True, timeout=180,
)
if result.returncode:
    raise SystemExit(f"engine failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
line = next((line for line in result.stdout.splitlines() if line.startswith("teacher ")), None)
if line is None:
    raise SystemExit("engine did not emit teacher predictions")
actual = [int(value) for value in line.split()[1:]]
expected = reference["teacher_forcing_ids"]
if actual != expected:
    raise SystemExit(f"teacher mismatch\nactual:   {actual}\nexpected: {expected}")
logit_line = next(line for line in result.stdout.splitlines() if line.startswith("last_logits "))
logits = [float(value) for value in logit_line.split()[1:]]
logit_error = max(abs(a - b) for a, b in zip(logits, reference["last_logits"], strict=True))
if logit_error > 2e-4:
    raise SystemExit(f"last-logit mismatch: max abs {logit_error}")
greedy = [int(line.split()[1]) for line in result.stdout.splitlines() if line.startswith("greedy ")]
if greedy != reference["greedy_new_ids"]:
    raise SystemExit(f"greedy mismatch\nactual:   {greedy}\nexpected: {reference['greedy_new_ids']}")
mode = "cached" if args.cached else "full"
print(f"PASS GLM-5.3 CPU {mode} engine: {len(actual)}/{len(expected)} teacher positions exact, "
      f"{len(greedy)} greedy tokens exact, logits max abs {logit_error:.3g}")
