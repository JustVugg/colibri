#!/usr/bin/env python3
"""Validate reproducible, one-variable Colibri experiment records."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _object(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _run(record, name):
    run = _object(record.get(name), name)
    config = _object(run.get("config"), f"{name}.config")
    samples = _object(run.get("samples"), f"{name}.samples")
    speeds = samples.get("tok_s")
    if (not isinstance(speeds, list) or len(speeds) < 3 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) or v <= 0 for v in speeds)):
        raise ValueError(f"{name}.samples.tok_s must contain at least 3 positive finite values")
    median = run.get("median_tok_s")
    if (isinstance(median, bool) or not isinstance(median, (int, float)) or
            not math.isclose(float(median), statistics.median(speeds), rel_tol=1e-6)):
        raise ValueError(f"{name}.median_tok_s must equal the sample median")
    evidence = _object(run.get("evidence"), f"{name}.evidence")
    _text(evidence.get("uri"), f"{name}.evidence.uri")
    digest = evidence.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{name}.evidence.sha256 must be 64 hex characters")
    try:
        bytes.fromhex(digest)
    except ValueError as error:
        raise ValueError(f"{name}.evidence.sha256 must be hexadecimal") from error
    quality = _object(run.get("quality"), f"{name}.quality")
    _text(quality.get("method"), f"{name}.quality.method")
    if quality.get("passed") is not True:
        raise ValueError(f"{name}.quality.passed must be true")
    return config


def validate(record):
    """Return a normalized summary or raise ValueError with an exact field."""
    if record.get("version") != 1:
        raise ValueError("version must be 1")
    for field in ("hypothesis", "commit", "model", "command", "prompt_hash"):
        _text(record.get(field), field)
    commit = record["commit"]
    if len(commit) != 40:
        raise ValueError("commit must be a full 40-character git SHA")
    try:
        bytes.fromhex(commit)
    except ValueError as error:
        raise ValueError("commit must be hexadecimal") from error
    hardware = _object(record.get("hardware"), "hardware")
    for field in ("cpu", "ram", "storage", "os"):
        _text(hardware.get(field), f"hardware.{field}")
    warmup = record.get("warmup_runs")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("warmup_runs must be a non-negative integer")

    baseline = _run(record, "baseline")
    trial = _run(record, "trial")
    keys = sorted(set(baseline) | set(trial))
    actual = [key for key in keys if baseline.get(key) != trial.get(key)]
    declared = record.get("changed_variables")
    if not isinstance(declared, list) or len(declared) != 1 or not isinstance(declared[0], str):
        raise ValueError("changed_variables must name exactly one variable")
    if actual != declared:
        raise ValueError(f"changed_variables {declared!r} does not match config diff {actual!r}")
    outcome = record.get("outcome")
    if outcome not in ("improvement", "regression", "no-change"):
        raise ValueError("outcome must be improvement, regression, or no-change")
    return {
        "variable": actual[0],
        "baseline_tok_s": record["baseline"]["median_tok_s"],
        "trial_tok_s": record["trial"]["median_tok_s"],
        "outcome": outcome,
    }


def validate_path(path):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    return validate(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    args = parser.parse_args(argv)
    failed = False
    for name in args.manifests:
        try:
            summary = validate_path(name)
            print(f"{name}: ok ({summary['variable']}, {summary['outcome']})")
        except (OSError, ValueError, json.JSONDecodeError) as error:
            failed = True
            print(f"{name}: {error}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
