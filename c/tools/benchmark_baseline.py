#!/usr/bin/env python3
"""Plan, collect and summarize a Colibri/SGLang/vLLM HTTP baseline."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

try:
    from . import benchmark_http_serving as http
except ImportError:
    import benchmark_http_serving as http

ENGINES = ("colibri", "sglang", "vllm")
SCHEMA = "colibri.serving-baseline/v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_fields(obj, names):
    require(isinstance(obj, dict), "expected an object")
    for name in names:
        require(isinstance(obj.get(name), str) and obj[name].strip()
                and not obj[name].startswith("REPLACE"), f"fill in {name}")


def load_manifest(path):
    spec = read_json(path)
    require(spec.get("schema") == SCHEMA, "unsupported manifest schema")
    require(spec.get("comparison") in ("matched_artifact", "deployment"),
            "comparison must be matched_artifact or deployment")
    text_fields(spec, ("experiment", "workload", "cache_policy", "reasoning_policy",
                       "speculation_policy", "quality_protocol", "residency"))
    require(spec["residency"] in ("fully_resident", "offload"), "invalid residency")
    text_fields(spec.get("hardware"), ("host", "cpu", "ram", "gpu", "storage", "os", "driver"))
    text_fields(spec.get("model"), ("source_revision", "tokenizer_sha256", "chat_template_sha256"))
    for key in ("tokenizer_sha256", "chat_template_sha256"):
        require(valid_hash(spec["model"][key]), f"invalid {key}")
    matrix = spec.get("matrix", {})
    require(set(matrix) == {"concurrency", "rounds", "repeats", "warmup_requests", "max_tokens",
                            "temperature", "timeout", "slo_first_output", "slo_duration"},
            "matrix fields differ from the documented schema")
    require(isinstance(matrix["concurrency"], list) and matrix["concurrency"], "empty concurrency")
    values = matrix["concurrency"]
    require(all(type(n) is int and n > 0 for n in values) and len(set(values)) == len(values),
            "concurrency must contain unique positive integers")
    for key in ("rounds", "repeats", "max_tokens"):
        require(type(matrix[key]) is int and matrix[key] > 0, f"invalid {key}")
    require(type(matrix["warmup_requests"]) is int and matrix["warmup_requests"] >= 0,
            "invalid warmup_requests")
    for key in ("temperature", "timeout", "slo_first_output", "slo_duration"):
        value = matrix[key]
        require(type(value) in (int, float) and math.isfinite(value)
                and (value >= 0 if key == "temperature" else value > 0), f"invalid {key}")
    engines = spec.get("engines", {})
    require(set(engines) == set(ENGINES), "provide exactly colibri, sglang and vllm")
    for engine in engines.values():
        text_fields(engine, ("base_url", "served_model", "revision", "launch_command",
                             "artifact_sha256", "weight_format", "quantization", "api_key_env"))
        http.endpoint(engine["base_url"])
        require(valid_hash(engine["artifact_sha256"]), "invalid artifact_sha256")
    if spec["comparison"] == "matched_artifact":
        identities = {(e["artifact_sha256"], e["weight_format"], e["quantization"])
                      for e in engines.values()}
        require(len(identities) == 1, "matched_artifact requires identical weights/format/quantization")
    workload_path = Path(path).resolve().parent / spec["workload"]
    workload, workload_hash = http.load_workload(workload_path)
    require(len(workload) * matrix["repeats"] >= max(values),
            "request count must reach the largest concurrency")
    return spec, workload, workload_hash


def valid_hash(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value) and len(set(value)) > 1)


def plan(spec):
    for number in range(1, spec["matrix"]["rounds"] + 1):
        shift = (number - 1) % len(ENGINES)
        for engine in ENGINES[shift:] + ENGINES[:shift]:
            yield {"round": number, "engine": engine,
                   "concurrency": spec["matrix"]["concurrency"]}


def collect(spec, workload, workload_hash, engine, number, output):
    require(engine in ENGINES, "unknown engine")
    require(1 <= number <= spec["matrix"]["rounds"], "round outside matrix")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{engine}-r{number}-c{c}.json" for c in spec["matrix"]["concurrency"]]
    require(not any(p.exists() for p in paths), "round already has results; use a new output directory")
    config = spec["engines"][engine]
    key = os.environ.get(config["api_key_env"], "")
    matrix = spec["matrix"]
    failed = False
    for concurrency, path in zip(matrix["concurrency"], paths):
        kwargs = dict(url=http.endpoint(config["base_url"]), model=config["served_model"],
                      concurrency=concurrency, max_tokens=matrix["max_tokens"],
                      temperature=matrix["temperature"], key=key, timeout=matrix["timeout"])
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        warmup = None
        if matrix["warmup_requests"]:
            prompts = [workload[i % len(workload)] for i in range(matrix["warmup_requests"])]
            rows, summary = http.run(workload=prompts, repeats=1, **kwargs)
            warmup = {"requests": rows, "summary": summary}
        rows, summary = [], None
        if warmup is None or warmup["summary"]["failed"] == 0:
            rows, summary = http.run(workload=workload, repeats=matrix["repeats"],
                                     slo_first_output=matrix["slo_first_output"],
                                     slo_duration=matrix["slo_duration"], **kwargs)
        report = {"schema": SCHEMA, "manifest": spec, "engine": engine, "round": number,
                  "concurrency": concurrency, "workload_sha256": workload_hash,
                  "harness_sha256": digest(http.__file__), "collector_sha256": digest(__file__),
                  "started_at": started, "status": "warmup_failed" if summary is None else "measured",
                  "warmup": warmup, "requests": rows, "summary": summary}
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
        failed |= summary is None or summary["failed"] > 0
        print(path)
        if summary is None:
            break
    return int(failed)


def spread(values):
    present = [v for v in values if v is not None]
    if len(present) != len(values):
        return {"count": len(present), "min": None, "median": None, "max": None}
    return {"count": len(values), "min": min(values),
            "median": statistics.median(values), "max": max(values)}


def compare(spec, workload, workload_hash, directory):
    expected = {(e, r, c) for e in ENGINES for r in range(1, spec["matrix"]["rounds"] + 1)
                for c in spec["matrix"]["concurrency"]}
    reports, issues, fingerprints = {}, [], set()
    files = sorted(Path(directory).glob("*-r*-c*.json"))
    for path in files:
        report = read_json(path)
        require(report.get("schema") == SCHEMA, f"unsupported report: {path}")
        require(report.get("manifest") == spec, f"manifest mismatch: {path}")
        require(report.get("workload_sha256") == workload_hash, f"workload mismatch: {path}")
        identity = (report["engine"], report["round"], report["concurrency"])
        require(identity in expected and identity not in reports, f"unexpected/duplicate cell: {path}")
        fingerprints.add((report["harness_sha256"], report["collector_sha256"]))
        reports[identity] = report
        summary = report["summary"]
        if summary is None:
            issues.append(f"{path.name}: measurement absent")
            continue
        count = len(workload) * spec["matrix"]["repeats"]
        rows = report["requests"]
        require(len(rows) == count and [r["index"] for r in rows] == list(range(count)),
                f"request set mismatch: {path}")
        require(type(summary["wall_seconds"]) in (int, float)
                and math.isfinite(summary["wall_seconds"]) and summary["wall_seconds"] > 0,
                f"invalid wall time: {path}")
        recalculated = http.summarize(rows, summary["wall_seconds"],
                                     spec["matrix"]["slo_first_output"], spec["matrix"]["slo_duration"])
        require(summary == recalculated, f"summary differs from raw requests: {path}")
        if summary["failed"]:
            issues.append(f"{path.name}: failed requests")
        if any(r["completion_tokens"] is None for r in rows if r["success"]):
            issues.append(f"{path.name}: missing token usage")
        if any(r["success"] and (r["first_output_seconds"] is None or r["completion_tokens"] == 0
                                 or r["finish_reason"] == "content_filter") for r in rows):
            issues.append(f"{path.name}: empty or filtered output")
    require(len(fingerprints) <= 1, "mixed benchmark implementations")
    for e, r, c in sorted(expected - reports.keys()):
        issues.append(f"missing {e}-r{r}-c{c}")
    table = []
    for engine in ENGINES:
        for concurrency in spec["matrix"]["concurrency"]:
            cells = [reports.get((engine, r, concurrency))
                     for r in range(1, spec["matrix"]["rounds"] + 1)]
            summaries = [cell["summary"] if cell else None for cell in cells]
            def metric(field, percentile=None):
                values = [s[field] if s else None for s in summaries]
                if percentile:
                    values = [v[percentile] if v else None for v in values]
                return spread(values)
            measured = [cell for cell in cells if cell and cell["summary"]]
            lengths = [row["completion_tokens"] for cell in measured for row in cell["requests"]
                       if row["success"] and row["completion_tokens"] is not None]
            table.append({"engine": engine, "concurrency": concurrency,
                          "rounds_measured": len(measured),
                          "aggregate_completion_tokens_per_second": metric("successful_completion_tokens_per_second"),
                          "first_output_p50_seconds": metric("successful_first_output_seconds", "p50"),
                          "first_output_p95_seconds": metric("successful_first_output_seconds", "p95"),
                          "first_output_p99_seconds": metric("successful_first_output_seconds", "p99"),
                          "duration_p95_seconds": metric("successful_duration_seconds", "p95"),
                          "failure_rate": metric("failure_rate"),
                          "slo_goodput_requests_per_second": metric("latency_slo", "goodput_requests_per_second"),
                          "reported_output_tokens": http.distribution(lengths)})
    return {"schema": SCHEMA, "manifest": spec, "workload_sha256": workload_hash,
            "sources": [{"file": p.name, "sha256": digest(p)} for p in files],
            "comparison": spec["comparison"],
            "status": "incomplete_or_failed" if issues else "protocol_complete",
            "quality": "not_assessed", "hardware_and_server_config": "operator_declared",
            "token_latency": "not_measured", "memory_and_power": "external_telemetry_required",
            "issues": issues, "cells": table}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "compare"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--round", type=int)
    parser.add_argument("--results", default="baseline-results")
    args = parser.parse_args()
    try:
        spec, workload, workload_hash = load_manifest(args.manifest)
        if args.action == "plan":
            print(json.dumps(list(plan(spec)), indent=2))
            return 0
        if args.action == "run":
            require(args.engine is not None and args.round is not None, "run needs --engine and --round")
            return collect(spec, workload, workload_hash, args.engine, args.round, args.results)
        result = compare(spec, workload, workload_hash, args.results)
        print(json.dumps(result, indent=2, allow_nan=False))
        return int(bool(result["issues"]))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
