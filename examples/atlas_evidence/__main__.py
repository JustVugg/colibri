"""Command-line interface for explicitly selected local evidence snapshots.

Copyright 2026 Albert Tan Sin Hon. Licensed under Apache-2.0.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

from .core import (MAX_RECORD_BYTES, assess, build_capture, canonical, digest,
                   snapshot, source_watch_manifest, strict_json, validate_capture,
                   validate_sources)


def _read(path):
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_RECORD_BYTES + 1)
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError("Input exceeds the 1 MiB byte limit")
    return strict_json(data.decode("utf-8"))


def _write(path, value):
    """Refuse existing output so a new run cannot silently replace evidence."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        if type(value) is dict and value.get("schema") == "atlas.evidence.capture.v1":
            handle.write(canonical(value).decode("utf-8"))
        else:
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")


def _sources(path):
    values = _read(path)
    if type(values) is not list:
        raise ValueError("Sources file must contain a JSON list")
    sources = []
    for value in values:
        if type(value) is not dict:
            raise ValueError("Every source must be an object")
        if "sha256" in value:
            sources.append(value)
        else:
            if set(value) - {"id", "text", "title", "path", "url"} or not {"id", "text"}.issubset(value):
                raise ValueError("Source fields must be id, text and optional title/path/url")
            sources.append(snapshot(value["id"], value["text"], value.get("title", ""), value.get("path"), value.get("url")))
    return validate_sources(sources, allow_empty=True)


def _fixture():
    text = "Fictional research policy. The review period is 30 days. This example is invented, not law."
    quote = "The review period is 30 days."
    source = snapshot("policy", text, "Invented policy for software testing", "sources/policy.txt")
    claim = {"id": "claim-review", "text": "The invented policy specifies a 30-day review period.",
             "source_id": "policy", "quote": quote, "start": text.index(quote), "end": text.index(quote) + len(quote)}
    from .client import build_request
    request = build_request([source], "deterministic-fixture")
    response = {"model": "deterministic-fixture", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps({"claims": [claim]})}}]}
    capture = build_capture([source], request, response, mode="fixture", model_revision="fixture-v1", engine_revision="not-run-fixture")
    # Fixed timestamp makes this synthetic case reproducible byte for byte.
    capture["created_at"] = "2026-11-13T00:00:00+00:00"
    capture["sha256"] = digest({key: value for key, value in capture.items() if key != "sha256"})
    validate_capture(capture)
    return capture


def demo(out, with_reweave=False):
    if with_reweave:
        from .replay import _dependency
        _dependency()
    directory = Path(out)
    if directory.exists():
        raise ValueError("Demo output directory already exists; choose a fresh directory")
    directory.mkdir(parents=True)
    capture = _fixture()
    valid = assess(capture)
    changed_sources = [snapshot("policy", capture["sources"][0]["text"].replace("30 days", "60 days"), capture["sources"][0]["title"], "sources/policy.txt")]
    drift = assess(capture, changed_sources)
    policy = assess(capture, min_quote_chars=80)
    fabricated = copy.deepcopy(capture)
    answer = strict_json(fabricated["response"]["choices"][0]["message"]["content"])
    answer["claims"][0]["quote"] = "The review period is 90 days."
    fabricated["response"]["choices"][0]["message"]["content"] = json.dumps(answer)
    fabricated["sha256"] = digest({key: value for key, value in fabricated.items() if key != "sha256"})
    invalid_quote = assess(fabricated)
    tampered = copy.deepcopy(capture)
    tampered["sources"][0]["text"] += " Altered."
    tamper_detected = False
    try:
        validate_capture(tampered)
    except ValueError:
        tamper_detected = True
    summary = {"schema": "atlas.evidence.demo.v1", "mode": "deterministic-fixture", "model_requests": 0,
               "checks": {"valid_span_passed": valid["passed"], "fabricated_quote_rejected": not invalid_quote["passed"],
                          "source_drift_rejected": not drift["passed"], "changed_policy_rejected": not policy["passed"],
                          "tamper_detected": tamper_detected}, "semantic_status": "unreviewed"}
    _write(directory / "sources.json", capture["sources"])
    _write(directory / "capture.json", capture)
    _write(directory / "report.json", valid)
    from .render import render_html
    (directory / "review.html").write_text(render_html(capture), encoding="utf-8")
    _write(directory / "fabricated-quote-report.json", invalid_quote)
    _write(directory / "source-drift-report.json", drift)
    _write(directory / "changed-policy-report.json", policy)
    _write(directory / "changed-sources.json", changed_sources)
    _write(directory / "source-watch.json", source_watch_manifest(capture, valid))
    source_path = directory / "sources/policy.txt"
    source_path.parent.mkdir()
    source_path.write_text(capture["sources"][0]["text"], encoding="utf-8")
    if with_reweave:
        from .replay import run_review
        store = directory / "reviews.sqlite3"
        reviews = {"original": run_review(capture, store), "policy_changed": run_review(capture, store, min_quote_chars=80), "source_changed": run_review(capture, store, current_sources=changed_sources)}
        _write(directory / "replay.json", reviews)
        summary["checks"].update({
            "replay_original_accepted": reviews["original"]["report_receipt"]["status"] == "accepted",
            "replay_policy_blocks_report": reviews["policy_changed"]["report_receipt"]["status"] == "blocked",
            "replay_source_blocks_report": reviews["source_changed"]["report_receipt"]["status"] == "blocked",
            "replay_policy_reuses_generation": reviews["policy_changed"]["stats"]["producer_calls"] == 0,
            "replay_source_reuses_generation": reviews["source_changed"]["stats"]["producer_calls"] == 0,
            "replay_keeps_work_identity": len({review["report_receipt"]["work_key"] for review in reviews.values()}) == 1,
            "replay_makes_zero_model_requests": all(review["model_requests"] == 0 for review in reviews.values()),
        })
        summary["replay"] = {key: {"status": value["report_receipt"]["status"], "model_requests": value["model_requests"], "stats": value["stats"]} for key, value in reviews.items()}
    _write(directory / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if all(summary["checks"].values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture and replay checks of candidate evidence. Exact spans do not establish semantic truth.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo_parser = sub.add_parser("demo", help="Run invented offline cases with zero model calls")
    demo_parser.add_argument("--out", required=True)
    demo_parser.add_argument("--with-reweave", action="store_true")
    snap_parser = sub.add_parser("snapshot", help="Record one explicitly selected UTF-8 source file")
    snap_parser.add_argument("--id", required=True)
    snap_parser.add_argument("--file", required=True)
    snap_parser.add_argument("--title", default="")
    snap_parser.add_argument("--path")
    snap_parser.add_argument("--url")
    snap_parser.add_argument("--out", required=True)
    capture_parser = sub.add_parser("capture", help="Send selected snapshots to a local Colibri server")
    capture_parser.add_argument("--sources", required=True)
    capture_parser.add_argument("--model", required=True)
    capture_parser.add_argument("--model-revision", required=True)
    capture_parser.add_argument("--engine-revision", required=True)
    capture_parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    capture_parser.add_argument("--timeout", type=float, default=60, help="Socket-operation timeout in seconds, at most 3600")
    capture_parser.add_argument("--max-tokens", type=int, default=2048)
    capture_parser.add_argument("--api-key-env", default="COLI_API_KEY", help="Environment variable name; its value is never captured")
    capture_parser.add_argument("--out", required=True)
    for name in ("check", "replay", "export-source-watch", "render"):
        item = sub.add_parser(name)
        item.add_argument("capture")
        item.add_argument("--sources")
        item.add_argument("--min-quote-chars", type=int, default=1)
        item.add_argument("--out", required=True)
        if name == "replay":
            item.add_argument("--store", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            return demo(args.out, args.with_reweave)
        if Path(args.out).exists():
            raise ValueError("Output already exists; choose a new path to preserve earlier evidence")
        if args.command == "snapshot":
            with Path(args.file).open("rb") as handle:
                data = handle.read(262_145)
            if len(data) > 262_144:
                raise ValueError("Source file exceeds 256 KiB")
            source = snapshot(args.id, data.decode("utf-8"), args.title, args.path, args.url)
            _write(args.out, [source])
            print(f"Saved source snapshot to {args.out}")
            return 0
        if args.command == "capture":
            from .client import build_request, generate
            sources = _sources(args.sources)
            validate_sources(sources)
            request = build_request(sources, args.model, max_tokens=args.max_tokens)
            response = generate(request, endpoint=args.endpoint, api_key=os.environ.get(args.api_key_env), timeout=args.timeout)
            capture = build_capture(sources, request, response, mode="live", model_revision=args.model_revision, engine_revision=args.engine_revision)
            _write(args.out, capture)
            print(f"Captured local response to {args.out}; run check before using candidate evidence.")
            return 0
        capture = _read(args.capture)
        current_sources = _sources(args.sources) if args.sources else None
        if args.command == "render":
            from .render import render_html
            html = render_html(capture, current_sources=current_sources, min_quote_chars=args.min_quote_chars)
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("x", encoding="utf-8") as handle:
                handle.write(html)
            return 0 if assess(capture, current_sources, args.min_quote_chars)["passed"] else 1
        if args.command == "replay":
            from .replay import run_review
            result = run_review(capture, args.store, current_sources=current_sources, min_quote_chars=args.min_quote_chars)
            _write(args.out, result)
            passed = result["report_receipt"]["status"] == "accepted"
        else:
            result = assess(capture, current_sources=current_sources, min_quote_chars=args.min_quote_chars)
            passed = result["passed"]
            if args.command == "export-source-watch":
                if not passed:
                    raise ValueError("Review failed; source-watch export is blocked")
                result = source_watch_manifest(capture, result)
            _write(args.out, result)
        print("Span checks passed; semantic support remains unreviewed." if passed else "Evidence checks failed; inspect the saved report.")
        return 0 if passed else 1
    except ImportError as exc:
        print(f"Optional integration unavailable: {exc}. Install examples/atlas_evidence/requirements.txt for replay.", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        print(f"Evidence Replay: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
