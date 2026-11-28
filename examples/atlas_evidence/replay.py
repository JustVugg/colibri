"""Recheck captured model evidence through the optional Atlas Reweave engine.

No inference happens in this module. Model requests belong to ``client.py``;
these registered callbacks only compute over already captured JSON snapshots.
Accepted means that the declared mechanical checks passed, never that a model's
claim is true or that a quotation semantically supports it.

Copyright 2026 Albert Tan Sin Hon. Licensed under Apache-2.0.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any

from .core import assess, validate_capture

_SCOPE = "colibri-atlas-evidence-v1"
_MAX_BYTES = 4 * 1024 * 1024
_NOTICE = (
    "Candidate evidence for human review. Exact quotation and source-version "
    "checks do not establish truth, authenticity, legal validity or entailment."
)


def _dependency():
    try:
        from reweave import Engine, Limits, Stage, Verdict
        from reweave.bundle import verify_bundle
        from reweave.codec import digest, snapshot
    except ModuleNotFoundError as exc:
        if exc.name == "reweave" or (exc.name or "").startswith("reweave."):
            raise ImportError(
                "Evidence replay requires Atlas Reweave. Install the pinned "
                "dependency with: python -m pip install -r "
                "examples/atlas_evidence/requirements.txt"
            ) from exc
        raise
    return Engine, Limits, Stage, Verdict, verify_bundle, digest, snapshot


def _revision() -> str:
    """Bind callbacks to the two source files which implement their behaviour."""
    digest = sha256()
    for name in ("core.py", "replay.py"):
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        digest.update(Path(__file__).with_name(name).read_bytes())
    return "atlas-evidence-v1:" + digest.hexdigest()


def make_stages(current_sources: list[dict[str, Any]], min_quote_chars: int = 1):
    """Build pure stages whose check policy binds the supplied current sources.

    ``current_sources`` is an explicit operator-supplied snapshot, not a live
    watch of a filesystem or website. Registering a changed snapshot makes old
    evidence subject to a new judgement without regenerating the captured work.
    """
    _, _, Stage, Verdict, _, digest, snapshot = _dependency()
    if type(min_quote_chars) is not int or min_quote_chars < 1:
        raise ValueError("min_quote_chars must be a positive integer")
    if type(current_sources) is not list:
        raise ValueError("current_sources must be a list of source snapshots")
    # The checker receives a detached snapshot. The digest binds even malformed
    # supplied values, so they cannot accidentally reuse an earlier acceptance.
    current = snapshot(current_sources, max_bytes=_MAX_BYTES)
    revision = _revision()
    policy = {
        "current_sources_digest": digest(current),
        "min_quote_chars": min_quote_chars,
        "semantic_review": "unreviewed",
    }

    def import_capture(value, parents):
        if parents:
            raise ValueError("Capture import accepts no parents")
        validate_capture(value)
        return value

    def check_capture(value, output, parents):
        validate_capture(output)
        valid = not parents and value == output
        return Verdict(valid, "Captured JSON structure and declared hashes checked")

    def evidence(value, parents):
        if value != {} or len(parents) != 1:
            raise ValueError("Evidence requires one captured parent and empty input")
        # This immutable computation is deliberately independent of the current
        # review policy. Policy-only changes belong to the verifier below.
        return assess(parents[0], min_quote_chars=1)

    def check_evidence(value, output, parents):
        if value != {} or len(parents) != 1:
            return Verdict(False, "Evidence stage has an invalid parent contract")
        expected = assess(parents[0], min_quote_chars=1)
        reviewed = assess(parents[0], current_sources=current,
                          min_quote_chars=min_quote_chars)
        valid = output == expected and reviewed.get("passed") is True
        return Verdict(
            valid,
            "Current quotation and source-version checks passed; semantic review remains unreviewed"
            if valid else "Current evidence checks failed; dependent report is withheld",
            reviewed,
        )

    def report(value, parents):
        if value != {} or len(parents) != 1:
            raise ValueError("Report requires one evidence parent and empty input")
        return {"schema": "atlas.evidence.review.v1", "notice": _NOTICE,
                "semantic_review": "unreviewed",
                "check_record": "Baseline capture checks; current acceptance is recorded by the receipt and its dependency judgements",
                "evidence": parents[0]}

    def check_report(value, output, parents):
        valid = value == {} and len(parents) == 1 and output == report(value, parents)
        return Verdict(valid, "Report matches its currently accepted evidence dependency")

    return [
        Stage("atlas.capture", revision, import_capture, revision, check_capture,
              description="Import an immutable, already recorded model response"),
        Stage("atlas.evidence", revision, evidence, revision, check_evidence,
              policy=policy,
              description="Check exact source passages against current supplied snapshots"),
        Stage("atlas.report", revision, report, revision, check_report,
              description="Release candidate evidence only while its dependency passes"),
    ]


def run_review(
    capture: dict[str, Any],
    store: str | Path,
    current_sources: list[dict[str, Any]] | None = None,
    min_quote_chars: int = 1,
) -> dict[str, Any]:
    """Record or recheck one captured response and export only accepted evidence.

    Reusing the same capture and store under changed source/check policies keeps
    work identities unchanged. ``stats.producer_calls`` counts pure local
    callbacks in this invocation, not model calls. ``model_requests`` is always
    zero because this function cannot invoke the model.
    """
    Engine, Limits, _, _, verify_bundle, _, snapshot = _dependency()
    recorded = snapshot(capture, max_bytes=_MAX_BYTES)
    validate_capture(recorded)
    current = snapshot(recorded["sources"] if current_sources is None else current_sources,
                       max_bytes=_MAX_BYTES)
    current_review = assess(recorded, current_sources=current,
                            min_quote_chars=min_quote_chars)
    limits = Limits(max_input_bytes=_MAX_BYTES, max_output_bytes=_MAX_BYTES,
                    max_verdict_bytes=_MAX_BYTES)
    with Engine(store, make_stages(current, min_quote_chars), scope=_SCOPE,
                limits=limits) as engine:
        imported = engine.run("atlas.capture", recorded)
        evidence = engine.run("atlas.evidence", {}, parents=[imported.work_key])
        report = engine.run("atlas.report", {}, parents=[evidence.work_key])
        bundle = None
        bundle_check = None
        if report.status == "accepted":
            # Export revalidates the complete dependency closure again.
            bundle = engine.export([report.work_key])
            bundle_check = verify_bundle(bundle, limits=limits)
        stats = engine.stats()
    return {
        "schema": "atlas.evidence.replay.v1",
        "notice": _NOTICE,
        "model_requests": 0,
        "current_review": current_review,
        "capture_receipt": imported.to_dict(),
        "evidence_receipt": evidence.to_dict(),
        "report_receipt": report.to_dict(),
        "stats": stats,
        "bundle": bundle,
        "bundle_check": bundle_check,
    }
