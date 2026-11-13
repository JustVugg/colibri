"""Pure checks for captured candidate evidence; never a semantic truth checker.

Copyright 2026 Albert Tan Sin Hon. Licensed under Apache-2.0.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit

SCHEMA = "atlas.evidence.capture.v1"
MAX_RECORD_BYTES = 1_048_576
MAX_SOURCE_BYTES = 262_144
MAX_SOURCES = 16
MAX_CLAIMS = 100
_ID = re.compile(r"[A-Za-z][A-Za-z0-9._-]{0,79}\Z")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def canonical(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError("Record must contain finite, well-formed JSON data") from exc


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def strict_json(text):
    if type(text) is not str or len(text.encode("utf-8")) > MAX_RECORD_BYTES:
        raise ValueError("JSON must be UTF-8 text of at most 1 MiB")
    try:
        value = json.loads(text, object_pairs_hook=_pairs,
                           parse_constant=lambda token: (_ for _ in ()).throw(ValueError("Non-finite JSON number")))
        canonical(value)
        return value
    except (json.JSONDecodeError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid JSON record") from exc


def _object(value, required, optional=()):
    if type(value) is not dict or not set(required).issubset(value) or set(value) - set(required) - set(optional):
        raise ValueError("Object fields do not match the declared schema")


def _text(value, label, maximum=8192, empty=False):
    if type(value) is not str or (not empty and not value.strip()) or len(value) > maximum:
        raise ValueError(f"{label} must be non-empty text of at most {maximum} characters")
    if _CONTROL.search(value):
        raise ValueError(f"{label} contains unsupported control characters")
    try:
        value.encode("utf-8")
        if len(value.encode("utf-16-le")) // 2 > maximum:
            raise ValueError(f"{label} exceeds {maximum} UTF-16 code units (export compatibility limit)")
    except UnicodeError as exc:
        raise ValueError(f"{label} contains malformed Unicode") from exc
    return value


def _identifier(value):
    if type(value) is not str or not _ID.fullmatch(value):
        raise ValueError("Identifier must begin with an ASCII letter and contain at most 80 letters, digits, dots, underscores or hyphens")


def _path(value):
    _text(value, "Source path", 1024)
    if value != value.strip() or any(c in value for c in "\\:\n\r\t") or value.startswith(("/", "-")) or "@{" in value:
        raise ValueError("Source path must be a safe repository-relative POSIX path")
    if any(part in ("", ".", "..") or part.lower() == ".git" for part in value.split("/")):
        raise ValueError("Source path contains an unsafe component")


def _source(source):
    _object(source, ("id", "text", "title", "sha256"), ("path", "url"))
    _identifier(source["id"])
    _text(source["text"], "Source text", MAX_SOURCE_BYTES)
    if len(source["text"].encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("Source exceeds the 256 KiB byte limit")
    _text(source["title"], "Source title", 1024, empty=True)
    if source["sha256"] != hashlib.sha256(source["text"].encode("utf-8")).hexdigest():
        raise ValueError("Source digest does not match its exact UTF-8 text")
    if "path" in source:
        _path(source["path"])
    if "url" in source:
        _text(source["url"], "Source URL", 2048)
        url = urlsplit(source["url"])
        if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or any(c.isspace() for c in source["url"]):
            raise ValueError("Source URL must be HTTP(S) without credentials or whitespace")
        try:
            port = url.port
            if port is not None and not 1 <= port <= 65535:
                raise ValueError("Source URL port is out of range")
            hostname = url.hostname
            if ":" in hostname or re.fullmatch(r"[0-9.]+", hostname):
                if "%" in hostname:
                    raise ValueError("Scoped source URL addresses are unsupported")
                ipaddress.ip_address(hostname)
            else:
                ascii_host = hostname.encode("idna").decode("ascii").rstrip(".")
                if len(ascii_host) > 253 or any(not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", label) for label in ascii_host.split(".")):
                    raise ValueError("Source URL has an invalid hostname")
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Source URL has an invalid hostname or port") from exc
    return source


def validate_sources(sources, allow_empty=False):
    if type(sources) is not list or not (0 if allow_empty else 1) <= len(sources) <= MAX_SOURCES:
        raise ValueError("Provide between 1 and 16 source snapshots")
    ids, paths = set(), set()
    for source in sources:
        _source(source)
        if source["id"] in ids:
            raise ValueError("Duplicate source identifier")
        path = source.get("path", f"sources/{source['id']}.txt")
        if path in paths:
            raise ValueError("Duplicate source path")
        ids.add(source["id"])
        paths.add(path)
    if len(canonical(sources)) > MAX_RECORD_BYTES:
        raise ValueError("Sources exceed the record byte limit")
    return sources


def snapshot(source_id, text, title="", path=None, url=None):
    if type(text) is not str:
        raise ValueError("Source text must be a string")
    try:
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    except UnicodeError as exc:
        raise ValueError("Source contains malformed Unicode") from exc
    source = {"id": source_id, "text": text, "title": title, "sha256": sha}
    if path is not None:
        source["path"] = path
    if url is not None:
        source["url"] = url
    return _source(source)


def build_capture(sources, request, response, mode="fixture", model_revision="unspecified", engine_revision="unspecified"):
    capture = {"schema": SCHEMA, "sources": sources, "request": request, "response": response,
               "mode": mode, "model_revision": model_revision, "engine_revision": engine_revision,
               "created_at": datetime.now(timezone.utc).isoformat()}
    capture["sha256"] = digest(capture)
    validate_capture(capture)
    return strict_json(canonical(capture).decode("utf-8"))


def validate_capture(capture):
    _object(capture, ("schema", "sources", "request", "response", "mode", "model_revision", "engine_revision", "created_at", "sha256"))
    if len(canonical(capture)) > MAX_RECORD_BYTES:
        raise ValueError("Capture exceeds the 1 MiB byte limit")
    if capture["schema"] != SCHEMA or capture["mode"] not in ("fixture", "live"):
        raise ValueError("Unsupported capture schema or mode")
    validate_sources(capture["sources"])
    if type(capture["request"]) is not dict or type(capture["response"]) is not dict:
        raise ValueError("Request and response must be JSON objects")
    for name in ("model_revision", "engine_revision"):
        _text(capture[name], name, 512)
    _text(capture["created_at"], "Capture timestamp", 64)
    try:
        stamp = datetime.fromisoformat(capture["created_at"])
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError("Timestamp must include UTC timezone")
    except (ValueError, AttributeError) as exc:
        raise ValueError("Capture timestamp must be an ISO UTC timestamp") from exc
    payload = {key: value for key, value in capture.items() if key != "sha256"}
    if type(capture["sha256"]) is not str or capture["sha256"] != digest(payload):
        raise ValueError("Capture digest does not match its content")
    return capture


def _candidate_claims(capture):
    response = capture["response"]
    choices = response.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("Exactly one model response choice is required")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Model output was incomplete or ended with a non-stop reason")
    message = choice.get("message")
    if type(message) is not dict or message.get("tool_calls") or message.get("function_call"):
        raise ValueError("A text answer without tool calls is required")
    content = message.get("content")
    if type(content) is not str:
        raise ValueError("Model response is missing text content")
    parsed = strict_json(content)
    _object(parsed, ("claims",))
    claims = parsed["claims"]
    if type(claims) is not list or not 1 <= len(claims) <= MAX_CLAIMS:
        raise ValueError("Model output must contain between 1 and 100 candidate claims")
    return claims


def assess(capture, current_sources=None, min_quote_chars=1):
    validate_capture(capture)
    if type(min_quote_chars) is not int or not 1 <= min_quote_chars <= MAX_SOURCE_BYTES:
        raise ValueError("Minimum quote length must be a positive integer within the source limit")
    current = capture["sources"] if current_sources is None else validate_sources(current_sources, allow_empty=True)
    originals = {source["id"]: source for source in capture["sources"]}
    present = {source["id"]: source for source in current}
    report = {"schema": "atlas.evidence.report.v1", "capture_sha256": capture["sha256"],
              "mode": capture["mode"], "passed": False, "semantic_status": "unreviewed",
              "check_scope": "Exact source text, offsets, snapshot hashes and stated quote-length policy only.",
              "current_source_hashes": {key: value["sha256"] for key, value in present.items()},
              "min_quote_chars": min_quote_chars, "claims": [], "errors": []}
    try:
        claims = _candidate_claims(capture)
    except ValueError as exc:
        report["errors"].append(str(exc))
        return report
    seen = set(originals)
    for candidate in claims:
        checked = {"candidate": candidate, "status": "rejected", "semantic_status": "unreviewed", "reasons": []}
        try:
            _object(candidate, ("id", "text", "source_id", "quote", "start", "end"))
            _identifier(candidate["id"])
            _identifier(candidate["source_id"])
            if candidate["id"] in seen:
                raise ValueError("Duplicate claim identifier or collision with a source identifier")
            seen.add(candidate["id"])
            _text(candidate["text"], "Candidate claim", 8192)
            _text(candidate["quote"], "Quoted passage", MAX_SOURCE_BYTES)
            source = originals.get(candidate["source_id"])
            if source is None:
                raise ValueError("Unknown source identifier")
            start, end = candidate["start"], candidate["end"]
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"]):
                raise ValueError("Offsets must be valid zero-based Unicode codepoint bounds")
            if source["text"][start:end] != candidate["quote"]:
                raise ValueError("Quoted passage does not match the exact source span")
            if len(candidate["quote"]) < min_quote_chars:
                raise ValueError("Quoted passage is shorter than the current policy minimum")
            current_source = present.get(candidate["source_id"])
            if current_source is None:
                raise ValueError("Source is missing from the current snapshot set")
            if current_source["sha256"] != source["sha256"]:
                raise ValueError("Source changed since capture; this candidate requires renewed review")
            checked["status"] = "span_verified"
        except ValueError as exc:
            checked["reasons"].append(str(exc))
        report["claims"].append(checked)
    report["passed"] = bool(report["claims"]) and all(item["status"] == "span_verified" for item in report["claims"])
    return report


def source_watch_manifest(capture, report):
    """Export context links only. This is not a declaration of semantic support."""
    if type(report) is not dict:
        raise ValueError("A complete passing report is required for export")
    original = assess(capture, min_quote_chars=report.get("min_quote_chars"))
    if not original["passed"] or canonical(report) != canonical(original):
        raise ValueError("Export requires the complete unmodified passing report under its stated policy and captured source hashes")
    sources = []
    for source in capture["sources"]:
        entry = {"id": source["id"], "path": source.get("path", f"sources/{source['id']}.txt"), "version": source["sha256"]}
        for key in ("title", "url"):
            if source.get(key):
                entry[key] = source[key]
        sources.append(entry)
    claims = []
    for item in original["claims"]:
        claim = item["candidate"]
        claims.append({"id": claim["id"], "text": claim["text"], "evidence": [{"source_id": claim["source_id"], "relation": "context", "pinpoint": f"Unicode codepoints {claim['start']}:{claim['end']} (end exclusive); semantic support unreviewed"}]})
    return {"schema_version": "1.0", "sources": sources, "claims": claims}
