#!/usr/bin/env python3
"""Validate the frozen MW-UR3 scoring corpus from raw engine evidence alone.

This adapter is deliberately scoped below any higher-level response-layer
HTTP surface.  It consumes group frames (the batched multi-option wire
kind), ordinary prompt-echo frames, and score-evidence lines, then joins
them to the frozen corpus/comparison/policy identities.  It never imports
or fabricates a response-layer logprob shape of its own.

The run manifest and raw request captures are capture authority, not
analyzer-derived summaries.  request_sha256 is checked against both the exact
bytes submitted and the frozen corpus; token_payload_sha256 is produced
independently from the run's frozen tokenizer.  The tokenizer and container
manifest are supplied as hashed capture inputs, and the result carries the
run-manifest digest.  Reconstructing any binding from output frames would
defeat the identity proof and is not a valid capture.

The group frames, and the score-evidence line format, are wire kinds that
only some engine builds emit.  Each is its own check family: a capture
missing one wire kind is not a malformed capture, it is one this adapter
cannot run that family's comparisons against, and that is reported by a
named per-family ``UNSUPPORTED`` status (never silently degraded to a
partial pass, and never conflated with a real FAIL) -- see
``family_support`` and the ``group_checks``/``score_checks`` result
fields.

Process exit status, as implemented in ``main``:

  0  PASS         result["verdict"] == "PASS"
  1  FAIL         result["verdict"] == "FAIL", or any EvidenceError whose
                  verdict is neither BLOCK nor UNSUPPORTED
  2  BLOCK        an EvidenceError raised with verdict="BLOCK", any other
                  uncaught exception (fail closed -- never emits a stale
                  PASS), or an OSError clearing a stale --output path
                  before the run starts
  3  STOP         result["verdict"] == "STOP"
  4  UNSUPPORTED  result["verdict"] == "UNSUPPORTED", or an EvidenceError
                  raised with verdict="UNSUPPORTED"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCHEMA = "mw-ur3-raw-result-v1"
RUN_SCHEMA = "mw-ur3-raw-run-v1"
CAPTURE_MAX_TOKENS = 1
REQUEST_ID_MAX = (1 << 64) - 1
TOKENIZER_ID_MAX = 1 << 21
EXPECTED = {
    "items": 20,
    "options": 80,
    "token_position": 240,
    "summed_score": 56,
    "prefix_swallowed": 24,
    "option_argmax_input": 80,
    "item_argmax": 20,
    "register": 420,
}
BOUNDARY = 2.624e-05
AUTHORITY_SHA256 = {
    "corpus": "d750ab07d611a140836f70ccba31fe28e586630d365131c4ce52e63120a07b6e",
    "comparisons": "b4c73d22a0d5f7e595f7833f8b31c4669bafbc928fa21192b585df74455f3476",
    "policy": "308734a31af638e912ccec790df07704d4d0f1d82b253353bddaa39aa261c56f",
}
UINT = re.compile(r"(?:0|[1-9][0-9]*)\Z")
NEG_FINITE = re.compile(
    r"-(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\Z"
    r"|0(?:\.0+)?(?:e[+-]?[0-9]+)?\Z",
    re.IGNORECASE,
)


class EvidenceError(Exception):
    def __init__(self, message: str, verdict: str = "FAIL",
                offset: int | None = None) -> None:
        super().__init__(message)
        self.verdict = verdict
        # The byte offset of the frame header this error was raised while
        # validating, when known -- lets a field-level error (a malformed
        # number deep inside a frame, say) be traced back to the same
        # header offset a truncated/malformed-header error already carries.
        self.offset = offset


GROUP_FRAME_KINDS = frozenset({"GRPP", "GRPG", "GRPS", "GRPE"})


def family_support(group_frames: list[dict], score_evidence_raw: bytes
                   ) -> tuple[bool, bool]:
    """Report, per check family, whether this capture's wire kinds let the
    adapter even attempt that family -- never raises.  ``group_frames`` is
    the already-parsed group-frame stream; ``score_evidence_raw`` is the
    raw score-evidence bytes (read once, not reopened) scanned for a bare
    ``SCORE `` line prefix rather than the strict :func:`validate_score`
    grammar, so a capture with no score-evidence lines at all is
    distinguished from one with malformed lines -- the latter is a real
    FAIL, not an UNSUPPORTED.
    """
    group_present = bool(
        {frame["kind"] for frame in group_frames} & GROUP_FRAME_KINDS)
    text = score_evidence_raw.decode("utf-8", errors="replace")
    score_present = any(line.startswith("SCORE ") for line in text.splitlines())
    return group_present, score_present


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokenizer_payloads(raw: bytes, token_ids: set[int],
                       label: str = "tokenizer capture") -> dict[int, bytes]:
    """Decode exact token payload bytes from the frozen ByteLevel tokenizer.

    This mirrors ``tok_decode`` in ``c/tok.h`` for one token at a time: added
    tokens render as their literal UTF-8 content, while model-vocabulary pieces
    are mapped through the inverse GPT-2/ByteLevel byte table.  The run manifest
    is deliberately not an input to this authority.
    """
    root = load_json_bytes(raw, label)
    if not isinstance(root, dict) or not isinstance(root.get("model"), dict):
        raise EvidenceError("tokenizer capture has no model object")
    vocab = root["model"].get("vocab")
    added = root.get("added_tokens", [])
    if not isinstance(vocab, dict) or not isinstance(added, list):
        raise EvidenceError("tokenizer capture has malformed vocabulary")

    id_to_piece: dict[int, str] = {}
    for piece, token in vocab.items():
        if not isinstance(piece, str) or type(token) is not int or \
                not 0 <= token <= TOKENIZER_ID_MAX:
            raise EvidenceError("tokenizer vocabulary entry is malformed")
        if "\x00" in piece or token in id_to_piece:
            raise EvidenceError("tokenizer vocabulary token ID is ambiguous")
        id_to_piece[token] = piece

    added_ids: set[int] = set()
    for record in added:
        if not isinstance(record, dict) or not {"id", "content"} <= set(record):
            raise EvidenceError("tokenizer added-token entry is malformed")
        token = record.get("id")
        content = record.get("content")
        if type(token) is not int or not 0 <= token <= TOKENIZER_ID_MAX or \
                not isinstance(content, str) or \
                "\x00" in content or token in added_ids:
            raise EvidenceError("tokenizer added-token entry is malformed")
        added_ids.add(token)
        id_to_piece[token] = content

    direct = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    cp_to_byte: dict[int, int] = {}
    displaced = 0
    for byte in range(256):
        codepoint = byte if byte in direct else 256 + displaced
        if byte not in direct:
            displaced += 1
        cp_to_byte[codepoint] = byte

    payloads: dict[int, bytes] = {}
    for token in token_ids:
        if type(token) is not int or token < 0 or token not in id_to_piece:
            raise EvidenceError(f"tokenizer has no captured token ID {token}")
        piece = id_to_piece[token]
        if token in added_ids:
            payloads[token] = piece.encode("utf-8")[:256]
            continue
        decoded = bytearray()
        for character in piece:
            byte = cp_to_byte.get(ord(character))
            if byte is not None:
                decoded.append(byte)
        payloads[token] = bytes(decoded[:256])
    return payloads


def load_json_bytes(raw: bytes, label: str):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label}: invalid UTF-8 JSON") from error
    return json.loads(text, object_pairs_hook=unique_object)


def load_json(path: Path):
    return load_json_bytes(path.read_bytes(), str(path))


def load_jsonl_bytes(raw: bytes, label: str):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label}: invalid UTF-8 JSONL") from error
    rows = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if not line.endswith("\n") or "\r" in line or not line.strip():
            raise EvidenceError(f"{label}:{number}: noncanonical JSONL record")
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise EvidenceError(f"{label}:{number}: {error}") from error
    return rows


def load_jsonl(path: Path):
    return load_jsonl_bytes(path.read_bytes(), str(path))


def uint(text: str, label: str) -> int:
    if not UINT.fullmatch(text):
        raise EvidenceError(f"noncanonical {label}: {text!r}")
    return int(text)


def request_id(text: str, label: str) -> str:
    """Return an exact canonical ID in the engine's positive uint64 domain."""
    value = uint(text, label)
    if not 1 <= value <= REQUEST_ID_MAX:
        raise EvidenceError(f"out-of-domain {label}: {text!r}")
    return str(value)


def fixed_metric(text: str, places: int, label: str,
                 upper: float | None = None) -> float:
    if not re.fullmatch(
            rf"-?(?:0|[1-9][0-9]*)\.[0-9]{{{places}}}", text):
        raise EvidenceError(f"noncanonical {label}: {text!r}")
    value = float(text)
    if not math.isfinite(value) or value < 0.0 or \
            (upper is not None and value > upper):
        raise EvidenceError(f"out-of-range {label}: {text!r}", "BLOCK")
    return value


def validate_prof(fields: list[str], label: str) -> tuple[int, int]:
    if len(fields) != 10:
        raise EvidenceError(f"malformed {label} PROF")
    fixed_metric(fields[1], 3, f"{label} PROF wall")
    prompt = uint(fields[2], f"{label} PROF prompt")
    completion = uint(fields[3], f"{label} PROF completion")
    for index, phase in enumerate(
            ("disk", "wait", "matmul", "attention", "head"), 4):
        fixed_metric(fields[index], 3, f"{label} PROF {phase}")
    uint(fields[9], f"{label} PROF forwards")
    return prompt, completion


def validate_done(fields: list[str], label: str) -> tuple[int, int, int]:
    if len(fields) != 9 or fields[2] != "STAT":
        raise EvidenceError(f"malformed {label} DONE")
    emitted = uint(fields[3], f"{label} DONE emitted")
    fixed_metric(fields[4], 2, f"{label} DONE throughput")
    fixed_metric(fields[5], 1, f"{label} DONE hit rate", 100.0)
    fixed_metric(fields[6], 2, f"{label} DONE RSS")
    prompt = uint(fields[7], f"{label} DONE prompt")
    flag = uint(fields[8], f"{label} DONE length flag")
    if flag not in (0, 1):
        raise EvidenceError(f"invalid {label} DONE length flag: {flag}")
    return emitted, prompt, flag


def logprob(text: str, label: str, allow_nan: bool = False) -> float | None:
    if allow_nan and text == "nan":
        return None
    if not NEG_FINITE.fullmatch(text):
        raise EvidenceError(f"invalid {label}: {text!r}")
    value = float(text)
    if not math.isfinite(value) or value > 0.0:
        raise EvidenceError(f"nonfinite/positive {label}: {text!r}", "BLOCK")
    return value


def validate_tail(fields: list[str], start: int, label: str,
                  allow_nan: bool = False,
                  expected_topk: int | None = None
                  ) -> tuple[float | None, int | None, float | None, int]:
    if len(fields) < start + 2:
        raise EvidenceError(f"truncated numeric tail: {label}")
    target = logprob(fields[start], f"{label} target", allow_nan)
    k = uint(fields[start + 1], f"{label} top-k count")
    if k > 32 or len(fields) != start + 2 + 2 * k:
        raise EvidenceError(f"wrong numeric-tail shape: {label}")
    if expected_topk is not None and k != expected_topk:
        raise EvidenceError(
            f"wrong top-k count for {label}: expected {expected_topk}, got {k}")
    if target is None and k != 0:
        raise EvidenceError(f"NO_SCORE row has top-k entries: {label}")
    if target is not None and k == 0:
        raise EvidenceError(f"scored row has no top-k entries: {label}")
    seen = set(); pairs = []
    for index in range(k):
        token = uint(fields[start + 2 + 2 * index], f"{label} top-k token")
        if token in seen:
            raise EvidenceError(f"duplicate top-k token: {label}")
        seen.add(token)
        value = logprob(fields[start + 3 + 2 * index],
                        f"{label} top-k logprob")
        pairs.append((token, value))
    if not pairs:
        return target, None, None, k
    # The engine deliberately emits the selected table in heap-slot order, not
    # score order.  Recover the engine's argmax by value, then its lowest-token-id
    # tie rule; the first emitted entry has no rank meaning.
    best_token, best_value = min(pairs, key=lambda pair: (-pair[1], pair[0]))
    return target, best_token, best_value, k


def parse_frames_bytes(raw: bytes, label: str) -> list[dict]:
    stream = io.BytesIO(raw)
    frames = []
    payload_kinds = {"DATA", "ECHO", "GRPP", "GRPG", "TOOL"}
    while stream.tell() < len(raw):
        offset = stream.tell()
        wire = stream.readline()
        if not wire.endswith(b"\n"):
            raise EvidenceError(f"{label}: truncated header at byte {offset}")
        if b"\r" in wire:
            raise EvidenceError(f"{label}: CR in header at byte {offset}")
        try:
            line = wire[:-1].decode("ascii")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{label}: non-ASCII header at byte {offset}") from error
        if not line:
            raise EvidenceError(f"{label}: empty header at byte {offset}")
        fields = line.split(" ")
        if not fields or not fields[0]:
            raise EvidenceError(f"{label}: malformed header at byte {offset}")
        kind = fields[0]
        payload = None
        if kind in payload_kinds:
            if len(fields) < 3:
                raise EvidenceError(f"{label}: truncated {kind} header")
            size = uint(fields[2], f"{kind} payload size")
            if size > 65536:
                raise EvidenceError(f"{label}: oversized {kind} payload")
            payload = stream.read(size)
            if len(payload) != size or stream.read(1) != b"\n":
                raise EvidenceError(f"{label}: truncated {kind} payload")
        frames.append({"kind": kind, "fields": fields, "payload": payload,
                       "offset": offset})
    return frames


def parse_frames(path: Path) -> list[dict]:
    return parse_frames_bytes(path.read_bytes(), str(path))


def ltr(values) -> float:
    total = 0.0
    for value in values:
        if value is None or not math.isfinite(value):
            raise EvidenceError("nonfinite value in binary64 sum", "BLOCK")
        total += value
        if not math.isfinite(total):
            raise EvidenceError("binary64 sum overflow", "BLOCK")
    return total


def lcp(sequences: list[list[int]]) -> int:
    length = min(map(len, sequences))
    index = 0
    while index < length and len({sequence[index] for sequence in sequences}) == 1:
        index += 1
    return index


def display_path(path: Path) -> str:
    """A relative, non-host-identifying rendering of a bound path for error
    text -- never the absolute host path a missing-file OSError would
    otherwise leak."""
    try:
        shown = os.path.relpath(path)
    except ValueError:
        shown = None
    return shown if shown and not shown.startswith("..") else path.name


def read_authority(role: str, path: Path) -> bytes:
    """Read one bound file's bytes exactly once (the same bytes are then
    both hashed and parsed, to avoid a time-of-check/time-of-use gap
    between the two).  A missing file is a named failure identifying the
    role and a relative path, never a raw host OSError with an absolute
    path."""
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise EvidenceError(
            f"missing authority: {role} ({display_path(path)})") from None
    except OSError as error:
        raise EvidenceError(
            f"unreadable authority: {role} ({display_path(path)}): "
            f"{error.strerror or error}") from None


def require_bindings(manifest: dict, paths: dict[str, Path],
                     capture_paths: dict[str, Path],
                     evidence_paths: dict[str, Path],
                     capture_snapshots: dict[str, bytes] | None = None
                     ) -> tuple[dict, dict, dict, dict[str, bytes]]:
    if manifest.get("schema") != RUN_SCHEMA:
        raise EvidenceError(f"run manifest schema must be {RUN_SCHEMA}")
    expected_keys = {
        "schema", "candidate_head", "candidate_tree", "build_id",
        "container_id", "model_id", "route_id", "run_id",
        "authority_sha256", "capture_sha256", "evidence_sha256",
        "group_request_ids", "sequential_request_ids", "logprobs",
        "request_sha256", "token_payload_sha256",
    }
    if set(manifest) != expected_keys:
        raise EvidenceError("run manifest field set is not exact")
    required = ("candidate_head", "candidate_tree", "build_id", "container_id",
                "model_id", "route_id", "run_id")
    for key in required:
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise EvidenceError(f"run manifest missing nonempty {key}")
    for key in ("candidate_head", "candidate_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise EvidenceError(f"run manifest {key} is not a canonical git object")

    read_bytes: dict[str, bytes] = {}
    for name, path in paths.items():
        read_bytes[f"authority:{name}"] = read_authority(name, path)
    hashes = {name: sha256_bytes(read_bytes[f"authority:{name}"]) for name in paths}
    if hashes != AUTHORITY_SHA256:
        raise EvidenceError("input authorities are not the frozen MW-UR3 objects")
    if manifest.get("authority_sha256") != hashes:
        raise EvidenceError("run manifest authority_sha256 is not exact")

    snapshots = capture_snapshots or {}
    if not set(snapshots) <= set(capture_paths):
        raise EvidenceError("capture snapshot field set is not a subset")
    capture_hashes = {}
    for name, path in capture_paths.items():
        read_bytes[f"capture:{name}"] = (
            snapshots[name] if name in snapshots else read_authority(name, path))
        capture_hashes[name] = sha256_bytes(read_bytes[f"capture:{name}"])
    if manifest.get("capture_sha256") != capture_hashes:
        raise EvidenceError("run manifest capture_sha256 is not exact")

    for name, path in evidence_paths.items():
        read_bytes[f"evidence:{name}"] = read_authority(name, path)
    evidence_hashes = {
        name: sha256_bytes(read_bytes[f"evidence:{name}"]) for name in evidence_paths}
    if manifest.get("evidence_sha256") != evidence_hashes:
        raise EvidenceError("run manifest evidence_sha256 is not exact")
    return hashes, capture_hashes, evidence_hashes, read_bytes


def validate_authorities(corpus_rows: list[dict], comparisons: list[dict],
                         policy: dict):
    if policy.get("schema") != "mw-ur3-policy-v1":
        raise EvidenceError("wrong MW-UR3 policy schema")
    required = policy.get("required_comparisons", {})
    policy_counts = {
        "token_position": required.get("token_positions"),
        "summed_score": required.get("summed_scores"),
        "prefix_swallowed": required.get("prefix_swallowed_semantics"),
        "option_argmax_input": required.get("option_argmax_inputs"),
        "item_argmax": required.get("item_argmax"),
    }
    if policy_counts != {key: EXPECTED[key] for key in policy_counts}:
        raise EvidenceError("policy denominator drift")
    if policy.get("historical_delta_boundary") != BOUNDARY:
        raise EvidenceError("policy stop boundary drift")

    corpus = {}
    for row in corpus_rows:
        key = (row.get("item"), row.get("option"))
        if row.get("schema") != "mw-ur3-corpus-v1" or key in corpus:
            raise EvidenceError(f"duplicate/wrong corpus identity: {key}")
        if row.get("context_length") != len(row.get("context_tokens", [])) or \
           row.get("continuation_length") != len(row.get("continuation_tokens", [])) or \
           row.get("prompt_tokens") != row.get("context_tokens", []) + row.get("continuation_tokens", []):
            raise EvidenceError(f"corpus token/length mismatch: {key}")
        corpus[key] = row
    expected_options = {(item, option) for item in range(20) for option in range(4)}
    if set(corpus) != expected_options or len(corpus_rows) != EXPECTED["options"]:
        raise EvidenceError("corpus is not exact 20x4 set")

    by_kind = defaultdict(list)
    identities = set()
    for row in comparisons:
        kind = row.get("kind")
        if row.get("schema") != "mw-ur3-comparison-v1" or kind not in policy_counts:
            raise EvidenceError(f"unexpected comparison kind/schema: {kind}")
        if kind == "token_position":
            ident = (kind, row.get("item"), row.get("option"), row.get("absolute_position"))
        else:
            ident = (kind, row.get("item"), row.get("option"))
        if ident in identities:
            raise EvidenceError(f"duplicate comparison identity: {ident}")
        identities.add(ident); by_kind[kind].append(row)
    counts = Counter(row["kind"] for row in comparisons)
    if len(comparisons) != EXPECTED["register"] or any(
            counts[key] != value for key, value in policy_counts.items()):
        raise EvidenceError(f"comparison census drift: {dict(counts)}")

    lcps = {}
    for item in range(20):
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        lcps[item] = lcp(prompts)
    expected_token = set()
    expected_semantic = set()
    expected_sum = set()
    for key, row in corpus.items():
        item, option = key; ctx = row["context_length"]; detected = lcps[item]
        if detected > ctx:
            expected_semantic.add((item, option, ctx, detected, detected - ctx))
        else:
            expected_token.update(
                (item, option, position, row["prompt_tokens"][position])
                for position in range(detected, len(row["prompt_tokens"])))
            expected_sum.add((item, option, tuple(range(ctx, len(row["prompt_tokens"])))))
    got_token = {(row["item"], row["option"], row["absolute_position"], row["token_id"])
                 for row in by_kind["token_position"]}
    got_semantic = {(row["item"], row["option"], row["context_length"],
                     row["detected_prefix_length"], row["continuation_positions_not_returned"])
                    for row in by_kind["prefix_swallowed"]}
    got_sum = {(row["item"], row["option"], tuple(row["positions"]))
               for row in by_kind["summed_score"]}
    if got_token != expected_token or got_semantic != expected_semantic or got_sum != expected_sum:
        raise EvidenceError("comparison identities disagree with corpus-derived LCP sets")
    option_rows = {(row.get("item"), row.get("option")): row
                   for row in by_kind["option_argmax_input"]}
    if set(option_rows) != expected_options or len(option_rows) != EXPECTED["options"]:
        raise EvidenceError("option-argmax identities are not the exact 20x4 set")
    for (item, option), row in option_rows.items():
        prompt = corpus[item, option]["prompt_tokens"]
        context = corpus[item, option]["context_length"]
        if row.get("group_absolute_positions") != list(range(lcps[item], len(prompt))) or \
           row.get("sequential_absolute_positions") != list(range(context, len(prompt))) or \
           row.get("prefix_swallowed_positions") != max(0, lcps[item] - context):
            raise EvidenceError(f"option-argmax position drift: {(item, option)}")
    item_rows = {row.get("item"): row for row in by_kind["item_argmax"]}
    if set(item_rows) != set(range(20)) or len(item_rows) != EXPECTED["items"]:
        raise EvidenceError("item-argmax identities are not exact items 0..19")
    for item, row in item_rows.items():
        if row.get("options") != [0, 1, 2, 3] or \
           row.get("option_input_identities") != [[item, option] for option in range(4)]:
            raise EvidenceError(f"item-argmax input drift: {item}")
    return corpus, by_kind, lcps


def token_record(tokens: list[int]) -> bytes:
    if not tokens:
        raise EvidenceError("empty canonical token request")
    return " ".join(map(str, tokens)).encode("ascii")


def parse_request_capture_bytes(raw: bytes, label: str, *, group: bool,
                                expected_topk: int) -> list[tuple[str, bytes]]:
    """Parse the exact canonical SUBMIT bytes written to the engine."""
    records = []
    stream = io.BytesIO(raw)
    while True:
        offset = stream.tell(); line = stream.readline()
        if not line:
            break
        if not line.endswith(b"\n") or b"\r" in line:
            raise EvidenceError(f"{label}:{offset}: noncanonical request header")
        try:
            fields = line[:-1].decode("ascii").split(" ")
        except UnicodeDecodeError as error:
            raise EvidenceError(f"{label}:{offset}: non-ASCII request header") from error
        expected_fields = 11 if group else 10
        if len(fields) != expected_fields or any(field == "" for field in fields) or \
                fields[0] != "SUBMIT":
            raise EvidenceError(f"{label}:{offset}: malformed request header")
        rid = request_id(fields[1], "captured request id")
        slot = uint(fields[2], "captured slot")
        size = uint(fields[3], "captured payload size")
        maximum = uint(fields[4], "captured max tokens")
        gbytes = uint(fields[7], "captured grammar bytes")
        if slot != 0 or maximum != CAPTURE_MAX_TOKENS or \
                fields[5:7] != ["0", "1"] or \
                gbytes != 0 or fields[8] != "ids=1" or \
                fields[9] != f"logprobs={expected_topk}" or \
                (group and fields[10] != "group=4"):
            raise EvidenceError(f"{label}:{offset}: wrong MW-UR3 request profile")
        payload = stream.read(size)
        if len(payload) != size or stream.read(1) != b"\n":
            raise EvidenceError(f"{label}:{offset}: truncated request payload")
        records.append((rid, payload))
    return records


def parse_request_capture(path: Path, *, group: bool,
                          expected_topk: int) -> list[tuple[str, bytes]]:
    return parse_request_capture_bytes(
        path.read_bytes(), str(path), group=group, expected_topk=expected_topk)


def request_topk_and_hashes(manifest: dict) -> tuple[int, int, dict, dict]:
    """Extract and structurally validate the group/sequential topk and the
    request-byte hash bindings the manifest is required to name -- shared
    by both the group and the sequential request-capture validators so the
    two can be run independently: group support does not gate
    sequential validation, and vice versa."""
    logprobs = manifest.get("logprobs")
    if not isinstance(logprobs, dict) or set(logprobs) != {"group", "sequential"}:
        raise EvidenceError("logprobs binding must name exact group/sequential paths")
    group_topk, sequential_topk = logprobs["group"], logprobs["sequential"]
    if type(group_topk) is not int or type(sequential_topk) is not int or \
            not 1 <= group_topk <= 32 or not 1 <= sequential_topk <= 32:
        raise EvidenceError("run logprobs bindings must be canonical integers 1..32")
    bindings = manifest.get("request_sha256")
    if not isinstance(bindings, dict) or set(bindings) != {"group", "sequential"}:
        raise EvidenceError("request_sha256 must name exact group/sequential paths")
    group_hashes, sequential_hashes = bindings["group"], bindings["sequential"]
    if not isinstance(group_hashes, dict) or \
            set(group_hashes) != {str(item) for item in range(20)}:
        raise EvidenceError("group request hashes are not exact items 0..19")
    expected_sequential = {f"{item}:{option}"
                           for item in range(20) for option in range(4)}
    if not isinstance(sequential_hashes, dict) or \
            set(sequential_hashes) != expected_sequential:
        raise EvidenceError("sequential request hashes are not exact 20x4 set")
    return group_topk, sequential_topk, group_hashes, sequential_hashes


def validate_group_requests(gids: dict, corpus: dict, lcps: dict,
                            group_requests_raw: bytes, label: str,
                            group_topk: int, group_hashes: dict) -> None:
    """Bind each raw group SUBMIT capture to the exact frozen token bytes."""
    captured_group = parse_request_capture_bytes(
        group_requests_raw, label, group=True, expected_topk=group_topk)
    expected_group_order = [gids[item] for item in range(20)]
    if [rid for rid, _ in captured_group] != expected_group_order:
        raise EvidenceError("captured group requests are not the exact ordered 20-set")
    for item in range(20):
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        detected = lcps[item]
        group_wire = b"\n".join(
            [token_record(prompts[0][:detected])] +
            [token_record(prompt[detected:]) for prompt in prompts])
        captured_wire = captured_group[item][1]
        if captured_wire != group_wire or \
                group_hashes[str(item)] != sha256_bytes(captured_wire):
            raise EvidenceError(f"group request-byte binding mismatch item {item}")


def validate_sequential_requests(sids: dict, corpus: dict,
                                 sequential_requests_raw: bytes, label: str,
                                 sequential_topk: int,
                                 sequential_hashes: dict) -> None:
    """Bind each raw sequential SUBMIT capture to the exact frozen token
    bytes.  Always run: today's engine emits ordinary sequential captures
    regardless of group or score-evidence support."""
    captured_sequential = parse_request_capture_bytes(
        sequential_requests_raw, label, group=False, expected_topk=sequential_topk)
    expected_sequential_order = [sids[item, option]
                                 for item in range(20) for option in range(4)]
    if [rid for rid, _ in captured_sequential] != expected_sequential_order:
        raise EvidenceError("captured sequential requests are not the exact ordered 80-set")
    for item in range(20):
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        for option, prompt in enumerate(prompts):
            key = f"{item}:{option}"
            captured_wire = captured_sequential[item * 4 + option][1]
            if captured_wire != token_record(prompt) or \
                    sequential_hashes[key] != sha256_bytes(captured_wire):
                raise EvidenceError(
                    f"sequential request-byte binding mismatch {item}/{option}")


def request_maps(manifest: dict):
    group = manifest.get("group_request_ids")
    sequential = manifest.get("sequential_request_ids")
    if not isinstance(group, dict) or set(group) != {str(i) for i in range(20)}:
        raise EvidenceError("group_request_ids must be exact items 0..19")
    expected_seq = {f"{i}:{o}" for i in range(20) for o in range(4)}
    if not isinstance(sequential, dict) or set(sequential) != expected_seq:
        raise EvidenceError("sequential_request_ids must be exact 20x4 set")
    gids = {int(item): request_id(str(rid), "group request id")
            for item, rid in group.items()}
    sids = {tuple(map(int, key.split(":"))):
            request_id(str(rid), "sequential request id")
            for key, rid in sequential.items()}
    all_ids = list(gids.values()) + list(sids.values())
    if len(set(all_ids)) != len(all_ids):
        raise EvidenceError("request IDs are not globally unique")
    return gids, sids


def correlate_group(frames: list[dict], gids: dict, corpus: dict, lcps: dict,
                    expected_topk: int):
    expected_ids = set(gids.values())
    accepted = {}; done = {}; prof = {}; prefixes = defaultdict(dict)
    members = defaultdict(dict); current = {}
    seen_ids = set(); open_rid = None; prof_seen = False
    accept_order = []
    expected_order = [gids[item] for item in sorted(gids)]
    item_for_rid = {rid: item for item, rid in gids.items()}
    for frame in frames:
        try:
            kind, fields = frame["kind"], frame["fields"]
            if kind == "PROF":
                if open_rid is None or current or len(fields) != 10 or prof_seen or \
                        set(members[open_rid]) != set(range(4)):
                    raise EvidenceError("misordered/malformed group PROF")
                prof[open_rid] = validate_prof(fields, "group")
                prof_seen = True
                continue
            if kind in {"HWINFO", "TIERS", "EMAP", "HITS", "STAT", "\x01\x01READY\x01\x01"}:
                if open_rid is not None:
                    raise EvidenceError(f"unexpected {kind} inside group lifecycle")
                continue
            if kind in {"ACCEPT", "DONE", "GRPP", "GRPS", "ECHO", "GRPG", "GRPE", "ERROR"}:
                if len(fields) < 2:
                    raise EvidenceError(f"truncated group frame {kind}")
                rid = request_id(fields[1], f"{kind} request id"); seen_ids.add(rid)
                if rid not in expected_ids:
                    raise EvidenceError(f"unexpected group request id {rid}")
            else:
                raise EvidenceError(f"unexpected raw group frame: {kind}")
            if kind == "ERROR":
                raise EvidenceError(f"group run contains ERROR for {rid}: {' '.join(fields[2:])}")
            if kind == "ACCEPT":
                if len(fields) != 3 or rid in accepted or open_rid is not None or \
                        len(accept_order) >= len(expected_order) or \
                        rid != expected_order[len(accept_order)]:
                    raise EvidenceError(f"duplicate/malformed group ACCEPT {rid}")
                accepted[rid] = uint(fields[2], "group ACCEPT total")
                accept_order.append(rid); open_rid = rid; prof_seen = False
            elif kind == "GRPP":
                if rid != open_rid or current or members[rid] or prof_seen or len(fields) < 6:
                    raise EvidenceError("misordered/truncated GRPP")
                position = uint(fields[3], "GRPP position")
                if position != len(prefixes[rid]):
                    raise EvidenceError(f"out-of-order GRPP position {rid}/{position}")
                tail = validate_tail(fields, 4, "GRPP", allow_nan=position == 0,
                                     expected_topk=0 if position == 0 else expected_topk)
                if position in prefixes[rid]:
                    raise EvidenceError(f"duplicate GRPP position {rid}/{position}")
                prefixes[rid][position] = (tail[0], frame["payload"], tail[1])
            elif kind == "GRPS":
                item = item_for_rid[rid]
                member = uint(fields[2], "GRPS member") if len(fields) == 5 else -1
                if rid != open_rid or len(fields) != 5 or current or prof_seen or \
                        len(prefixes[rid]) != lcps[item] or member != len(members[rid]):
                    raise EvidenceError("malformed GRPS")
                if member in members[rid]:
                    raise EvidenceError(f"duplicate/nested GRPS {rid}/{member}")
                members[rid][member] = {"ctx": uint(fields[3], "GRPS ctx"),
                                        "len": uint(fields[4], "GRPS len"),
                                        "echo": {},
                                        "grpg": None, "grpe": None}
                current[rid] = member
            elif kind == "ECHO":
                if rid != open_rid or rid not in current or len(fields) < 6:
                    raise EvidenceError(f"unattributed group ECHO {rid}")
                position = uint(fields[3], "group ECHO position")
                tail = validate_tail(fields, 4, "group ECHO",
                                     expected_topk=expected_topk)
                record = members[rid][current[rid]]
                if record["grpg"] is not None or position != record["ctx"] + len(record["echo"]):
                    raise EvidenceError(f"out-of-order group ECHO {rid}/{position}")
                if position in record["echo"]:
                    raise EvidenceError(f"duplicate group ECHO {rid}/{position}")
                record["echo"][position] = (tail[0], frame["payload"], tail[1])
            elif kind == "GRPG":
                if rid != open_rid or rid not in current or len(fields) < 7:
                    raise EvidenceError(f"unattributed GRPG {rid}")
                member = uint(fields[3], "GRPG member")
                position = uint(fields[4], "GRPG position")
                if member != current[rid] or member not in members[rid]:
                    raise EvidenceError(f"misattributed/duplicate GRPG {rid}/{member}")
                record = members[rid][member]
                if record["grpg"] is not None or \
                        len(record["echo"]) != record["len"] or \
                        position != record["ctx"] + record["len"]:
                    raise EvidenceError(f"misattributed/duplicate GRPG {rid}/{member}")
                tail = validate_tail(fields, 5, "GRPG",
                                     expected_topk=expected_topk)
                if tail[0] != tail[2]:
                    raise EvidenceError(f"GRPG target is not its argmax {rid}/{member}")
                record["grpg"] = (position, tail[0], tail[1], frame["payload"])
            elif kind == "GRPE":
                if rid != open_rid or rid not in current or len(fields) != 6:
                    raise EvidenceError(f"unattributed/malformed GRPE {rid}")
                member = uint(fields[2], "GRPE member")
                if member != current[rid] or members[rid][member]["grpg"] is None:
                    raise EvidenceError(f"misattributed GRPE {rid}/{member}")
                score = logprob(fields[3], "GRPE score")
                members[rid][member]["grpe"] = (
                    score, uint(fields[4], "GRPE len"), uint(fields[5], "GRPE greedy"))
                del current[rid]
            elif kind == "DONE":
                if rid != open_rid or len(fields) != 9 or rid in done or current or \
                        not prof_seen or set(members[rid]) != set(range(4)):
                    raise EvidenceError(f"duplicate/malformed group DONE {rid}")
                done[rid] = validate_done(fields, "group")
                open_rid = None; prof_seen = False
        except EvidenceError as error:
            if error.offset is not None:
                raise
            raise EvidenceError(
                f"byte {frame['offset']}: {error}", error.verdict,
                offset=frame['offset']) from error
    if current or open_rid is not None or seen_ids != expected_ids or accept_order != expected_order:
        raise EvidenceError("incomplete group member lifecycle/request set")

    values = {}; shared = {}; payloads = {}; shared_payloads = {}; finals = {}
    for item, rid in gids.items():
        detected = lcps[item]
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        lengths = [len(prompt) - detected for prompt in prompts]
        if accepted.get(rid) != detected + sum(lengths):
            raise EvidenceError(f"wrong group ACCEPT total item {item}")
        if set(prefixes[rid]) != set(range(detected)):
            raise EvidenceError(f"wrong GRPP positions item {item}")
        if prefixes[rid].get(0, ("missing",))[0] is not None:
            raise EvidenceError(f"GRPP pos0 is not NO_SCORE item {item}")
        if set(members[rid]) != set(range(4)):
            raise EvidenceError(f"wrong group member set item {item}")
        shared[item] = {position: record[0]
                        for position, record in prefixes[rid].items()}
        shared_payloads[item] = {position: record[1]
                                 for position, record in prefixes[rid].items()}
        for option in range(4):
            record = members[rid][option]; prompt = prompts[option]
            expected_positions = set(range(detected, len(prompt)))
            if record["ctx"] != detected or record["len"] != lengths[option] or \
               set(record["echo"]) != expected_positions:
                raise EvidenceError(f"wrong group member shape item/option {item}/{option}")
            if record["grpg"] is None or record["grpg"][0] != len(prompt):
                raise EvidenceError(f"missing/wrong GRPG item/option {item}/{option}")
            score = ltr(record["echo"][position][0]
                        for position in sorted(expected_positions))
            expected_greedy = int(all(
                record["echo"][position][2] == prompt[position]
                for position in sorted(expected_positions)))
            if record["grpe"] is None or record["grpe"][0] != score or \
               record["grpe"][1] != lengths[option] or \
               record["grpe"][2] != expected_greedy:
                raise EvidenceError(f"GRPE sum/shape mismatch item/option {item}/{option}")
            for position, evidence in record["echo"].items():
                values[item, option, position] = evidence[0]
                payloads[item, option, position] = evidence[1]
            finals[item, option] = {
                "absolute_position": record["grpg"][0],
                "target_logprob": record["grpg"][1],
                "token_id": record["grpg"][2],
                "payload": record["grpg"][3],
            }
        summary = done.get(rid)
        if prof.get(rid) != (accepted[rid], 0):
            raise EvidenceError(f"wrong group PROF item {item}")
        if summary is None or summary != (0, accepted[rid], 0):
            raise EvidenceError(f"wrong group DONE item {item}")
    return values, shared, payloads, shared_payloads, finals


def correlate_sequential(frames: list[dict], sids: dict, corpus: dict,
                         expected_topk: int):
    expected_ids = set(sids.values()); accepted = {}; done = {}; prof = {}
    echoes = defaultdict(dict); generations = {}
    data_count = defaultdict(int)
    seen_ids = set(); open_rid = None; prof_seen = False; accept_order = []
    expected_order = [sids[key] for key in sorted(sids)]
    key_for_rid = {rid: key for key, rid in sids.items()}
    for frame in frames:
        try:
            kind, fields = frame["kind"], frame["fields"]
            if kind == "PROF":
                if open_rid is None or len(fields) != 10 or prof_seen:
                    raise EvidenceError("misordered/malformed sequential PROF")
                key = key_for_rid[open_rid]
                if set(echoes[open_rid]) != set(range(len(corpus[key]["prompt_tokens"]))):
                    raise EvidenceError("sequential PROF before complete ECHO evidence")
                prof[open_rid] = validate_prof(fields, "sequential")
                prof_seen = True
                continue
            if kind in {"HWINFO", "TIERS", "EMAP", "HITS", "STAT", "\x01\x01READY\x01\x01"}:
                continue
            if kind in {"ACCEPT", "DONE", "ECHO", "DATA", "ERROR"}:
                if len(fields) < 2:
                    raise EvidenceError(f"truncated sequential {kind}")
                rid = request_id(fields[1], f"{kind} request id"); seen_ids.add(rid)
                if rid not in expected_ids:
                    raise EvidenceError(f"unexpected sequential request id {rid}")
            else:
                raise EvidenceError(f"unexpected raw sequential frame: {kind}")
            if kind == "ERROR":
                raise EvidenceError(f"sequential run contains ERROR for {rid}")
            if kind == "ACCEPT":
                if len(fields) != 3 or rid in accepted or open_rid is not None or \
                        len(accept_order) >= len(expected_order) or \
                        rid != expected_order[len(accept_order)]:
                    raise EvidenceError(f"duplicate/malformed sequential ACCEPT {rid}")
                accepted[rid] = uint(fields[2], "sequential ACCEPT total")
                accept_order.append(rid); open_rid = rid; prof_seen = False
            elif kind == "ECHO":
                if rid != open_rid or prof_seen or len(fields) < 6:
                    raise EvidenceError("truncated sequential ECHO")
                position = uint(fields[3], "sequential ECHO position")
                if position != len(echoes[rid]):
                    raise EvidenceError(f"out-of-order sequential ECHO {rid}/{position}")
                tail = validate_tail(
                    fields, 4, "sequential ECHO", allow_nan=position == 0,
                    expected_topk=0 if position == 0 else expected_topk)
                if position in echoes[rid]:
                    raise EvidenceError(f"duplicate sequential ECHO {rid}/{position}")
                echoes[rid][position] = (tail[0], frame["payload"], tail[1])
            elif kind == "DATA":
                if rid != open_rid or prof_seen or len(fields) < 6:
                    raise EvidenceError(f"misordered sequential DATA {rid}")
                ordinal = data_count[rid]
                if ordinal >= CAPTURE_MAX_TOKENS:
                    raise EvidenceError(f"wrong sequential DATA lifecycle {rid}/{ordinal}")
                tail = validate_tail(fields, 3, "sequential DATA",
                                     expected_topk=expected_topk)
                if tail[0] != tail[2]:
                    raise EvidenceError(f"sequential DATA target is not its argmax {rid}")
                generations[rid] = {
                    "ordinal": ordinal,
                    "target_logprob": tail[0],
                    "token_id": tail[1],
                    "payload": frame["payload"],
                }
                data_count[rid] += 1
            elif kind == "DONE":
                if rid != open_rid or len(fields) != 9 or rid in done or not prof_seen:
                    raise EvidenceError(f"duplicate/malformed sequential DONE {rid}")
                done[rid] = validate_done(fields, "sequential")
                open_rid = None; prof_seen = False
        except EvidenceError as error:
            if error.offset is not None:
                raise
            raise EvidenceError(
                f"byte {frame['offset']}: {error}", error.verdict,
                offset=frame['offset']) from error
    if open_rid is not None or seen_ids != expected_ids or accept_order != expected_order:
        raise EvidenceError("incomplete sequential request set")
    values = {}; top_tokens = {}; payloads = {}; generated = {}
    for key, rid in sids.items():
        prompt = corpus[key]["prompt_tokens"]
        if accepted.get(rid) != len(prompt) or set(echoes[rid]) != set(range(len(prompt))):
            raise EvidenceError(f"wrong sequential ECHO shape {key}")
        if echoes[rid][0][0] is not None:
            raise EvidenceError(f"sequential pos0 is not NO_SCORE {key}")
        summary = done.get(rid)
        if prof.get(rid) != (len(prompt), CAPTURE_MAX_TOKENS):
            raise EvidenceError(f"wrong sequential PROF {key}")
        if data_count[rid] != CAPTURE_MAX_TOKENS or rid not in generations:
            raise EvidenceError(f"missing sequential DATA lifecycle {key}")
        if summary != (CAPTURE_MAX_TOKENS, len(prompt), 1):
            raise EvidenceError(f"wrong sequential DONE {key}")
        for position, evidence in echoes[rid].items():
            values[key[0], key[1], position] = evidence[0]
            payloads[key[0], key[1], position] = evidence[1]
            top_tokens[key[0], key[1], position] = evidence[2]
        generated[key] = generations[rid]
    return values, top_tokens, payloads, generated


def validate_payload_attribution(manifest: dict, tokenizer_raw: bytes,
                                 tokenizer_label: str,
                                 corpus: dict, lcps: dict,
                                 sequential_payloads: dict, generations: dict,
                                 group_payloads: dict | None = None,
                                 shared_payloads: dict | None = None,
                                 finals: dict | None = None,
                                 ) -> tuple[list[dict], list[dict]]:
    """Join every payload byte string to an independently bound token ID.

    ``group_payloads``/``shared_payloads``/``finals`` are ``None`` when the
    group check family is unsupported for this capture: the
    sequential-side attribution (every prompt position, every generation)
    still runs unconditionally, but the group/sequential cross-check and
    the group-final-distribution binding are skipped rather than raising.
    """
    token_hashes = manifest.get("token_payload_sha256")
    if not isinstance(token_hashes, dict):
        raise EvidenceError("missing token_payload_sha256 identity bindings")
    expected_tokens = {
        token for row in corpus.values() for token in row["prompt_tokens"]
    }
    if finals is not None:
        expected_tokens.update(record["token_id"] for record in finals.values())
    expected_tokens.update(record["token_id"] for record in generations.values())
    derived_payloads = tokenizer_payloads(
        tokenizer_raw, expected_tokens, tokenizer_label)
    if set(token_hashes) != {str(token) for token in expected_tokens} or any(
            not isinstance(digest, str) or
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in token_hashes.values()):
        raise EvidenceError("token payload bindings are not the exact observed token set")
    for token, payload in derived_payloads.items():
        if token_hashes[str(token)] != sha256_bytes(payload):
            raise EvidenceError(
                f"token payload identity is not tokenizer-derived: {token}")

    def bind(token: int, payload: bytes, label: str) -> None:
        if payload != derived_payloads[token] or \
                token_hashes[str(token)] != sha256_bytes(payload):
            raise EvidenceError(f"token payload identity mismatch: {label}")

    for item in range(20):
        prompts = [corpus[item, option]["prompt_tokens"] for option in range(4)]
        detected = lcps[item]
        if group_payloads is not None:
            for position in range(detected):
                shared_payload = shared_payloads[item][position]
                token = prompts[0][position]
                bind(token, shared_payload, f"group shared {item}/{position}")
                for option in range(4):
                    sequential_payload = sequential_payloads[item, option, position]
                    bind(token, sequential_payload,
                         f"sequential shared {item}/{option}/{position}")
                    if shared_payload != sequential_payload:
                        raise EvidenceError(
                            f"group/sequential shared payload mismatch {item}/{option}/{position}")
            for option, prompt in enumerate(prompts):
                for position in range(detected, len(prompt)):
                    group_payload = group_payloads[item, option, position]
                    sequential_payload = sequential_payloads[item, option, position]
                    token = prompt[position]
                    bind(token, group_payload,
                         f"group continuation {item}/{option}/{position}")
                    bind(token, sequential_payload,
                         f"sequential continuation {item}/{option}/{position}")
                    if group_payload != sequential_payload:
                        raise EvidenceError(
                            f"group/sequential payload mismatch {item}/{option}/{position}")
        else:
            for option, prompt in enumerate(prompts):
                for position in range(len(prompt)):
                    sequential_payload = sequential_payloads[item, option, position]
                    bind(token := prompt[position], sequential_payload,
                         f"sequential {item}/{option}/{position}")

    final_rows = []
    if finals is not None:
        for (item, option), record in sorted(finals.items()):
            token = record["token_id"]
            bind(token, record["payload"], f"group final {item}/{option}")
            final_rows.append({
                "item": item,
                "option": option,
                "absolute_position": record["absolute_position"],
                "token_id": token,
                "target_logprob": record["target_logprob"],
                "payload_sha256": sha256_bytes(record["payload"]),
            })
    generation_rows = []
    for (item, option), record in sorted(generations.items()):
        token = record["token_id"]
        bind(token, record["payload"], f"sequential generation {item}/{option}")
        generation_rows.append({
            "item": item,
            "option": option,
            "ordinal": record["ordinal"],
            "token_id": token,
            "target_logprob": record["target_logprob"],
            "payload_sha256": sha256_bytes(record["payload"]),
        })
    return final_rows, generation_rows


def validate_score_bytes(raw: bytes, label: str, corpus_rows: list[dict],
                         sequential: dict, sequential_top1: dict):
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceError(f"{label}: non-ASCII SCORE evidence") from error
    rows = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        fields = line.rstrip("\n").split(" ") if line.endswith("\n") else []
        if len(fields) != 6 or fields[0] != "SCORE":
            raise EvidenceError(f"{label}:{number}: malformed SCORE evidence")
        ordinal = uint(fields[1], "SCORE ordinal")
        if ordinal != len(rows) or not re.fullmatch(r"[0-9a-f]{64}", fields[2]):
            raise EvidenceError(f"{label}:{number}: SCORE identity/order mismatch")
        value = logprob(fields[3], "SCORE sum")
        contlen = uint(fields[4], "SCORE continuation length")
        greedy = uint(fields[5], "SCORE greedy")
        if greedy not in (0, 1):
            raise EvidenceError(f"{label}:{number}: bad SCORE greedy")
        corpus = corpus_rows[ordinal]
        wire = (f"{corpus['context_length']} {corpus['continuation_length']} " +
                " ".join(map(str, corpus["prompt_tokens"])) + "\n").encode()
        digest = hashlib.sha256(wire).hexdigest()
        positions = range(corpus["context_length"], len(corpus["prompt_tokens"]))
        expected = ltr(sequential[corpus["item"], corpus["option"], p]
                       for p in positions)
        expected_greedy = int(all(
            sequential_top1[corpus["item"], corpus["option"], position] ==
            corpus["prompt_tokens"][position] for position in positions))
        if fields[2] != digest or contlen != corpus["continuation_length"] or \
                value != expected or greedy != expected_greedy:
            raise EvidenceError(f"{label}:{number}: SCORE/ECHO mismatch")
        rows.append(value)
    if len(rows) != EXPECTED["options"]:
        raise EvidenceError("SCORE evidence is not exact 80-row set")
    return rows


def validate_score(path: Path, corpus_rows: list[dict], sequential: dict,
                   sequential_top1: dict):
    return validate_score_bytes(
        path.read_bytes(), str(path), corpus_rows, sequential, sequential_top1)


# "worst of the families" ranking: a genuine FAIL always outranks
# an UNSUPPORTED family, a boundary STOP outranks a plain UNSUPPORTED, and
# only two families that both actually ran and passed can ever produce the
# overall PASS.
_VERDICT_RANK = {"PASS": 0, "UNSUPPORTED": 1, "STOP": 2, "FAIL": 3}


def build_group_comparisons(by_kind: dict, group: dict, sequential: dict,
                            shared: dict) -> dict:
    """Join the group-dependent comparison kinds against the group and
    sequential per-position values.  Isolated from :func:`analyze` so the
    "both families supported" and "group-only" paths share one oracle."""
    token_rows = []
    for row in by_kind["token_position"]:
        key = (row["item"], row["option"], row["absolute_position"])
        if key not in group or key not in sequential:
            raise EvidenceError(f"missing token-position evidence {key}")
        delta = group[key] - sequential[key]
        token_rows.append({**row, "group_logprob": group[key],
                           "sequential_logprob": sequential[key], "delta": delta})
    max_abs = max(abs(row["delta"]) for row in token_rows)

    sum_rows = []
    for row in by_kind["summed_score"]:
        item, option = row["item"], row["option"]
        positions = row["positions"]
        gs = ltr(group[item, option, position] for position in positions)
        ss = ltr(sequential[item, option, position] for position in positions)
        sum_rows.append({**row, "group_score": gs, "sequential_score": ss,
                         "delta": gs - ss})

    semantic_rows = []
    for row in by_kind["prefix_swallowed"]:
        item, option = row["item"], row["option"]
        positions = list(range(row["context_length"], row["detected_prefix_length"]))
        semantic_rows.append({**row, "positions": positions,
            "shared_prefix_logprobs": [shared[item][position] for position in positions],
            "sequential_logprobs": [sequential[item, option, position]
                                      for position in positions]})

    option_rows = []
    option_scores = {}
    for row in by_kind["option_argmax_input"]:
        item, option = row["item"], row["option"]
        gs = ltr(group[item, option, position]
                 for position in row["group_absolute_positions"])
        ss = ltr(sequential[item, option, position]
                 for position in row["sequential_absolute_positions"])
        option_scores[item, option] = (gs, ss)
        option_rows.append({**row, "group_score": gs, "sequential_score": ss})

    item_rows = []
    flips = 0
    for row in by_kind["item_argmax"]:
        item = row["item"]
        group_order = sorted(range(4), key=lambda option: (-option_scores[item, option][0], option))
        seq_order = sorted(range(4), key=lambda option: (-option_scores[item, option][1], option))
        ga, sa = group_order[0], seq_order[0]
        gm = option_scores[item, ga][0] - option_scores[item, group_order[1]][0]
        sm = option_scores[item, sa][1] - option_scores[item, seq_order[1]][1]
        flip = ga != sa; flips += int(flip)
        item_rows.append({**row, "group_argmax": ga, "sequential_argmax": sa,
                          "flip": flip, "group_margin": gm,
                          "sequential_margin": sm, "minimum_margin": min(gm, sm)})
    return {"token_rows": token_rows, "sum_rows": sum_rows,
           "semantic_rows": semantic_rows, "option_rows": option_rows,
           "item_rows": item_rows, "max_abs": max_abs, "flips": flips}


def analyze(args) -> dict:
    paths = {"corpus": args.corpus, "comparisons": args.comparisons,
             "policy": args.policy}
    capture_paths = {"tokenizer": args.tokenizer,
                     "container_manifest": args.container_manifest}
    evidence_paths = {"group_frames": args.group_frames,
                      "sequential_frames": args.sequential_frames,
                      "score_evidence": args.score_evidence,
                      "group_requests": args.group_requests,
                      "sequential_requests": args.sequential_requests}
    run_manifest_raw = read_authority("run manifest", args.run_manifest)
    run_manifest_sha256 = sha256_bytes(run_manifest_raw)
    manifest = load_json_bytes(run_manifest_raw, str(args.run_manifest))
    tokenizer_raw = read_authority("tokenizer", args.tokenizer)
    authority_hashes, capture_hashes, evidence_hashes, read_bytes = require_bindings(
        manifest, paths, capture_paths, evidence_paths,
        capture_snapshots={"tokenizer": tokenizer_raw})
    corpus_rows = load_jsonl_bytes(read_bytes["authority:corpus"], "corpus")
    comparisons = load_jsonl_bytes(read_bytes["authority:comparisons"], "comparisons")
    policy = load_json_bytes(read_bytes["authority:policy"], "policy")
    corpus, by_kind, lcps = validate_authorities(corpus_rows, comparisons, policy)
    gids, sids = request_maps(manifest)
    group_topk, sequential_topk, group_hashes, sequential_hashes = \
        request_topk_and_hashes(manifest)

    group_frames_raw = read_bytes["evidence:group_frames"]
    sequential_frames_raw = read_bytes["evidence:sequential_frames"]
    score_evidence_raw = read_bytes["evidence:score_evidence"]
    group_frames = parse_frames_bytes(group_frames_raw, str(args.group_frames))
    sequential_frames = parse_frames_bytes(
        sequential_frames_raw, str(args.sequential_frames))
    group_supported, score_supported = family_support(
        group_frames, score_evidence_raw)

    # Sequential is always attempted: today's engine already emits ordinary
    # ECHO/DATA frames and sequential SUBMIT captures regardless of group
    # or score-evidence support.
    validate_sequential_requests(
        sids, corpus, read_bytes["evidence:sequential_requests"],
        str(args.sequential_requests), sequential_topk, sequential_hashes)
    sequential, sequential_top1, sequential_payloads, generations = correlate_sequential(
        sequential_frames, sids, corpus, sequential_topk)

    group = shared = group_payloads = shared_payloads = finals = None
    if group_supported:
        validate_group_requests(
            gids, corpus, lcps, read_bytes["evidence:group_requests"],
            str(args.group_requests), group_topk, group_hashes)
        group, shared, group_payloads, shared_payloads, finals = correlate_group(
            group_frames, gids, corpus, lcps, group_topk)

    final_rows, generation_rows = validate_payload_attribution(
        manifest, tokenizer_raw, str(args.tokenizer), corpus, lcps,
        sequential_payloads, generations,
        group_payloads=group_payloads, shared_payloads=shared_payloads,
        finals=finals)

    score_checks = "UNSUPPORTED"
    if score_supported:
        validate_score_bytes(
            score_evidence_raw, str(args.score_evidence),
            corpus_rows, sequential, sequential_top1)
        score_checks = "PASS"

    bindings = {key: manifest[key] for key in (
        "candidate_head", "candidate_tree", "build_id", "container_id",
        "model_id", "route_id", "run_id")}
    common = {"schema": SCHEMA,
             "bindings": bindings,
             "run_manifest_sha256": run_manifest_sha256,
             "authority_sha256": authority_hashes,
             "capture_sha256": capture_hashes,
             "evidence_sha256": evidence_hashes,
             "request_sha256": manifest["request_sha256"],
             "logprobs": manifest["logprobs"],
             "request_profile": {"max_tokens": CAPTURE_MAX_TOKENS,
                                 "temperature": 0, "top_p": 1,
                                 "ids": 1, "grammar_bytes": 0},
             "token_payload_sha256": manifest["token_payload_sha256"],
             "token_position_stop_boundary": BOUNDARY,
             "final_distributions": final_rows,
             "sequential_generations": generation_rows}

    if not group_supported:
        group_checks = "UNSUPPORTED"
        overall = max(group_checks, score_checks, key=_VERDICT_RANK.get)
        disposition = (
            "group frame kinds absent (UNSUPPORTED); "
            f"score-evidence checks {score_checks.lower()} "
            f"(ran: {'score' if score_supported else 'none'})")
        return {**common, "verdict": overall, "disposition": disposition,
                "group_checks": group_checks, "score_checks": score_checks,
                "checks_ran": (["score"] if score_supported else [])}

    joined = build_group_comparisons(by_kind, group, sequential, shared)
    observed = {"items": len(joined["item_rows"]),
               "options": len(joined["option_rows"]),
               "token_position": len(joined["token_rows"]),
               "summed_score": len(joined["sum_rows"]),
               "prefix_swallowed": len(joined["semantic_rows"]),
               "option_argmax_input": len(joined["option_rows"]),
               "item_argmax": len(joined["item_rows"]),
               "register": len(comparisons)}
    if observed != EXPECTED:
        raise EvidenceError(f"result denominator mismatch: {observed}")

    group_checks = "PASS"
    if joined["flips"]:
        group_checks = "FAIL"
    elif joined["max_abs"] > BOUNDARY:
        group_checks = "STOP"
    overall = max(group_checks, score_checks, key=_VERDICT_RANK.get)

    if overall == "PASS":
        disposition = "complete exact-authority raw-engine evidence"
    elif group_checks == "FAIL":
        disposition = "one or more item argmax flips"
    elif group_checks == "STOP":
        disposition = "paired token-position boundary exceeded; return raw evidence"
    else:
        disposition = (
            "group checks passed but score-evidence lines are absent "
            "(UNSUPPORTED); ran: group")

    return {**common, "verdict": overall, "disposition": disposition,
            "denominators": observed,
            "group_checks": group_checks, "score_checks": score_checks,
            "checks_ran": ["group"] + (["score"] if score_supported else []),
            "maximum_absolute_token_position_delta": joined["max_abs"],
            "argmax_flips": joined["flips"],
            "token_positions": joined["token_rows"],
            "summed_scores": joined["sum_rows"],
            "prefix_swallowed_semantics": joined["semantic_rows"],
            "option_argmax_inputs": joined["option_rows"],
            "items": joined["item_rows"]}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    for name in ("corpus", "comparisons", "policy", "run-manifest",
                 "tokenizer", "container-manifest", "group-requests",
                 "sequential-requests", "group-frames", "sequential-frames",
                 "score-evidence"):
        result.add_argument(f"--{name}", required=True, type=Path)
    result.add_argument("--output", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.output:
        try:
            args.output.unlink(missing_ok=True)
        except OSError as error:
            sys.stderr.write(f"cannot clear stale evidence output: {error}\n")
            return 2
    try:
        result = analyze(args)
        status = {"PASS": 0, "FAIL": 1, "STOP": 3, "UNSUPPORTED": 4}[result["verdict"]]
    except EvidenceError as error:
        result = {"schema": SCHEMA, "verdict": error.verdict,
                  "disposition": str(error)}
        status = {"BLOCK": 2, "UNSUPPORTED": 4}.get(error.verdict, 1)
    except Exception as error:  # fail closed: never preserve/emit a stale PASS
        result = {"schema": SCHEMA, "verdict": "BLOCK",
                  "disposition": f"internal evidence failure: {type(error).__name__}: {error}"}
        status = 2
    wire = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        temporary = args.output.with_name(
            f".{args.output.name}.tmp.{os.getpid()}")
        try:
            with temporary.open("x", encoding="utf-8", newline="") as handle:
                handle.write(wire); handle.flush(); os.fsync(handle.fileno())
            os.replace(temporary, args.output)
        except OSError as error:
            temporary.unlink(missing_ok=True)
            sys.stderr.write(f"cannot commit evidence output: {error}\n")
            return 2
    else:
        try:
            sys.stdout.write(wire); sys.stdout.flush()
        except OSError:
            return 2
    return status


if __name__ == "__main__":
    raise SystemExit(main())
