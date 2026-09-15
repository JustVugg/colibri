#!/usr/bin/env python3
"""Validate one complete ABLATE evidence artifact against a config.json."""

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys

# The canonical-manifest rule is shared with the engine and with the other
# evidence checkers, so it lives in one module. Imported both ways because this
# file is run as a script from the engine directory and imported as part of the
# tools package by the tests.
try:
    from tools.engine_evidence import (
        ManifestFormError, canonical_manifest_bytes)
except ImportError:                       # run directly: tools/ is on the path
    from engine_evidence import ManifestFormError, canonical_manifest_bytes


DOMAIN = b"coli-ablate-manifest/2\n"
_INT = re.compile(r"(?:0|[1-9][0-9]*|-[1-9][0-9]*)")
_SHA256 = re.compile(r"[0-9a-f]{64}")

# The wire schema (``coli-ablate/2``) is a fixed-width LP64 domain, chosen so
# a producer and a checker on different host ABIs agree byte-for-byte: every
# manifest integer and every completion counter is signed 64-bit, while a
# value the engine narrows to C ``int`` (a layer, an expert, a token id) is
# signed 32-bit.
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1
_ENGINE_VOCAB_MAX = 1 << 24
_ENGINE_LAYERS_MAX = 128
_ENGINE_EXPERTS_MAX = 4096
_ENGINE_TEXT_MAX_BYTES = 256 << 20


class AblateEvidenceError(ValueError):
    """The artifact cannot prove a complete ABLATE denominator."""


def _checked_engine_text_size(length, label):
    if type(length) is not int or not 0 <= length <= _ENGINE_TEXT_MAX_BYTES:
        raise AblateEvidenceError(
            f"{label} exceeds the inclusive 256 MiB engine limit")
    return length


def _reject_constant(value):
    raise AblateEvidenceError(f"non-JSON constant: {value}")


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise AblateEvidenceError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_record(raw, label):
    try:
        return json.loads(
            raw.decode("ascii"), parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError,
            AblateEvidenceError) as exc:
        raise AblateEvidenceError(f"invalid {label} JSON: {exc}") from exc


def _bounded_config_bytes(config_path):
    path = pathlib.Path(config_path)
    with path.open("rb") as source:
        source.seek(0, 2)
        length = source.tell()
        _checked_engine_text_size(length, "external config.json")
        source.seek(0)
        raw = source.read(_ENGINE_TEXT_MAX_BYTES + 1)
    _checked_engine_text_size(len(raw), "external config.json")
    return raw


def _config_identity(config_path):
    raw = _bounded_config_bytes(config_path)
    if not raw:
        raise AblateEvidenceError("external config.json is empty")
    try:
        config = json.loads(
            raw.decode("utf-8"), parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError,
            AblateEvidenceError) as exc:
        raise AblateEvidenceError(f"external config.json is invalid: {exc}") from exc
    if not isinstance(config, dict):
        raise AblateEvidenceError("external config.json root is not an object")
    try:
        identity = {
            "vocab": _bounded_int(
                config["vocab_size"], "config vocab_size",
                1, _ENGINE_VOCAB_MAX),
            "n_layers": _bounded_int(
                config["num_hidden_layers"], "config num_hidden_layers",
                1, _ENGINE_LAYERS_MAX),
            "n_experts": _bounded_int(
                config["n_routed_experts"], "config n_routed_experts",
                1, _ENGINE_EXPERTS_MAX),
        }
        identity["first_dense"] = _bounded_int(
            config["first_k_dense_replace"],
            "config first_k_dense_replace", 0, identity["n_layers"])
    except (KeyError, AblateEvidenceError) as exc:
        raise AblateEvidenceError(
            "external config.json topology is incomplete or invalid") from exc
    identity["config_sha256"] = hashlib.sha256(raw).hexdigest()
    return identity


def _bounded_int(value, label, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise AblateEvidenceError(
            f"{label} is outside {minimum}..{maximum}")
    return value


def _manifest_i64(text, line_number):
    try:
        value = int(text)
    except ValueError as exc:
        raise AblateEvidenceError(
            f"manifest line {line_number} integer is too large") from exc
    return _bounded_int(
        value, f"manifest line {line_number} integer",
        _INT64_MIN, _INT64_MAX)


def _count_add(current, increment, label):
    _bounded_int(current, label, 0, _INT64_MAX)
    _bounded_int(increment, f"{label} increment", 0, _INT64_MAX)
    if increment > _INT64_MAX - current:
        raise AblateEvidenceError(f"{label} exceeds signed 64-bit domain")
    return current + increment


def _manifest_proof(raw, vocab, n_layers, first_dense, n_experts):
    # The engine accepts a manifest saved with either line ending, and with or
    # without a terminator on its last line, and digests the canonical form
    # rather than the bytes on disk. Reproduce that here from the shared rule,
    # or this checker would reject a file the producer ran and would compute a
    # different digest for one it accepted.
    try:
        raw = canonical_manifest_bytes(raw)
    except ManifestFormError as exc:
        raise AblateEvidenceError(f"manifest is not canonical text: {exc}") from exc
    _bounded_int(vocab, "external vocabulary", 1, _ENGINE_VOCAB_MAX)
    _bounded_int(n_layers, "external n_layers", 1, _ENGINE_LAYERS_MAX)
    _bounded_int(first_dense, "header first_dense", 0, n_layers)
    _bounded_int(n_experts, "external n_experts", 1, _ENGINE_EXPERTS_MAX)
    items = []
    seen = set()
    item_count = targets = 0
    for line_number, raw_line in enumerate(raw[:-1].split(b"\n"), 1):
        try:
            text = raw_line.decode("ascii")
        except UnicodeDecodeError as exc:
            raise AblateEvidenceError(
                f"manifest line {line_number} is not ASCII") from exc
        parts = text.split(" ")
        if (not parts or any(not _INT.fullmatch(part) for part in parts) or
                " ".join(parts) != text):
            raise AblateEvidenceError(
                f"manifest line {line_number} is not canonical integer grammar")
        values = [_manifest_i64(part, line_number) for part in parts]
        if len(values) < 5:
            raise AblateEvidenceError(f"manifest line {line_number} is truncated")
        item, length, prompt, mode, cells = values[:5]
        if (item < 0 or item in seen or
                not 2 <= length <= _INT32_MAX or prompt < 1 or
                prompt >= length or mode not in range(4) or
                cells not in range(17) or
                (mode == 0 and cells != 0) or
                (mode != 0 and cells == 0)):
            raise AblateEvidenceError(
                f"manifest line {line_number} has invalid fields/denominator")
        expected = 5 + 3 * cells + length
        if len(values) != expected:
            raise AblateEvidenceError(
                f"manifest line {line_number} has invalid fields/denominator")
        triples = []
        source_cells = set()
        cursor = 5
        for _ in range(cells):
            layer, expert, applied = values[cursor:cursor + 3]
            cursor += 3
            if (not first_dense <= layer < n_layers or
                    not 0 <= expert < n_experts or
                    not _INT32_MIN <= applied <= _INT32_MAX or
                    (mode == 3 and
                     (not 0 <= applied < n_experts or applied == expert)) or
                    (mode != 3 and applied != -1) or
                    (layer, expert) in source_cells):
                raise AblateEvidenceError(
                    f"manifest line {line_number} has invalid cell")
            source_cells.add((layer, expert))
            triples.append([layer, expert, applied])
        tokens = values[cursor:]
        if any(token < 0 or token >= vocab for token in tokens):
            raise AblateEvidenceError(
                f"manifest line {line_number} has out-of-vocabulary token")
        seen.add(item)
        positions = range(prompt - 1, length - 1)
        row_targets = length - prompt
        item_count = _count_add(item_count, 1, "manifest item count")
        targets = _count_add(targets, row_targets, "manifest target count")
        items.append({
            "item": item, "T": length, "n_prompt": prompt,
            "mode": mode, "ncells": cells, "cells": triples,
            "positions": positions, "tokens": tuple(tokens),
        })
    if item_count <= 0 or targets <= 0:
        raise AblateEvidenceError("manifest denominator is not positive")
    return {
        "sha256": hashlib.sha256(DOMAIN + raw).hexdigest(),
        "items": tuple(items), "item_count": item_count, "targets": targets,
    }


def validate_ablate_evidence(manifest_path, evidence_path, config_path):
    identity = _config_identity(config_path)
    manifest_raw = pathlib.Path(manifest_path).read_bytes()
    evidence_raw = pathlib.Path(evidence_path).read_bytes()
    if (not evidence_raw or not evidence_raw.endswith(b"\n") or
            b"\r" in evidence_raw or b"\0" in evidence_raw):
        raise AblateEvidenceError("evidence is not nonempty canonical LF JSONL")
    records = [_json_record(line, f"record {index}")
               for index, line in enumerate(evidence_raw.splitlines(), 1)]
    if not records:
        raise AblateEvidenceError("evidence has no header")
    header = records[0]
    if (not isinstance(header, dict) or set(header) != {
            "t", "schema", "vocab", "topk", "manifest_sha256",
            "n_layers", "first_dense", "n_experts",
            "config_sha256",
            "expected_items", "expected_targets"} or
            header.get("t") != "hdr" or header.get("schema") != "coli-ablate/2" or
            type(header.get("topk")) is not int or
            not isinstance(header.get("config_sha256"), str) or
            not _SHA256.fullmatch(header["config_sha256"]) or
            not isinstance(header.get("manifest_sha256"), str) or
            not _SHA256.fullmatch(header["manifest_sha256"])):
        raise AblateEvidenceError("header keys or values are not exact")
    try:
        _bounded_int(header["vocab"], "header vocabulary",
                     1, _ENGINE_VOCAB_MAX)
        if header["topk"] != min(32, header["vocab"]):
            raise AblateEvidenceError("header topk is not producer-exact")
        _bounded_int(header["n_layers"], "header n_layers",
                     1, _ENGINE_LAYERS_MAX)
        _bounded_int(header["first_dense"], "header first_dense",
                     0, header["n_layers"])
        _bounded_int(header["n_experts"], "header n_experts",
                     1, _ENGINE_EXPERTS_MAX)
        _bounded_int(header["expected_items"], "header expected_items",
                     1, _INT64_MAX)
        _bounded_int(header["expected_targets"], "header expected_targets",
                     1, _INT64_MAX)
    except (KeyError, AblateEvidenceError) as exc:
        raise AblateEvidenceError(
            f"header keys or values are not exact: {exc}") from exc
    if any(header[key] != identity[key] for key in (
            "vocab", "n_layers", "first_dense", "n_experts",
            "config_sha256")):
        raise AblateEvidenceError(
            "header does not match the external config identity")
    proof = _manifest_proof(
        manifest_raw, identity["vocab"], identity["n_layers"],
        identity["first_dense"], identity["n_experts"])
    if (header["manifest_sha256"] != proof["sha256"] or
            header["expected_items"] != proof["item_count"] or
            header["expected_targets"] != proof["targets"]):
        raise AblateEvidenceError("header does not bind the source manifest proof")

    cursor = 1
    completed_items = completed_targets = 0
    logit_keys = {
        "t", "item", "pos", "gold", "nll", "glogit", "molo", "mgn",
        "am", "amlogit", "logZ", "corr", "tk",
    }
    for expected in proof["items"]:
        if cursor >= len(records):
            raise AblateEvidenceError("missing item header")
        item_header = records[cursor]
        cursor += 1
        if (not isinstance(item_header, dict) or set(item_header) != {
                "t", "item", "mode", "ncells", "T", "n_prompt", "cells"} or
                any(type(item_header.get(key)) is not int for key in (
                    "item", "mode", "ncells", "T", "n_prompt")) or
                not isinstance(item_header.get("cells"), list) or
                any(not isinstance(cell, list) or len(cell) != 3 or
                    any(type(value) is not int for value in cell)
                    for cell in item_header["cells"]) or
                item_header != {key: expected[key] for key in (
                    "item", "mode", "ncells", "T", "n_prompt", "cells")} |
                {"t": "ah"}):
            raise AblateEvidenceError("item header does not match manifest order")
        for position in expected["positions"]:
            if cursor >= len(records):
                raise AblateEvidenceError("missing target row")
            row = records[cursor]
            cursor += 1
            if (not isinstance(row, dict) or set(row) != logit_keys or
                    row.get("t") != "lg" or type(row.get("item")) is not int or
                    row["item"] != expected["item"] or
                    type(row.get("pos")) is not int or row["pos"] != position or
                    type(row.get("gold")) is not int or
                    row["gold"] != expected["tokens"][position + 1] or
                    type(row.get("am")) is not int or
                    row["am"] not in range(header["vocab"]) or
                    type(row.get("corr")) is not int or row["corr"] not in (0, 1)):
                raise AblateEvidenceError("target row identity/fields are invalid")
            for field in ("nll", "glogit", "molo", "mgn", "amlogit", "logZ"):
                if (type(row.get(field)) not in (int, float) or
                        not math.isfinite(row[field])):
                    raise AblateEvidenceError(f"target row {field} is nonfinite")
            # These three hold for every row the producer can emit: nll is a
            # negated log-probability (always <= 0 before negation); corr is
            # defined as the argmax/gold agreement, not sampled separately;
            # and amlogit is the row's own max logit, so no field can exceed
            # it -- least of all the gold token's own logit.
            if row["nll"] < 0:
                raise AblateEvidenceError("target row nll is negative")
            if row["corr"] != int(row["am"] == row["gold"]):
                raise AblateEvidenceError(
                    "target row corr does not match its am/gold agreement")
            if row["amlogit"] < row["glogit"]:
                raise AblateEvidenceError(
                    "target row amlogit is below glogit")
            topk = row.get("tk")
            if (not isinstance(topk, list) or len(topk) != header["topk"] or
                    any(not isinstance(pair, list) or len(pair) != 2 or
                        type(pair[0]) is not int or not 0 <= pair[0] < header["vocab"] or
                        type(pair[1]) not in (int, float) or not math.isfinite(pair[1])
                        for pair in topk) or
                    len({pair[0] for pair in topk}) != len(topk)):
                raise AblateEvidenceError("target row top-k is invalid")
            completed_targets = _count_add(
                completed_targets, 1, "completed target count")
        completed_items = _count_add(
            completed_items, 1, "completed item count")

    if cursor >= len(records):
        raise AblateEvidenceError("missing terminal completion record")
    done = records[cursor]
    cursor += 1
    if (not isinstance(done, dict) or set(done) != {
            "t", "manifest_sha256", "completed_items", "completed_targets"}):
        raise AblateEvidenceError("terminal completion proof is invalid")
    try:
        _bounded_int(done["completed_items"], "done completed_items",
                     1, _INT64_MAX)
        _bounded_int(done["completed_targets"], "done completed_targets",
                     1, _INT64_MAX)
    except (KeyError, AblateEvidenceError) as exc:
        raise AblateEvidenceError("terminal completion proof is invalid") from exc
    if done != {"t": "done", "manifest_sha256": proof["sha256"],
                "completed_items": completed_items,
                "completed_targets": completed_targets}:
        raise AblateEvidenceError("terminal completion proof is invalid")
    if cursor != len(records):
        raise AblateEvidenceError("records follow terminal completion proof")
    if (completed_items != proof["item_count"] or
            completed_targets != proof["targets"]):
        raise AblateEvidenceError("completed denominator does not match manifest")
    return {
        "manifest_sha256": proof["sha256"], "items": completed_items,
        "targets": completed_targets,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("evidence")
    parser.add_argument("--config", required=True,
                        help="independently supplied loaded-model config.json")
    args = parser.parse_args(argv)
    try:
        result = validate_ablate_evidence(
            args.manifest, args.evidence, args.config)
    except (OSError, AblateEvidenceError) as exc:
        print(f"[ablate-evidence] INCOMPLETE: {exc}", file=sys.stderr)
        return 1
    print(f"[ablate-evidence] PASS manifest={result['manifest_sha256']} "
          f"items={result['items']} targets={result['targets']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
