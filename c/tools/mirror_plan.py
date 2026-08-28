#!/usr/bin/env python3
"""Plan, stage, and verify a usage-ranked partial model mirror."""

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path


SCHEMA = "colibri.partial-mirror.v1"
RECEIPT = ".colibri-mirror.json"
GIB = 1024 ** 3
MAX_HEADER = 256 * 1024 * 1024
COPY_CHUNK = 16 * 1024 * 1024
GLM_EXPERT_KEY = re.compile(
    r"(?:^|\.)layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(?:\.qs)?$"
)
K3_EXPERT_KEY = re.compile(
    r"(?:language_model\.)?model\.layers\.(\d+)\.block_sparse_moe\."
    r"experts\.(\d+)\.(w1|w2|w3)\.weight_(packed|scale)$"
)
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


class MirrorError(RuntimeError):
    pass


def load_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MirrorError(f"Cannot read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise MirrorError(f"Expected a JSON object in {path}")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(COPY_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def usage_counts(path):
    counts = {}
    if not path.is_file():
        return counts
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as error:
        raise MirrorError(f"Cannot read usage history {path}: {error}") from error
    for line in lines:
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            layer, expert, count = map(int, fields)
        except ValueError:
            continue
        if layer >= 0 and expert >= 0 and count > 0:
            key = (layer, expert)
            counts[key] = counts.get(key, 0) + count
    return counts


def expert_tensor(name):
    match = GLM_EXPERT_KEY.search(name)
    if match is not None:
        return ((int(match.group(1)), int(match.group(2))),
                match.group(3) == "gate_proj" and name.endswith(".weight"))
    match = K3_EXPERT_KEY.fullmatch(name)
    if match is not None:
        return ((int(match.group(1)), int(match.group(2))),
                match.group(3) == "w1" and match.group(4) == "packed")
    return None


def shard_metadata(path):
    size = path.stat().st_size
    if size < 8:
        raise MirrorError(f"Safetensors shard is truncated: {path}")
    with path.open("rb") as stream:
        prefix = stream.read(8)
        header_size = struct.unpack("<Q", prefix)[0]
        if not 0 < header_size <= MAX_HEADER or header_size > size - 8:
            raise MirrorError(f"Invalid safetensors header length in {path}")
        try:
            header = json.loads(stream.read(header_size))
        except (UnicodeError, json.JSONDecodeError) as error:
            raise MirrorError(f"Invalid safetensors header in {path}: {error}") from error
    if not isinstance(header, dict):
        raise MirrorError(f"Safetensors header is not an object: {path}")

    expert_bytes = {}
    anchor_experts = set()
    for name, metadata in header.items():
        if name == "__metadata__" or not isinstance(name, str) or not isinstance(metadata, dict):
            continue
        tensor = expert_tensor(name)
        offsets = metadata.get("data_offsets")
        if tensor is None or not isinstance(offsets, list) or len(offsets) != 2:
            continue
        start, end = offsets
        if (isinstance(start, bool) or isinstance(end, bool) or
                not isinstance(start, int) or not isinstance(end, int) or
                start < 0 or end < start or end > size - 8 - header_size):
            raise MirrorError(f"Invalid tensor offsets for {name} in {path}")
        expert, is_anchor = tensor
        expert_bytes[expert] = expert_bytes.get(expert, 0) + end - start
        if is_anchor:
            anchor_experts.add(expert)
    return size, expert_bytes, anchor_experts


def discover_shards(model, source_dirs):
    directories = []
    for directory in (model, *source_dirs):
        resolved = Path(directory).expanduser().resolve()
        if resolved not in directories:
            directories.append(resolved)
    candidates = []
    seen = set()
    for directory in directories:
        if not directory.is_dir():
            raise MirrorError(f"Model shard directory does not exist: {directory}")
        for path in sorted(directory.glob("*.safetensors")):
            if path.name in seen or not path.is_file():
                continue
            seen.add(path.name)
            size, expert_bytes, anchor_experts = shard_metadata(path)
            candidates.append({
                "name": path.name,
                "size": size,
                "source": path,
                "expert_bytes": expert_bytes,
                "anchor_experts": anchor_experts,
            })
    if not candidates:
        raise MirrorError("No safetensors shards were found in the model source directories")
    return directories, candidates


def select_shards(candidates, counts, budget):
    # The engine decides an expert's replica from its first tensor (gate_proj for
    # GLM, packed w1 for K3), so score companions only after that anchor is present.
    # The byte-weighted usage score estimates I/O moved off the primary drive.
    selected = []
    activated = set()
    remaining = list(candidates)
    used = 0
    while True:
        choices = []
        for item in remaining:
            if used + item["size"] > budget:
                continue
            eligible = activated | item["anchor_experts"]
            benefit = sum(counts.get(expert, 0) * byte_count
                          for expert, byte_count in item["expert_bytes"].items()
                          if expert in eligible)
            if benefit:
                choices.append((item, benefit))
        if not choices:
            break
        item, benefit = min(
            choices,
            key=lambda choice: (
                -Fraction(choice[1], choice[0]["size"]),
                -choice[1],
                choice[0]["size"],
                choice[0]["name"],
            ),
        )
        newly_activated = item["anchor_experts"] - activated
        selected.append({**item, "weighted_expert_bytes": benefit,
                         "activated_experts": len(newly_activated)})
        activated.update(item["anchor_experts"])
        remaining.remove(item)
        used += item["size"]
    return selected


def existing_parent(path):
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def public_file(item):
    return {
        "name": item["name"],
        "size": item["size"],
        "weighted_expert_bytes": item["weighted_expert_bytes"],
        "activated_experts": item["activated_experts"],
    }


def create_plan(model, mirror, source_dirs, usage, budget, reserve):
    model = Path(model).expanduser().resolve()
    mirror = Path(mirror).expanduser().resolve()
    usage = Path(usage).expanduser().resolve()
    if model == mirror:
        raise MirrorError("Mirror directory must differ from the primary model directory")
    if budget <= 0 or reserve < 0:
        raise MirrorError("Mirror budget must be positive and reserve cannot be negative")
    directories, candidates = discover_shards(model, source_dirs)
    counts = usage_counts(usage)
    selected = select_shards(candidates, counts, budget)
    selected_bytes = sum(item["size"] for item in selected)
    existing_bytes = sum(item["size"] for item in selected
                         if not (mirror / item["name"]).is_symlink()
                         and (mirror / item["name"]).is_file()
                         and (mirror / item["name"]).stat().st_size == item["size"])
    remaining = selected_bytes - existing_bytes
    free = shutil.disk_usage(existing_parent(mirror)).free
    warnings = []
    mirror_device = existing_parent(mirror).stat().st_dev
    if any(directory.stat().st_dev == mirror_device for directory in directories):
        warnings.append("mirror path shares a filesystem device with a source directory")
    if not counts:
        reason = "usage_history_missing"
    elif not selected:
        reason = "no_usage_ranked_shards_fit_the_budget"
    elif free - remaining < reserve:
        reason = "free_space_reserve"
    else:
        reason = None
    payload = {
        "schema": SCHEMA,
        "model_root": str(model),
        "source_dirs": [str(path) for path in directories],
        "mirror_root": str(mirror),
        "usage_file": str(usage),
        "usage_history_present": bool(counts),
        "usage_selections": sum(counts.values()),
        "budget_bytes": budget,
        "reserve_bytes": reserve,
        "free_bytes": free,
        "selected_bytes": selected_bytes,
        "remaining_copy_bytes": remaining,
        "projected_free_bytes": free - remaining,
        "admitted": reason is None,
        "reason": reason,
        "warnings": warnings,
        "files": [public_file(item) for item in selected],
    }
    return payload, selected


def fsync_directory(path):
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in (errno.EBADF, errno.EINVAL, errno.ENOTSUP):
                raise
    finally:
        os.close(descriptor)


def atomic_json(path, payload):
    temporary = path.with_name(f".{path.name}.part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def copy_verified(source, target, expected_digest):
    temporary = target.with_name(f".{target.name}.part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            for chunk in iter(lambda: input_stream.read(COPY_CHUNK), b""):
                output_stream.write(chunk)
                digest.update(chunk)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if digest.hexdigest() != expected_digest or sha256(temporary) != expected_digest:
            raise MirrorError(f"Copied shard failed SHA-256 verification: {source.name}")
        os.replace(temporary, target)
        fsync_directory(target.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def stage_mirror(model, mirror, source_dirs, usage, budget, reserve):
    plan, selected = create_plan(model, mirror, source_dirs, usage, budget, reserve)
    if not selected or not plan["usage_history_present"]:
        return plan
    mirror = Path(mirror).expanduser().resolve()
    mirror.mkdir(parents=True, exist_ok=True)
    remaining = 0
    for item in selected:
        item["sha256"] = sha256(item["source"])
        target = mirror / item["name"]
        item["already_verified"] = (
            not target.is_symlink() and target.is_file()
            and target.stat().st_size == item["size"]
            and sha256(target) == item["sha256"]
        )
        if not item["already_verified"]:
            remaining += item["size"]
    free = shutil.disk_usage(mirror).free
    plan["free_bytes"] = free
    plan["remaining_copy_bytes"] = remaining
    plan["projected_free_bytes"] = free - remaining
    if free - remaining < reserve:
        plan["admitted"] = False
        plan["reason"] = "free_space_reserve"
        return plan
    plan["admitted"] = True
    plan["reason"] = None
    for item in selected:
        if not item["already_verified"]:
            copy_verified(item["source"], mirror / item["name"], item["sha256"])
    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_root": plan["model_root"],
        "usage_file": plan["usage_file"],
        "budget_bytes": budget,
        "reserve_bytes": reserve,
        "files": [{**public_file(item), "sha256": item["sha256"]} for item in selected],
    }
    atomic_json(mirror / RECEIPT, receipt)
    result = verify_mirror(mirror)
    result["plan"] = plan
    return result


def safe_receipt_name(value):
    return (isinstance(value, str) and value and "/" not in value and "\\" not in value
            and Path(value).name == value and value.endswith(".safetensors"))


def verify_mirror(mirror):
    mirror = Path(mirror).expanduser().resolve()
    receipt_path = mirror / RECEIPT
    if not receipt_path.is_file():
        return {"schema": SCHEMA, "ready": False, "mirror_root": str(mirror),
                "reason": "receipt_missing", "failures": []}
    receipt = load_json(receipt_path)
    failures = []
    files = receipt.get("files")
    if receipt.get("schema") != SCHEMA or not isinstance(files, list) or not files:
        return {"schema": SCHEMA, "ready": False, "mirror_root": str(mirror),
                "reason": "invalid_receipt", "failures": []}
    mirrored_bytes = 0
    names = set()
    for item in files:
        if not isinstance(item, dict) or not safe_receipt_name(item.get("name")):
            failures.append("invalid_receipt_entry")
            continue
        name = item["name"]
        if name in names:
            failures.append(name + " (duplicate)")
            continue
        names.add(name)
        size = item.get("size")
        digest = item.get("sha256")
        target = mirror / name
        if (isinstance(size, bool) or not isinstance(size, int) or size <= 0 or
                not isinstance(digest, str) or SHA256_HEX.fullmatch(digest) is None):
            failures.append(name + " (metadata)")
            continue
        mirrored_bytes += size
        if target.is_symlink() or not target.is_file() or target.stat().st_size != size:
            failures.append(name + " (size)")
        elif sha256(target) != digest:
            failures.append(name + " (sha256)")
    return {
        "schema": SCHEMA,
        "ready": not failures,
        "mirror_root": str(mirror),
        "file_count": len(files),
        "mirrored_bytes": mirrored_bytes,
        "failures": failures,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "stage", "verify"))
    parser.add_argument("--model", required=True, help="primary Colibri model directory")
    parser.add_argument("--mirror", required=True, help="partial mirror directory")
    parser.add_argument("--source-dir", action="append", default=[],
                        help="additional split-model shard directory (repeatable)")
    parser.add_argument("--usage", help="usage history (default: MODEL/.coli_usage)")
    parser.add_argument("--budget-gib", type=float, default=0,
                        help="maximum total mirror size; required for plan/stage")
    parser.add_argument("--reserve-gib", type=float, default=10,
                        help="free space to preserve on the mirror filesystem")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model = Path(args.model)
    mirror = Path(args.mirror)
    if args.action == "verify":
        payload = verify_mirror(mirror)
        print(json.dumps(payload, indent=2))
        return 0 if payload["ready"] else 4
    if (not math.isfinite(args.budget_gib) or args.budget_gib <= 0 or
            not math.isfinite(args.reserve_gib) or args.reserve_gib < 0):
        raise MirrorError("--budget-gib must be positive and --reserve-gib cannot be negative")
    usage = Path(args.usage) if args.usage else model / ".coli_usage"
    budget = int(args.budget_gib * GIB)
    reserve = int(args.reserve_gib * GIB)
    if args.action == "plan":
        payload, _selected = create_plan(model, mirror, args.source_dir, usage, budget, reserve)
    else:
        payload = stage_mirror(model, mirror, args.source_dir, usage, budget, reserve)
    print(json.dumps(payload, indent=2))
    ready = payload.get("ready", payload.get("admitted", False))
    return 0 if ready else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MirrorError, OSError) as error:
        print(json.dumps({"schema": SCHEMA, "ready": False, "error": str(error)}),
              file=sys.stderr)
        raise SystemExit(1)
