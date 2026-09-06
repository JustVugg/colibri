#!/usr/bin/env python3
"""Capture or validate one hash-bound native-MTP accepted-token witness.

The decisive future run is launched directly, never through a shell
pipeline. The capture command takes only absolute paths and supplies the
resolved model snapshot to the child through ``SNAP`` in the recorded
environment::

    python3 tests/check_native_mtp_witness.py capture \
      --binary /absolute/build/colibri \
      --snapshot /absolute/model \
      --container-manifest /absolute/model-payloads.sha256 \
      --input /absolute/submit.raw --run-dir /absolute/evidence/run-001 \
      --id 7 --topk 5 --vocab 154880

It resolves the snapshot once, then launches and binds that exact root
through ``SNAP``. The frozen twelve-key CPU environment includes
``KVSAVE=0``, ``USAGE_SAVE=0``, ``COLI_NO_OMP_TUNE=1``, and C locales
rather than inheriting ambient state. It captures stdout and stderr
separately, writes the direct child status, then freezes every artifact
and protocol payload in one content-addressed binding.

The container manifest is an exact sorted ``sha256sum`` inventory of every
regular snapshot file (``<lowercase-sha256><two spaces><relative-path><LF>``);
symlinks and undeclared or missing payloads fail closed before the child
runs. Exit zero requires the full compound witness -- a configuration
banner alone, or a bare acceptance marker alone, is never enough; both are
refused independently below.

``validate`` re-checks an already-frozen binding against the artifacts it
names. That replay proves the recorded bytes are internally consistent,
but it cannot itself prove the artifacts came from one common run -- only
a live ``capture`` can claim that, so ``validate`` always reports its
provenance as incomplete even when the replay is clean. Both subcommands
apply the same "exactly one accepted token" requirement to the underlying
evidence: a replay that binds zero or more than one accepted token is
refused exactly like a live capture would be.

Explicit-CUDA drafting and CUDA auto-off are separate future runs. The
decisive witness above retains explicit positive ``DRAFT``. The auto-off
run must omit ``DRAFT`` and may legitimately report the engine's inactive-
draft form; it cannot claim an accepted native-MTP witness and is not a
``capture`` success for this instrument -- the frozen environment always
binds an explicit positive ``DRAFT`` and refuses before the child ever
launches otherwise.

Exit status is one of four values, each with its own stderr prefix so the
caller can tell them apart without parsing the message body: ``0`` is a
live-capture ``PASS`` (stdout only, no stderr); ``1`` is every ordinary
``WitnessError``/``OSError`` (a malformed, incomplete, or inconsistent
witness, or an offline ``validate`` replay, which is always incomplete on
provenance even when clean) and prints ``[native-mtp] INCOMPLETE: ...``;
``2`` is reserved by ``argparse`` for command-line usage errors; ``3`` is
the distinct ``EngineWitnessUnsupported`` outcome below and prints
``[native-mtp] UNSUPPORTED: ...`` instead of the generic incomplete
prefix, because that case is not a malformed witness -- it is proof the
engine build cannot produce this witness at all.

Current limitation: the accepted-token proof below depends on the engine
printing an explicit per-emission stderr record (guarded by the
``MTP_DEBUG`` environment key this tool always sets) naming which decoded
token came from an accepted native-MTP draft. A build of the engine that
does not print that record can still show every other sign of an active,
proposing native-MTP session -- the loaded banner, generated tokens, HIT
proposals -- without ever proving that any generated token was itself the
accepted one. This tool tells that apart from an ordinary validation
failure with a distinct, named status rather than reporting either a bare
failure or a false success; whichever run recipe launches the engine
should confirm at build time that this record is compiled in before
trusting a bare "no witness" result to mean "no acceptance happened".

No model is run by the committed tests -- they drive this tool against an
injected stand-in for the direct child launch.
"""

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
from types import MappingProxyType


HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from engine_evidence import PreambleError, parse_engine_loaded


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GAPS = _load_module("native_mtp_gap_checker", HERE / "check_data_logprob_gaps.py")


SCHEMA = "colibri-b1-native-mtp-witness/2"
SNAPSHOT_SCHEMA = "colibri-snapshot-inventory/1"
CAPTURE_OUTCOME_SCHEMA = "colibri-b1-native-mtp-live-capture-outcome/1"
_INT32_MAX = 2**31 - 1
_MAX_PROMPT_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SUBMIT_RE = re.compile(
    rb"SUBMIT (?P<id>[1-9][0-9]*) 0 (?P<size>0|[1-9][0-9]*) "
    rb"(?P<maximum>0|[1-9][0-9]*) 0 1 0 logprobs=(?P<topk>[1-9][0-9]*)")
_STOP_RE = re.compile(
    r"^\[stop\] (?P<count>0|[1-9][0-9]*) stop tokens:"
    r"(?P<ids>(?: (?:0|[1-9][0-9]*))*)"
    r"(?: \((?P<special>0|[1-9][0-9]*) from the tokenizer's special set\))?$")
_MTPDBG_RE = re.compile(
    r"^\[mtpdbg\] draft0=(?P<draft>0|[1-9][0-9]*) "
    r"verified=(?P<verified>0|[1-9][0-9]*) (?P<result>HIT|miss)$")
_MTPEMIT_RE = re.compile(
    r"^\[mtpemit\] request=(?P<request>[1-9][0-9]*) "
    r"ordinal=(?P<ordinal>0|[1-9][0-9]*) "
    r"token=(?P<token>0|[1-9][0-9]*)$")
_SNAPSHOT_MANIFEST_RE = re.compile(
    rb"(?P<digest>[0-9a-f]{64})  (?P<path>[\x21-\x7e]+)")


class WitnessError(ValueError):
    """The bundle cannot prove the complete native-MTP witness."""


class EngineWitnessUnsupported(WitnessError):
    """The captured engine build never printed the accepted-token record.

    Distinct from every other WitnessError: this is not a malformed or
    incomplete witness, it is a witness the engine build cannot produce at
    all because the per-emission stderr record this tool binds to
    (guarded by ``MTP_DEBUG`` in the frozen environment) was never
    printed, even though the run otherwise shows an active, proposing
    native-MTP session.
    """


@dataclass(frozen=True)
class _DirectCaptureRun:
    root: pathlib.Path
    binary: pathlib.Path
    container: pathlib.Path
    snapshot: pathlib.Path
    environment: pathlib.Path
    input: pathlib.Path
    status: pathlib.Path
    stdout: pathlib.Path
    stderr: pathlib.Path
    request_id: int
    topk: int
    vocab: int

    @property
    def artifacts(self):
        return MappingProxyType({
            "binary": self.binary,
            "container": self.container,
            "environment": self.environment,
            "input": self.input,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
        })


def _reject_constant(value):
    raise WitnessError(f"non-JSON constant: {value}")


def _canonical_json(value):
    return (json.dumps(value, ensure_ascii=True, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("ascii")


def _strict_json_bytes(raw, label, require_canonical=True):
    try:
        value = json.loads(raw.decode("ascii"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, WitnessError) as exc:
        raise WitnessError(f"invalid {label} JSON: {exc}") from exc
    if require_canonical and raw != _canonical_json(value):
        raise WitnessError(f"noncanonical {label} JSON bytes")
    return value


def _sha256_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def _absolute_file(path, label):
    candidate = pathlib.Path(path)
    if not candidate.is_absolute() or str(candidate) != os.path.normpath(str(candidate)):
        raise WitnessError(f"{label} path must be canonical absolute: {path!r}")
    if not candidate.is_file():
        raise WitnessError(f"{label} is not a file: {path!r}")
    return candidate


def _read_artifact(path, label):
    """Read one regular artifact once and derive metadata from those bytes."""
    path = _absolute_file(path, label).resolve()
    raw = path.read_bytes()
    return (_artifact_from_bytes(path, raw), raw)


def _snapshot_stat(st_result):
    """Metadata that changes on replacement or mutate-then-restore."""
    return (
        st_result.st_dev, st_result.st_ino, stat.S_IFMT(st_result.st_mode),
        st_result.st_size, st_result.st_mtime_ns, st_result.st_ctime_ns,
    )


def _parse_snapshot_manifest(raw):
    if not raw or not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise WitnessError(
            "snapshot manifest must be nonempty canonical LF records")
    entries = []
    previous = None
    for raw_line in raw.splitlines():
        match = _SNAPSHOT_MANIFEST_RE.fullmatch(raw_line)
        if not match:
            raise WitnessError(f"malformed snapshot manifest record: {raw_line!r}")
        path_raw = match.group("path")
        try:
            path_text = path_raw.decode("ascii")
        except UnicodeDecodeError as exc:
            raise WitnessError("snapshot manifest path is not ASCII") from exc
        pure = pathlib.PurePosixPath(path_text)
        if (not pure.parts or path_text in (".", "..") or
                pure.is_absolute() or path_text != pure.as_posix() or
                path_text.startswith("/") or "\\" in path_text or
                re.match(r"^[A-Za-z]:", path_text) or
                any(part in ("", ".", "..") for part in pure.parts) or
                "//" in path_text or any(ord(char) < 0x21 for char in path_text)):
            raise WitnessError(
                f"snapshot manifest path is not canonical relative: {path_text!r}")
        if previous is not None and path_text <= previous:
            raise WitnessError("snapshot manifest paths are duplicate or unsorted")
        previous = path_text
        entries.append((path_text, match.group("digest").decode("ascii")))
    if not entries:
        raise WitnessError("snapshot manifest has no payload denominator")
    return tuple(entries)


def _walk_snapshot(root):
    """Return every regular payload, byte-sorted by relative path (the same
    order a sorted ``sha256sum`` manifest uses), and a no-follow stability
    fingerprint. The payload order and the manifest's declared order must
    agree independent of directory-tree shape -- a filesystem walk that is
    merely sorted per-directory (e.g. "model" before "model.json", because
    the walk compares bare entry names) does not agree with a manifest
    sorted by full relative path (where "model.json" < "model/inner.bin"
    because "." sorts before "/"), so the payload list is explicitly
    re-sorted by its own full path text before being handed back.
    """
    root = pathlib.Path(root)
    try:
        root_stat = os.lstat(root)
    except FileNotFoundError as exc:
        raise WitnessError("snapshot directory is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise WitnessError("snapshot root must be a real directory, not a symlink")
    payloads = []
    fingerprint = [(".", _snapshot_stat(root_stat))]

    def visit(directory, prefix):
        try:
            with os.scandir(directory) as scan:
                entries = sorted(scan, key=lambda entry: entry.name)
        except OSError as exc:
            raise WitnessError(f"cannot inventory snapshot directory {directory}") from exc
        for entry in entries:
            if "/" in entry.name or entry.name in (".", ".."):
                raise WitnessError("snapshot contains a noncanonical entry name")
            rel = entry.name if not prefix else f"{prefix}/{entry.name}"
            try:
                item_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise WitnessError(f"cannot stat snapshot entry {rel!r}") from exc
            fingerprint.append((rel, _snapshot_stat(item_stat)))
            if stat.S_ISLNK(item_stat.st_mode):
                raise WitnessError(f"snapshot symlink is forbidden: {rel!r}")
            if stat.S_ISDIR(item_stat.st_mode):
                visit(pathlib.Path(directory) / entry.name, rel)
            elif stat.S_ISREG(item_stat.st_mode):
                payloads.append(rel)
            else:
                raise WitnessError(f"snapshot entry is not regular: {rel!r}")

    visit(root, "")
    return tuple(sorted(payloads)), tuple(fingerprint)


def _hash_snapshot_payload(root, relative):
    if sys.platform == "win32":
        # The identity check below requires POSIX stat semantics: fstat of the
        # open descriptor must agree with lstat of the path on device and
        # inode. Windows derives those differently for a handle and a path
        # (CI observed disagreement on a subset of files), so the witness
        # refuses rather than report a spurious "payload changed".
        raise WitnessError(
            "snapshot payload identity requires POSIX stat semantics "
            "(fstat/lstat agreement on device and inode); Windows is unsupported")
    path = pathlib.Path(root).joinpath(*relative.split("/"))
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WitnessError(f"cannot open snapshot payload {relative!r}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise WitnessError(f"snapshot payload is not regular: {relative!r}")
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            size += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        linked = os.lstat(path)
    except FileNotFoundError as exc:
        raise WitnessError(f"snapshot payload vanished: {relative!r}") from exc
    if (_snapshot_stat(before) != _snapshot_stat(after) or
            _snapshot_stat(after) != _snapshot_stat(linked)):
        raise WitnessError(f"snapshot payload changed while hashing: {relative!r}")
    return digest.hexdigest(), size


def _apply_snapshot_manifest(snapshot, manifest_raw):
    """Apply exact manifest bytes to the complete no-follow snapshot inventory."""
    root = pathlib.Path(snapshot)
    entries = _parse_snapshot_manifest(manifest_raw)
    declared = tuple(path for path, _ in entries)
    actual, before = _walk_snapshot(root)
    if actual != declared:
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        raise WitnessError(
            f"snapshot inventory mismatch; missing={missing!r} extra={extra!r}")
    total = 0
    for relative, expected in entries:
        observed, size = _hash_snapshot_payload(root, relative)
        if observed != expected:
            raise WitnessError(f"snapshot payload SHA-256 mismatch: {relative!r}")
        total += size
    actual_after, after = _walk_snapshot(root)
    if actual_after != actual or after != before:
        raise WitnessError("snapshot inventory changed while applying manifest")
    inventory_digest = _sha256_bytes(
        (SNAPSHOT_SCHEMA + "\n").encode("ascii") + manifest_raw)
    summary = {
        "schema": SNAPSHOT_SCHEMA, "files": len(entries),
        "bytes": total, "sha256": inventory_digest,
    }
    return summary, after


def _artifact_from_bytes(path, raw):
    path = pathlib.Path(path).resolve()
    return {
        "path": str(path), "size": len(raw), "sha256": _sha256_bytes(raw),
    }


def _materialize_capture_artifact(path, root, label, raw):
    """Materialize immutable capture bytes without accepting a supplied path."""
    stream = _open_capture_file(path, label)
    try:
        stream.write(raw)
        stream.flush()
        opened = os.fstat(stream.fileno())
        try:
            entry = os.lstat(path)
        except FileNotFoundError as exc:
            raise WitnessError(f"{label} capture path was removed") from exc
        if (not stat.S_ISREG(entry.st_mode) or
                (entry.st_dev, entry.st_ino) != (opened.st_dev, opened.st_ino)):
            raise WitnessError(f"{label} capture path was replaced")
        resolved = pathlib.Path(path).resolve(strict=True)
        if resolved.parent != pathlib.Path(root).resolve(strict=True):
            raise WitnessError(f"{label} capture path escaped the run root")
        if opened.st_size != len(raw):
            raise WitnessError(f"{label} capture size changed while materializing")
        return (_artifact_from_bytes(resolved, raw), raw)
    finally:
        stream.close()


def _freeze_anonymous_stream(stream):
    """Read immutable bytes from an anonymous descriptor after child exit."""
    stream.flush()
    stream.seek(0)
    return stream.read()


def _open_capture_file(path, label):
    """Create one capture-owned regular file without following an entry."""
    try:
        return open(path, "x+b")
    except FileExistsError as exc:
        raise WitnessError(f"{label} capture path was precreated") from exc


def _payload_records(stdout_blob):
    frames, _framing = GAPS.parse_frames(stdout_blob)
    records = []
    for fields, payload, _offset in frames:
        if fields[0] not in (b"DATA", b"ECHO") or payload is None:
            continue
        try:
            request_id = int(fields[1].decode("ascii"))
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise WitnessError(f"cannot bind malformed payload frame: {fields!r}") from exc
        records.append({
            "kind": fields[0].decode("ascii"), "id": request_id,
            "size": len(payload), "sha256": _sha256_bytes(payload),
        })
    return records


def _binding_id(record):
    body = dict(record)
    body.pop("binding_id", None)
    return _sha256_bytes(_canonical_json(body))


def _build_capture_binding(run, external_before, owned_capture,
                           snapshot_inventory):
    """Freeze the one direct-capture run; no post-hoc path accepts artifacts."""
    run_artifacts = run.artifacts
    owned_names = {"environment", "input", "status", "stdout", "stderr"}
    if set(owned_capture) != owned_names:
        raise WitnessError("owned capture denominator is not exact")
    for name in owned_names:
        if run_artifacts[name].parent != run.root:
            raise WitnessError(f"{name} is outside the direct-capture run root")
    artifacts = {name: owned_capture[name][0] for name in owned_names}
    blobs = {name: owned_capture[name][1] for name in owned_names}
    for name in ("binary", "container"):
        path = run_artifacts[name]
        item, raw = _read_artifact(path, name)
        if name == "container" and not raw:
            raise WitnessError("container artifact must be a nonempty regular file")
        if item != external_before[name]:
            raise WitnessError(f"{name} changed during direct capture")
        artifacts[name] = item
        blobs[name] = raw
    record = {
        "schema": SCHEMA,
        "snapshot": str(run.snapshot),
        "request": {
            "id": run.request_id, "topk": run.topk, "vocab": run.vocab,
        },
        "snapshot_inventory": snapshot_inventory,
        "artifacts": {name: artifacts[name] for name in sorted(artifacts)},
        "payloads": _payload_records(blobs["stdout"]),
    }
    record["binding_id"] = _binding_id(record)
    return record


def _write_binding(path, record):
    output = pathlib.Path(path)
    raw = _canonical_json(record)
    with open(output, "xb") as stream:
        if stream.write(raw) != len(raw):
            raise WitnessError("short write while freezing binding")
        stream.flush()
        os.fsync(stream.fileno())


def _parse_binding(raw):
    """Strictly parse one already-frozen binding byte image."""
    record = _strict_json_bytes(raw, "binding")
    if set(record) != {
            "schema", "binding_id", "snapshot", "request", "artifacts",
            "payloads", "snapshot_inventory"}:
        raise WitnessError("binding keys are not exact")
    if record["schema"] != SCHEMA:
        raise WitnessError(f"unknown binding schema: {record['schema']!r}")
    if (not isinstance(record["binding_id"], str) or
            not _SHA256_RE.fullmatch(record["binding_id"])):
        raise WitnessError("binding_id is not canonical SHA-256")
    if record["binding_id"] != _binding_id(record):
        raise WitnessError("binding_id does not bind this record")
    return record


def _load_binding(path):
    raw = pathlib.Path(path).read_bytes()
    return _parse_binding(raw)


def _validate_artifacts(record):
    artifacts = record["artifacts"]
    expected_names = {
        "binary", "container", "environment", "input", "status", "stdout",
        "stderr",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != expected_names:
        raise WitnessError("artifact denominator is not exact")
    paths = []
    blobs = {}
    for name in sorted(expected_names):
        item = artifacts[name]
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise WitnessError(f"{name} artifact keys are not exact")
        if (not isinstance(item["path"], str) or
                type(item["size"]) is not int or item["size"] < 0 or
                not isinstance(item["sha256"], str) or
                not _SHA256_RE.fullmatch(item["sha256"])):
            raise WitnessError(f"{name} artifact metadata is malformed")
        actual, raw = _read_artifact(item["path"], name)
        if actual["path"] != item["path"]:
            raise WitnessError(f"{name} artifact path is not resolved canonical")
        paths.append(actual["path"])
        if actual["size"] != item["size"]:
            raise WitnessError(f"{name} artifact size mismatch")
        if actual["sha256"] != item["sha256"]:
            raise WitnessError(f"{name} artifact SHA-256 mismatch")
        if name == "container" and not raw:
            raise WitnessError("container artifact must be a nonempty regular file")
        blobs[name] = raw
    if len(set(paths)) != len(paths):
        raise WitnessError("artifact paths must be distinct")
    return MappingProxyType(blobs)


def _validate_request_meta(record):
    request = record["request"]
    if not isinstance(request, dict) or set(request) != {"id", "topk", "vocab"}:
        raise WitnessError("request binding keys are not exact")
    request_id, topk, vocab = request["id"], request["topk"], request["vocab"]
    if (type(request_id) is not int or not 1 <= request_id <= 2**64 - 1 or
            type(topk) is not int or not 1 <= topk <= 32 or
            type(vocab) is not int or not 1 <= vocab <= _INT32_MAX):
        raise WitnessError("request binding values are outside their domains")
    return request_id, topk, vocab


def _validate_environment(raw, snapshot):
    environment = _strict_json_bytes(raw, "environment")
    if (not isinstance(environment, dict) or
            any(not isinstance(k, str) or not isinstance(v, str)
                for k, v in environment.items())):
        raise WitnessError("environment must be an exact string map")
    required = {
        "SNAP": snapshot, "SERVE": "1", "SERVE_BATCH": "1",
        "KV_SLOTS": "1", "MTP_DEBUG": "1", "KVSAVE": "0",
        "USAGE_SAVE": "0", "COLI_NO_OMP_TUNE": "1",
        "LANG": "C", "LC_ALL": "C",
    }
    for key, expected in required.items():
        if environment.get(key) != expected:
            raise WitnessError(f"environment {key} must be {expected!r}")
    draft = environment.get("DRAFT")
    if (draft is None or not re.fullmatch(r"(?:[1-9]|[1-5][0-9]|6[0-3])", draft)):
        raise WitnessError("environment DRAFT must be explicit positive 1..63")
    ctx = environment.get("CTX")
    if (ctx is None or not re.fullmatch(r"(?:0|[1-9][0-9]*)", ctx) or
            not 1 <= int(ctx) <= _INT32_MAX):
        raise WitnessError("environment CTX must be canonical positive int32")
    if set(environment) != set(required) | {"DRAFT", "CTX"}:
        raise WitnessError("environment keys must equal the exact CPU witness map")
    return environment


def _validate_input(blob, request_id, topk):
    newline = blob.find(b"\n")
    if newline < 0:
        raise WitnessError("input lacks a complete SUBMIT header")
    match = _SUBMIT_RE.fullmatch(blob[:newline])
    if not match:
        raise WitnessError("input must contain one canonical greedy grammar-free SUBMIT")
    values = {name: int(match.group(name))
              for name in ("id", "size", "maximum", "topk")}
    if values["id"] != request_id or values["topk"] != topk:
        raise WitnessError("input SUBMIT does not match bound request")
    if not 1 <= values["size"] <= _MAX_PROMPT_BYTES:
        raise WitnessError("input prompt payload must be 1..16 MiB")
    if not 1 <= values["maximum"] <= _INT32_MAX:
        raise WitnessError("input maximum must be a positive int32")
    payload_start = newline + 1
    payload_end = payload_start + values["size"]
    if (payload_end >= len(blob) or blob[payload_end:payload_end + 1] != b"\n" or
            payload_end + 1 != len(blob)):
        raise WitnessError("input must end after exactly one framed SUBMIT payload")
    payload = blob[payload_start:payload_end]
    if b"\0" in payload:
        raise WitnessError("input prompt payload contains NUL")
    return values, payload


def _validate_status(raw):
    if raw != b"0\n":
        raise WitnessError(f"engine status is not exact zero: {raw!r}")


def _validate_stdout(blob, request_id, topk, vocab, expected_payloads):
    frames, framing = GAPS.parse_frames(blob)
    if GAPS.capture_mode(frames) != "full-process":
        raise WitnessError("stdout is not a complete-process capture")
    data, echo, problems, _mode, _forms = GAPS.check(
        frames, framing, str(request_id), topk, vocab)
    if problems:
        raise WitnessError("stdout gap check failed: " + "; ".join(problems))
    # data/echo are both guaranteed >= 1 here: GAPS.check() itself
    # reports "no DATA frames found" whenever data == 0, and the same
    # ACCEPT-prompt/ECHO-position accounting forces echo >= 1 whenever
    # ACCEPT is present at all -- a locally repeated ">= 1" guard here
    # would be unreachable dead code given that guarantee.
    for fields, _payload, _offset in frames:
        if fields[0] in GAPS._GLOBAL_KINDS:
            continue
        if fields[0] not in (b"ACCEPT", b"ECHO", b"DATA", b"DONE"):
            raise WitnessError(f"stdout contains an unowned request record: {fields!r}")
        try:
            frame_id = int(fields[1].decode("ascii"))
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise WitnessError(f"stdout request id is malformed: {fields!r}") from exc
        if frame_id != request_id:
            raise WitnessError(f"stdout contains second request id {frame_id}")

    lines = blob.split(b"\n", 2)
    if len(lines) != 3:
        raise WitnessError("stdout lacks complete banner/load prefix")
    try:
        loaded = parse_engine_loaded(lines[1].decode("ascii"))
    except (UnicodeDecodeError, PreambleError) as exc:
        raise WitnessError(f"stdout load record is invalid: {exc}") from exc
    if loaded["mtp"] != "ACTIVE" or not 1 <= loaded["draft"] <= 63:
        raise WitnessError("stdout does not prove active positive-depth native MTP")
    actual_payloads = _payload_records(blob)
    if actual_payloads != expected_payloads:
        raise WitnessError("bound protocol payload hashes do not match stdout")
    data_rows = []
    for fields, _payload, _offset in frames:
        if fields[0] != b"DATA":
            continue
        try:
            row_topk = int(fields[4].decode("ascii"))
            pairs = {
                int(fields[5 + 2 * index].decode("ascii")):
                fields[6 + 2 * index]
                for index in range(row_topk)
            }
        except (IndexError, UnicodeDecodeError, ValueError) as exc:
            raise WitnessError(f"cannot bind malformed DATA row: {fields!r}") from exc
        if len(pairs) != row_topk:
            raise WitnessError("DATA top-k token identities are not unique")
        data_rows.append({"target": fields[3], "topk": pairs})
    if len(data_rows) != data:
        raise WitnessError("DATA denominator changed while binding rows")
    return loaded, data_rows


def _validate_stderr(raw, vocab, request_id, data_rows):
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise WitnessError("stderr must be nonempty canonical newline records")
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise WitnessError("stderr is not UTF-8") from exc
    if any(line.startswith("[GRAMMAR]") or line.startswith("[CORPUS]")
           for line in lines):
        raise WitnessError("stderr reports an alternate grammar/corpus draft source")

    stop_matches = []
    stop_index = None
    for index, line in enumerate(lines):
        match = _STOP_RE.fullmatch(line)
        if match:
            stop_matches.append(match)
            stop_index = index
        elif line.startswith("[stop] ") and " stop tokens:" in line:
            raise WitnessError(f"malformed armed-stop record: {line!r}")
    if len(stop_matches) != 1:
        raise WitnessError(f"expected one armed-stop record, found {len(stop_matches)}")
    match = stop_matches[0]
    count = int(match.group("count"))
    ids_text = match.group("ids").strip()
    stop_ids = [] if not ids_text else [int(value) for value in ids_text.split(" ")]
    if count != len(stop_ids) or count > 64 or len(set(stop_ids)) != len(stop_ids):
        raise WitnessError("armed-stop count/uniqueness mismatch")
    if any(token < 0 or token >= vocab for token in stop_ids):
        raise WitnessError("armed-stop token is outside the vocabulary")
    special = match.group("special")
    if special is not None and not 1 <= int(special) <= count:
        raise WitnessError("armed-stop special-token count is invalid")

    qualifying = []
    pending = []
    bound_ordinals = set()
    markers = 0
    mtpemit_records = 0
    for index, line in enumerate(lines):
        if line.startswith("[mtpemit]"):
            mtpemit_records += 1
            bound = _MTPEMIT_RE.fullmatch(line)
            if not bound:
                raise WitnessError(f"malformed mtpemit record: {line!r}")
            if index <= stop_index:
                raise WitnessError("mtpemit record precedes the same-run stop set")
            bound_request = int(bound.group("request"))
            ordinal = int(bound.group("ordinal"))
            token = int(bound.group("token"))
            if bound_request != request_id:
                raise WitnessError("mtpemit request does not match the bound request")
            if token >= vocab:
                raise WitnessError("mtpemit token is outside the vocabulary")
            if not pending:
                raise WitnessError("mtpemit has no preceding qualifying HIT")
            if token != pending.pop(0):
                raise WitnessError("mtpemit token does not match its qualifying HIT")
            if ordinal in bound_ordinals:
                raise WitnessError("duplicate/replayed mtpemit emission ordinal")
            if bound_ordinals and ordinal <= max(bound_ordinals):
                raise WitnessError("mtpemit emission ordinals are not strictly ordered")
            if ordinal >= len(data_rows):
                raise WitnessError("mtpemit emission ordinal has no DATA row")
            row = data_rows[ordinal]
            if token not in row["topk"]:
                raise WitnessError("mtpemit token is absent from its DATA top-k")
            if row["topk"][token] != row["target"]:
                raise WitnessError("mtpemit token does not own its DATA target logprob")
            bound_ordinals.add(ordinal)
            continue
        if not line.startswith("[mtpdbg]"):
            continue
        marker = _MTPDBG_RE.fullmatch(line)
        if not marker:
            raise WitnessError(f"malformed mtpdbg record: {line!r}")
        markers += 1
        if index <= stop_index:
            raise WitnessError("mtpdbg proposal precedes the same-run stop set")
        draft = int(marker.group("draft"))
        verified = int(marker.group("verified"))
        result = marker.group("result")
        if draft >= vocab or verified >= vocab:
            raise WitnessError("mtpdbg token is outside the vocabulary")
        expected = "HIT" if draft == verified else "miss"
        if result != expected:
            raise WitnessError("mtpdbg HIT/miss label contradicts token equality")
        if result == "HIT" and draft not in stop_ids:
            qualifying.append(draft)
            pending.append(draft)
    if markers < 1:
        raise WitnessError("stderr contains no mtpdbg proposal")
    if not qualifying:
        raise WitnessError("stderr has no equal non-stop native-MTP HIT")
    if mtpemit_records < 1:
        raise EngineWitnessUnsupported(
            "engine does not emit the accepted-token witness line "
            "(requires MTP_DEBUG support)")
    if pending:
        raise WitnessError("qualifying HIT is missing its mtpemit/DATA binding")
    if len(bound_ordinals) != len(qualifying):
        raise WitnessError("qualifying HIT/mtpemit denominator mismatch")
    return qualifying


def _validate_components(record, blobs):
    """Apply the one semantic validator to one record and immutable byte map."""
    request_id, topk, vocab = _validate_request_meta(record)
    snapshot = record["snapshot"]
    if (not isinstance(snapshot, str) or not pathlib.Path(snapshot).is_absolute() or
            snapshot != os.path.normpath(snapshot)):
        raise WitnessError("bound snapshot path is not canonical absolute")
    if not pathlib.Path(snapshot).is_dir():
        raise WitnessError("bound snapshot directory is unavailable")
    inventory, _ = _apply_snapshot_manifest(snapshot, blobs["container"])
    if record["snapshot_inventory"] != inventory:
        raise WitnessError("bound snapshot inventory does not match applied manifest")
    environment = _validate_environment(blobs["environment"], snapshot)
    input_meta, _ = _validate_input(blobs["input"], request_id, topk)
    _validate_status(blobs["status"])
    loaded, data_rows = _validate_stdout(
        blobs["stdout"], request_id, topk, vocab, record["payloads"])
    qualifying = _validate_stderr(
        blobs["stderr"], vocab, request_id, data_rows)
    if int(environment["DRAFT"]) != loaded["draft"]:
        raise WitnessError("configured DRAFT does not equal loaded effective draft")
    if len(data_rows) > input_meta["maximum"]:
        raise WitnessError("emitted DATA count exceeds bound input maximum")
    return {
        "binding_id": record["binding_id"], "request_id": request_id,
        "accepted_tokens": qualifying,
    }


def validate_binding(path):
    """Replay a frozen bundle without claiming common-process provenance."""
    record = _load_binding(path)
    blobs = _validate_artifacts(record)
    result = _validate_components(record, blobs)
    if len(result["accepted_tokens"]) != 1:
        raise WitnessError("replay does not bind exactly one accepted token")
    result.update({
        "replay_verdict": "COMPLETE",
        "provenance_verdict": "INCOMPLETE",
    })
    return result


def _write_capture_outcome(path, outcome):
    """Exclusively freeze and byte-revalidate one live capture outcome."""
    expected = {
        "schema", "verdict", "binding_id", "binding_sha256", "request_id",
        "accepted_tokens", "run_root",
    }
    if (not isinstance(outcome, dict) or set(outcome) != expected or
            outcome["schema"] != CAPTURE_OUTCOME_SCHEMA or
            outcome["verdict"] != "PASS" or
            not isinstance(outcome["binding_id"], str) or
            not _SHA256_RE.fullmatch(outcome["binding_id"]) or
            not isinstance(outcome["binding_sha256"], str) or
            not _SHA256_RE.fullmatch(outcome["binding_sha256"]) or
            type(outcome["request_id"]) is not int or
            not 1 <= outcome["request_id"] <= 2**64 - 1 or
            not isinstance(outcome["accepted_tokens"], list) or
            len(outcome["accepted_tokens"]) != 1 or
            type(outcome["accepted_tokens"][0]) is not int or
            not isinstance(outcome["run_root"], str) or
            not pathlib.Path(outcome["run_root"]).is_absolute() or
            outcome["run_root"] != os.path.normpath(outcome["run_root"])):
        raise WitnessError("live capture outcome is malformed")
    raw = _canonical_json(outcome)
    output = pathlib.Path(path)
    with open(output, "xb") as stream:
        if stream.write(raw) != len(raw):
            raise WitnessError("short write while freezing live capture outcome")
        stream.flush()
        os.fsync(stream.fileno())
    frozen = output.read_bytes()
    if frozen != raw or _strict_json_bytes(
            frozen, "live capture outcome") != outcome:
        raise WitnessError("live capture outcome failed byte revalidation")
    return _sha256_bytes(frozen)


def _capture_environment(snapshot, draft, ctx):
    return {
        "SNAP": snapshot, "SERVE": "1", "SERVE_BATCH": "1",
        "KV_SLOTS": "1", "DRAFT": str(draft), "CTX": str(ctx),
        "MTP_DEBUG": "1", "KVSAVE": "0", "USAGE_SAVE": "0",
        "COLI_NO_OMP_TUNE": "1", "LANG": "C", "LC_ALL": "C",
    }


def capture_bundle(binary, snapshot, container, input_path, run_dir,
                   request_id, topk, vocab, draft=1, ctx=4096,
                   run_fn=subprocess.run):
    binary_path = _absolute_file(binary, "binary").resolve()
    container_path = _absolute_file(container, "container").resolve()
    input_source = _absolute_file(input_path, "input").resolve()
    snapshot_path = pathlib.Path(snapshot)
    output = pathlib.Path(run_dir)
    if (not snapshot_path.is_absolute() or
            str(snapshot_path) != os.path.normpath(str(snapshot_path)) or
            not snapshot_path.is_dir()):
        raise WitnessError("snapshot must be an existing canonical absolute directory")
    try:
        snapshot_entry = os.lstat(snapshot_path)
    except OSError as exc:
        raise WitnessError("snapshot directory is unavailable") from exc
    if stat.S_ISLNK(snapshot_entry.st_mode):
        raise WitnessError("snapshot root symlink is forbidden")
    if (not output.is_absolute() or
            str(output) != os.path.normpath(str(output)) or output.exists()):
        raise WitnessError("run directory must be a new canonical absolute path")
    if (type(draft) is not int or not 1 <= draft <= 63 or
            type(ctx) is not int or not 1 <= ctx <= _INT32_MAX):
        raise WitnessError("capture draft/CTX is outside the decisive domain")
    if (type(request_id) is not int or not 1 <= request_id <= 2**64 - 1 or
            type(topk) is not int or not 1 <= topk <= 32 or
            type(vocab) is not int or not 1 <= vocab <= _INT32_MAX):
        raise WitnessError("capture request metadata is outside its domain")
    binary_before, binary_raw = _read_artifact(binary_path, "binary")
    container_before, container_raw = _read_artifact(
        container_path, "container")
    resolved_snapshot = snapshot_path.resolve(strict=True)
    if (container_path == resolved_snapshot or
            resolved_snapshot in container_path.parents):
        raise WitnessError("snapshot manifest must be outside the snapshot")
    try:
        resolved_output_parent = output.parent.resolve(strict=True)
    except OSError as exc:
        raise WitnessError("run directory parent is unavailable") from exc
    if (resolved_output_parent == resolved_snapshot or
            resolved_snapshot in resolved_output_parent.parents):
        raise WitnessError("run directory must be outside the snapshot")
    snapshot_before, stability_before = _apply_snapshot_manifest(
        resolved_snapshot, container_raw)
    _, input_raw = _read_artifact(input_source, "input")
    _validate_input(input_raw, request_id, topk)
    output.mkdir(mode=0o700)
    output = output.resolve(strict=True)

    paths = {
        "environment": output / "environment.json",
        "input": output / "request.raw",
        "status": output / "engine_status.txt",
        "stdout": output / "engine_stdout.raw",
        "stderr": output / "engine_stderr.raw",
    }
    environment = _capture_environment(str(resolved_snapshot), draft, ctx)
    environment_raw = _canonical_json(environment)
    with tempfile.TemporaryFile(mode="w+b") as stdin, \
            tempfile.TemporaryFile(mode="w+b") as stdout, \
            tempfile.TemporaryFile(mode="w+b") as stderr:
        stdin.write(input_raw)
        stdin.flush()
        stdin.seek(0)
        result = run_fn(
            [str(binary_path)], stdin=stdin, stdout=stdout, stderr=stderr,
            env=environment, cwd=str(output), check=False, shell=False)
        stdout_raw = _freeze_anonymous_stream(stdout)
        stderr_raw = _freeze_anonymous_stream(stderr)
    snapshot_after_child, stability_after_child = _apply_snapshot_manifest(
        resolved_snapshot, container_raw)
    if (snapshot_after_child != snapshot_before or
            stability_after_child != stability_before):
        raise WitnessError("snapshot changed during direct child execution")
    captured_raw = {
        "environment": environment_raw, "input": input_raw,
        "status": f"{result.returncode}\n".encode("ascii"),
        "stdout": stdout_raw, "stderr": stderr_raw,
    }
    owned_capture = {
        name: _materialize_capture_artifact(
            paths[name], output, name, captured_raw[name])
        for name in ("environment", "input", "status", "stdout", "stderr")
    }
    snapshot_after_capture, stability_after_capture = _apply_snapshot_manifest(
        resolved_snapshot, container_raw)
    if (snapshot_after_capture != snapshot_before or
            stability_after_capture != stability_before):
        raise WitnessError("snapshot changed while materializing capture")

    run = _DirectCaptureRun(
        root=output, binary=binary_path, container=container_path,
        snapshot=resolved_snapshot, environment=paths["environment"],
        input=paths["input"], status=paths["status"],
        stdout=paths["stdout"], stderr=paths["stderr"],
        request_id=request_id, topk=topk, vocab=vocab)
    record = _build_capture_binding(
        run, MappingProxyType({
            "binary": binary_before, "container": container_before,
        }), MappingProxyType(owned_capture), snapshot_before)
    snapshot_before_binding, stability_before_binding = _apply_snapshot_manifest(
        resolved_snapshot, container_raw)
    if (snapshot_before_binding != snapshot_before or
            stability_before_binding != stability_before):
        raise WitnessError("snapshot changed before binding issuance")
    binding_path = output / "binding.json"
    component_blobs = MappingProxyType({
        "binary": binary_raw, "container": container_raw,
        **captured_raw,
    })
    _write_binding(binding_path, record)
    expected_binding_raw = _canonical_json(record)
    try:
        binding_raw = binding_path.read_bytes()
    except OSError as exc:
        raise WitnessError("frozen binding is unavailable") from exc
    if binding_raw != expected_binding_raw:
        raise WitnessError("frozen binding failed exact byte revalidation")
    frozen_record = _parse_binding(binding_raw)
    if frozen_record != record:
        raise WitnessError("frozen binding does not equal the issued record")
    live_result = _validate_components(frozen_record, component_blobs)
    if len(live_result["accepted_tokens"]) != 1:
        raise WitnessError("live capture requires exactly one accepted token")
    outcome = {
        "schema": CAPTURE_OUTCOME_SCHEMA,
        "verdict": "PASS",
        "binding_id": frozen_record["binding_id"],
        "binding_sha256": _sha256_bytes(binding_raw),
        "request_id": live_result["request_id"],
        "accepted_tokens": live_result["accepted_tokens"],
        "run_root": str(output),
    }
    outcome_sha256 = _write_capture_outcome(
        output / "capture_outcome.json", outcome)
    live_result.update({
        "replay_verdict": "COMPLETE",
        "provenance_verdict": "PASS",
        "outcome_sha256": outcome_sha256,
    })
    return live_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a frozen binding")
    validate.add_argument("binding")

    capture = sub.add_parser("capture", help="run one decisive direct capture")
    capture.add_argument("--binary", required=True)
    capture.add_argument("--snapshot", required=True)
    capture.add_argument("--container-manifest", required=True)
    capture.add_argument("--input", required=True)
    capture.add_argument("--run-dir", required=True)
    capture.add_argument("--id", dest="request_id", type=int, required=True)
    capture.add_argument("--topk", type=int, required=True)
    capture.add_argument("--vocab", type=int, required=True)
    capture.add_argument("--draft", type=int, default=1)
    capture.add_argument("--ctx", type=int, default=4096)

    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_binding(args.binding)
            print(f"[native-mtp] REPLAY binding={result['binding_id']} "
                  f"request={result['request_id']} "
                  f"accepted={len(result['accepted_tokens'])}")
            print("[native-mtp] INCOMPLETE: offline replay cannot establish "
                  "common-run provenance", file=sys.stderr)
            return 1
        result = capture_bundle(
            args.binary, args.snapshot, args.container_manifest, args.input,
            args.run_dir, args.request_id, args.topk, args.vocab,
            args.draft, args.ctx)
        print(f"[native-mtp] PASS provenance=live-capture "
              f"binding={result['binding_id']} "
              f"request={result['request_id']} "
              f"accepted={len(result['accepted_tokens'])} "
              f"outcome_sha256={result['outcome_sha256']}")
        return 0
    except EngineWitnessUnsupported as exc:
        print(f"[native-mtp] UNSUPPORTED: {exc}", file=sys.stderr)
        return 3
    except (OSError, WitnessError) as exc:
        print(f"[native-mtp] INCOMPLETE: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
